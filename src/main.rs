mod arith_encode;
mod delta_indices;
mod index_stream;
mod matrix_io;
mod mst_codec;
mod sorted_indices;

use clap::{Args, Parser, Subcommand, ValueEnum};
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use index_stream::IndexStreamCodec;
use matrix_io::{load_10x_no_positions, load_10x_with_positions, InputPosType, Platform};
use mimalloc::MiMalloc;
use mst_codec::{
    encode_subarray_column, encode_subarray_mst_with_metric, DatalessPoint, EncodedClusterBlock,
    EncodedColumnBlock, EncodedDiffsMST, KnnDistanceMetric, MstWeightMode, Point,
};
use ndarray::Array2;
use rayon::prelude::*;
use sorted_indices::SortedIndexCodec;
use sprs::CsMat;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use tracing::{info, warn};
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::fmt;
use tracing_subscriber::prelude::*;
use tracing_subscriber::EnvFilter;

use linfa::prelude::{Fit, Predict};
use linfa::DatasetBase;
use linfa_clustering::KMeans;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Clone)]
struct CompressionResult {
    quantizers_requested: usize,
    quantizers_used: usize,
    quantizer_bins: usize,
    total_mst_bytes: usize,
    gzip_bytes_estimate: usize,
    rate_bits_per_value: f64,
    mse: f64,
    rmse: f64,
    cluster_sizes: Vec<usize>,
    encoded_blocks: Vec<EncodedClusterBlock>,
    positions: Vec<DatalessPoint>,
    row_order: Vec<u32>,
    gene_order: Vec<u32>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum KnnMetricArg {
    L0,
    L2,
    Hamming,
    Jaccard,
}

impl From<KnnMetricArg> for KnnDistanceMetric {
    fn from(value: KnnMetricArg) -> Self {
        match value {
            KnnMetricArg::L0 => KnnDistanceMetric::L0,
            KnnMetricArg::L2 => KnnDistanceMetric::L2,
            KnnMetricArg::Hamming => KnnDistanceMetric::Hamming,
            KnnMetricArg::Jaccard => KnnDistanceMetric::Jaccard,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum IndexCodecArg {
    Arithmetic,
    StreamVbyte,
}

impl From<IndexCodecArg> for IndexStreamCodec {
    fn from(value: IndexCodecArg) -> Self {
        match value {
            IndexCodecArg::Arithmetic => IndexStreamCodec::Arithmetic,
            IndexCodecArg::StreamVbyte => IndexStreamCodec::StreamVByte,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum SortedIndexCodecArg {
    Delta,
    EliasFano,
}

impl From<SortedIndexCodecArg> for SortedIndexCodec {
    fn from(value: SortedIndexCodecArg) -> Self {
        match value {
            SortedIndexCodecArg::Delta => SortedIndexCodec::Delta,
            SortedIndexCodecArg::EliasFano => SortedIndexCodec::EliasFano,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum MstWeightArg {
    Metric,
    EncodingCost,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ClusterEncodingArg {
    Row,
    Column,
    Hybrid,
}

impl From<MstWeightArg> for MstWeightMode {
    fn from(value: MstWeightArg) -> Self {
        match value {
            MstWeightArg::Metric => MstWeightMode::Metric,
            MstWeightArg::EncodingCost => MstWeightMode::EncodingCost,
        }
    }
}

#[derive(Debug, Clone)]
struct SweepMetric {
    quantizers_requested: usize,
    quantizers_used: usize,
    gzip_bytes_estimate: usize,
    rate_bits_per_value: f64,
    mse: f64,
    rmse: f64,
}

fn normalize_quantizer_counts(selected: usize, sweep: &[usize]) -> Vec<usize> {
    let selected = selected.max(1);
    let mut counts = Vec::new();

    if sweep.is_empty() {
        counts.push(selected);
        return counts;
    }

    for &value in sweep {
        let normalized = value.max(1);
        if !counts.contains(&normalized) {
            counts.push(normalized);
        }
    }

    if !counts.contains(&selected) {
        counts.push(selected);
    }

    counts
}

fn build_cell_features(
    points: &[Point],
    data: &CsMat<u16>,
    feature_dims: usize,
) -> anyhow::Result<Array2<f64>> {
    let feature_dims = feature_dims.max(8);
    let mut features = Array2::<f64>::zeros((points.len(), feature_dims));

    for (point_idx, point) in points.iter().enumerate() {
        let row = data
            .outer_view(point.row_index)
            .ok_or_else(|| anyhow::anyhow!("Missing CSR row {}", point.row_index))?;
        for (gene_idx, &value) in row.iter() {
            let bucket = gene_idx % feature_dims;
            features[(point_idx, bucket)] += (value as f64).ln_1p();
        }
    }

    for row_idx in 0..features.nrows() {
        let mut norm_sq = 0.0;
        for col_idx in 0..features.ncols() {
            let v = features[(row_idx, col_idx)];
            norm_sq += v * v;
        }
        let norm = norm_sq.sqrt();
        if norm > 0.0 {
            for col_idx in 0..features.ncols() {
                features[(row_idx, col_idx)] /= norm;
            }
        }
    }

    Ok(features)
}

fn cluster_cells_with_kmeans(
    features: &Array2<f64>,
    num_clusters: usize,
) -> anyhow::Result<Vec<usize>> {
    let nrows = features.nrows();
    if nrows == 0 {
        return Ok(Vec::new());
    }

    let k = num_clusters.max(1).min(nrows);
    let dataset = DatasetBase::from(features.clone());
    let model = KMeans::params(k)
        .max_n_iterations(20)
        .fit(&dataset)
        .map_err(|e| anyhow::anyhow!("k-means clustering failed: {}", e))?;

    Ok(model.predict(&dataset).to_vec())
}

fn group_points_by_cluster(assignments: &[usize], num_clusters: usize) -> Vec<Vec<usize>> {
    let mut clusters = vec![Vec::new(); num_clusters];
    for (point_idx, &cluster_id) in assignments.iter().enumerate() {
        if cluster_id < num_clusters {
            clusters[cluster_id].push(point_idx);
        }
    }
    clusters.retain(|cluster| !cluster.is_empty());
    clusters
}

fn slice_feature_rows(features: &Array2<f64>, row_ids: &[usize]) -> Array2<f64> {
    let mut sliced = Array2::<f64>::zeros((row_ids.len(), features.ncols()));
    for (dst_row, &src_row) in row_ids.iter().enumerate() {
        for col in 0..features.ncols() {
            sliced[(dst_row, col)] = features[(src_row, col)];
        }
    }
    sliced
}

fn split_oversized_clusters(
    initial_clusters: Vec<Vec<usize>>,
    features: &Array2<f64>,
    max_cluster_size: Option<usize>,
) -> anyhow::Result<Vec<Vec<usize>>> {
    let Some(max_cluster_size_raw) = max_cluster_size else {
        return Ok(initial_clusters);
    };
    let max_cluster_size = max_cluster_size_raw.max(1);

    let mut pending = initial_clusters;
    let mut final_clusters = Vec::new();

    while let Some(cluster) = pending.pop() {
        if cluster.len() <= max_cluster_size {
            final_clusters.push(cluster);
            continue;
        }

        let split_k = ((cluster.len() + max_cluster_size - 1) / max_cluster_size)
            .max(2)
            .min(cluster.len());

        info!(
            "Re-clustering oversized cluster: size={} target_subclusters={} max_cluster_size={}",
            cluster.len(),
            split_k,
            max_cluster_size
        );

        let subfeatures = slice_feature_rows(features, &cluster);
        let sub_assignments = match cluster_cells_with_kmeans(&subfeatures, split_k) {
            Ok(a) => a,
            Err(err) => {
                warn!(
                    "Sub-clustering failed ({}). Falling back to deterministic chunk split.",
                    err
                );
                let mut offset = 0usize;
                while offset < cluster.len() {
                    let end = (offset + max_cluster_size).min(cluster.len());
                    final_clusters.push(cluster[offset..end].to_vec());
                    offset = end;
                }
                continue;
            }
        };

        let local_clusters = group_points_by_cluster(&sub_assignments, split_k);
        let max_local_size = local_clusters.iter().map(|c| c.len()).max().unwrap_or(0);
        if local_clusters.len() <= 1 || max_local_size == cluster.len() {
            warn!(
                "Sub-clustering made no progress (size={}): using deterministic chunk split.",
                cluster.len()
            );
            let mut offset = 0usize;
            while offset < cluster.len() {
                let end = (offset + max_cluster_size).min(cluster.len());
                final_clusters.push(cluster[offset..end].to_vec());
                offset = end;
            }
            continue;
        }

        for local_cluster in local_clusters {
            let mapped: Vec<usize> = local_cluster
                .into_iter()
                .map(|local_idx| cluster[local_idx])
                .collect();
            if mapped.len() > max_cluster_size {
                pending.push(mapped);
            } else {
                final_clusters.push(mapped);
            }
        }
    }

    final_clusters.sort_unstable_by_key(|cluster| cluster[0]);
    Ok(final_clusters)
}

fn projection_weight(col: usize, seed: u64) -> f64 {
    let mut z = (col as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(seed);
    z ^= z >> 30;
    z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z ^= z >> 27;
    z = z.wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    let unit = (z as f64) / (u64::MAX as f64);
    (unit * 2.0) - 1.0
}

fn reorder_rows_within_clusters(
    clusters: &[Vec<usize>],
    features: &Array2<f64>,
) -> Vec<Vec<usize>> {
    let ncols = features.ncols();
    let w1: Vec<f64> = (0..ncols)
        .map(|c| projection_weight(c, 0x1234_5678_9ABC_DEF0))
        .collect();
    let w2: Vec<f64> = (0..ncols)
        .map(|c| projection_weight(c, 0x0FED_CBA9_8765_4321))
        .collect();

    clusters
        .par_iter()
        .map(|cluster| {
            let mut scored = Vec::with_capacity(cluster.len());
            for &point_idx in cluster {
                let mut p1 = 0.0f64;
                let mut p2 = 0.0f64;
                for col in 0..ncols {
                    let v = features[(point_idx, col)];
                    p1 += v * w1[col];
                    p2 += v * w2[col];
                }
                scored.push((point_idx, p1, p2));
            }

            scored.sort_unstable_by(|a, b| {
                a.1.total_cmp(&b.1)
                    .then_with(|| a.2.total_cmp(&b.2))
                    .then_with(|| a.0.cmp(&b.0))
            });
            scored.into_iter().map(|(idx, _, _)| idx).collect()
        })
        .collect()
}

fn compute_gene_permutation_from_row_stream(
    points: &[Point],
    data: &CsMat<u16>,
    row_stream_point_indices: &[usize],
) -> anyhow::Result<(Vec<u32>, Vec<u32>)> {
    let ncols = data.cols();
    let mut counts = vec![0u64; ncols];
    let mut sum_positions = vec![0u128; ncols];
    let mut sig_a = vec![0f64; ncols];
    let mut sig_b = vec![0f64; ncols];
    let mut sig_c = vec![0f64; ncols];
    let mut sig_d = vec![0f64; ncols];

    for (stream_pos, &point_idx) in row_stream_point_indices.iter().enumerate() {
        let point = points
            .get(point_idx)
            .ok_or_else(|| anyhow::anyhow!("Invalid point index in row stream: {}", point_idx))?;

        let pa = projection_weight(stream_pos, 0xA5A5_A5A5_A5A5_A5A5);
        let pb = projection_weight(stream_pos, 0xC3C3_C3C3_C3C3_C3C3);
        let pc = projection_weight(stream_pos, 0x5A5A_5A5A_5A5A_5A5A);
        let pd = projection_weight(stream_pos, 0x3C3C_3C3C_3C3C_3C3C);

        if let Some(row) = data.outer_view(point.row_index) {
            for (gene_idx, &value) in row.iter() {
                if value == 0 {
                    continue;
                }
                counts[gene_idx] += 1;
                sum_positions[gene_idx] += stream_pos as u128;
                sig_a[gene_idx] += pa;
                sig_b[gene_idx] += pb;
                sig_c[gene_idx] += pc;
                sig_d[gene_idx] += pd;
            }
        }
    }

    let mut new_to_old: Vec<usize> = (0..ncols).collect();
    new_to_old.sort_unstable_by(|&ga, &gb| {
        let ca = counts[ga];
        let cb = counts[gb];
        match (ca == 0, cb == 0) {
            (true, true) => ga.cmp(&gb),
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => {
                let na1 = sig_a[ga] / ca as f64;
                let nb1 = sig_a[gb] / cb as f64;
                let na2 = sig_b[ga] / ca as f64;
                let nb2 = sig_b[gb] / cb as f64;
                let na3 = sig_c[ga] / ca as f64;
                let nb3 = sig_c[gb] / cb as f64;
                let na4 = sig_d[ga] / ca as f64;
                let nb4 = sig_d[gb] / cb as f64;
                let left = sum_positions[ga].saturating_mul(cb as u128);
                let right = sum_positions[gb].saturating_mul(ca as u128);
                na1.total_cmp(&nb1)
                    .then_with(|| na2.total_cmp(&nb2))
                    .then_with(|| na3.total_cmp(&nb3))
                    .then_with(|| na4.total_cmp(&nb4))
                    .then_with(|| left.cmp(&right))
                    .then_with(|| cb.cmp(&ca))
                    .then_with(|| ga.cmp(&gb))
            }
        }
    });

    let mut old_to_new = vec![0u32; ncols];
    for (new_idx, &old_idx) in new_to_old.iter().enumerate() {
        old_to_new[old_idx] = new_idx as u32;
    }

    Ok((
        new_to_old.into_iter().map(|g| g as u32).collect(),
        old_to_new,
    ))
}

fn downsample_values(values: &[u16], max_values: usize) -> Vec<u16> {
    if values.len() <= max_values {
        return values.to_vec();
    }
    let step = ((values.len() as f64) / (max_values as f64)).ceil() as usize;
    values.iter().step_by(step.max(1)).copied().collect()
}

fn nearest_center(value: u16, centers: &[u16]) -> u16 {
    let mut best = centers[0];
    let mut best_dist = value.abs_diff(best);
    for &center in centers.iter().skip(1) {
        let dist = value.abs_diff(center);
        if dist < best_dist {
            best = center;
            best_dist = dist;
        }
    }
    best
}

fn train_quantizer_with_kmeans(values: &[u16], bins: usize) -> anyhow::Result<Vec<u16>> {
    if values.is_empty() {
        return Ok(vec![0]);
    }

    let mut unique = values.to_vec();
    unique.sort_unstable();
    unique.dedup();

    let capped_bins = bins.max(1).min(unique.len());
    if capped_bins == unique.len() {
        return Ok(unique);
    }

    let sampled = downsample_values(values, 200_000);
    let mut samples = Array2::<f64>::zeros((sampled.len(), 1));
    for (i, &value) in sampled.iter().enumerate() {
        samples[(i, 0)] = value as f64;
    }

    let dataset = DatasetBase::from(samples);
    let model = KMeans::params(capped_bins)
        .max_n_iterations(40)
        .fit(&dataset)
        .map_err(|e| anyhow::anyhow!("k-means quantizer training failed: {}", e))?;

    let mut centers: Vec<u16> = model
        .centroids()
        .column(0)
        .iter()
        .map(|&c| c.round().clamp(0.0, u16::MAX as f64) as u16)
        .collect();

    centers.sort_unstable();
    centers.dedup();
    if centers.is_empty() {
        centers.push(0);
    }

    Ok(centers)
}

fn quantize_matrix_by_cluster(
    data: &CsMat<u16>,
    points: &[Point],
    cluster_assignments: &[usize],
    num_clusters: usize,
    bins: usize,
) -> anyhow::Result<CsMat<u16>> {
    let mut row_to_cluster = vec![usize::MAX; data.rows()];
    for (point_idx, point) in points.iter().enumerate() {
        let cluster_id = *cluster_assignments
            .get(point_idx)
            .ok_or_else(|| anyhow::anyhow!("Missing cluster assignment for point {}", point_idx))?;
        if cluster_id >= num_clusters {
            anyhow::bail!(
                "Invalid cluster assignment {} for point {}",
                cluster_id,
                point_idx
            );
        }
        row_to_cluster[point.row_index] = cluster_id;
    }

    let indptr_binding = data.indptr();
    let indptr = indptr_binding.raw_storage();
    let indices = data.indices().to_vec();
    let values = data.data();

    let mut values_by_cluster: Vec<Vec<u16>> = vec![Vec::new(); num_clusters];
    for row_idx in 0..data.rows() {
        let cluster_id = row_to_cluster[row_idx];
        if cluster_id == usize::MAX {
            continue;
        }
        for pos in indptr[row_idx]..indptr[row_idx + 1] {
            values_by_cluster[cluster_id].push(values[pos]);
        }
    }

    let mut quantizers = Vec::with_capacity(num_clusters);
    for cluster_values in &values_by_cluster {
        quantizers.push(train_quantizer_with_kmeans(cluster_values, bins)?);
    }

    let mut quantized_values = values.to_vec();
    for row_idx in 0..data.rows() {
        let cluster_id = row_to_cluster[row_idx];
        if cluster_id == usize::MAX {
            continue;
        }
        let centers = &quantizers[cluster_id];
        for pos in indptr[row_idx]..indptr[row_idx + 1] {
            quantized_values[pos] = nearest_center(quantized_values[pos], centers);
        }
    }

    Ok(CsMat::new(
        (data.rows(), data.cols()),
        indptr.to_vec(),
        indices,
        quantized_values,
    ))
}

fn sparse_quantization_sse(original: &CsMat<u16>, quantized: &CsMat<u16>) -> anyhow::Result<f64> {
    if original.shape() != quantized.shape() {
        anyhow::bail!("Shape mismatch between original and quantized matrix");
    }
    if original.nnz() != quantized.nnz() {
        anyhow::bail!("NNZ mismatch between original and quantized matrix");
    }

    let sse = original
        .data()
        .iter()
        .zip(quantized.data().iter())
        .map(|(&orig, &quant)| {
            let diff = orig as f64 - quant as f64;
            diff * diff
        })
        .sum();

    Ok(sse)
}

fn estimate_gzip_payload_size(
    encoded_blocks: &[EncodedClusterBlock],
    positions: &[DatalessPoint],
    row_order: &[u32],
    gene_order: &[u32],
) -> anyhow::Result<usize> {
    let config = bincode::config::standard()
        .with_little_endian()
        .with_fixed_int_encoding();
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    bincode::encode_into_std_write(encoded_blocks, &mut encoder, config)?;
    bincode::encode_into_std_write(positions, &mut encoder, config)?;
    bincode::encode_into_std_write(row_order, &mut encoder, config)?;
    bincode::encode_into_std_write(gene_order, &mut encoder, config)?;
    Ok(encoder.finish()?.len())
}

#[derive(Default)]
struct SymbolHistogram {
    counts: HashMap<u32, u64>,
    total: u64,
}

impl SymbolHistogram {
    fn add_slice(&mut self, values: &[u32]) {
        self.total += values.len() as u64;
        for &value in values {
            *self.counts.entry(value).or_insert(0) += 1;
        }
    }

    fn entropy_bits(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        let total = self.total as f64;
        let mut bits = 0.0;
        for &count in self.counts.values() {
            let p = (count as f64) / total;
            bits -= (count as f64) * p.log2();
        }
        bits
    }
}

#[derive(Default)]
struct IndexBoundsReport {
    actual_bits_total: f64,
    actual_bits_row: f64,
    actual_bits_column: f64,
    lower_bound_bits_total: f64,
    lower_bound_bits_row: f64,
    lower_bound_bits_column: f64,
    invalid_constraints: u64,
    symbols: SymbolHistogram,
}

#[derive(Clone, Copy)]
enum ColumnSupportPlan {
    Empty,
    Raw {
        count: usize,
    },
    Ref {
        parent_idx: usize,
        remove_count: usize,
        add_count: usize,
    },
    Template {
        template_idx: usize,
        remove_count: usize,
        add_count: usize,
    },
}

fn build_log2_factorial(max_n: usize) -> Vec<f64> {
    let mut table = vec![0.0f64; max_n + 1];
    for i in 1..=max_n {
        table[i] = table[i - 1] + (i as f64).log2();
    }
    table
}

fn log2_choose(log2_fact: &[f64], n: usize, k: usize, invalid_constraints: &mut u64) -> f64 {
    if k > n || n >= log2_fact.len() {
        *invalid_constraints += 1;
        return 0.0;
    }
    log2_fact[n] - log2_fact[k] - log2_fact[n - k]
}

fn zigzag_decode_i64(v: u32) -> i64 {
    ((v >> 1) as i64) ^ (-((v & 1) as i64))
}

fn resolve_column_support_size(
    idx: usize,
    plans: &[ColumnSupportPlan],
    template_sizes: &[usize],
    memo: &mut [Option<usize>],
    visiting: &mut [bool],
    universe: usize,
    invalid_constraints: &mut u64,
) -> usize {
    if let Some(value) = memo.get(idx).and_then(|v| *v) {
        return value;
    }
    if idx >= plans.len() {
        *invalid_constraints += 1;
        return 0;
    }
    if visiting[idx] {
        *invalid_constraints += 1;
        return 0;
    }

    visiting[idx] = true;
    let value = match plans[idx] {
        ColumnSupportPlan::Empty => 0,
        ColumnSupportPlan::Raw { count } => count.min(universe),
        ColumnSupportPlan::Ref {
            parent_idx,
            remove_count,
            add_count,
        } => {
            let parent = if parent_idx == idx {
                *invalid_constraints += 1;
                0
            } else {
                resolve_column_support_size(
                    parent_idx,
                    plans,
                    template_sizes,
                    memo,
                    visiting,
                    universe,
                    invalid_constraints,
                )
            };
            let remove = remove_count.min(parent);
            let add = add_count.min(universe.saturating_sub(parent));
            parent.saturating_sub(remove) + add
        }
        ColumnSupportPlan::Template {
            template_idx,
            remove_count,
            add_count,
        } => {
            let template_support = template_sizes
                .get(template_idx)
                .copied()
                .unwrap_or_else(|| {
                    *invalid_constraints += 1;
                    0
                });
            let template_support = template_support.min(universe);
            let remove = remove_count.min(template_support);
            let add = add_count.min(universe.saturating_sub(template_support));
            template_support.saturating_sub(remove) + add
        }
    };
    visiting[idx] = false;
    memo[idx] = Some(value);
    value
}

fn analyze_row_block_index_bounds(block: &EncodedDiffsMST, report: &mut IndexBoundsReport) {
    const ROW_MODE_PARENT: u32 = 0;
    const ROW_MODE_FULL: u32 = 1;
    const ROW_MODE_TEMPLATE: u32 = 2;

    let (_, _, _, op_index_bytes, _, _) = block.bytes_breakdown();
    let actual_bits = (op_index_bytes as f64) * 8.0;
    report.actual_bits_total += actual_bits;
    report.actual_bits_row += actual_bits;

    let child_modes = block.child_modes.decode_all().unwrap_or_default();
    let full_counts = block.child_full_counts.decode_all().unwrap_or_default();
    let remove_counts = block.child_remove_counts.decode_all().unwrap_or_default();
    let add_counts = block.child_add_counts.decode_all().unwrap_or_default();
    let update_counts = block.child_update_counts.decode_all().unwrap_or_default();
    let row_template_counts = block.row_template_counts.decode_all().unwrap_or_default();
    let child_template_ids = block.child_template_ids.decode_all();

    report.symbols.add_slice(&child_modes);
    report.symbols.add_slice(&full_counts);
    report.symbols.add_slice(&remove_counts);
    report.symbols.add_slice(&add_counts);
    report.symbols.add_slice(&update_counts);
    report.symbols.add_slice(&row_template_counts);
    report
        .symbols
        .add_slice(&block.row_template_first_genes.decode_all());
    report
        .symbols
        .add_slice(&block.row_template_gene_gaps.decode_all());
    report.symbols.add_slice(&child_template_ids);
    report
        .symbols
        .add_slice(&block.child_full_first_genes.decode_all());
    report
        .symbols
        .add_slice(&block.child_full_gene_gaps.decode_all());
    report
        .symbols
        .add_slice(&block.child_remove_first_genes.decode_all());
    report
        .symbols
        .add_slice(&block.child_remove_gene_gaps.decode_all());
    report
        .symbols
        .add_slice(&block.child_add_first_genes.decode_all());
    report
        .symbols
        .add_slice(&block.child_add_gene_gaps.decode_all());
    report
        .symbols
        .add_slice(&block.child_update_first_genes.decode_all());
    report
        .symbols
        .add_slice(&block.child_update_gene_gaps.decode_all());

    let universe = block.num_genes as usize;
    let log2_fact = build_log2_factorial(universe);
    let parent_offsets = block.parent_offset.decode_all().unwrap_or_default();
    if parent_offsets.is_empty() {
        return;
    }

    let mut support_sizes = vec![0usize; parent_offsets.len()];
    support_sizes[0] = block.root_indices.decode_all_u32().len().min(universe);
    let template_support_sizes: Vec<usize> = row_template_counts
        .iter()
        .map(|&c| (c as usize).min(universe))
        .collect();
    let mut template_id_cursor = 0usize;

    let mut lb_bits = 0.0f64;
    for dfs_pos in 1..support_sizes.len() {
        let parent_offset = parent_offsets.get(dfs_pos).copied().unwrap_or(0) as usize;
        let parent_pos = dfs_pos.saturating_sub(parent_offset);
        let parent_support = support_sizes.get(parent_pos).copied().unwrap_or(0);
        let child_idx = dfs_pos - 1;
        let mode = child_modes
            .get(child_idx)
            .copied()
            .unwrap_or(ROW_MODE_PARENT);

        if mode == ROW_MODE_FULL {
            let full_count = full_counts.get(child_idx).copied().unwrap_or(0) as usize;
            lb_bits += log2_choose(
                &log2_fact,
                universe,
                full_count,
                &mut report.invalid_constraints,
            );
            support_sizes[dfs_pos] = full_count.min(universe);
            continue;
        }

        let remove_count = remove_counts.get(child_idx).copied().unwrap_or(0) as usize;
        let add_count = add_counts.get(child_idx).copied().unwrap_or(0) as usize;
        let update_count = update_counts.get(child_idx).copied().unwrap_or(0) as usize;
        let base_support = if mode == ROW_MODE_TEMPLATE {
            let template_idx_raw = child_template_ids
                .get(template_id_cursor)
                .copied()
                .unwrap_or(0);
            template_id_cursor += 1;
            template_support_sizes
                .get(template_idx_raw as usize)
                .copied()
                .unwrap_or_else(|| {
                    report.invalid_constraints += 1;
                    parent_support
                })
        } else {
            parent_support
        };

        lb_bits += log2_choose(
            &log2_fact,
            base_support,
            remove_count,
            &mut report.invalid_constraints,
        );
        lb_bits += log2_choose(
            &log2_fact,
            universe.saturating_sub(parent_support),
            add_count,
            &mut report.invalid_constraints,
        );
        let common_support = base_support.saturating_sub(remove_count.min(base_support));
        lb_bits += log2_choose(
            &log2_fact,
            common_support,
            update_count,
            &mut report.invalid_constraints,
        );

        let child_support = base_support
            .saturating_sub(remove_count.min(base_support))
            .saturating_add(add_count.min(universe.saturating_sub(base_support)));
        support_sizes[dfs_pos] = child_support.min(universe);
    }

    report.lower_bound_bits_total += lb_bits;
    report.lower_bound_bits_row += lb_bits;
}

fn analyze_column_block_index_bounds(block: &EncodedColumnBlock, report: &mut IndexBoundsReport) {
    const MODE_RAW: u32 = 0;
    const MODE_REF: u32 = 1;
    const MODE_TEMPLATE: u32 = 2;

    let (_, _, _, op_index_bytes, _, _) = block.bytes_breakdown();
    let actual_bits = (op_index_bytes as f64) * 8.0;
    report.actual_bits_total += actual_bits;
    report.actual_bits_column += actual_bits;

    let raw_firsts = block.raw_row_firsts.decode_all();
    let raw_gaps = block.raw_row_gaps.decode_all();
    let template_row_firsts = block.template_row_firsts.decode_all();
    let template_row_gaps = block.template_row_gaps.decode_all();
    let ref_parent_deltas = block.ref_parent_deltas.decode_all();
    let ref_remove_firsts = block.ref_remove_firsts.decode_all();
    let ref_remove_gaps = block.ref_remove_gaps.decode_all();
    let ref_add_firsts = block.ref_add_firsts.decode_all();
    let ref_add_gaps = block.ref_add_gaps.decode_all();
    let template_ids = block.template_ids.decode_all();
    let template_remove_firsts = block.template_remove_firsts.decode_all();
    let template_remove_gaps = block.template_remove_gaps.decode_all();
    let template_add_firsts = block.template_add_firsts.decode_all();
    let template_add_gaps = block.template_add_gaps.decode_all();

    report.symbols.add_slice(&raw_firsts);
    report.symbols.add_slice(&raw_gaps);
    report.symbols.add_slice(&template_row_firsts);
    report.symbols.add_slice(&template_row_gaps);
    report.symbols.add_slice(&ref_parent_deltas);
    report.symbols.add_slice(&ref_remove_firsts);
    report.symbols.add_slice(&ref_remove_gaps);
    report.symbols.add_slice(&ref_add_firsts);
    report.symbols.add_slice(&ref_add_gaps);
    report.symbols.add_slice(&template_ids);
    report.symbols.add_slice(&template_remove_firsts);
    report.symbols.add_slice(&template_remove_gaps);
    report.symbols.add_slice(&template_add_firsts);
    report.symbols.add_slice(&template_add_gaps);

    let num_genes = block.local_to_global.decode_all_u32().len();
    if num_genes == 0 {
        return;
    }

    let modes = block.posting_modes.decode_all().unwrap_or_default();
    let counts = block.posting_counts.decode_all().unwrap_or_default();
    let template_counts = block.template_counts.decode_all().unwrap_or_default();
    let ref_remove_counts = block.ref_remove_counts.decode_all().unwrap_or_default();
    let ref_add_counts = block.ref_add_counts.decode_all().unwrap_or_default();
    let template_remove_counts = block
        .template_remove_counts
        .decode_all()
        .unwrap_or_default();
    let template_add_counts = block.template_add_counts.decode_all().unwrap_or_default();

    let universe = block.num_cells as usize;
    let template_sizes: Vec<usize> = template_counts
        .iter()
        .map(|&count| (count as usize).min(universe))
        .collect();

    let mut raw_first_cursor = 0usize;
    let mut raw_gap_cursor = 0usize;
    let mut template_row_first_cursor = 0usize;
    let mut template_row_gap_cursor = 0usize;
    let mut ref_parent_cursor = 0usize;
    let mut ref_count_cursor = 0usize;
    let mut ref_remove_first_cursor = 0usize;
    let mut ref_remove_gap_cursor = 0usize;
    let mut ref_add_first_cursor = 0usize;
    let mut ref_add_gap_cursor = 0usize;
    let mut template_ref_cursor = 0usize;
    let mut template_ref_count_cursor = 0usize;
    let mut template_remove_first_cursor = 0usize;
    let mut template_remove_gap_cursor = 0usize;
    let mut template_add_first_cursor = 0usize;
    let mut template_add_gap_cursor = 0usize;

    for &count in &template_sizes {
        if count > 0 {
            template_row_first_cursor = template_row_first_cursor.saturating_add(1);
            template_row_gap_cursor = template_row_gap_cursor.saturating_add(count - 1);
        }
    }

    let mut plans = Vec::with_capacity(num_genes);
    for gene_pos in 0..num_genes {
        let count = counts.get(gene_pos).copied().unwrap_or(0) as usize;
        let mode = modes.get(gene_pos).copied().unwrap_or(MODE_RAW);

        if count == 0 {
            plans.push(ColumnSupportPlan::Empty);
            continue;
        }

        if mode == MODE_REF {
            let parent_delta = ref_parent_deltas
                .get(ref_parent_cursor)
                .copied()
                .unwrap_or(0);
            ref_parent_cursor += 1;
            let parent_idx = (gene_pos as i64)
                .checked_add(zigzag_decode_i64(parent_delta))
                .filter(|idx| *idx >= 0 && (*idx as usize) < num_genes)
                .map(|idx| idx as usize)
                .unwrap_or_else(|| {
                    report.invalid_constraints += 1;
                    gene_pos
                });
            let remove_count = ref_remove_counts
                .get(ref_count_cursor)
                .copied()
                .unwrap_or(0) as usize;
            let add_count = ref_add_counts.get(ref_count_cursor).copied().unwrap_or(0) as usize;
            ref_count_cursor += 1;

            if remove_count > 0 {
                ref_remove_first_cursor = ref_remove_first_cursor.saturating_add(1);
                ref_remove_gap_cursor =
                    ref_remove_gap_cursor.saturating_add(remove_count.saturating_sub(1));
            }
            if add_count > 0 {
                ref_add_first_cursor = ref_add_first_cursor.saturating_add(1);
                ref_add_gap_cursor = ref_add_gap_cursor.saturating_add(add_count.saturating_sub(1));
            }

            plans.push(ColumnSupportPlan::Ref {
                parent_idx,
                remove_count,
                add_count,
            });
        } else if mode == MODE_TEMPLATE {
            let template_idx_raw = template_ids.get(template_ref_cursor).copied().unwrap_or(0);
            template_ref_cursor += 1;
            let template_idx =
                (template_idx_raw as usize).min(template_sizes.len().saturating_sub(1));
            if template_sizes.is_empty() || template_idx_raw as usize >= template_sizes.len() {
                report.invalid_constraints += 1;
            }

            let remove_count = template_remove_counts
                .get(template_ref_count_cursor)
                .copied()
                .unwrap_or(0) as usize;
            let add_count = template_add_counts
                .get(template_ref_count_cursor)
                .copied()
                .unwrap_or(0) as usize;
            template_ref_count_cursor += 1;

            if remove_count > 0 {
                template_remove_first_cursor = template_remove_first_cursor.saturating_add(1);
                template_remove_gap_cursor =
                    template_remove_gap_cursor.saturating_add(remove_count.saturating_sub(1));
            }
            if add_count > 0 {
                template_add_first_cursor = template_add_first_cursor.saturating_add(1);
                template_add_gap_cursor =
                    template_add_gap_cursor.saturating_add(add_count.saturating_sub(1));
            }

            plans.push(ColumnSupportPlan::Template {
                template_idx,
                remove_count,
                add_count,
            });
        } else {
            if count > 0 {
                raw_first_cursor = raw_first_cursor.saturating_add(1);
                raw_gap_cursor = raw_gap_cursor.saturating_add(count.saturating_sub(1));
            }
            plans.push(ColumnSupportPlan::Raw { count });
        }
    }

    let log2_fact = build_log2_factorial(universe);
    let mut memo = vec![None; num_genes];
    let mut visiting = vec![false; num_genes];
    let mut support_sizes = vec![0usize; num_genes];
    for gene_pos in 0..num_genes {
        support_sizes[gene_pos] = resolve_column_support_size(
            gene_pos,
            &plans,
            &template_sizes,
            &mut memo,
            &mut visiting,
            universe,
            &mut report.invalid_constraints,
        );
    }

    let mut lb_bits = 0.0f64;
    for gene_pos in 0..num_genes {
        match plans[gene_pos] {
            ColumnSupportPlan::Empty => {}
            ColumnSupportPlan::Raw { count } => {
                lb_bits +=
                    log2_choose(&log2_fact, universe, count, &mut report.invalid_constraints);
            }
            ColumnSupportPlan::Ref {
                parent_idx,
                remove_count,
                add_count,
            } => {
                let parent_support = support_sizes
                    .get(parent_idx)
                    .copied()
                    .unwrap_or(0)
                    .min(universe);
                lb_bits += log2_choose(
                    &log2_fact,
                    parent_support,
                    remove_count,
                    &mut report.invalid_constraints,
                );
                lb_bits += log2_choose(
                    &log2_fact,
                    universe.saturating_sub(parent_support),
                    add_count,
                    &mut report.invalid_constraints,
                );
            }
            ColumnSupportPlan::Template {
                template_idx,
                remove_count,
                add_count,
            } => {
                let template_support = template_sizes
                    .get(template_idx)
                    .copied()
                    .unwrap_or(0)
                    .min(universe);
                lb_bits += log2_choose(
                    &log2_fact,
                    template_support,
                    remove_count,
                    &mut report.invalid_constraints,
                );
                lb_bits += log2_choose(
                    &log2_fact,
                    universe.saturating_sub(template_support),
                    add_count,
                    &mut report.invalid_constraints,
                );
            }
        }
    }

    report.lower_bound_bits_total += lb_bits;
    report.lower_bound_bits_column += lb_bits;
}

fn compute_index_bounds_report(encoded_blocks: &[EncodedClusterBlock]) -> IndexBoundsReport {
    let mut report = IndexBoundsReport::default();
    for block in encoded_blocks {
        match block {
            EncodedClusterBlock::RowMst(row_block) => {
                analyze_row_block_index_bounds(row_block, &mut report);
            }
            EncodedClusterBlock::Column(column_block) => {
                analyze_column_block_index_bounds(column_block, &mut report);
            }
        }
    }
    report
}

fn log_index_bounds_report(report: &IndexBoundsReport, nnz: usize) {
    let entropy_bits = report.symbols.entropy_bits();
    let symbols = report.symbols.total.max(1) as f64;
    let nnz_denom = nnz.max(1) as f64;

    let actual_over_lb = if report.lower_bound_bits_total > 0.0 {
        report.actual_bits_total / report.lower_bound_bits_total
    } else {
        0.0
    };
    let actual_over_h0 = if entropy_bits > 0.0 {
        report.actual_bits_total / entropy_bits
    } else {
        0.0
    };

    info!(
        "Index bounds report (op_indices): actual_bits={:.0} (row={:.0}, column={:.0})",
        report.actual_bits_total, report.actual_bits_row, report.actual_bits_column
    );
    info!(
        "  Support lower bound={:.0} bits (actual/lower_bound={:.3})",
        report.lower_bound_bits_total, actual_over_lb
    );
    info!(
        "  Zero-order entropy bound={:.0} bits over {} symbols ({} unique, actual/H0={:.3})",
        entropy_bits,
        report.symbols.total,
        report.symbols.counts.len(),
        actual_over_h0
    );
    info!(
        "  Bits per symbol: actual={:.3}, H0={:.3}",
        report.actual_bits_total / symbols,
        entropy_bits / symbols
    );
    info!(
        "  Bits per nnz: actual={:.3}, lower_bound={:.3}, H0={:.3}",
        report.actual_bits_total / nnz_denom,
        report.lower_bound_bits_total / nnz_denom,
        entropy_bits / nnz_denom
    );
    info!(
        "  Lower-bound split: row={:.0} bits, column={:.0} bits",
        report.lower_bound_bits_row, report.lower_bound_bits_column
    );
    if report.invalid_constraints > 0 {
        info!(
            "  Note: {} bound terms had invalid constraints and were clamped/ignored",
            report.invalid_constraints
        );
    }
}

fn run_clustered_compression(
    points: &[Point],
    data: &CsMat<u16>,
    quantizers_requested: usize,
    quantizer_bins: usize,
    quantize_values: bool,
    verify_lossless: bool,
    max_cluster_size: Option<usize>,
    knn_metric: KnnDistanceMetric,
    mst_weight_mode: MstWeightMode,
    index_codec: IndexStreamCodec,
    sorted_index_codec: SortedIndexCodec,
    full_row_fallback_ratio: Option<f32>,
    forest_cut_factor: Option<f32>,
    cluster_encoding: ClusterEncodingArg,
    column_template_count: usize,
    column_template_adaptive: bool,
    column_template_max: usize,
    row_template_adaptive: bool,
    row_template_max: usize,
) -> anyhow::Result<CompressionResult> {
    if points.is_empty() {
        anyhow::bail!("Cannot encode empty point set");
    }

    let requested_clusters = quantizers_requested.max(1).min(points.len());
    let features = build_cell_features(points, data, 24)?;
    let assignments = match cluster_cells_with_kmeans(&features, requested_clusters) {
        Ok(a) => a,
        Err(err) => {
            warn!(
                "Cell clustering failed ({}). Falling back to round-robin assignment.",
                err
            );
            (0..points.len())
                .map(|idx| idx % requested_clusters)
                .collect::<Vec<_>>()
        }
    };

    let initial_quantizers_used = assignments
        .iter()
        .copied()
        .max()
        .map(|m| m + 1)
        .unwrap_or(1);

    let initial_clusters = group_points_by_cluster(&assignments, initial_quantizers_used);
    let initial_max = initial_clusters.iter().map(|c| c.len()).max().unwrap_or(0);
    let clusters = split_oversized_clusters(initial_clusters, &features, max_cluster_size)?;
    let clusters = reorder_rows_within_clusters(&clusters, &features);
    let final_max = clusters.iter().map(|c| c.len()).max().unwrap_or(0);

    info!(
        "Cluster layout: requested={} initial={} final={} largest_initial={} largest_final={} max_cluster_size={:?}",
        quantizers_requested,
        initial_quantizers_used,
        clusters.len(),
        initial_max,
        final_max,
        max_cluster_size
    );

    let mut refined_assignments = vec![usize::MAX; points.len()];
    for (cluster_id, cluster) in clusters.iter().enumerate() {
        for &point_idx in cluster {
            refined_assignments[point_idx] = cluster_id;
        }
    }
    if refined_assignments.iter().any(|&x| x == usize::MAX) {
        anyhow::bail!("Internal error: some points were not assigned to a final cluster");
    }

    let quantized_data = if quantize_values {
        quantize_matrix_by_cluster(
            data,
            points,
            &refined_assignments,
            clusters.len(),
            quantizer_bins,
        )?
    } else {
        data.clone()
    };

    let row_stream_point_indices: Vec<usize> = clusters
        .iter()
        .flat_map(|cluster| cluster.iter().copied())
        .collect();
    if row_stream_point_indices.len() != points.len() {
        anyhow::bail!(
            "Internal error: row stream covers {} points, expected {}",
            row_stream_point_indices.len(),
            points.len()
        );
    }

    let (gene_order, gene_old_to_new) = compute_gene_permutation_from_row_stream(
        points,
        &quantized_data,
        &row_stream_point_indices,
    )?;

    let mut encoded_blocks = Vec::new();
    let mut positions = Vec::new();
    let mut row_order = Vec::new();
    let mut cluster_sizes = Vec::new();
    let mut total_mst_bytes = 0usize;

    let cluster_results: Vec<
        anyhow::Result<
            Option<(
                usize,
                EncodedClusterBlock,
                Vec<DatalessPoint>,
                Vec<u32>,
                usize,
                usize,
                bool,
                bool,
            )>,
        >,
    > = clusters
        .par_iter()
        .enumerate()
        .map(|(cluster_idx, cluster)| {
            if cluster.is_empty() {
                return Ok(None);
            }

            let cluster_points: Vec<Point> =
                cluster.iter().map(|&idx| points[idx].clone()).collect();
            let mut row_candidate: Option<(EncodedClusterBlock, Vec<u32>, usize)> = None;
            let mut col_candidate: Option<(EncodedClusterBlock, Vec<u32>, usize)> = None;
            let mut row_selected_adaptive = false;
            let mut col_selected_adaptive = false;

            if matches!(
                cluster_encoding,
                ClusterEncodingArg::Row | ClusterEncodingArg::Hybrid
            ) {
                let mut best_row: Option<(EncodedClusterBlock, Vec<u32>, usize)> = None;
                let mut try_row_encode = |adaptive: bool| {
                    if let Some((row_block, dfs_order)) = encode_subarray_mst_with_metric(
                        &cluster_points,
                        &quantized_data,
                        knn_metric,
                        mst_weight_mode,
                        Some(&gene_old_to_new),
                        index_codec,
                        sorted_index_codec,
                        full_row_fallback_ratio,
                        forest_cut_factor,
                        adaptive,
                        row_template_max,
                    ) {
                        let bytes = row_block.total_bytes();
                        if best_row.as_ref().map(|b| bytes < b.2).unwrap_or(true) {
                            best_row =
                                Some((EncodedClusterBlock::RowMst(row_block), dfs_order, bytes));
                            row_selected_adaptive = adaptive;
                        }
                    }
                };

                try_row_encode(row_template_adaptive);
                if row_template_adaptive {
                    try_row_encode(false);
                }
                row_candidate = best_row;
            }

            if matches!(
                cluster_encoding,
                ClusterEncodingArg::Column | ClusterEncodingArg::Hybrid
            ) {
                let mut best_col: Option<(EncodedClusterBlock, Vec<u32>, usize)> = None;
                let mut try_col_encode = |adaptive: bool| {
                    if let Some(col_block) = encode_subarray_column(
                        &cluster_points,
                        &quantized_data,
                        Some(&gene_old_to_new),
                        index_codec,
                        sorted_index_codec,
                        column_template_count,
                        adaptive,
                        column_template_max,
                    ) {
                        let (col_block, order) = col_block;
                        let bytes = col_block.total_bytes();
                        if best_col.as_ref().map(|b| bytes < b.2).unwrap_or(true) {
                            best_col = Some((EncodedClusterBlock::Column(col_block), order, bytes));
                            col_selected_adaptive = adaptive;
                        }
                    }
                };

                try_col_encode(column_template_adaptive);
                if column_template_adaptive {
                    try_col_encode(false);
                }
                col_candidate = best_col;
            }

            let row_available = row_candidate.is_some();
            let col_available = col_candidate.is_some();
            let (encoded, local_order, bytes) = match cluster_encoding {
                ClusterEncodingArg::Row => row_candidate.ok_or_else(|| {
                    anyhow::anyhow!("Row-MST encoding failed for cluster {}", cluster_idx)
                })?,
                ClusterEncodingArg::Column => col_candidate.ok_or_else(|| {
                    anyhow::anyhow!("Column encoding failed for cluster {}", cluster_idx)
                })?,
                ClusterEncodingArg::Hybrid => match (row_candidate, col_candidate) {
                    (Some(r), Some(c)) => {
                        if r.2 <= c.2 {
                            r
                        } else {
                            c
                        }
                    }
                    (Some(r), None) => r,
                    (None, Some(c)) => c,
                    (None, None) => {
                        return Err(anyhow::anyhow!(
                            "Both row and column encoding failed for cluster {}",
                            cluster_idx
                        ))
                    }
                },
            };

            let mut cluster_positions = Vec::with_capacity(cluster_points.len());
            let mut cluster_row_order = Vec::with_capacity(cluster_points.len());

            for &local_idx_u32 in &local_order {
                let local_idx = local_idx_u32 as usize;
                let point = cluster_points
                    .get(local_idx)
                    .ok_or_else(|| anyhow::anyhow!("Invalid local order index {}", local_idx))?;
                cluster_positions.push(DatalessPoint::new(point.x, point.y));
                cluster_row_order.push(point.row_index as u32);
            }

            Ok(Some((
                cluster_idx,
                encoded,
                cluster_positions,
                cluster_row_order,
                cluster_points.len(),
                bytes,
                row_template_adaptive && row_available && !row_selected_adaptive,
                column_template_adaptive && col_available && !col_selected_adaptive,
            )))
        })
        .collect();

    let mut ordered = Vec::new();
    for result in cluster_results {
        if let Some(v) = result? {
            ordered.push(v);
        }
    }
    ordered.sort_unstable_by_key(|(cluster_idx, _, _, _, _, _, _, _)| *cluster_idx);

    let mut row_block_count = 0usize;
    let mut column_block_count = 0usize;
    let mut row_adaptive_fallback_wins = 0usize;
    let mut col_adaptive_fallback_wins = 0usize;
    let mut col_value_model_global_blocks = 0usize;
    let mut col_value_model_per_gene_blocks = 0usize;
    for (
        _idx,
        encoded,
        cluster_positions,
        cluster_row_order,
        cluster_ncells,
        bytes,
        row_fallback_used,
        col_fallback_used,
    ) in ordered
    {
        if row_fallback_used {
            row_adaptive_fallback_wins += 1;
        }
        if col_fallback_used {
            col_adaptive_fallback_wins += 1;
        }
        match &encoded {
            EncodedClusterBlock::RowMst(_) => row_block_count += 1,
            EncodedClusterBlock::Column(block) => {
                column_block_count += 1;
                if block.uses_per_gene_values() {
                    col_value_model_per_gene_blocks += 1;
                } else {
                    col_value_model_global_blocks += 1;
                }
            }
        }
        total_mst_bytes += bytes;
        cluster_sizes.push(cluster_ncells);
        positions.extend(cluster_positions);
        row_order.extend(cluster_row_order);
        encoded_blocks.push(encoded);
    }

    info!(
        "Cluster payload selection: row_blocks={} column_blocks={}",
        row_block_count, column_block_count
    );
    if row_template_adaptive || column_template_adaptive {
        info!(
            "Adaptive fallback selection: row_nonadaptive_wins={} column_nonadaptive_wins={}",
            row_adaptive_fallback_wins, col_adaptive_fallback_wins
        );
    }
    if column_block_count > 0 {
        info!(
            "Column value model selection: global_blocks={} per_gene_blocks={}",
            col_value_model_global_blocks, col_value_model_per_gene_blocks
        );
    }

    if verify_lossless && !quantize_values {
        let reconstructed = reconstruct_csr_from_clustered_payload(
            &encoded_blocks,
            &row_order,
            &gene_order,
            data.rows(),
            data.cols(),
        )?;
        compare_csr_exact(data, &reconstructed)?;
    }

    let sse = if quantize_values {
        sparse_quantization_sse(data, &quantized_data)?
    } else {
        0.0
    };
    let total_entries = (points.len() * data.cols()).max(1);
    let mse = sse / total_entries as f64;
    let rmse = mse.sqrt();
    let gzip_bytes_estimate =
        estimate_gzip_payload_size(&encoded_blocks, &positions, &row_order, &gene_order)?;
    let rate_bits_per_value = (gzip_bytes_estimate as f64 * 8.0) / total_entries as f64;

    Ok(CompressionResult {
        quantizers_requested,
        quantizers_used: clusters.len(),
        quantizer_bins,
        total_mst_bytes,
        gzip_bytes_estimate,
        rate_bits_per_value,
        mse,
        rmse,
        cluster_sizes,
        encoded_blocks,
        positions,
        row_order,
        gene_order,
    })
}

fn decode_clustered_payload(
    input: &Path,
) -> anyhow::Result<(
    Vec<EncodedClusterBlock>,
    Vec<DatalessPoint>,
    Vec<u32>,
    Vec<u32>,
)> {
    let file = File::open(input)?;
    let mut reader = BufReader::new(file);
    let gz = GzDecoder::new(&mut reader);
    let mut gz_reader = BufReader::new(gz);
    let config = bincode::config::standard()
        .with_little_endian()
        .with_fixed_int_encoding();

    let encoded_blocks: Vec<EncodedClusterBlock> =
        bincode::decode_from_std_read(&mut gz_reader, config)?;
    let positions: Vec<DatalessPoint> = bincode::decode_from_std_read(&mut gz_reader, config)?;
    let row_order: Vec<u32> = bincode::decode_from_std_read(&mut gz_reader, config)?;
    let gene_order: Vec<u32> = bincode::decode_from_std_read(&mut gz_reader, config)?;
    Ok((encoded_blocks, positions, row_order, gene_order))
}

fn reconstruct_csr_from_clustered_payload(
    encoded_blocks: &[EncodedClusterBlock],
    row_order: &[u32],
    gene_order: &[u32],
    nrows: usize,
    ncols: usize,
) -> anyhow::Result<CsMat<u16>> {
    let mut tri = sprs::TriMatI::<u16, usize>::new((nrows, ncols));
    let mut cursor = 0usize;

    for block in encoded_blocks {
        for sparse_row in block.decode_rows() {
            let row_idx = *row_order
                .get(cursor)
                .ok_or_else(|| anyhow::anyhow!("Row-order mapping shorter than decoded rows"))?
                as usize;
            if row_idx >= nrows {
                anyhow::bail!(
                    "Row-order entry {} out of bounds for {} rows",
                    row_idx,
                    nrows
                );
            }
            for (gene_idx, value) in sparse_row {
                let mapped_gene = if gene_order.is_empty() {
                    gene_idx as usize
                } else {
                    *gene_order.get(gene_idx as usize).ok_or_else(|| {
                        anyhow::anyhow!(
                            "Gene-order mapping missing index {} (len={})",
                            gene_idx,
                            gene_order.len()
                        )
                    })? as usize
                };
                if mapped_gene >= ncols {
                    anyhow::bail!(
                        "Mapped gene index {} out of bounds for {} columns",
                        mapped_gene,
                        ncols
                    );
                }
                tri.add_triplet(row_idx, mapped_gene, value);
            }
            cursor += 1;
        }
    }

    if cursor != row_order.len() {
        anyhow::bail!(
            "Decoded row count ({}) differs from row-order length ({})",
            cursor,
            row_order.len()
        );
    }

    Ok(tri.to_csr::<usize>())
}

fn compare_csr_exact(expected: &CsMat<u16>, actual: &CsMat<u16>) -> anyhow::Result<()> {
    if expected.shape() != actual.shape() {
        anyhow::bail!(
            "Shape mismatch: expected {}x{}, actual {}x{}",
            expected.rows(),
            expected.cols(),
            actual.rows(),
            actual.cols()
        );
    }

    for row_idx in 0..expected.rows() {
        let exp = expected.outer_view(row_idx);
        let act = actual.outer_view(row_idx);
        match (exp, act) {
            (None, None) => {}
            (Some(e), Some(a)) => {
                if e.nnz() != a.nnz() {
                    anyhow::bail!(
                        "Row {} nnz mismatch: expected {}, actual {}",
                        row_idx,
                        e.nnz(),
                        a.nnz()
                    );
                }

                let mut e_vals: Vec<(usize, u16)> = e.iter().map(|(i, &v)| (i, v)).collect();
                let mut a_vals: Vec<(usize, u16)> = a.iter().map(|(i, &v)| (i, v)).collect();
                e_vals.sort_unstable_by_key(|(i, _)| *i);
                a_vals.sort_unstable_by_key(|(i, _)| *i);

                if e_vals != a_vals {
                    anyhow::bail!("Row {} values differ", row_idx);
                }
            }
            (Some(e), None) => {
                anyhow::bail!(
                    "Row {} missing in reconstructed matrix (nnz={})",
                    row_idx,
                    e.nnz()
                );
            }
            (None, Some(a)) => {
                anyhow::bail!("Row {} unexpectedly present (nnz={})", row_idx, a.nnz());
            }
        }
    }

    Ok(())
}

fn resolve_position_columns(
    platform: Option<Platform>,
    pos_x_col: Option<usize>,
    pos_y_col: Option<usize>,
) -> (usize, usize) {
    match platform {
        Some(Platform::Visium) => (4, 5),
        Some(Platform::Xenium) => (1, 2),
        Some(Platform::SingleCell) => (pos_x_col.unwrap_or(0), pos_y_col.unwrap_or(1)),
        None => (pos_x_col.unwrap_or(1), pos_y_col.unwrap_or(2)),
    }
}

#[derive(Parser)]
#[command(version, about = "Clustered MST encoder (lossless + lossy)")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    #[command(arg_required_else_help = true)]
    Build(BuildCommand),
    #[command(arg_required_else_help = true)]
    RoundtripCheck(RoundtripCheckCommand),
}

#[derive(Debug, Args)]
struct BuildCommand {
    #[arg(short = 'i', long)]
    input: PathBuf,
    /// Optional positions file (CSV or Parquet). Required unless platform is `single-cell`.
    #[arg(short = 'p', long = "input-pos")]
    input_pos: Option<PathBuf>,
    #[arg(short = 'o', long)]
    output: Option<PathBuf>,
    /// Optional CSV path to append compression statistics per run.
    #[arg(long = "stats-csv")]
    stats_csv: Option<PathBuf>,
    #[arg(short = 'F', long = "pos-format", value_enum, default_value_t = InputPosType::Parquet)]
    pos_format: InputPosType,
    #[arg(short = 'P', long = "platform", value_enum)]
    platform: Option<Platform>,
    #[arg(long = "pos-x-col")]
    pos_x_col: Option<usize>,
    #[arg(long = "pos-y-col")]
    pos_y_col: Option<usize>,
    #[arg(long = "max-cells")]
    max_cells: Option<usize>,
    #[arg(long, default_value_t = false)]
    lossy: bool,
    #[arg(long = "lossy-lossless", default_value_t = false)]
    lossy_lossless: bool,
    #[arg(long = "lossy-verify-lossless", default_value_t = false)]
    lossy_verify_lossless: bool,
    #[arg(long = "lossy-quantizers", default_value_t = 8)]
    lossy_quantizers: usize,
    #[arg(long = "lossy-bins", default_value_t = 16)]
    lossy_bins: usize,
    #[arg(long = "lossy-sweep", value_delimiter = ',')]
    lossy_sweep: Vec<usize>,
    /// Maximum allowed cells per cluster (applies to both lossy and lossless modes).
    /// Oversized clusters are recursively re-clustered.
    #[arg(long = "max-cluster-size")]
    max_cluster_size: Option<usize>,
    /// Distance metric used for KNN graph construction prior to MST.
    #[arg(long = "knn-metric", value_enum, default_value_t = KnnMetricArg::L0)]
    knn_metric: KnnMetricArg,
    /// Edge weighting used during MST construction.
    #[arg(long = "mst-weight", value_enum, default_value_t = MstWeightArg::Metric)]
    mst_weight: MstWeightArg,
    /// Codec used for index-like streams in MST payloads.
    #[arg(long = "index-codec", value_enum, default_value_t = IndexCodecArg::Arithmetic)]
    index_codec: IndexCodecArg,
    /// Codec used for sorted support/index lists (root supports, column gene ids, local dictionaries).
    #[arg(long = "sorted-index-codec", value_enum, default_value_t = SortedIndexCodecArg::Delta)]
    sorted_index_codec: SortedIndexCodecArg,
    /// If set, disable per-child full-row fallback when edit deltas are too large.
    #[arg(long = "disable-full-row-fallback", default_value_t = false)]
    disable_full_row_fallback: bool,
    /// Trigger full-row storage for a child when `delta_edits > ratio * child_nnz`.
    #[arg(long = "full-row-fallback-ratio", default_value_t = 1.0)]
    full_row_fallback_ratio: f32,
    /// If set, cut MST edges larger than `median_edge_weight * factor`, producing a forest.
    #[arg(long = "forest-cut-factor")]
    forest_cut_factor: Option<f32>,
    /// Cluster payload strategy: row MST ops, column postings, or choose smaller per-cluster.
    #[arg(long = "cluster-encoding", value_enum, default_value_t = ClusterEncodingArg::Hybrid)]
    cluster_encoding: ClusterEncodingArg,
    /// Fixed number of prototype supports per cluster for column encoding (used when adaptive is off).
    #[arg(long = "column-template-count", default_value_t = 0)]
    column_template_count: usize,
    /// Enable adaptive selection of column template count per cluster.
    #[arg(long = "column-template-adaptive", default_value_t = false)]
    column_template_adaptive: bool,
    /// Maximum number of column templates to consider in adaptive mode.
    #[arg(long = "column-template-max", default_value_t = 32)]
    column_template_max: usize,
    /// Enable adaptive row-template mode for row-MST payloads.
    #[arg(long = "row-template-adaptive", default_value_t = false)]
    row_template_adaptive: bool,
    /// Maximum number of row templates to consider in adaptive mode.
    #[arg(long = "row-template-max", default_value_t = 16)]
    row_template_max: usize,
    /// If set, emit combinatorial and entropy diagnostics for op_indices.
    #[arg(long = "report-index-bounds", default_value_t = false)]
    report_index_bounds: bool,
}

fn infer_stats_label(input: &Path) -> String {
    input
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
        .map(|s| s.to_string())
        .or_else(|| {
            input
                .file_name()
                .and_then(|s| s.to_str())
                .map(|s| s.to_string())
        })
        .unwrap_or_else(|| input.display().to_string())
}

fn append_build_stats_csv(
    csv_path: &Path,
    stats_label: &str,
    input_matrix_bytes: usize,
    input_positions_bytes: usize,
    mst_uncompressed_bytes: usize,
    gzip_actual_bytes: usize,
    rows_times_cols: usize,
    nnz: usize,
    actual_rate_bits_per_value: f64,
    topology_parent_offset_bytes: usize,
    values_root_plus_ops_bytes: usize,
    metadata_num_genes_bytes: usize,
    column_order_bytes: usize,
    values_root_indices_bytes: usize,
    values_root_vals_bytes: usize,
    values_op_indices_bytes: usize,
    values_op_vals_bytes: usize,
) -> anyhow::Result<()> {
    let input_bytes_sum = input_matrix_bytes + input_positions_bytes;
    let gzip_to_input_sum_ratio = if input_bytes_sum > 0 {
        gzip_actual_bytes as f64 / input_bytes_sum as f64
    } else {
        0.0
    };

    let file_exists = csv_path.exists();
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(csv_path)?;
    let mut wtr = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(file);

    if !file_exists || std::fs::metadata(csv_path)?.len() == 0 {
        wtr.write_record([
            "stats_csv",
            "input_matrix_bytes",
            "input_positions_bytes",
            "input_bytes_sum",
            "mst_uncompressed_bytes",
            "gzip_actual_bytes",
            "gzip_to_input_sum_ratio",
            "rows_times_cols",
            "nnz",
            "actual_rate_bits_per_value",
            "topology_parent_offset_bytes",
            "values_root_plus_ops_bytes",
            "metadata_num_genes_bytes",
            "column_order_bytes",
            "values_root_indices_bytes",
            "values_root_vals_bytes",
            "values_op_indices_bytes",
            "values_op_vals_bytes",
        ])?;
    }

    wtr.write_record([
        stats_label.to_string(),
        input_matrix_bytes.to_string(),
        input_positions_bytes.to_string(),
        input_bytes_sum.to_string(),
        mst_uncompressed_bytes.to_string(),
        gzip_actual_bytes.to_string(),
        gzip_to_input_sum_ratio.to_string(),
        rows_times_cols.to_string(),
        nnz.to_string(),
        actual_rate_bits_per_value.to_string(),
        topology_parent_offset_bytes.to_string(),
        values_root_plus_ops_bytes.to_string(),
        metadata_num_genes_bytes.to_string(),
        column_order_bytes.to_string(),
        values_root_indices_bytes.to_string(),
        values_root_vals_bytes.to_string(),
        values_op_indices_bytes.to_string(),
        values_op_vals_bytes.to_string(),
    ])?;
    wtr.flush()?;
    Ok(())
}

#[derive(Debug, Args)]
struct RoundtripCheckCommand {
    #[arg(short = 'e', long = "encoded")]
    encoded: PathBuf,
    #[arg(short = 'i', long)]
    input: PathBuf,
    #[arg(short = 'p', long = "input-pos")]
    input_pos: PathBuf,
    #[arg(short = 'F', long = "pos-format", value_enum, default_value_t = InputPosType::Parquet)]
    pos_format: InputPosType,
    #[arg(short = 'P', long = "platform", value_enum)]
    platform: Option<Platform>,
    #[arg(long = "pos-x-col")]
    pos_x_col: Option<usize>,
    #[arg(long = "pos-y-col")]
    pos_y_col: Option<usize>,
    #[arg(long = "max-cells")]
    max_cells: Option<usize>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(
            EnvFilter::builder()
                .with_default_directive(LevelFilter::INFO.into())
                .from_env_lossy()
                .add_directive("ureq=warn".parse()?),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Build(args) => {
            let (csr, points) = match args.platform {
                Some(Platform::SingleCell) => {
                    // Single-cell mode: no positions file required; generate dummy points.
                    load_10x_no_positions(&args.input, args.max_cells)?
                }
                _ => {
                    // Spatial modes: positions file is required.
                    let pos_path = args.input_pos.as_ref().ok_or_else(|| {
                        anyhow::anyhow!(
                            "input-pos is required unless --platform single-cell is used"
                        )
                    })?;
                    let (pos_x_col, pos_y_col) =
                        resolve_position_columns(args.platform, args.pos_x_col, args.pos_y_col);
                    load_10x_with_positions(
                        &args.input,
                        pos_path,
                        args.pos_format,
                        pos_x_col,
                        pos_y_col,
                        args.max_cells,
                    )?
                }
            };

            let quantize_values = args.lossy && !args.lossy_lossless;
            let target_quantizers = args.lossy_quantizers.max(1);
            let quantizer_bins = args.lossy_bins.max(1);
            let sweep_counts = normalize_quantizer_counts(target_quantizers, &args.lossy_sweep);
            let index_codec = IndexStreamCodec::from(args.index_codec);
            let sorted_index_codec = SortedIndexCodec::from(args.sorted_index_codec);
            let mst_weight_mode = MstWeightMode::from(args.mst_weight);
            let fallback_ratio = if args.disable_full_row_fallback {
                None
            } else {
                Some(args.full_row_fallback_ratio.max(0.0))
            };
            let forest_cut_factor = args.forest_cut_factor.map(|f| f.max(0.0));
            let cluster_encoding = args.cluster_encoding;

            info!(
                "Encoding mode: {} | quantizers={} bins={} sweep={:?} max_cluster_size={:?} knn_metric={:?} mst_weight={:?} index_codec={:?} sorted_index_codec={:?} full_row_fallback={:?} forest_cut_factor={:?} cluster_encoding={:?} column_template_count={} column_template_adaptive={} column_template_max={} row_template_adaptive={} row_template_max={}",
                if quantize_values { "lossy" } else { "lossless" },
                target_quantizers,
                quantizer_bins,
                sweep_counts,
                args.max_cluster_size,
                args.knn_metric,
                args.mst_weight,
                args.index_codec,
                args.sorted_index_codec,
                fallback_ratio,
                forest_cut_factor,
                cluster_encoding,
                args.column_template_count,
                args.column_template_adaptive,
                args.column_template_max,
                args.row_template_adaptive,
                args.row_template_max
            );

            let mut metrics = Vec::new();
            let mut selected: Option<CompressionResult> = None;

            for &quantizer_count in &sweep_counts {
                let result = run_clustered_compression(
                    &points,
                    &csr,
                    quantizer_count,
                    quantizer_bins,
                    quantize_values,
                    args.lossy_verify_lossless,
                    args.max_cluster_size,
                    args.knn_metric.into(),
                    mst_weight_mode,
                    index_codec,
                    sorted_index_codec,
                    fallback_ratio,
                    forest_cut_factor,
                    cluster_encoding,
                    args.column_template_count,
                    args.column_template_adaptive,
                    args.column_template_max,
                    args.row_template_adaptive,
                    args.row_template_max,
                )?;

                info!(
                    "q={} (used={}): gzip≈{} bytes, rate={:.4} bits/value, mse={:.6}, rmse={:.6}",
                    result.quantizers_requested,
                    result.quantizers_used,
                    result.gzip_bytes_estimate,
                    result.rate_bits_per_value,
                    result.mse,
                    result.rmse
                );

                metrics.push(SweepMetric {
                    quantizers_requested: result.quantizers_requested,
                    quantizers_used: result.quantizers_used,
                    gzip_bytes_estimate: result.gzip_bytes_estimate,
                    rate_bits_per_value: result.rate_bits_per_value,
                    mse: result.mse,
                    rmse: result.rmse,
                });

                if quantizer_count == target_quantizers {
                    selected = Some(result);
                }
            }

            let selected = selected.ok_or_else(|| {
                anyhow::anyhow!("Failed to build selected setting q={}", target_quantizers)
            })?;

            info!("Rate-distortion sweep summary:");
            for metric in &metrics {
                info!(
                    "  q={} (used={}): rate={:.4}, mse={:.6}, rmse={:.6}, gzip≈{} bytes",
                    metric.quantizers_requested,
                    metric.quantizers_used,
                    metric.rate_bits_per_value,
                    metric.mse,
                    metric.rmse,
                    metric.gzip_bytes_estimate
                );
            }

            let output = args
                .output
                .unwrap_or_else(|| PathBuf::from("output.bin.gz"));
            let config = bincode::config::standard()
                .with_little_endian()
                .with_fixed_int_encoding();

            let file = File::create(&output)?;
            let writer = BufWriter::new(file);
            let mut encoder = GzEncoder::new(writer, Compression::default());
            bincode::encode_into_std_write(&selected.encoded_blocks, &mut encoder, config)?;
            bincode::encode_into_std_write(&selected.positions, &mut encoder, config)?;
            bincode::encode_into_std_write(&selected.row_order, &mut encoder, config)?;
            bincode::encode_into_std_write(&selected.gene_order, &mut encoder, config)?;
            let _ = encoder.finish()?;

            let actual_gzip_bytes = std::fs::metadata(&output)?.len() as usize;
            let total_entries = (points.len() * csr.cols()).max(1);
            let actual_rate = (actual_gzip_bytes as f64 * 8.0) / total_entries as f64;

            let mut bytes_parent = 0usize;
            let mut bytes_root_indices = 0usize;
            let mut bytes_root_vals = 0usize;
            let mut bytes_indices = 0usize;
            let mut bytes_delta_vals = 0usize;
            let mut bytes_num_genes = 0usize;

            for block in &selected.encoded_blocks {
                let (p, ri, rv, i, dv, ng) = block.bytes_breakdown();
                bytes_parent += p;
                bytes_root_indices += ri;
                bytes_root_vals += rv;
                bytes_indices += i;
                bytes_delta_vals += dv;
                bytes_num_genes += ng;
            }

            let topology_bytes = bytes_parent;
            let value_bytes =
                bytes_root_indices + bytes_root_vals + bytes_indices + bytes_delta_vals;
            let mst_payload_bytes = topology_bytes + value_bytes + bytes_num_genes;
            let gene_order_bytes = selected.gene_order.len() * std::mem::size_of::<u32>();
            let payload_with_col_order = mst_payload_bytes + gene_order_bytes;

            info!("Saved encoded payload to {}", output.display());
            info!(
                "Selected setting: q={} (used={}), bins={}, clusters={}",
                selected.quantizers_requested,
                selected.quantizers_used,
                selected.quantizer_bins,
                selected.cluster_sizes.len()
            );
            info!(
                "Size: mst_uncompressed={} bytes, gzip_estimate={} bytes, gzip_actual={} bytes",
                selected.total_mst_bytes, selected.gzip_bytes_estimate, actual_gzip_bytes
            );
            info!(
                "Distortion: mse={:.6}, rmse={:.6}, actual_rate={:.4} bits/value",
                selected.mse, selected.rmse, actual_rate
            );
            info!(
                "MST breakdown: topology(parent_offset)={} bytes, values(root+ops)={} bytes, metadata(num_genes)={} bytes, column_order={} bytes, total={} bytes",
                topology_bytes,
                value_bytes,
                bytes_num_genes,
                gene_order_bytes,
                payload_with_col_order
            );
            info!(
                "  values breakdown: root_indices={} root_vals={} op_indices={} op_vals={}",
                bytes_root_indices, bytes_root_vals, bytes_indices, bytes_delta_vals
            );
            if let Some(stats_path) = args.stats_csv.as_ref() {
                let input_matrix_bytes = std::fs::metadata(&args.input)
                    .map(|m| m.len() as usize)
                    .unwrap_or(0);
                let input_positions_bytes = args
                    .input_pos
                    .as_ref()
                    .and_then(|p| std::fs::metadata(p).ok())
                    .map(|m| m.len() as usize)
                    .unwrap_or(0);
                let stats_label = infer_stats_label(&args.input);
                append_build_stats_csv(
                    stats_path,
                    &stats_label,
                    input_matrix_bytes,
                    input_positions_bytes,
                    selected.total_mst_bytes,
                    actual_gzip_bytes,
                    points.len() * csr.cols(),
                    csr.nnz(),
                    actual_rate,
                    topology_bytes,
                    value_bytes,
                    bytes_num_genes,
                    gene_order_bytes,
                    bytes_root_indices,
                    bytes_root_vals,
                    bytes_indices,
                    bytes_delta_vals,
                )?;
                info!("Appended build stats CSV row to {}", stats_path.display());
            }
            if args.report_index_bounds {
                let bounds_report = compute_index_bounds_report(&selected.encoded_blocks);
                log_index_bounds_report(&bounds_report, csr.nnz());
            }
        }
        Commands::RoundtripCheck(args) => {
            let (pos_x_col, pos_y_col) =
                resolve_position_columns(args.platform, args.pos_x_col, args.pos_y_col);

            let (encoded_blocks, _positions, row_order, gene_order) =
                decode_clustered_payload(&args.encoded)?;
            info!(
                "Loaded payload: blocks={}, mapped_rows={}, gene_order_len={}",
                encoded_blocks.len(),
                row_order.len(),
                gene_order.len()
            );

            let (csr_truth, _points) = load_10x_with_positions(
                &args.input,
                &args.input_pos,
                args.pos_format,
                pos_x_col,
                pos_y_col,
                args.max_cells,
            )?;

            info!(
                "Ground truth CSR: rows={}, cols={}, nnz={}",
                csr_truth.rows(),
                csr_truth.cols(),
                csr_truth.nnz()
            );

            let reconstructed = reconstruct_csr_from_clustered_payload(
                &encoded_blocks,
                &row_order,
                &gene_order,
                csr_truth.rows(),
                csr_truth.cols(),
            )?;

            info!(
                "Reconstructed CSR: rows={}, cols={}, nnz={}",
                reconstructed.rows(),
                reconstructed.cols(),
                reconstructed.nnz()
            );

            compare_csr_exact(&csr_truth, &reconstructed)?;
            info!("Round-trip check PASSED: reconstructed CSR exactly matches input.");
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sprs::TriMatI;

    #[test]
    fn lossy_mode_respects_max_cluster_size() {
        let n_cells = 24usize;
        let n_genes = 12usize;
        let max_cluster_size = 5usize;

        let mut tri = TriMatI::<u16, usize>::new((n_cells, n_genes));
        for cell in 0..n_cells {
            let g0 = cell % n_genes;
            let g1 = (cell * 3 + 1) % n_genes;
            let g2 = (cell * 5 + 2) % n_genes;
            tri.add_triplet(cell, g0, ((cell % 7) + 1) as u16);
            tri.add_triplet(cell, g1, ((cell % 5) + 2) as u16);
            tri.add_triplet(cell, g2, ((cell % 3) + 1) as u16);
        }
        let csr = tri.to_csr::<usize>();

        let points: Vec<Point> = (0..n_cells)
            .map(|i| Point::new(i as f64, (i % 4) as f64, i))
            .collect();

        let result = run_clustered_compression(
            &points,
            &csr,
            1,
            8,
            true,  // lossy path
            false, // no lossless verification
            Some(max_cluster_size),
            KnnDistanceMetric::L0,
            MstWeightMode::Metric,
            IndexStreamCodec::Arithmetic,
            SortedIndexCodec::EliasFano,
            Some(1.0),
            None,
            ClusterEncodingArg::Hybrid,
            0,
            false,
            32,
            false,
            16,
        )
        .expect("lossy compression should succeed");

        assert_eq!(result.cluster_sizes.iter().sum::<usize>(), n_cells);
        assert!(result.cluster_sizes.iter().all(|&s| s <= max_cluster_size));
    }
}
