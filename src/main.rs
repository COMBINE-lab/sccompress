mod arith_encode;
mod cluster;
mod delta_indices;
mod h5_utils;
mod index_stream;
mod matrix_io;
mod mst_codec;
mod sorted_indices;

use clap::{Args, Parser, Subcommand, ValueEnum};
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use index_stream::IndexStreamCodec;
use matrix_io::{load_10x_oriented, InputPosType, MatrixOrientation, Platform};
use mimalloc::MiMalloc;
use mst_codec::{
    codec_timing_stats, column_memory_stats, encode_subarray_column,
    encode_subarray_mst_with_metric, reset_codec_timing_stats, reset_column_memory_stats,
    reset_row_mst_order_stats, row_mst_order_stats, DatalessPoint, EncodedClusterBlock,
    EncodedColumnBlock, EncodedDiffsMST, GeneKruskalMstParams, KnnDistanceMetric, MstWeightMode,
    Point,
};
use rayon::prelude::*;
use sorted_indices::SortedIndexCodec;
use sprs::CsMat;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Cursor, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use tracing::{info, warn};
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::fmt;
use tracing_subscriber::prelude::*;
use tracing_subscriber::EnvFilter;
use zstd::stream::{Decoder as ZstdDecoder, Encoder as ZstdEncoder};

use cluster::{
    group_points_by_cluster, joint_svd_seriation, projection_weight, split_oversized_clusters,
};
use nalgebra::DMatrix;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn current_rss_bytes() -> Option<u64> {
    let output = Command::new("ps")
        .arg("-o")
        .arg("rss=")
        .arg("-p")
        .arg(std::process::id().to_string())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let rss_kb = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse::<u64>()
        .ok()?;
    Some(rss_kb * 1024)
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn log_memory_checkpoint(label: &str) {
    if let Some(rss) = current_rss_bytes() {
        info!("Memory checkpoint: {} current_rss={:.1} MiB", label, mib(rss));
    } else {
        info!("Memory checkpoint: {} current_rss=unavailable", label);
    }
}

fn estimate_csr_heap_bytes(matrix: &CsMat<u16>) -> u64 {
    let values = matrix.nnz() * std::mem::size_of::<u16>();
    let indices = matrix.nnz() * std::mem::size_of::<usize>();
    let indptr = (matrix.rows() + 1) * std::mem::size_of::<usize>();
    (values + indices + indptr) as u64
}

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
    timing_cluster_prep_ms: u64,
    timing_gene_order_ms: u64,
    timing_encode_tiles_ms: u64,
    timing_row_mst_encode_ms: u64,
    timing_column_encode_ms: u64,
    cluster_sizes: Vec<usize>,
    encoded_blocks: Vec<EncodedClusterBlock>,
    positions: Vec<DatalessPoint>,
    tile_local_orders: Vec<Vec<u32>>,
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
enum PayloadCompressionArg {
    Gzip,
    Zstd,
    SevenZip,
}

impl PayloadCompressionArg {
    fn extension(self) -> &'static str {
        match self {
            PayloadCompressionArg::Gzip => "gz",
            PayloadCompressionArg::Zstd => "zst",
            PayloadCompressionArg::SevenZip => "7z",
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum MstWeightArg {
    Metric,
    EncodingCost,
}

/// How to compute the gene (column) permutation that maps original gene indices
/// to their serialized order.  A good gene ordering places genes with similar
/// cell support patterns adjacent, shrinking gap-coded posting-list costs and
/// MST ref-chain edit sizes.
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum GeneReorderMethod {
    /// Hash-projection seriation (current default): for each gene, accumulate
    /// deterministic hash-based random-projection scores weighted by the row
    /// stream order, then sort genes lexicographically by those scores.
    #[default]
    Projection,
    /// SVD-based dual seriation: build a truncated SVD of the (optionally
    /// normalized) sparse cell×gene matrix, then sort genes by their first
    /// right singular vector coordinate (`v₁`), breaking ties with `v₂`.
    Svd,
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

/// SVD-based gene reordering: build a sparse cell×gene matrix from the cluster
/// row stream, compute a truncated SVD, and sort genes by their coordinates in
/// the first two right singular vectors (v₁, v₂).  Genes with similar cell
/// support patterns end up adjacent, which improves gap-coded posting-list
/// compression and gene-level MST ref-chain costs.
fn compute_gene_permutation_svd(
    points: &[Point],
    data: &CsMat<u16>,
    row_stream_point_indices: &[usize],
) -> anyhow::Result<(Vec<u32>, Vec<u32>)> {
    let ncols = data.cols();
    if ncols == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    let n_cells = row_stream_point_indices.len();
    if n_cells == 0 {
        let identity: Vec<u32> = (0..ncols as u32).collect();
        return Ok((identity.clone(), identity));
    }

    // Collect per-gene occurrence counts and the set of cells that express each gene.
    // We work on the binary support (presence/absence) to keep the SVD focused on
    // co-occurrence structure rather than magnitude.
    let mut col_nnz = vec![0u64; ncols];
    let mut row_nnz = vec![0u64; n_cells];
    let mut triplets: Vec<(usize, usize, f64)> = Vec::new();

    for (stream_pos, &point_idx) in row_stream_point_indices.iter().enumerate() {
        let point = &points[point_idx];
        if let Some(row) = data.outer_view(point.row_index) {
            for (gene_idx, &value) in row.iter() {
                if value != 0 {
                    col_nnz[gene_idx] += 1;
                    row_nnz[stream_pos] += 1;
                    triplets.push((stream_pos, gene_idx, 1.0));
                }
            }
        }
    }

    // Find genes that actually appear; genes with zero support keep their original index
    let active_genes: Vec<usize> = (0..ncols).filter(|&g| col_nnz[g] > 0).collect();
    if active_genes.len() <= 2 {
        let identity: Vec<u32> = (0..ncols as u32).collect();
        return Ok((identity.clone(), identity));
    }

    // Cap the dense matrix size: if (n_cells × active_genes) would be huge, subsample
    // rows to keep the dense SVD tractable.
    let max_svd_rows = 4000usize;
    let max_svd_cols = 2000usize;

    let sampled_rows: Vec<usize> = if n_cells > max_svd_rows {
        let step = n_cells as f64 / max_svd_rows as f64;
        (0..max_svd_rows)
            .map(|i| (i as f64 * step) as usize)
            .collect()
    } else {
        (0..n_cells).collect()
    };

    let active_cols: Vec<usize> = if active_genes.len() > max_svd_cols {
        // Keep the most frequent genes
        let mut by_freq: Vec<(usize, u64)> =
            active_genes.iter().map(|&g| (g, col_nnz[g])).collect();
        by_freq.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        by_freq.truncate(max_svd_cols);
        by_freq.sort_unstable_by_key(|&(g, _)| g);
        by_freq.iter().map(|&(g, _)| g).collect()
    } else {
        active_genes.clone()
    };

    // Build a column index map for the subsetted genes
    let mut col_local: HashMap<usize, usize> = HashMap::with_capacity(active_cols.len());
    for (local, &global) in active_cols.iter().enumerate() {
        col_local.insert(global, local);
    }

    let nr = sampled_rows.len();
    let nc = active_cols.len();

    // Build dense matrix with TF-IDF-like normalization:
    //   value = (1 if gene present) / sqrt(col_nnz[gene])
    // This down-weights ubiquitous genes and emphasizes discriminative ones.
    let mut dense = vec![0.0f64; nr * nc];
    let sampled_set: HashMap<usize, usize> = sampled_rows
        .iter()
        .enumerate()
        .map(|(i, &r)| (r, i))
        .collect();

    for &(stream_pos, gene_idx, _) in &triplets {
        if let Some(&local_row) = sampled_set.get(&stream_pos) {
            if let Some(&local_col) = col_local.get(&gene_idx) {
                dense[local_row * nc + local_col] =
                    1.0 / (col_nnz[gene_idx] as f64).sqrt().max(1e-12);
            }
        }
    }

    let mat = DMatrix::from_row_slice(nr, nc, &dense);
    let svd = mat.svd(false, true);
    let v_t = match svd.v_t {
        Some(vt) => vt,
        None => {
            info!("SVD gene reorder: V^T not available, falling back to projection method");
            return compute_gene_permutation_from_row_stream(
                points,
                data,
                row_stream_point_indices,
            );
        }
    };

    // Extract the first two right singular vector coordinates for each active column
    let n_sv = v_t.nrows().min(2);
    let mut scores: Vec<(usize, f64, f64)> = Vec::with_capacity(ncols);
    for &global_gene in &active_cols {
        let local_col = col_local[&global_gene];
        let s1 = if n_sv >= 1 { v_t[(0, local_col)] } else { 0.0 };
        let s2 = if n_sv >= 2 { v_t[(1, local_col)] } else { 0.0 };
        scores.push((global_gene, s1, s2));
    }
    // Inactive genes (not in SVD subset) get a deterministic tiebreak score
    for g in 0..ncols {
        if col_nnz[g] == 0 || !col_local.contains_key(&g) {
            scores.push((g, f64::INFINITY, g as f64));
        }
    }

    scores.sort_unstable_by(|a, b| {
        a.1.total_cmp(&b.1)
            .then_with(|| a.2.total_cmp(&b.2))
            .then_with(|| a.0.cmp(&b.0))
    });

    let mut old_to_new = vec![0u32; ncols];
    let mut new_to_old = Vec::with_capacity(ncols);
    for (new_idx, &(old_idx, _, _)) in scores.iter().enumerate() {
        old_to_new[old_idx] = new_idx as u32;
        new_to_old.push(old_idx as u32);
    }

    info!(
        "SVD gene reorder: {} genes reordered ({} active in SVD, {} total)",
        active_cols.len(),
        nc,
        ncols
    );

    Ok((new_to_old, old_to_new))
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

fn train_quantizer_from_quantiles(values: &[u16], bins: usize) -> anyhow::Result<Vec<u16>> {
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
    let mut sampled_sorted = sampled;
    sampled_sorted.sort_unstable();
    let mut centers = Vec::with_capacity(capped_bins);
    for bin in 0..capped_bins {
        let pos = if capped_bins == 1 {
            sampled_sorted.len() / 2
        } else {
            bin * (sampled_sorted.len().saturating_sub(1)) / (capped_bins - 1)
        };
        centers.push(sampled_sorted[pos]);
    }

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
        quantizers.push(train_quantizer_from_quantiles(cluster_values, bins)?);
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

/// Serialized layout (in order):
///   1. `encoded_blocks`
///   2. `positions`
///   3. `row_order` placeholder (empty `Vec<u32>` for current builds; reserved slot for
///      future "exact-order" payloads)
///   4. `gene_order` placeholder (empty `Vec<u32>`; same reservation)
///   5. `cluster_sizes: Vec<u32>` - one entry per cluster, used by the count-matrix
///      decoder to group tiles within a cluster (`gene_blocks = blocks / clusters`).
///   6. `ncols: u32` - number of columns in the original matrix (post-orientation).
///   7. `tile_local_orders: Vec<Vec<u32>>` - one permutation per encoded block, mapping
///      each tile's local row index back to its cluster-local row index. Required to
///      merge per-gene-block tiles into a single row in the count matrix.
///
/// Empty placeholders for `row_order`/`gene_order` keep this format readable by older
/// readers that probed for those fields. The new tail
/// (`cluster_sizes` + `ncols` + `tile_local_orders`) enables permutation-tolerant
/// decompression back to a count matrix.
fn serialize_payload(
    encoded_blocks: &[EncodedClusterBlock],
    positions: &[DatalessPoint],
    cluster_sizes: &[u32],
    ncols: u32,
    tile_local_orders: &[Vec<u32>],
) -> anyhow::Result<Vec<u8>> {
    let config = bincode::config::standard()
        .with_little_endian()
        .with_fixed_int_encoding();
    let mut payload = Vec::new();
    bincode::encode_into_std_write(encoded_blocks, &mut payload, config)?;
    bincode::encode_into_std_write(positions, &mut payload, config)?;
    let empty_order: Vec<u32> = Vec::new();
    bincode::encode_into_std_write(&empty_order, &mut payload, config)?;
    bincode::encode_into_std_write(&empty_order, &mut payload, config)?;
    bincode::encode_into_std_write(&cluster_sizes.to_vec(), &mut payload, config)?;
    bincode::encode_into_std_write(&ncols, &mut payload, config)?;
    bincode::encode_into_std_write(&tile_local_orders.to_vec(), &mut payload, config)?;
    Ok(payload)
}

fn compress_payload_bytes(
    raw_payload: &[u8],
    compression: PayloadCompressionArg,
) -> anyhow::Result<Vec<u8>> {
    match compression {
        PayloadCompressionArg::Gzip => {
            let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
            encoder.write_all(raw_payload)?;
            Ok(encoder.finish()?)
        }
        PayloadCompressionArg::Zstd => {
            let mut encoder = ZstdEncoder::new(Vec::new(), 3)?;
            encoder.write_all(raw_payload)?;
            Ok(encoder.finish()?)
        }
        PayloadCompressionArg::SevenZip => {
            let nonce = format!(
                "{}_{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0)
            );
            let raw_path = std::env::temp_dir().join(format!("quadtree_payload_{}.bin", nonce));
            let archive_path = std::env::temp_dir().join(format!("quadtree_payload_{}.7z", nonce));
            std::fs::write(&raw_path, raw_payload)?;
            let status = Command::new("7z")
                .arg("a")
                .arg("-t7z")
                .arg("-mx=9")
                .arg("-y")
                .arg(&archive_path)
                .arg(&raw_path)
                .status()?;
            if !status.success() {
                let _ = std::fs::remove_file(&raw_path);
                let _ = std::fs::remove_file(&archive_path);
                anyhow::bail!("7z compression failed");
            }
            let archive = std::fs::read(&archive_path)?;
            let _ = std::fs::remove_file(&raw_path);
            let _ = std::fs::remove_file(&archive_path);
            Ok(archive)
        }
    }
}

fn estimate_compressed_payload_size(
    compression: PayloadCompressionArg,
    encoded_blocks: &[EncodedClusterBlock],
    positions: &[DatalessPoint],
    cluster_sizes: &[u32],
    ncols: u32,
    tile_local_orders: &[Vec<u32>],
) -> anyhow::Result<usize> {
    let payload = serialize_payload(
        encoded_blocks,
        positions,
        cluster_sizes,
        ncols,
        tile_local_orders,
    )?;
    Ok(compress_payload_bytes(&payload, compression)?.len())
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
    cell_blocks_requested: usize,
    quantizer_bins: usize,
    quantize_values: bool,
    verify_lossless: bool,
    max_cluster_size: Option<usize>,
    knn_metric: KnnDistanceMetric,
    mst_weight_mode: MstWeightMode,
    index_codec: IndexStreamCodec,
    sorted_index_codec: SortedIndexCodec,
    row_parent_ref_cost_ratio: Option<f32>,
    forest_cut_factor: Option<f32>,
    row_mst_window: usize,
    column_template_count: usize,
    column_template_adaptive: bool,
    column_template_max: usize,
    row_template_adaptive: bool,
    row_template_max: usize,
    disable_row_mst_candidate: bool,
    disable_column_candidate: bool,
    disable_row_parent_ref: bool,
    disable_column_ref: bool,
    prefer_column_on_tie: bool,
    report_memory: bool,
    cluster_seed: Option<u64>,
    gene_reorder_method: GeneReorderMethod,
    joint_svd_fast: bool,
    gene_blocks: usize,
    payload_compression: PayloadCompressionArg,
) -> anyhow::Result<CompressionResult> {
    if points.is_empty() {
        anyhow::bail!("Cannot encode empty point set");
    }
    let cluster_prep_start = Instant::now();

    let requested_clusters = cell_blocks_requested.max(1).min(points.len());

    // Joint SVD is the only cell-clustering path. If it fails, keep the run
    // deterministic by assigning cells round-robin and computing gene order separately.
    let joint_result = match joint_svd_seriation(
        points,
        data,
        requested_clusters,
        cluster_seed,
        joint_svd_fast,
    ) {
        Ok(result) => Some(result),
        Err(e) => {
            warn!(
                "Joint SVD seriation failed ({}), falling back to round-robin cell labels",
                e
            );
            None
        }
    };

    let assignments = if let Some((ref cell_labels, _, _)) = joint_result {
        cell_labels.clone()
    } else {
        (0..points.len())
            .map(|idx| idx % requested_clusters)
            .collect::<Vec<_>>()
    };

    let initial_quantizers_used = assignments
        .iter()
        .copied()
        .max()
        .map(|m| m + 1)
        .unwrap_or(1);

    let initial_clusters = group_points_by_cluster(&assignments, initial_quantizers_used);
    let initial_max = initial_clusters.iter().map(|c| c.len()).max().unwrap_or(0);
    let clusters = split_oversized_clusters(initial_clusters, max_cluster_size)?;
    let final_max = clusters.iter().map(|c| c.len()).max().unwrap_or(0);

    info!(
        "Cluster layout: requested={} initial={} final={} largest_initial={} largest_final={} max_cluster_size={:?}",
        cell_blocks_requested,
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

    let quantized_data_owned = if quantize_values {
        Some(quantize_matrix_by_cluster(
            data,
            points,
            &refined_assignments,
            clusters.len(),
            quantizer_bins,
        )?)
    } else {
        None
    };
    let quantized_data = quantized_data_owned.as_ref().unwrap_or(data);
    if report_memory {
        log_memory_checkpoint("after quantized_data selection");
        info!(
            "Memory estimate: active_csr_heap={:.1} MiB rows={} cols={} nnz={}",
            mib(estimate_csr_heap_bytes(quantized_data)),
            quantized_data.rows(),
            quantized_data.cols(),
            quantized_data.nnz()
        );
    }

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
    let timing_cluster_prep_ms = cluster_prep_start.elapsed().as_millis() as u64;

    let gene_order_start = Instant::now();
    let (gene_order, gene_old_to_new) = if let Some((_, ref new_to_old, ref old_to_new)) =
        joint_result
    {
        info!("Using gene ordering from joint SVD seriation");
        (new_to_old.clone(), old_to_new.clone())
    } else {
        match gene_reorder_method {
            GeneReorderMethod::Projection => compute_gene_permutation_from_row_stream(
                points,
                &quantized_data,
                &row_stream_point_indices,
            )?,
            GeneReorderMethod::Svd => {
                compute_gene_permutation_svd(points, &quantized_data, &row_stream_point_indices)?
            }
        }
    };
    let timing_gene_order_ms = gene_order_start.elapsed().as_millis() as u64;

    // Build per-gene-block column masks.  When gene_blocks <= 1 the single "block" is
    // the full gene set and we skip the column-filtering overhead entirely.
    let n_genes = data.cols();
    let effective_gene_blocks = gene_blocks.max(1).min(n_genes);
    let gene_block_ranges: Vec<std::ops::Range<u32>> = {
        let block_sz = n_genes / effective_gene_blocks;
        let remainder = n_genes % effective_gene_blocks;
        let mut ranges = Vec::with_capacity(effective_gene_blocks);
        let mut start = 0u32;
        for b in 0..effective_gene_blocks {
            let sz = block_sz + if b < remainder { 1 } else { 0 };
            ranges.push(start..start + sz as u32);
            start += sz as u32;
        }
        ranges
    };

    let mut encoded_blocks = Vec::new();
    let mut positions = Vec::new();
    let mut row_order = Vec::new();
    let mut cluster_sizes = Vec::new();
    let mut tile_local_orders: Vec<Vec<u32>> = Vec::new();
    let mut total_mst_bytes = 0usize;
    let row_mst_encode_us = AtomicU64::new(0);
    let column_encode_us = AtomicU64::new(0);

    // Precompute per-gene-block old→new mappings once (shared across all cell clusters).
    let block_gene_o2ns: Vec<Vec<u32>> = gene_block_ranges
        .iter()
        .map(|range| {
            if effective_gene_blocks > 1 {
                let mut o2n = vec![u32::MAX; n_genes];
                for (new, old_remapped) in range.clone().enumerate() {
                    let orig = gene_order[old_remapped as usize] as usize;
                    if orig < n_genes {
                        o2n[orig] = new as u32;
                    }
                }
                o2n
            } else {
                gene_old_to_new.clone()
            }
        })
        .collect();
    if report_memory {
        let gene_order_bytes = (gene_order.capacity() + gene_old_to_new.capacity())
            * std::mem::size_of::<u32>();
        let block_maps_bytes = block_gene_o2ns.capacity() * std::mem::size_of::<Vec<u32>>()
            + block_gene_o2ns
                .iter()
                .map(|map| map.capacity() * std::mem::size_of::<u32>())
                .sum::<usize>();
        log_memory_checkpoint("after gene order and block maps");
        info!(
            "Memory estimate: gene_order_maps={:.1} MiB block_gene_o2ns={:.1} MiB gene_blocks={}",
            mib(gene_order_bytes as u64),
            mib(block_maps_bytes as u64),
            block_gene_o2ns.len()
        );
    }

    let encode_tile = |cluster_points: &[Point],
                       block_gene_o2n: &[u32],
                       gene_block_size: u32,
                       cluster_idx: usize|
     -> anyhow::Result<(EncodedClusterBlock, Vec<u32>, usize, bool, bool)> {
        // `local_to_global` is always the identity over the tile's gene block (size B).
        // The decoder recovers the global gene as `gene_block_start + in_block_offset`,
        // so the per-tile dictionary collapses to ~0 bytes after entropy coding.
        let gene_block_size_hint: Option<u32> = Some(gene_block_size);
        let mut row_selected_adaptive = false;
        let mut col_selected_adaptive = false;

        let row_candidate = if disable_row_mst_candidate {
            None
        } else {
            let mut best_row: Option<(EncodedClusterBlock, Vec<u32>, usize)> = None;
            let mut try_row_encode = |adaptive: bool| {
                let t = Instant::now();
                let encoded = encode_subarray_mst_with_metric(
                    cluster_points,
                    &quantized_data,
                    knn_metric,
                    mst_weight_mode,
                    Some(block_gene_o2n),
                    index_codec,
                    sorted_index_codec,
                    row_parent_ref_cost_ratio,
                    forest_cut_factor,
                    row_mst_window,
                    adaptive,
                    row_template_max,
                    !disable_row_parent_ref,
                    gene_block_size_hint,
                );
                row_mst_encode_us.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
                if let Some((row_block, traversal_order)) = encoded {
                    let bytes = row_block.total_bytes();
                    if best_row.as_ref().map(|b| bytes < b.2).unwrap_or(true) {
                        best_row = Some((
                            EncodedClusterBlock::RowMst(row_block),
                            traversal_order,
                            bytes,
                        ));
                        row_selected_adaptive = adaptive;
                    }
                }
            };

            try_row_encode(row_template_adaptive);
            if row_template_adaptive {
                try_row_encode(false);
            }
            best_row
        };

        let col_candidate = if disable_column_candidate {
            None
        } else {
            let mut best_col: Option<(EncodedClusterBlock, Vec<u32>, usize)> = None;
            let mut try_col_encode = |adaptive: bool| {
                let t = Instant::now();
                let gene_kruskal_cfg = GeneKruskalMstParams {
                    knn_metric,
                    mst_weight_mode,
                    index_codec,
                    full_row_fallback_ratio: row_parent_ref_cost_ratio,
                    forest_cut_factor,
                    row_mst_window,
                    allow_ref: !disable_column_ref,
                };
                if let Some(col_block) = encode_subarray_column(
                    cluster_points,
                    &quantized_data,
                    Some(block_gene_o2n),
                    index_codec,
                    sorted_index_codec,
                    column_template_count,
                    adaptive,
                    column_template_max,
                    gene_kruskal_cfg,
                    gene_block_size_hint,
                ) {
                    let (col_block, order) = col_block;
                    let bytes = col_block.total_bytes();
                    if best_col.as_ref().map(|b| bytes < b.2).unwrap_or(true) {
                        best_col = Some((EncodedClusterBlock::Column(col_block), order, bytes));
                        col_selected_adaptive = adaptive;
                    }
                }
                column_encode_us.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
            };

            try_col_encode(column_template_adaptive);
            if column_template_adaptive {
                try_col_encode(false);
            }
            best_col
        };

        let row_available = row_candidate.is_some();
        let col_available = col_candidate.is_some();
        let (encoded, local_order, bytes) = match (row_candidate, col_candidate) {
            (Some(r), Some(c)) => {
                // Default: row wins ties (smaller-or-equal beats column).
                // With --prefer-column-on-tie: column wins ties, since column
                // tiles decode ~3-4x faster than row-MST tiles per tile and a
                // byte-for-byte tie should be broken on decode speed.
                let pick_row = if prefer_column_on_tie {
                    r.2 < c.2
                } else {
                    r.2 <= c.2
                };
                if pick_row {
                    r
                } else {
                    c
                }
            }
            (Some(r), None) => r,
            (None, Some(c)) => c,
            (None, None) => {
                return Err(anyhow::anyhow!(
                    "Both row and column encoding failed for cluster {} gene-block",
                    cluster_idx
                ))
            }
        };

        Ok((
            encoded,
            local_order,
            bytes,
            row_template_adaptive && row_available && !row_selected_adaptive,
            column_template_adaptive && col_available && !col_selected_adaptive,
        ))
    };

    // Each cluster produces one or more encoded blocks (one per gene block).
    // positions / row_order come from the first gene block's DFS order.
    let encode_tiles_start = Instant::now();
    reset_row_mst_order_stats();
    reset_codec_timing_stats();
    reset_column_memory_stats();
    let cluster_results: Vec<
        anyhow::Result<
            Option<(
                usize,
                Vec<EncodedClusterBlock>,
                Vec<Vec<u32>>,
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

            // Encode all gene blocks for this cluster concurrently, then re-order by
            // block index to keep output deterministic.
            let tile_results: Vec<
                anyhow::Result<(usize, EncodedClusterBlock, Vec<u32>, usize, bool, bool)>,
            > = block_gene_o2ns
                .par_iter()
                .enumerate()
                .map(|(gb_idx, block_gene_o2n)| {
                    let gene_block_range = &gene_block_ranges[gb_idx];
                    let gene_block_size = gene_block_range.end - gene_block_range.start;
                    let (encoded, local_order, bytes, row_fb, col_fb) =
                        encode_tile(&cluster_points, block_gene_o2n, gene_block_size, cluster_idx)?;
                    Ok((gb_idx, encoded, local_order, bytes, row_fb, col_fb))
                })
                .collect();

            let mut ordered_tiles = Vec::with_capacity(effective_gene_blocks);
            for result in tile_results {
                ordered_tiles.push(result?);
            }
            ordered_tiles.sort_unstable_by_key(|(gb_idx, _, _, _, _, _)| *gb_idx);

            let mut tile_blocks = Vec::with_capacity(effective_gene_blocks);
            let mut tile_local_orders: Vec<Vec<u32>> =
                Vec::with_capacity(effective_gene_blocks);
            let mut total_bytes = 0usize;
            let mut first_order: Option<Vec<u32>> = None;
            let mut any_row_fallback = false;
            let mut any_col_fallback = false;

            for (_gb_idx, encoded, local_order, bytes, row_fb, col_fb) in ordered_tiles {
                total_bytes += bytes;
                if first_order.is_none() {
                    first_order = Some(local_order.clone());
                }
                any_row_fallback |= row_fb;
                any_col_fallback |= col_fb;
                tile_blocks.push(encoded);
                tile_local_orders.push(local_order);
            }

            let local_order =
                first_order.unwrap_or_else(|| (0..cluster_points.len() as u32).collect());
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
                tile_blocks,
                tile_local_orders,
                cluster_positions,
                cluster_row_order,
                cluster_points.len(),
                total_bytes,
                any_row_fallback,
                any_col_fallback,
            )))
        })
        .collect();

    let mut ordered = Vec::new();
    for result in cluster_results {
        if let Some(v) = result? {
            ordered.push(v);
        }
    }
    ordered.sort_unstable_by_key(|(cluster_idx, _, _, _, _, _, _, _, _)| *cluster_idx);

    let mut row_block_count = 0usize;
    let mut column_block_count = 0usize;
    let mut row_mode_parent_total = 0usize;
    let mut row_mode_full_total = 0usize;
    let mut row_mode_template_total = 0usize;
    let mut row_adaptive_fallback_wins = 0usize;
    let mut col_adaptive_fallback_wins = 0usize;
    let mut col_value_model_global_blocks = 0usize;
    let mut col_value_model_per_gene_blocks = 0usize;
    let mut col_mode_raw_total = 0usize;
    let mut col_mode_ref_total = 0usize;
    let mut col_mode_template_total = 0usize;
    let mut col_blocks_with_raw = 0usize;
    let mut col_blocks_with_ref = 0usize;
    let mut col_blocks_with_template = 0usize;
    let mut col_blocks_raw_only = 0usize;
    let mut col_blocks_ref_only = 0usize;
    let mut col_blocks_template_only = 0usize;
    let mut col_blocks_mixed_modes = 0usize;
    for (
        _idx,
        tile_blocks,
        cluster_tile_local_orders,
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
        for encoded in &tile_blocks {
            match encoded {
                EncodedClusterBlock::RowMst(block) => {
                    row_block_count += 1;
                    for mode in block.child_modes.decode_all().unwrap_or_default() {
                        match mode {
                            0 => row_mode_parent_total += 1,
                            1 => row_mode_full_total += 1,
                            2 => row_mode_template_total += 1,
                            _ => {}
                        }
                    }
                }
                EncodedClusterBlock::Column(block) => {
                    column_block_count += 1;
                    if block.uses_per_gene_values() {
                        col_value_model_per_gene_blocks += 1;
                    } else {
                        col_value_model_global_blocks += 1;
                    }
                    let modes = block.posting_modes.decode_all().unwrap_or_default();
                    let mut has_raw = false;
                    let mut has_ref = false;
                    let mut has_template = false;
                    for mode in modes {
                        if mode == EncodedColumnBlock::MODE_RAW {
                            col_mode_raw_total += 1;
                            has_raw = true;
                        } else if mode == EncodedColumnBlock::MODE_REF {
                            col_mode_ref_total += 1;
                            has_ref = true;
                        } else if mode == EncodedColumnBlock::MODE_TEMPLATE {
                            col_mode_template_total += 1;
                            has_template = true;
                        }
                    }
                    let mode_kinds = (has_raw as u8) + (has_ref as u8) + (has_template as u8);
                    if has_raw {
                        col_blocks_with_raw += 1;
                    }
                    if has_ref {
                        col_blocks_with_ref += 1;
                    }
                    if has_template {
                        col_blocks_with_template += 1;
                    }
                    if mode_kinds > 1 {
                        col_blocks_mixed_modes += 1;
                    } else if has_raw {
                        col_blocks_raw_only += 1;
                    } else if has_ref {
                        col_blocks_ref_only += 1;
                    } else if has_template {
                        col_blocks_template_only += 1;
                    }
                }
            }
        }
        total_mst_bytes += bytes;
        cluster_sizes.push(cluster_ncells);
        positions.extend(cluster_positions);
        row_order.extend(cluster_row_order);
        encoded_blocks.extend(tile_blocks);
        tile_local_orders.extend(cluster_tile_local_orders);
    }
    let timing_encode_tiles_ms = encode_tiles_start.elapsed().as_millis() as u64;
    if report_memory {
        let col_mem = column_memory_stats();
        log_memory_checkpoint("after encode tiles");
        info!(
            "Column memory estimates total/max_tile MiB: expressions={:.1}/{:.1} reordered={:.1}/{:.1} postings={:.1}/{:.1} posting_rows={:.1}/{:.1} gene_sparse={:.1}/{:.1} vals_flat={:.1}/{:.1}",
            mib(col_mem.expressions_bytes_total),
            mib(col_mem.expressions_bytes_max_tile),
            mib(col_mem.reordered_bytes_total),
            mib(col_mem.reordered_bytes_max_tile),
            mib(col_mem.postings_bytes_total),
            mib(col_mem.postings_bytes_max_tile),
            mib(col_mem.posting_rows_bytes_total),
            mib(col_mem.posting_rows_bytes_max_tile),
            mib(col_mem.gene_sparse_bytes_total),
            mib(col_mem.gene_sparse_bytes_max_tile),
            mib(col_mem.vals_flat_bytes_total),
            mib(col_mem.vals_flat_bytes_max_tile),
        );
    }
    let timing_row_mst_encode_ms = row_mst_encode_us.load(Ordering::Relaxed) / 1_000;
    let timing_column_encode_ms = column_encode_us.load(Ordering::Relaxed) / 1_000;
    let codec_timing = codec_timing_stats();

    info!(
        "Cluster payload selection: row_blocks={} column_blocks={}",
        row_block_count, column_block_count
    );
    if row_block_count > 0 {
        info!(
            "Row-MST child mode usage: parent={} full={} template={} total={}",
            row_mode_parent_total,
            row_mode_full_total,
            row_mode_template_total,
            row_mode_parent_total + row_mode_full_total + row_mode_template_total
        );
    }
    let row_order_stats = row_mst_order_stats();
    info!(
        "Row-MST stream order usage: projection_full_backrefs={}",
        row_order_stats.projection_full_backrefs
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
        let col_mode_total = col_mode_raw_total + col_mode_ref_total + col_mode_template_total;
        info!(
            "Column posting mode usage: raw={} ref={} template={} total={}",
            col_mode_raw_total, col_mode_ref_total, col_mode_template_total, col_mode_total
        );
        info!(
            "Column block mode coverage: blocks={} with_raw={} with_ref={} with_template={} raw_only={} ref_only={} template_only={} mixed={}",
            column_block_count,
            col_blocks_with_raw,
            col_blocks_with_ref,
            col_blocks_with_template,
            col_blocks_raw_only,
            col_blocks_ref_only,
            col_blocks_template_only,
            col_blocks_mixed_modes
        );
    }
    info!(
        "Row-MST candidate timing breakdown: expressions={}ms build_mst={}ms traversal_setup={}ms template_select={}ms child_modes={}ms stream_encode={}ms",
        codec_timing.row_expressions_us / 1_000,
        codec_timing.row_build_mst_us / 1_000,
        codec_timing.row_traversal_setup_us / 1_000,
        codec_timing.row_template_select_us / 1_000,
        codec_timing.row_child_modes_us / 1_000,
        codec_timing.row_stream_encode_us / 1_000
    );
    info!(
        "Column candidate timing breakdown: expressions_postings={}ms gene_mst={}ms template_select={}ms posting_decisions={}ms stream_encode={}ms",
        codec_timing.column_expressions_postings_us / 1_000,
        codec_timing.column_gene_mst_us / 1_000,
        codec_timing.column_template_select_us / 1_000,
        codec_timing.column_posting_decisions_us / 1_000,
        codec_timing.column_stream_encode_us / 1_000
    );

    if verify_lossless && !quantize_values {
        let cluster_sizes_u32: Vec<u32> = cluster_sizes.iter().map(|&s| s as u32).collect();
        let reconstructed = reconstruct_count_matrix_from_payload(
            &encoded_blocks,
            &cluster_sizes_u32,
            data.cols(),
            &tile_local_orders,
        )?;
        compare_count_matrix_permutation_invariant(data, &reconstructed)?;
        info!(
            "In-memory decompress verify PASSED: rows={} cols={} nnz={} (matches input under row/column permutation)",
            reconstructed.rows(),
            reconstructed.cols(),
            reconstructed.nnz()
        );
    }

    let sse = if quantize_values {
        sparse_quantization_sse(data, &quantized_data)?
    } else {
        0.0
    };
    let total_entries = (points.len() * data.cols()).max(1);
    let mse = sse / total_entries as f64;
    let rmse = mse.sqrt();
    let cluster_sizes_u32: Vec<u32> = cluster_sizes.iter().map(|&s| s as u32).collect();
    let ncols_u32 = data.cols() as u32;
    let compressed_bytes_estimate = estimate_compressed_payload_size(
        payload_compression,
        &encoded_blocks,
        &positions,
        &cluster_sizes_u32,
        ncols_u32,
        &tile_local_orders,
    )?;
    let rate_bits_per_value = (compressed_bytes_estimate as f64 * 8.0) / total_entries as f64;

    Ok(CompressionResult {
        quantizers_requested: cell_blocks_requested,
        quantizers_used: clusters.len(),
        quantizer_bins,
        total_mst_bytes,
        gzip_bytes_estimate: compressed_bytes_estimate,
        rate_bits_per_value,
        mse,
        rmse,
        timing_cluster_prep_ms,
        timing_gene_order_ms,
        timing_encode_tiles_ms,
        timing_row_mst_encode_ms,
        timing_column_encode_ms,
        cluster_sizes,
        encoded_blocks,
        positions,
        tile_local_orders,
    })
}

struct DecodedPayload {
    encoded_blocks: Vec<EncodedClusterBlock>,
    positions: Vec<DatalessPoint>,
    row_order: Vec<u32>,
    gene_order: Vec<u32>,
    cluster_sizes: Vec<u32>,
    ncols: u32,
    tile_local_orders: Vec<Vec<u32>>,
}

/// Fine-grained breakdown of the load phase, in microseconds. Useful for figuring out
/// whether `decompress_load_ms` is dominated by the outer (zstd/gzip/7z) stream read
/// or by bincode-deserialising the tile structs.
#[derive(Clone, Copy, Debug, Default)]
struct DecodeLoadStats {
    outer_decompress_us: u128,
    bincode_decode_us: u128,
    raw_payload_bytes: usize,
}

fn decode_clustered_payload(
    input: &Path,
    input_compression: Option<PayloadCompressionArg>,
) -> anyhow::Result<(DecodedPayload, DecodeLoadStats)> {
    let compression = input_compression.unwrap_or_else(|| {
        input
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| {
                if ext.eq_ignore_ascii_case("gz") {
                    PayloadCompressionArg::Gzip
                } else if ext.eq_ignore_ascii_case("7z") {
                    PayloadCompressionArg::SevenZip
                } else {
                    PayloadCompressionArg::Zstd
                }
            })
            .unwrap_or(PayloadCompressionArg::Zstd)
    });
    let config = bincode::config::standard()
        .with_little_endian()
        .with_fixed_int_encoding();

    let t_outer = Instant::now();
    let raw_payload = match compression {
        PayloadCompressionArg::Gzip => {
            let file = File::open(input)?;
            let mut reader = BufReader::new(file);
            let mut gz_reader = BufReader::new(GzDecoder::new(&mut reader));
            let mut raw = Vec::new();
            std::io::Read::read_to_end(&mut gz_reader, &mut raw)?;
            raw
        }
        PayloadCompressionArg::Zstd => {
            let file = File::open(input)?;
            let mut reader = BufReader::new(file);
            let mut zstd_reader = BufReader::new(ZstdDecoder::new(&mut reader)?);
            let mut raw = Vec::new();
            std::io::Read::read_to_end(&mut zstd_reader, &mut raw)?;
            raw
        }
        PayloadCompressionArg::SevenZip => {
            let output = Command::new("7z").arg("x").arg("-so").arg(input).output()?;
            if !output.status.success() {
                anyhow::bail!("7z decode failed for {}", input.display());
            }
            output.stdout
        }
    };
    let outer_decompress_us = t_outer.elapsed().as_micros();
    let raw_payload_bytes = raw_payload.len();
    let mut payload_reader = Cursor::new(raw_payload);

    let t_bincode = Instant::now();
    let encoded_blocks: Vec<EncodedClusterBlock> =
        bincode::decode_from_std_read(&mut payload_reader, config)?;
    let positions: Vec<DatalessPoint> = bincode::decode_from_std_read(&mut payload_reader, config)?;
    let end = payload_reader.get_ref().len();
    let pos = payload_reader.position() as usize;
    let row_order: Vec<u32> = if pos < end {
        bincode::decode_from_std_read(&mut payload_reader, config)?
    } else {
        Vec::new()
    };
    let pos = payload_reader.position() as usize;
    let gene_order: Vec<u32> = if pos < end {
        bincode::decode_from_std_read(&mut payload_reader, config)?
    } else {
        Vec::new()
    };
    let pos = payload_reader.position() as usize;
    let cluster_sizes: Vec<u32> = if pos < end {
        bincode::decode_from_std_read(&mut payload_reader, config)?
    } else {
        Vec::new()
    };
    let pos = payload_reader.position() as usize;
    let ncols: u32 = if pos < end {
        bincode::decode_from_std_read(&mut payload_reader, config)?
    } else {
        0
    };
    let pos = payload_reader.position() as usize;
    let tile_local_orders: Vec<Vec<u32>> = if pos < end {
        bincode::decode_from_std_read(&mut payload_reader, config)?
    } else {
        Vec::new()
    };
    let bincode_decode_us = t_bincode.elapsed().as_micros();
    Ok((
        DecodedPayload {
            encoded_blocks,
            positions,
            row_order,
            gene_order,
            cluster_sizes,
            ncols,
            tile_local_orders,
        },
        DecodeLoadStats {
            outer_decompress_us,
            bincode_decode_us,
            raw_payload_bytes,
        },
    ))
}

/// Reconstruct a sparse count matrix from the encoded payload using the
/// `cluster_sizes` + `ncols` + `tile_local_orders` metadata embedded in the payload.
///
/// The returned matrix has the same shape and nnz as the original, but its row order
/// is the encoder's cluster + SVD order and its column order is the encoder's
/// gene-SVD order. Both are permutations of the original input matrix.
///
/// Within a cluster, each gene-block tile may have chosen its own row ordering (DFS
/// for row-MST tiles, value-aware reorder for column tiles). `tile_local_orders[k]`
/// gives the permutation that maps tile `k`'s emitted-row index to the cluster-local
/// row index, so we can recombine the partial rows from each tile into a single global
/// row.
/// Per-phase timing accumulators in microseconds. Each thread adds to the atomics, so the
/// reported numbers are *total CPU work* (sum across threads) for that phase, not wall.
#[derive(Default)]
struct ReconstructPhaseStats {
    tile_decode_us: AtomicU64,
    row_assemble_us: AtomicU64,
    sort_check_us: AtomicU64,
}

fn record_us(slot: &AtomicU64, t: Instant) {
    let us = t.elapsed().as_micros() as u64;
    slot.fetch_add(us, Ordering::Relaxed);
}

/// Reconstruct the sparse count matrix from the decoded tile structs.
///
/// Tile-parallel: spawns one Rayon task per tile (`n_clusters * gene_blocks_per_cluster`
/// tasks total) so the work-stealer can load-balance across all available threads.
/// Per-cluster row assembly and CSR stitch are cheap and run separately.
fn reconstruct_count_matrix_from_payload(
    encoded_blocks: &[EncodedClusterBlock],
    cluster_sizes: &[u32],
    ncols: usize,
    tile_local_orders: &[Vec<u32>],
) -> anyhow::Result<CsMat<u16>> {
    if cluster_sizes.is_empty() {
        anyhow::bail!(
            "Cannot reconstruct count matrix: payload has no cluster_sizes (pre-decoder format)"
        );
    }
    if encoded_blocks.is_empty() {
        anyhow::bail!("Cannot reconstruct count matrix: payload has no encoded blocks");
    }
    if tile_local_orders.len() != encoded_blocks.len() {
        anyhow::bail!(
            "tile_local_orders.len() ({}) differs from encoded_blocks.len() ({})",
            tile_local_orders.len(),
            encoded_blocks.len()
        );
    }
    let n_clusters = cluster_sizes.len();
    if encoded_blocks.len() % n_clusters != 0 {
        anyhow::bail!(
            "encoded_blocks.len() ({}) is not a multiple of cluster_sizes.len() ({})",
            encoded_blocks.len(),
            n_clusters
        );
    }
    let gene_blocks_per_cluster = encoded_blocks.len() / n_clusters;
    let nrows: usize = cluster_sizes.iter().map(|&s| s as usize).sum();

    // Reconstruct the per-tile gene-block offsets. With `gene_blocks_per_cluster > 1`,
    // each tile encodes genes using block-local indices (0..block_size-1), so different
    // tiles within a cluster would collide on the same `(row, gene)` if we didn't offset
    // them into a shared post-SVD column space. The encoder partitions `ncols` into
    // `gene_blocks_per_cluster` contiguous ranges, distributing the remainder to the
    // first blocks.
    let gene_block_starts: Vec<usize> = {
        let mut starts = Vec::with_capacity(gene_blocks_per_cluster);
        let block_sz = ncols / gene_blocks_per_cluster;
        let remainder = ncols % gene_blocks_per_cluster;
        let mut acc = 0usize;
        for b in 0..gene_blocks_per_cluster {
            starts.push(acc);
            let sz = block_sz + if b < remainder { 1 } else { 0 };
            acc += sz;
        }
        starts
    };

    // Validate tile-level invariants once, up front, so the hot paths don't have to.
    for c in 0..n_clusters {
        let csize = cluster_sizes[c] as usize;
        let start = c * gene_blocks_per_cluster;
        for tile_offset in 0..gene_blocks_per_cluster {
            let tile_global = start + tile_offset;
            let order = &tile_local_orders[tile_global];
            if order.len() != csize {
                anyhow::bail!(
                    "tile_local_orders[{}] len {} differs from cluster_sizes[{}]={}",
                    tile_global,
                    order.len(),
                    c,
                    csize
                );
            }
            if encoded_blocks[tile_global].num_cells() != csize {
                anyhow::bail!(
                    "Cluster {} tile {} num_cells={} differs from cluster_sizes[{}]={}",
                    c,
                    tile_offset,
                    encoded_blocks[tile_global].num_cells(),
                    c,
                    csize
                );
            }
        }
    }

    let phase = ReconstructPhaseStats::default();

    // --- Phase A: decode tile bytes -> sparse rows -----------------------------------
    // What this owns: arithmetic decoding, MST traversal, parent/template edits, value
    // bin lookups. This is the heavy CPU step (~95% of reconstruct time in practice).
    //
    // Output shape: `tile_sparse_rows[tile_global][emitted_row_idx] = Vec<(gene, value)>`
    // with `gene` still in *tile-local* coordinates (0..gene_block_size).
    let decode_one_tile = |tile_global: usize| -> Vec<Vec<(u32, u16)>> {
        let t = Instant::now();
        let rows = encoded_blocks[tile_global].decode_rows();
        record_us(&phase.tile_decode_us, t);
        rows
    };

    let tile_sparse_rows: Vec<Vec<Vec<(u32, u16)>>> = (0..encoded_blocks.len())
        .into_par_iter()
        .map(decode_one_tile)
        .collect();

    // --- Phase B: assemble per-cluster row buffers -----------------------------------
    // What this owns: applying `tile_local_orders` to remap emitted-row index back to
    // cluster-local row, offsetting tile-local gene id to global column, and concat'ing
    // the disjoint gene-block ranges into one sorted per-row vector.
    //
    // Clusters are disjoint so this is trivially parallel over clusters.
    let per_cluster_rows: anyhow::Result<Vec<Vec<Vec<(u32, u16)>>>> = {
        let assemble_one_cluster = |c: usize| -> anyhow::Result<Vec<Vec<(u32, u16)>>> {
            let t_assemble = Instant::now();
            let csize = cluster_sizes[c] as usize;
            let start = c * gene_blocks_per_cluster;
            let mut rows: Vec<Vec<(u32, u16)>> = vec![Vec::new(); csize];
            for tile_offset in 0..gene_blocks_per_cluster {
                let tile_global = start + tile_offset;
                let order = &tile_local_orders[tile_global];
                let gene_offset = gene_block_starts[tile_offset];
                let tile_rows = &tile_sparse_rows[tile_global];
                for (encoded_row_idx, sparse_row) in tile_rows.iter().enumerate() {
                    let cluster_local_row = order[encoded_row_idx] as usize;
                    if cluster_local_row >= csize {
                        anyhow::bail!(
                            "tile_local_orders[{}][{}] = {} >= cluster_size {}",
                            tile_global,
                            encoded_row_idx,
                            cluster_local_row,
                            csize
                        );
                    }
                    let row_buf = &mut rows[cluster_local_row];
                    row_buf.reserve(sparse_row.len());
                    for &(gene, value) in sparse_row.iter() {
                        if value == 0 {
                            continue;
                        }
                        let gene_idx = gene_offset + gene as usize;
                        if gene_idx >= ncols {
                            anyhow::bail!(
                                "Decoded gene index {} >= ncols {} in cluster {} tile {} row {}",
                                gene_idx,
                                ncols,
                                c,
                                tile_offset,
                                encoded_row_idx
                            );
                        }
                        row_buf.push((gene_idx as u32, value));
                    }
                }
            }
            record_us(&phase.row_assemble_us, t_assemble);

            // Different gene blocks contribute disjoint, ascending column ranges, so the
            // concatenated per-row buffer is already sorted across blocks. Within a single
            // tile, `decode_rows()` produces (gene, value) pairs in ascending gene order.
            // Still, defend against future changes with a cheap sorted-check + sort.
            let t_sort = Instant::now();
            for row in rows.iter_mut() {
                let mut sorted = true;
                for w in row.windows(2) {
                    if w[0].0 > w[1].0 {
                        sorted = false;
                        break;
                    }
                }
                if !sorted {
                    row.sort_unstable_by_key(|&(col, _)| col);
                }
            }
            record_us(&phase.sort_check_us, t_sort);
            Ok(rows)
        };
        (0..n_clusters)
            .into_par_iter()
            .map(assemble_one_cluster)
            .collect()
    };
    let per_cluster_rows = per_cluster_rows?;

    // --- Phase C: stitch per-cluster CSR portions into the final CsMat ---------------
    // Pass 1 (serial, cheap): walk per_cluster_rows reading only Vec lengths
    // to build `indptr` and `cluster_nnz_offsets` (offset into `indices`/
    // `data` where each cluster's slice starts). Costs O(nrows + n_clusters).
    //
    // Pass 2 (parallel over clusters): each cluster has a *disjoint* output
    // slice [cluster_nnz_offsets[c]..cluster_nnz_offsets[c+1]), so writes are
    // race-free without any synchronisation. Uses `set_len` + raw pointer
    // writes to skip Vec::push's per-element capacity check.
    let t_stitch = Instant::now();

    let mut indptr: Vec<usize> = Vec::with_capacity(nrows + 1);
    indptr.push(0usize);
    let mut cluster_nnz_offsets: Vec<usize> = Vec::with_capacity(n_clusters + 1);
    cluster_nnz_offsets.push(0usize);
    let mut running = 0usize;
    for (c, rows) in per_cluster_rows.iter().enumerate() {
        if rows.len() != cluster_sizes[c] as usize {
            anyhow::bail!(
                "per_cluster_rows[{}].len()={} differs from cluster_sizes[{}]={}",
                c,
                rows.len(),
                c,
                cluster_sizes[c]
            );
        }
        for row in rows.iter() {
            running = running.saturating_add(row.len());
            indptr.push(running);
        }
        cluster_nnz_offsets.push(running);
    }
    let total_nnz = running;

    let mut indices: Vec<usize> = Vec::with_capacity(total_nnz);
    let mut data: Vec<u16> = Vec::with_capacity(total_nnz);
    // SAFETY: `with_capacity(N)` guarantees at least N elements of backing
    // storage. We initialise every slot in the parallel loop below before the
    // Vecs are observed by any external code (CsMat::new reads len, not
    // capacity).
    unsafe {
        indices.set_len(total_nnz);
        data.set_len(total_nnz);
    }
    // Cast the mutable raw pointers to `usize` so they can be sent across
    // Rayon worker boundaries. Raw pointers are `!Send` in safe Rust; the
    // pattern `as usize` is the standard idiom for sharing them, valid here
    // because the per-cluster slices we write to are disjoint by construction.
    let indices_addr = indices.as_mut_ptr() as usize;
    let data_addr = data.as_mut_ptr() as usize;
    let nnz_offsets = &cluster_nnz_offsets;
    per_cluster_rows
        .into_par_iter()
        .enumerate()
        .try_for_each(|(c, rows)| -> anyhow::Result<()> {
            let start = nnz_offsets[c];
            let end = nnz_offsets[c + 1];
            let mut off = start;
            let indices_ptr = indices_addr as *mut usize;
            let data_ptr = data_addr as *mut u16;
            for row in rows.into_iter() {
                for (col, val) in row.into_iter() {
                    if off >= end {
                        return Err(anyhow::anyhow!(
                            "stitch overflow: cluster {} wrote past its slice [{}, {})",
                            c,
                            start,
                            end
                        ));
                    }
                    // SAFETY: `off` is in cluster c's exclusive range
                    // [start, end), and clusters have disjoint ranges, so no
                    // two threads write to the same address.
                    unsafe {
                        indices_ptr.add(off).write(col as usize);
                        data_ptr.add(off).write(val);
                    }
                    off += 1;
                }
            }
            debug_assert_eq!(
                off, end,
                "cluster {} wrote {} entries but expected slice [{},{}) of len {}",
                c,
                off - start,
                start,
                end,
                end - start
            );
            Ok(())
        })?;
    let stitch_us = t_stitch.elapsed().as_micros();

    let tile_decode_us = phase.tile_decode_us.load(Ordering::Relaxed);
    let row_assemble_us = phase.row_assemble_us.load(Ordering::Relaxed);
    let sort_check_us = phase.sort_check_us.load(Ordering::Relaxed);
    tracing::info!(
        "Decompress phase breakdown (threads={}): tile_decode={}us row_assemble={}us sort_check={}us stitch={}us  [sums across threads; total_tiles={}, n_clusters={}, gene_blocks_per_cluster={}]",
        rayon::current_num_threads(),
        tile_decode_us,
        row_assemble_us,
        sort_check_us,
        stitch_us,
        encoded_blocks.len(),
        n_clusters,
        gene_blocks_per_cluster,
    );

    Ok(CsMat::new((nrows, ncols), indptr, indices, data))
}

#[allow(dead_code)]
fn reconstruct_csr_from_clustered_payload(
    encoded_blocks: &[EncodedClusterBlock],
    row_order: &[u32],
    gene_order: &[u32],
    nrows: usize,
    ncols: usize,
) -> anyhow::Result<CsMat<u16>> {
    if row_order.is_empty() {
        anyhow::bail!(
            "Cannot reconstruct CSR: payload has no row_order (and possibly no gene_order); current encoder omits these from the bincode stream"
        );
    }
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

/// Compare two CSR matrices up to row + column permutation.
///
/// Verifies:
///   - shape (rows, cols) match
///   - nnz match
///   - total value sum matches
///   - sorted row "fingerprints" match (each row = sorted multiset of its non-zero values)
///   - sorted column "fingerprints" match
///
/// This is strong enough to confirm the decoder reconstructs the same count matrix
/// (counts, structure) as the input, even though row order is encoder-driven (clustered +
/// SVD-seriated) and column order is gene-SVD-seriated, neither of which matches the input.
fn compare_count_matrix_permutation_invariant(
    expected: &CsMat<u16>,
    actual: &CsMat<u16>,
) -> anyhow::Result<()> {
    if expected.shape() != actual.shape() {
        anyhow::bail!(
            "Shape mismatch: expected {}x{}, actual {}x{}",
            expected.rows(),
            expected.cols(),
            actual.rows(),
            actual.cols()
        );
    }
    if expected.nnz() != actual.nnz() {
        anyhow::bail!(
            "nnz mismatch: expected {}, actual {}",
            expected.nnz(),
            actual.nnz()
        );
    }

    fn row_fingerprints(m: &CsMat<u16>) -> Vec<Vec<u16>> {
        let mut out: Vec<Vec<u16>> = (0..m.rows())
            .map(|r| {
                let row = m.outer_view(r);
                let mut vals: Vec<u16> = row
                    .map(|view| view.iter().map(|(_, &v)| v).collect())
                    .unwrap_or_default();
                vals.sort_unstable();
                vals
            })
            .collect();
        out.sort_unstable();
        out
    }

    fn col_fingerprints(m: &CsMat<u16>) -> Vec<Vec<u16>> {
        let csc = m.to_csc();
        let mut out: Vec<Vec<u16>> = (0..csc.cols())
            .map(|c| {
                let col = csc.outer_view(c);
                let mut vals: Vec<u16> = col
                    .map(|view| view.iter().map(|(_, &v)| v).collect())
                    .unwrap_or_default();
                vals.sort_unstable();
                vals
            })
            .collect();
        out.sort_unstable();
        out
    }

    let exp_sum: u64 = expected.data().iter().map(|&v| v as u64).sum();
    let act_sum: u64 = actual.data().iter().map(|&v| v as u64).sum();
    if exp_sum != act_sum {
        anyhow::bail!(
            "Total value sum mismatch: expected {}, actual {}",
            exp_sum,
            act_sum
        );
    }

    let exp_rows = row_fingerprints(expected);
    let act_rows = row_fingerprints(actual);
    if exp_rows != act_rows {
        anyhow::bail!("Row fingerprints differ (sorted by value multiset per row)");
    }
    let exp_cols = col_fingerprints(expected);
    let act_cols = col_fingerprints(actual);
    if exp_cols != act_cols {
        anyhow::bail!("Column fingerprints differ (sorted by value multiset per column)");
    }
    Ok(())
}

#[allow(dead_code)]
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
    platform: Platform,
    pos_x_col: Option<usize>,
    pos_y_col: Option<usize>,
) -> (usize, usize) {
    match platform {
        Platform::Visium => (4, 5),
        Platform::Xenium => (1, 2),
        Platform::SingleCell => (pos_x_col.unwrap_or(0), pos_y_col.unwrap_or(1)),
    }
}

fn coordinate_mode_label(
    platform: Platform,
    input_pos: Option<&PathBuf>,
    matrix_orientation: MatrixOrientation,
) -> &'static str {
    match (platform, input_pos, matrix_orientation) {
        (Platform::SingleCell, _, _) | (_, None, _) => "none",
        (_, Some(_), MatrixOrientation::CellGene) => "coordinates",
        (_, Some(_), MatrixOrientation::GeneCell) => "requested-but-dummy-gene-rows",
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
    Dump(DumpCommand),
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
    /// If set, do not write compressed payload to disk.
    #[arg(long = "no-output", default_value_t = false)]
    no_output: bool,
    /// Compression codec for payload output and size estimation.
    #[arg(long = "output-compression", value_enum, default_value_t = PayloadCompressionArg::Zstd)]
    output_compression: PayloadCompressionArg,
    /// Optional CSV path to append compression statistics per run.
    #[arg(long = "stats-csv")]
    stats_csv: Option<PathBuf>,
    #[arg(short = 'F', long = "pos-format", value_enum, default_value_t = InputPosType::Parquet)]
    pos_format: InputPosType,
    /// Platform of the input matrix. Defaults to `single-cell` when not specified
    /// (no positions required); use `visium` or `xenium` for spatial data with positions.
    #[arg(short = 'P', long = "platform", value_enum, default_value_t = Platform::SingleCell)]
    platform: Platform,
    /// Matrix row/column orientation used by the encoder.
    #[arg(long = "matrix-orientation", value_enum, default_value_t = MatrixOrientation::GeneCell)]
    matrix_orientation: MatrixOrientation,
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
    /// Number of cell clusters (blocks) used by the cell clustering stage.
    #[arg(long = "cell-blocks", default_value_t = 6)]
    cell_blocks: usize,
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
    /// If set, cut MST edges larger than `median_edge_weight * factor`, producing a forest.
    #[arg(long = "forest-cut-factor")]
    forest_cut_factor: Option<f32>,
    /// Number of nearby rows on each side to connect for row-MST candidate edges.
    #[arg(long = "row-mst-window", default_value_t = 8)]
    row_mst_window: usize,
    /// Optional RNG seed for cell clustering/SVD stages.
    #[arg(long = "set-seed", visible_alias = "cluster-seed")]
    set_seed: Option<u64>,
    /// How to compute the gene (column) reordering: `projection` (hash-projection seriation, fast)
    /// or `svd` (truncated SVD on the sparse matrix, uses right singular vectors).
    #[arg(long = "gene-reorder-method", value_enum, default_value_t = GeneReorderMethod::Projection)]
    gene_reorder_method: GeneReorderMethod,
    /// Use randomized joint SVD for cell clusters and gene order.
    #[arg(long = "joint-svd-fast", default_value_t = false)]
    joint_svd_fast: bool,
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
    /// Split genes into this many contiguous blocks (after reordering) and build
    /// MST independently on each (cell_cluster × gene_block) tile.  Default 1 = no splitting.
    #[arg(long = "gene-blocks", default_value_t = 5)]
    gene_blocks: usize,
    /// Maximum number of row templates to consider in adaptive mode.
    #[arg(long = "row-template-max", default_value_t = 16)]
    row_template_max: usize,
    /// Disable the row-MST full-row fallback. When enabled, row-MST children
    /// must use parent/template edits unless no valid parent is available.
    #[arg(long = "disable-row-full-fallback", default_value_t = false)]
    disable_row_full_fallback: bool,
    /// Disable the row-MST candidate entirely, forcing tiles to use column encoding.
    #[arg(long = "disable-row-mst-candidate", default_value_t = false)]
    disable_row_mst_candidate: bool,
    /// Force every tile to use column encoding only.
    #[arg(long = "column-only", default_value_t = false)]
    column_only: bool,
    /// Disable the column candidate entirely, forcing tiles to use row-MST encoding.
    #[arg(long = "disable-column-candidate", default_value_t = false)]
    disable_column_candidate: bool,
    /// Disable parent/reference edit mode inside row-MST blocks.
    #[arg(long = "disable-row-parent-ref", default_value_t = false)]
    disable_row_parent_ref: bool,
    /// Disable reference posting mode inside column blocks.
    #[arg(long = "disable-column-ref", default_value_t = false)]
    disable_column_ref: bool,
    /// When the row-MST and column candidates produce the same compressed
    /// byte size for a tile, prefer the column candidate. Default behaviour
    /// prefers row-MST on ties (which can pick the slower-to-decode encoding).
    #[arg(long = "prefer-column-on-tie", default_value_t = false)]
    prefer_column_on_tie: bool,
    /// If set, emit combinatorial and entropy diagnostics for op_indices.
    #[arg(long = "report-index-bounds", default_value_t = false)]
    report_index_bounds: bool,
    /// If set, log RSS checkpoints and approximate heap sizes for key structures.
    #[arg(long = "report-memory", default_value_t = false)]
    report_memory: bool,
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

fn output_path_with_label_suffix(base: &Path, label: &str) -> PathBuf {
    let file_name = base
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("output.bin.zst");
    let parent = base.parent().unwrap_or_else(|| Path::new(""));

    if file_name.ends_with(".bin.gz") {
        let stem = file_name.trim_end_matches(".bin.gz");
        parent.join(format!("{}_{}.bin.gz", stem, label))
    } else if file_name.ends_with(".bin.zst") {
        let stem = file_name.trim_end_matches(".bin.zst");
        parent.join(format!("{}_{}.bin.zst", stem, label))
    } else if file_name.ends_with(".bin.7z") {
        let stem = file_name.trim_end_matches(".bin.7z");
        parent.join(format!("{}_{}.bin.7z", stem, label))
    } else if let Some(dot) = file_name.rfind('.') {
        let (stem, ext) = file_name.split_at(dot);
        parent.join(format!("{}_{}{}", stem, label, ext))
    } else {
        parent.join(format!("{}_{}", file_name, label))
    }
}

fn append_build_stats_csv(
    csv_path: &Path,
    stats_label: &str,
    cell_cluster_method: &str,
    gene_reorder_method: &str,
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
            "cell_cluster_method",
            "gene_reorder_method",
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
        cell_cluster_method.to_string(),
        gene_reorder_method.to_string(),
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
struct DumpCommand {
    #[arg(short = 'e', long = "encoded")]
    encoded: PathBuf,
    /// Override encoded payload compression format (`gzip` or `zstd`).
    #[arg(long = "input-compression", value_enum)]
    input_compression: Option<PayloadCompressionArg>,
    /// If set, write the decoded CSR count matrix to this path as a
    /// 10x-style HDF5 file. The dumped matrix is in decoder (cluster + SVD)
    /// order with placeholder `cell_*` / `gene_*` barcodes and feature names,
    /// since the payload does not store the row/column permutation back to
    /// the input H5.
    #[arg(short = 'o', long = "output")]
    output: Option<PathBuf>,
    #[arg(short = 'i', long)]
    input: PathBuf,
    /// Optional positions file (CSV or Parquet). Required unless platform is `single-cell`.
    #[arg(short = 'p', long = "input-pos")]
    input_pos: Option<PathBuf>,
    #[arg(short = 'F', long = "pos-format", value_enum, default_value_t = InputPosType::Parquet)]
    pos_format: InputPosType,
    /// Platform of the encoded payload. Defaults to `single-cell` when not specified;
    /// use `visium` or `xenium` for spatial payloads (and pass `--input-pos`).
    #[arg(short = 'P', long = "platform", value_enum, default_value_t = Platform::SingleCell)]
    platform: Platform,
    /// Matrix row/column orientation used when loading the ground-truth input.
    #[arg(long = "matrix-orientation", value_enum, default_value_t = MatrixOrientation::GeneCell)]
    matrix_orientation: MatrixOrientation,
    #[arg(long = "pos-x-col")]
    pos_x_col: Option<usize>,
    #[arg(long = "pos-y-col")]
    pos_y_col: Option<usize>,
    #[arg(long = "max-cells")]
    max_cells: Option<usize>,
}

/// Number of Rayon worker threads we install when the user has not set
/// `RAYON_NUM_THREADS`. Picked to match the SLURM-default `--cpus-per-task` in
/// `scripts/submit_patro_datasets_parallel_single_cell.sh` so running the bare
/// binary locally produces the same parallelism profile as on the cluster.
const DEFAULT_RAYON_THREADS: usize = 4;

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

    // Install a 4-thread global Rayon pool by default. If the user already set
    // `RAYON_NUM_THREADS` (or built a global pool by some other means) we leave
    // it alone -- `build_global` returns an Err on the second call.
    if std::env::var_os("RAYON_NUM_THREADS").is_none() {
        match rayon::ThreadPoolBuilder::new()
            .num_threads(DEFAULT_RAYON_THREADS)
            .build_global()
        {
            Ok(()) => tracing::debug!(
                "Installed default Rayon pool with {} threads (RAYON_NUM_THREADS unset)",
                DEFAULT_RAYON_THREADS
            ),
            Err(e) => tracing::debug!("Rayon global pool already initialised: {}", e),
        }
    }

    let cli = Cli::parse();

    match cli.command {
        Commands::Build(args) => {
            let positions_arg: Option<&PathBuf> = if args.platform == Platform::SingleCell {
                None
            } else {
                args.input_pos.as_ref()
            };
            let coordinate_mode =
                coordinate_mode_label(args.platform, positions_arg, args.matrix_orientation);
            let (pos_x_col, pos_y_col) =
                resolve_position_columns(args.platform, args.pos_x_col, args.pos_y_col);
            let positions_spec = positions_arg
                .map(|pos_path| (pos_path.as_path(), args.pos_format, pos_x_col, pos_y_col));
            if positions_spec.is_none()
                && args.platform != Platform::SingleCell
                && args.matrix_orientation == MatrixOrientation::CellGene
            {
                return Err(anyhow::anyhow!(
                    "input-pos is required for --platform {:?} with --matrix-orientation cell-gene",
                    args.platform
                )
                .into());
            }
            if positions_spec.is_some() && args.matrix_orientation == MatrixOrientation::GeneCell {
                warn!(
                    "input positions were provided with gene-cell orientation; gene rows cannot use cell coordinates, so dummy coordinates are used"
                );
            }
            let (csr, points) = load_10x_oriented(
                &args.input,
                positions_spec,
                args.matrix_orientation,
                args.max_cells,
            )?;
            if args.report_memory {
                log_memory_checkpoint("after input load");
                info!(
                    "Memory estimate: input_csr_heap={:.1} MiB points_heap={:.1} MiB rows={} cols={} nnz={} points={}",
                    mib(estimate_csr_heap_bytes(&csr)),
                    mib((points.capacity() * std::mem::size_of::<Point>()) as u64),
                    csr.rows(),
                    csr.cols(),
                    csr.nnz(),
                    points.len()
                );
            }

            let quantize_values = args.lossy && !args.lossy_lossless;
            let target_quantizers = args.lossy_quantizers.max(1);
            let target_cell_blocks = args.cell_blocks.max(1);
            let quantizer_bins = args.lossy_bins.max(1);
            let sweep_counts = normalize_quantizer_counts(target_quantizers, &args.lossy_sweep);
            let index_codec = IndexStreamCodec::from(args.index_codec);
            let sorted_index_codec = SortedIndexCodec::from(args.sorted_index_codec);
            let mst_weight_mode = MstWeightMode::from(args.mst_weight);
            let row_parent_ref_cost_ratio = if args.disable_row_full_fallback {
                None
            } else {
                Some(1.0)
            };
            let forest_cut_factor = args.forest_cut_factor.map(|f| f.max(0.0));
            let row_mst_window = args.row_mst_window.max(1);
            let disable_row_mst_candidate = args.disable_row_mst_candidate || args.column_only;

            info!(
                "Encoding mode: {} | output_compression={:?} lossy_quantizers={} cell_blocks={} bins={} sweep={:?} max_cluster_size={:?} cell_cluster_method=svd-joint knn_metric={:?} mst_weight={:?} row_mst_window={} row_mst_stream=projection-full-backrefs row_mst_column_order=existing column_row_order=projection matrix_orientation={:?} coordinate_mode={} input_rows={} input_cols={} input_nnz={} joint_svd_fast={} joint_svd_row_vectors=2 row_full_fallback={} disable_row_mst_candidate={} disable_column_candidate={} disable_row_parent_ref={} disable_column_ref={} prefer_column_on_tie={} local_to_global_mode=identity index_codec={:?} sorted_index_codec={:?} forest_cut_factor={:?} cluster_encoding=Hybrid column_template_count={} column_template_adaptive={} column_template_max={} row_template_adaptive={} row_template_max={}",
                if quantize_values { "lossy" } else { "lossless" },
                args.output_compression,
                target_quantizers,
                target_cell_blocks,
                quantizer_bins,
                sweep_counts,
                args.max_cluster_size,
                args.knn_metric,
                args.mst_weight,
                row_mst_window,
                args.matrix_orientation,
                coordinate_mode,
                csr.rows(),
                csr.cols(),
                csr.nnz(),
                args.joint_svd_fast,
                !args.disable_row_full_fallback,
                disable_row_mst_candidate,
                args.disable_column_candidate,
                args.disable_row_parent_ref,
                args.disable_column_ref,
                args.prefer_column_on_tie,
                args.index_codec,
                args.sorted_index_codec,
                forest_cut_factor,
                args.column_template_count,
                args.column_template_adaptive,
                args.column_template_max,
                args.row_template_adaptive,
                args.row_template_max
            );

            let sweep_configs: Vec<(GeneReorderMethod, String)> =
                vec![(args.gene_reorder_method, "svd-joint".to_string())];

            for (gene_reorder_method, config_label) in &sweep_configs {
                let gene_method_label = config_label.clone();
                info!(
                    "Running build: cell={}, gene={}",
                    config_label, gene_method_label
                );
                let mut metrics = Vec::new();
                let mut selected: Option<CompressionResult> = None;

                for &quantizer_count in &sweep_counts {
                    let result = run_clustered_compression(
                        &points,
                        &csr,
                        target_cell_blocks,
                        quantizer_bins,
                        quantize_values,
                        args.lossy_verify_lossless,
                        args.max_cluster_size,
                        args.knn_metric.into(),
                        mst_weight_mode,
                        index_codec,
                        sorted_index_codec,
                        row_parent_ref_cost_ratio,
                        forest_cut_factor,
                        row_mst_window,
                        args.column_template_count,
                        args.column_template_adaptive,
                        args.column_template_max,
                        args.row_template_adaptive,
                        args.row_template_max,
                        disable_row_mst_candidate,
                        args.disable_column_candidate,
                        args.disable_row_parent_ref,
                        args.disable_column_ref,
                        args.prefer_column_on_tie,
                        args.report_memory,
                        args.set_seed,
                        *gene_reorder_method,
                        args.joint_svd_fast,
                        args.gene_blocks,
                        args.output_compression,
                    )?;

                    info!(
                        "q={} (used={}): compressed≈{} bytes, rate={:.4} bits/value, mse={:.6}, rmse={:.6}",
                        result.quantizers_requested,
                        result.quantizers_used,
                        result.gzip_bytes_estimate,
                        result.rate_bits_per_value,
                        result.mse,
                        result.rmse
                    );
                    info!(
                        "timing: cluster_prep={}ms gene_order={}ms encode_tiles={}ms row_mst_encode={}ms column_encode={}ms",
                        result.timing_cluster_prep_ms,
                        result.timing_gene_order_ms,
                        result.timing_encode_tiles_ms,
                        result.timing_row_mst_encode_ms,
                        result.timing_column_encode_ms
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
                        "  q={} (used={}): rate={:.4}, mse={:.6}, rmse={:.6}, compressed≈{} bytes",
                        metric.quantizers_requested,
                        metric.quantizers_used,
                        metric.rate_bits_per_value,
                        metric.mse,
                        metric.rmse,
                        metric.gzip_bytes_estimate
                    );
                }

                let selected_cluster_sizes_u32: Vec<u32> =
                    selected.cluster_sizes.iter().map(|&s| s as u32).collect();
                let selected_ncols_u32 = csr.cols() as u32;
                let actual_gzip_bytes = if args.no_output {
                    // Keep size metrics available while skipping disk writes.
                    let payload = serialize_payload(
                        &selected.encoded_blocks,
                        &selected.positions,
                        &selected_cluster_sizes_u32,
                        selected_ncols_u32,
                        &selected.tile_local_orders,
                    )?;
                    compress_payload_bytes(&payload, args.output_compression)?.len()
                } else {
                    let mut output = args.output.clone().unwrap_or_else(|| {
                        PathBuf::from(format!(
                            "output.bin.{}",
                            args.output_compression.extension()
                        ))
                    });
                    if sweep_configs.len() > 1 {
                        output = output_path_with_label_suffix(&output, config_label);
                    }
                    let payload = serialize_payload(
                        &selected.encoded_blocks,
                        &selected.positions,
                        &selected_cluster_sizes_u32,
                        selected_ncols_u32,
                        &selected.tile_local_orders,
                    )?;
                    let compressed = compress_payload_bytes(&payload, args.output_compression)?;
                    let file = File::create(&output)?;
                    let mut writer = BufWriter::new(file);
                    writer.write_all(&compressed)?;
                    writer.flush()?;
                    info!("Saved encoded payload to {}", output.display());
                    std::fs::metadata(&output)?.len() as usize
                };
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
                let gene_order_bytes = csr.cols() * std::mem::size_of::<u32>();
                let payload_with_col_order = mst_payload_bytes + gene_order_bytes;

                if args.no_output {
                    info!("--no-output enabled: skipped writing compressed payload file");
                }
                info!(
                    "Selected setting: q={} (used={}), bins={}, clusters={}",
                    selected.quantizers_requested,
                    selected.quantizers_used,
                    selected.quantizer_bins,
                    selected.cluster_sizes.len()
                );
                info!(
                "Size: mst_uncompressed={} bytes, compressed_estimate={} bytes, compressed_actual={} bytes",
                selected.total_mst_bytes, selected.gzip_bytes_estimate, actual_gzip_bytes
            );
                info!(
                    "Distortion: mse={:.6}, rmse={:.6}, actual_rate={:.4} bits/value",
                    selected.mse, selected.rmse, actual_rate
                );
                info!(
                "MST breakdown: topology(parent_offset)={} bytes, values(root+ops)={} bytes, metadata(num_genes)={} bytes, gene_order_if_serialized={} bytes, total={} bytes",
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
                    let stats_label = stats_path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .map(str::to_owned)
                        .unwrap_or_else(|| infer_stats_label(&args.input));
                    append_build_stats_csv(
                        stats_path,
                        &stats_label,
                        config_label,
                        &gene_method_label,
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
        }
        Commands::Dump(args) => {
            let (pos_x_col, pos_y_col) =
                resolve_position_columns(args.platform, args.pos_x_col, args.pos_y_col);
            let positions_spec = if args.platform == Platform::SingleCell {
                None
            } else {
                let input_pos = args.input_pos.as_ref().ok_or_else(|| {
                    anyhow::anyhow!(
                        "--input-pos is required for --platform {:?}",
                        args.platform
                    )
                })?;
                Some((
                    input_pos.as_path(),
                    args.pos_format,
                    pos_x_col,
                    pos_y_col,
                ))
            };

            let t_total = Instant::now();

            let t_load = Instant::now();
            let (decoded, load_stats) =
                decode_clustered_payload(&args.encoded, args.input_compression)?;
            let load_ms = t_load.elapsed().as_millis();
            info!(
                "Decompress: payload load (outer-decompress + bincode) took {}ms; \
                 outer_decompress={}us, bincode_decode={}us, raw_payload={}KB; \
                 blocks={}, clusters={}, ncols={}",
                load_ms,
                load_stats.outer_decompress_us,
                load_stats.bincode_decode_us,
                load_stats.raw_payload_bytes / 1024,
                decoded.encoded_blocks.len(),
                decoded.cluster_sizes.len(),
                decoded.ncols
            );
            if decoded.cluster_sizes.is_empty() {
                return Err(anyhow::anyhow!(
                    "Payload predates the count-matrix decoder (no cluster_sizes metadata). Rebuild the input with a current encoder."
                )
                .into());
            }

            let t_reconstruct = Instant::now();
            let reconstructed = reconstruct_count_matrix_from_payload(
                &decoded.encoded_blocks,
                &decoded.cluster_sizes,
                decoded.ncols as usize,
                &decoded.tile_local_orders,
            )?;
            let reconstruct_ms = t_reconstruct.elapsed().as_millis();
            info!(
                "Decompress: reconstruct (tile decode + CSR build) took {}ms; rows={}, cols={}, nnz={}",
                reconstruct_ms,
                reconstructed.rows(),
                reconstructed.cols(),
                reconstructed.nnz()
            );

            let decode_only_ms = load_ms + reconstruct_ms;
            info!(
                "Decompress: total decode time = {}ms (load {}ms + reconstruct {}ms)",
                decode_only_ms, load_ms, reconstruct_ms
            );

            let t_truth = Instant::now();
            let (csr_truth, _points) = load_10x_oriented(
                &args.input,
                positions_spec,
                args.matrix_orientation,
                args.max_cells,
            )?;
            let truth_ms = t_truth.elapsed().as_millis();
            info!(
                "Decompress: ground-truth H5 load took {}ms; rows={}, cols={}, nnz={}, matrix_orientation={:?}",
                truth_ms,
                csr_truth.rows(),
                csr_truth.cols(),
                csr_truth.nnz(),
                args.matrix_orientation
            );

            let t_verify = Instant::now();
            compare_count_matrix_permutation_invariant(&csr_truth, &reconstructed)?;
            let verify_ms = t_verify.elapsed().as_millis();
            let total_ms = t_total.elapsed().as_millis();
            info!(
                "Decompression PASSED: matches input under row/column permutation. \
                 verify={}ms, end-to-end={}ms (decode={}ms + truth-load={}ms + verify={}ms)",
                verify_ms, total_ms, decode_only_ms, truth_ms, verify_ms
            );

            if let Some(out_path) = args.output.as_ref() {
                let t_write = Instant::now();
                crate::h5_utils::write_csr_10x_h5(out_path, &reconstructed)?;
                let write_ms = t_write.elapsed().as_millis();
                info!(
                    "Dumped decoded CSR matrix to {}: rows={}, cols={}, nnz={}, write={}ms \
                     (placeholder barcodes/features; matrix is in decoder order)",
                    out_path.display(),
                    reconstructed.rows(),
                    reconstructed.cols(),
                    reconstructed.nnz(),
                    write_ms
                );
            }
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
            8,
            0,
            false,
            32,
            false,
            16,
            false,
            false,
            false,
            false,
            false,
            false,
            None,
            GeneReorderMethod::Projection,
            false,
            1,
            PayloadCompressionArg::Zstd,
        )
        .expect("lossy compression should succeed");

        assert_eq!(result.cluster_sizes.iter().sum::<usize>(), n_cells);
        assert!(result.cluster_sizes.iter().all(|&s| s <= max_cluster_size));
    }

    #[test]
    fn svd_joint_seriation_runs_on_toy_matrix() {
        let n_cells = 12usize;
        let n_genes = 8usize;
        let mut tri = TriMatI::<u16, usize>::new((n_cells, n_genes));
        for cell in 0..n_cells {
            tri.add_triplet(cell, cell % n_genes, (cell as u16 + 1) * 3);
            tri.add_triplet(cell, (cell + 1) % n_genes, 5u16);
        }
        let csr = tri.to_csr::<usize>();
        let points: Vec<Point> = (0..n_cells)
            .map(|i| Point::new(i as f64, (i % 3) as f64, i))
            .collect();
        let result = run_clustered_compression(
            &points,
            &csr,
            3,
            8,
            false,
            false,
            None,
            KnnDistanceMetric::L0,
            MstWeightMode::Metric,
            IndexStreamCodec::Arithmetic,
            SortedIndexCodec::EliasFano,
            Some(1.0),
            None,
            8,
            0,
            false,
            32,
            false,
            16,
            false,
            false,
            false,
            false,
            false,
            false,
            Some(42),
            GeneReorderMethod::Projection,
            false,
            1,
            PayloadCompressionArg::Zstd,
        )
        .expect("svd-joint seriation path");
        assert_eq!(result.cluster_sizes.iter().sum::<usize>(), n_cells);
    }
}
