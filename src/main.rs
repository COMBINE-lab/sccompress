mod arith_encode;
mod cluster;
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
    EncodedColumnBlock, EncodedDiffsMST, HnswBuildConfig, KnnDistanceMetric, MstWeightMode, Point,
    RowMstNeighborMode,
};
use ndarray::Array2;
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
    build_cell_features, cluster_cells, group_points_by_cluster, joint_svd_seriation,
    projection_weight, reorder_rows_within_clusters, split_oversized_clusters,
    CellClusterMethodArg, SpatialGraphParams, TensorGridDisc, TensorGridParams,
};
use linfa::prelude::{Fit, Predict};
use linfa::DatasetBase;
use linfa_clustering::KMeans;
use nalgebra::DMatrix;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

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
    timing_cluster_prep_ms: u64,
    timing_gene_order_ms: u64,
    timing_encode_tiles_ms: u64,
    timing_row_mst_encode_ms: u64,
    timing_column_encode_ms: u64,
    cluster_sizes: Vec<usize>,
    encoded_blocks: Vec<EncodedClusterBlock>,
    positions: Vec<DatalessPoint>,
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

#[derive(Debug, Clone, Copy, ValueEnum)]
enum HnswProfileArg {
    Default,
    Fast,
    Faster,
}

impl HnswProfileArg {
    fn to_config(self) -> HnswBuildConfig {
        match self {
            HnswProfileArg::Default => HnswBuildConfig::default(),
            HnswProfileArg::Fast => HnswBuildConfig {
                max_nb_connection: 12,
                ef_construction: 64,
                ef_search: 24,
            },
            HnswProfileArg::Faster => HnswBuildConfig {
                max_nb_connection: 8,
                ef_construction: 40,
                ef_search: 16,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum RowMstNeighborArg {
    Hnsw,
    LocalWindow,
}

impl From<RowMstNeighborArg> for RowMstNeighborMode {
    fn from(value: RowMstNeighborArg) -> Self {
        match value {
            RowMstNeighborArg::Hnsw => RowMstNeighborMode::Hnsw,
            RowMstNeighborArg::LocalWindow => RowMstNeighborMode::LocalWindow,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ClusterEncodingArg {
    Row,
    Column,
    Hybrid,
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
    /// K-means on SVD right-singular-vector embeddings: cluster genes into groups with similar
    /// cell-support patterns, then sort within each cluster by `v₁`.  Genes sharing a cluster
    /// are placed contiguously, so posting-list deltas and MST edit chains benefit from block
    /// structure that pure seriation may miss when the manifold is not 1-D.
    Kmeans,
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

fn l2_normalize_rows_inplace(a: &mut Array2<f64>) {
    for mut row in a.rows_mut() {
        let norm_sq: f64 = row.iter().map(|v| v * v).sum();
        let nrm = norm_sq.sqrt();
        if nrm > 1e-15 {
            row.mapv_inplace(|x| x / nrm);
        }
    }
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

/// K-means on right-singular-vector gene embeddings, then sort within each cluster by v₁.
///
/// Builds the same TF-IDF-normalized sparse matrix and SVD as [`compute_gene_permutation_svd`],
/// but instead of a global 1-D sort by `(v₁,v₂)`, it:
///   1. Embeds each gene in `ℝ^{n_comp}` via the right singular vectors.
///   2. Runs k-means on genes to find groups with similar cell-support patterns.
///   3. Sorts clusters by their centroid's v₁, and genes within each cluster by v₁.
///
/// This captures block structure (gene modules) that a pure 1-D seriation can miss.
fn compute_gene_permutation_kmeans(
    points: &[Point],
    data: &CsMat<u16>,
    row_stream_point_indices: &[usize],
    cluster_seed: Option<u64>,
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

    let mut col_nnz = vec![0u64; ncols];
    let mut triplets: Vec<(usize, usize)> = Vec::new();

    for (stream_pos, &point_idx) in row_stream_point_indices.iter().enumerate() {
        let point = &points[point_idx];
        if let Some(row) = data.outer_view(point.row_index) {
            for (gene_idx, &value) in row.iter() {
                if value != 0 {
                    col_nnz[gene_idx] += 1;
                    triplets.push((stream_pos, gene_idx));
                }
            }
        }
    }

    let active_genes: Vec<usize> = (0..ncols).filter(|&g| col_nnz[g] > 0).collect();
    if active_genes.len() <= 2 {
        let identity: Vec<u32> = (0..ncols as u32).collect();
        return Ok((identity.clone(), identity));
    }

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
        let mut by_freq: Vec<(usize, u64)> =
            active_genes.iter().map(|&g| (g, col_nnz[g])).collect();
        by_freq.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        by_freq.truncate(max_svd_cols);
        by_freq.sort_unstable_by_key(|&(g, _)| g);
        by_freq.iter().map(|&(g, _)| g).collect()
    } else {
        active_genes.clone()
    };

    let mut col_local: HashMap<usize, usize> = HashMap::with_capacity(active_cols.len());
    for (local, &global) in active_cols.iter().enumerate() {
        col_local.insert(global, local);
    }

    let nr = sampled_rows.len();
    let nc = active_cols.len();

    let mut dense = vec![0.0f64; nr * nc];
    let sampled_set: HashMap<usize, usize> = sampled_rows
        .iter()
        .enumerate()
        .map(|(i, &r)| (r, i))
        .collect();

    for &(stream_pos, gene_idx) in &triplets {
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
            info!("K-means gene reorder: V^T not available, falling back to SVD seriation");
            return compute_gene_permutation_svd(points, data, row_stream_point_indices);
        }
    };

    let n_sv = v_t.nrows();
    let n_comp = n_sv.min(8).max(1);

    // Build gene embedding matrix (active genes only)
    let mut gene_embed = Array2::<f64>::zeros((nc, n_comp));
    for (local_col, _) in active_cols.iter().enumerate() {
        for j in 0..n_comp {
            gene_embed[(local_col, j)] = v_t[(j, local_col)];
        }
    }
    l2_normalize_rows_inplace(&mut gene_embed);

    // K-means on gene embeddings — use sqrt(nc) clusters, capped
    let gene_k = (nc as f64).sqrt().ceil() as usize;
    let gene_k = gene_k.max(2).min(nc);
    let dataset = DatasetBase::from(gene_embed.clone());
    let model = if let Some(seed) = cluster_seed {
        KMeans::params_with_rng(
            gene_k,
            Xoshiro256Plus::seed_from_u64(seed.wrapping_add(9001)),
        )
        .max_n_iterations(30)
        .fit(&dataset)
        .map_err(|e| anyhow::anyhow!("gene k-means failed: {}", e))?
    } else {
        KMeans::params(gene_k)
            .max_n_iterations(30)
            .fit(&dataset)
            .map_err(|e| anyhow::anyhow!("gene k-means failed: {}", e))?
    };
    let gene_labels: Vec<usize> = model.predict(&dataset).to_vec();

    // For each gene cluster, compute centroid v₁ for inter-cluster ordering
    let mut cluster_v1_sum = vec![0.0f64; gene_k];
    let mut cluster_count = vec![0usize; gene_k];
    for (local_col, &label) in gene_labels.iter().enumerate() {
        cluster_v1_sum[label] += v_t[(0, local_col)];
        cluster_count[label] += 1;
    }
    let mut cluster_order: Vec<usize> = (0..gene_k).collect();
    cluster_order.sort_unstable_by(|&a, &b| {
        let mean_a = if cluster_count[a] > 0 {
            cluster_v1_sum[a] / cluster_count[a] as f64
        } else {
            f64::INFINITY
        };
        let mean_b = if cluster_count[b] > 0 {
            cluster_v1_sum[b] / cluster_count[b] as f64
        } else {
            f64::INFINITY
        };
        mean_a.total_cmp(&mean_b)
    });
    let mut cluster_rank = vec![0usize; gene_k];
    for (rank, &cid) in cluster_order.iter().enumerate() {
        cluster_rank[cid] = rank;
    }

    // Build final ordering: (cluster_rank, v₁ within cluster, global_gene_idx)
    let mut scores: Vec<(usize, usize, f64, f64)> = Vec::with_capacity(ncols);
    for (local_col, &global_gene) in active_cols.iter().enumerate() {
        let label = gene_labels[local_col];
        let rank = cluster_rank[label];
        let s1 = v_t[(0, local_col)];
        let s2 = if n_sv >= 2 { v_t[(1, local_col)] } else { 0.0 };
        scores.push((global_gene, rank, s1, s2));
    }
    for g in 0..ncols {
        if col_nnz[g] == 0 || !col_local.contains_key(&g) {
            scores.push((g, usize::MAX, f64::INFINITY, g as f64));
        }
    }

    scores.sort_unstable_by(|a, b| {
        a.1.cmp(&b.1)
            .then_with(|| a.2.total_cmp(&b.2))
            .then_with(|| a.3.total_cmp(&b.3))
            .then_with(|| a.0.cmp(&b.0))
    });

    let mut old_to_new = vec![0u32; ncols];
    let mut new_to_old = Vec::with_capacity(ncols);
    for (new_idx, &(old_idx, _, _, _)) in scores.iter().enumerate() {
        old_to_new[old_idx] = new_idx as u32;
        new_to_old.push(old_idx as u32);
    }

    info!(
        "K-means gene reorder: {} genes -> {} gene clusters ({} active in SVD, {} total)",
        active_cols.len(),
        gene_k,
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

/// Serialized layout: `encoded_blocks`, then `positions`. Row/gene order permutations are
/// not stored on disk (smaller archives; [`decode_clustered_payload`] returns empty order
/// vectors for new payloads).
fn serialize_payload(
    encoded_blocks: &[EncodedClusterBlock],
    positions: &[DatalessPoint],
) -> anyhow::Result<Vec<u8>> {
    let config = bincode::config::standard()
        .with_little_endian()
        .with_fixed_int_encoding();
    let mut payload = Vec::new();
    bincode::encode_into_std_write(encoded_blocks, &mut payload, config)?;
    bincode::encode_into_std_write(positions, &mut payload, config)?;
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
) -> anyhow::Result<usize> {
    let payload = serialize_payload(encoded_blocks, positions)?;
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
    full_row_fallback_ratio: Option<f32>,
    forest_cut_factor: Option<f32>,
    hnsw_build: HnswBuildConfig,
    row_mst_neighbor_mode: RowMstNeighborMode,
    row_mst_window: usize,
    cluster_encoding: ClusterEncodingArg,
    column_template_count: usize,
    column_template_adaptive: bool,
    column_template_max: usize,
    row_template_adaptive: bool,
    row_template_max: usize,
    cell_cluster_method: CellClusterMethodArg,
    spatial_graph: SpatialGraphParams,
    tensor_grid: TensorGridParams,
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
    let features = build_cell_features(points, data, 24)?;

    // Joint path: one operation produces both cell assignments and gene ordering.
    let joint_result = if cell_cluster_method == CellClusterMethodArg::SvdJoint {
        match joint_svd_seriation(
            points,
            data,
            requested_clusters,
            cluster_seed,
            joint_svd_fast,
        ) {
            Ok(result) => Some(result),
            Err(e) => {
                warn!(
                    "Joint SVD seriation failed ({}), falling back to separate cell+gene",
                    e
                );
                None
            }
        }
    } else {
        None
    };

    let assignments = if let Some((ref cell_labels, _, _)) = joint_result {
        cell_labels.clone()
    } else {
        match cluster_cells(
            cell_cluster_method,
            &features,
            requested_clusters,
            points,
            spatial_graph,
            tensor_grid,
            cluster_seed,
        ) {
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
    let clusters = split_oversized_clusters(
        points,
        initial_clusters,
        &features,
        max_cluster_size,
        cell_cluster_method,
        spatial_graph,
        tensor_grid,
        cluster_seed,
    )?;
    let clusters = reorder_rows_within_clusters(&clusters, &features);
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
            GeneReorderMethod::Kmeans => compute_gene_permutation_kmeans(
                points,
                &quantized_data,
                &row_stream_point_indices,
                cluster_seed,
            )?,
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

    if effective_gene_blocks > 1 {
        info!(
            "Gene blocking: {} blocks, sizes {:?}",
            effective_gene_blocks,
            gene_block_ranges
                .iter()
                .map(|r| r.len())
                .collect::<Vec<_>>()
        );
    }

    let mut encoded_blocks = Vec::new();
    let mut positions = Vec::new();
    let mut row_order = Vec::new();
    let mut cluster_sizes = Vec::new();
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

    let encode_tile = |cluster_points: &[Point],
                       block_gene_o2n: &[u32],
                       cluster_idx: usize|
     -> anyhow::Result<(EncodedClusterBlock, Vec<u32>, usize, bool, bool)> {
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
                let t = Instant::now();
                let encoded = encode_subarray_mst_with_metric(
                    cluster_points,
                    &quantized_data,
                    knn_metric,
                    mst_weight_mode,
                    Some(block_gene_o2n),
                    index_codec,
                    sorted_index_codec,
                    full_row_fallback_ratio,
                    forest_cut_factor,
                    hnsw_build,
                    row_mst_neighbor_mode,
                    row_mst_window,
                    adaptive,
                    row_template_max,
                );
                row_mst_encode_us.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
                if let Some((row_block, dfs_order)) = encoded {
                    let bytes = row_block.total_bytes();
                    if best_row.as_ref().map(|b| bytes < b.2).unwrap_or(true) {
                        best_row = Some((EncodedClusterBlock::RowMst(row_block), dfs_order, bytes));
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
                let t = Instant::now();
                if let Some(col_block) = encode_subarray_column(
                    cluster_points,
                    &quantized_data,
                    Some(block_gene_o2n),
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
                column_encode_us.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
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
                        "Both row and column encoding failed for cluster {} gene-block",
                        cluster_idx
                    ))
                }
            },
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
    let cluster_results: Vec<
        anyhow::Result<
            Option<(
                usize,
                Vec<EncodedClusterBlock>,
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
                    let (encoded, local_order, bytes, row_fb, col_fb) =
                        encode_tile(&cluster_points, block_gene_o2n, cluster_idx)?;
                    Ok((gb_idx, encoded, local_order, bytes, row_fb, col_fb))
                })
                .collect();

            let mut ordered_tiles = Vec::with_capacity(effective_gene_blocks);
            for result in tile_results {
                ordered_tiles.push(result?);
            }
            ordered_tiles.sort_unstable_by_key(|(gb_idx, _, _, _, _, _)| *gb_idx);

            let mut tile_blocks = Vec::with_capacity(effective_gene_blocks);
            let mut total_bytes = 0usize;
            let mut first_order: Option<Vec<u32>> = None;
            let mut any_row_fallback = false;
            let mut any_col_fallback = false;

            for (_gb_idx, encoded, local_order, bytes, row_fb, col_fb) in ordered_tiles {
                total_bytes += bytes;
                if first_order.is_none() {
                    first_order = Some(local_order);
                }
                any_row_fallback |= row_fb;
                any_col_fallback |= col_fb;
                tile_blocks.push(encoded);
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
    ordered.sort_unstable_by_key(|(cluster_idx, _, _, _, _, _, _, _)| *cluster_idx);

    let mut row_block_count = 0usize;
    let mut column_block_count = 0usize;
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
                EncodedClusterBlock::RowMst(_) => row_block_count += 1,
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
    }
    let timing_encode_tiles_ms = encode_tiles_start.elapsed().as_millis() as u64;
    let timing_row_mst_encode_ms = row_mst_encode_us.load(Ordering::Relaxed) / 1_000;
    let timing_column_encode_ms = column_encode_us.load(Ordering::Relaxed) / 1_000;

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
    let compressed_bytes_estimate =
        estimate_compressed_payload_size(payload_compression, &encoded_blocks, &positions)?;
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
    })
}

fn decode_clustered_payload(
    input: &Path,
    input_compression: Option<PayloadCompressionArg>,
) -> anyhow::Result<(
    Vec<EncodedClusterBlock>,
    Vec<DatalessPoint>,
    Vec<u32>,
    Vec<u32>,
)> {
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
    let mut payload_reader = Cursor::new(raw_payload);

    let encoded_blocks: Vec<EncodedClusterBlock> =
        bincode::decode_from_std_read(&mut payload_reader, config)?;
    let positions: Vec<DatalessPoint> = bincode::decode_from_std_read(&mut payload_reader, config)?;
    let end = payload_reader.get_ref().len();
    let pos = payload_reader.position() as usize;
    let row_order = if pos < end {
        bincode::decode_from_std_read(&mut payload_reader, config)?
    } else {
        Vec::new()
    };
    let pos = payload_reader.position() as usize;
    let gene_order = if pos < end {
        bincode::decode_from_std_read(&mut payload_reader, config)?
    } else {
        Vec::new()
    };
    Ok((encoded_blocks, positions, row_order, gene_order))
}

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
    /// If set, do not write compressed payload to disk.
    #[arg(long = "no-output", default_value_t = false)]
    no_output: bool,
    /// Compression codec for payload output and size estimation.
    #[arg(long = "output-compression", value_enum, default_value_t = PayloadCompressionArg::SevenZip)]
    output_compression: PayloadCompressionArg,
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
    /// Number of cell clusters (blocks) used by the cell clustering stage.
    #[arg(long = "cell-blocks", default_value_t = 8)]
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
    /// If set, disable per-child full-row fallback when edit deltas are too large.
    #[arg(long = "disable-full-row-fallback", default_value_t = false)]
    disable_full_row_fallback: bool,
    /// Trigger full-row storage for a child when `delta_edits > ratio * child_nnz`.
    #[arg(long = "full-row-fallback-ratio", default_value_t = 1.0)]
    full_row_fallback_ratio: f32,
    /// If set, cut MST edges larger than `median_edge_weight * factor`, producing a forest.
    #[arg(long = "forest-cut-factor")]
    forest_cut_factor: Option<f32>,
    /// HNSW construction/search preset for MST KNN graph build.
    #[arg(long = "hnsw-fast-profile", value_enum, default_value_t = HnswProfileArg::Default)]
    hnsw_fast_profile: HnswProfileArg,
    /// Candidate graph used before row-MST: HNSW kNN, or local row-order window.
    #[arg(long = "row-mst-neighbor-mode", value_enum, default_value_t = RowMstNeighborArg::LocalWindow)]
    row_mst_neighbor_mode: RowMstNeighborArg,
    /// Number of nearby rows on each side to connect when `--row-mst-neighbor-mode=local-window`.
    #[arg(long = "row-mst-window", default_value_t = 8)]
    row_mst_window: usize,
    /// How to cluster cells (`kmeans` or `svd-joint`).
    #[arg(long = "cell-cluster-method", value_enum, default_value_t = CellClusterMethodArg::Kmeans)]
    cell_cluster_method: CellClusterMethodArg,
    /// Optional RNG seed for cell clustering/SVD stages.
    #[arg(long = "set-seed", visible_alias = "cluster-seed")]
    set_seed: Option<u64>,
    /// If set, run all cell-cluster methods and append one stats row per method.
    #[arg(
        long = "cell-cluster-method-sweep-all",
        visible_alias = "cluster-all",
        default_value_t = false
    )]
    cell_cluster_method_sweep_all: bool,
    /// For `spatial-graph`: number of spatial \((x,y)\) nearest neighbors per cell.
    #[arg(long = "cell-cluster-spatial-knn", default_value_t = 12)]
    cell_cluster_spatial_knn: usize,
    /// For `spatial-graph`: number of expression (dot-product) nearest neighbors per cell.
    #[arg(long = "cell-cluster-expr-knn", default_value_t = 12)]
    cell_cluster_expr_knn: usize,
    /// For `spatial-graph`: weight in \([0,1]\) on spatial kNN affinity (`1 - blend` on expression kNN).
    #[arg(long = "cell-cluster-spatial-blend", default_value_t = 0.45)]
    cell_cluster_spatial_blend: f64,
    /// For `tensor-grid`: number of bins along x (rect) or hex resolution hint.
    #[arg(long = "cell-cluster-grid-nx", default_value_t = 8)]
    cell_cluster_grid_nx: usize,
    /// For `tensor-grid`: number of bins along y (rect) or hex resolution hint.
    #[arg(long = "cell-cluster-grid-ny", default_value_t = 8)]
    cell_cluster_grid_ny: usize,
    /// For `tensor-grid`: scale of tile coordinates / one-hot vs bucket features.
    #[arg(long = "cell-cluster-grid-tile-weight", default_value_t = 0.55)]
    cell_cluster_grid_tile_weight: f64,
    /// For `tensor-grid`: `rect` = axis-aligned grid; `hex` = pointy-top hex bins.
    #[arg(long = "cell-cluster-grid-disc", value_enum, default_value_t = TensorGridDisc::Rect)]
    cell_cluster_grid_disc: TensorGridDisc,
    /// For `tensor-grid`: use one-hot tile dims when number of tiles ≤ this (else 2D tile centers).
    #[arg(long = "cell-cluster-grid-onehot-max", default_value_t = 64)]
    cell_cluster_grid_onehot_max: usize,
    /// How to compute the gene (column) reordering: `projection` (hash-projection seriation, fast)
    /// or `svd` (truncated SVD on the sparse matrix, uses right singular vectors).
    #[arg(long = "gene-reorder-method", value_enum, default_value_t = GeneReorderMethod::Projection)]
    gene_reorder_method: GeneReorderMethod,
    /// For `cell-cluster-method=svd-joint`: use randomized joint SVD for cell clusters and gene order.
    #[arg(long = "joint-svd-fast", default_value_t = false)]
    joint_svd_fast: bool,
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
    /// Split genes into this many contiguous blocks (after reordering) and build
    /// MST independently on each (cell_cluster × gene_block) tile.  Default 1 = no splitting.
    #[arg(long = "gene-blocks", default_value_t = 4)]
    gene_blocks: usize,
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

fn cell_cluster_method_name(method: CellClusterMethodArg) -> String {
    method
        .to_possible_value()
        .map(|v| v.get_name().to_string())
        .unwrap_or_else(|| format!("{:?}", method).to_lowercase())
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
struct RoundtripCheckCommand {
    #[arg(short = 'e', long = "encoded")]
    encoded: PathBuf,
    /// Override encoded payload compression format (`gzip` or `zstd`).
    #[arg(long = "input-compression", value_enum)]
    input_compression: Option<PayloadCompressionArg>,
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
            let target_cell_blocks = args.cell_blocks.max(1);
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
            let hnsw_build = args.hnsw_fast_profile.to_config();
            let row_mst_neighbor_mode = RowMstNeighborMode::from(args.row_mst_neighbor_mode);
            let row_mst_window = args.row_mst_window.max(1);
            let cluster_encoding = args.cluster_encoding;
            let spatial_graph = SpatialGraphParams {
                spatial_knn: args.cell_cluster_spatial_knn.max(1),
                expr_knn: args.cell_cluster_expr_knn.max(1),
                blend: args.cell_cluster_spatial_blend.clamp(0.0, 1.0),
            };
            let tensor_grid = TensorGridParams {
                nx: args.cell_cluster_grid_nx.max(1),
                ny: args.cell_cluster_grid_ny.max(1),
                tile_weight: args.cell_cluster_grid_tile_weight.max(0.0),
                disc: args.cell_cluster_grid_disc,
                onehot_max_tiles: args.cell_cluster_grid_onehot_max.max(1),
            };

            info!(
                "Encoding mode: {} | output_compression={:?} lossy_quantizers={} cell_blocks={} bins={} sweep={:?} max_cluster_size={:?} cell_cluster_method={:?} knn_metric={:?} mst_weight={:?} hnsw_profile={:?} hnsw_M={} hnsw_ef_construction={} hnsw_query_ef={} row_mst_neighbor_mode={:?} row_mst_window={} joint_svd_fast={} index_codec={:?} sorted_index_codec={:?} full_row_fallback={:?} forest_cut_factor={:?} cluster_encoding={:?} column_template_count={} column_template_adaptive={} column_template_max={} row_template_adaptive={} row_template_max={}",
                if quantize_values { "lossy" } else { "lossless" },
                args.output_compression,
                target_quantizers,
                target_cell_blocks,
                quantizer_bins,
                sweep_counts,
                args.max_cluster_size,
                args.cell_cluster_method,
                args.knn_metric,
                args.mst_weight,
                args.hnsw_fast_profile,
                hnsw_build.max_nb_connection,
                hnsw_build.ef_construction,
                hnsw_build.ef_search,
                args.row_mst_neighbor_mode,
                row_mst_window,
                args.joint_svd_fast,
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

            // Each sweep config: (cell_method, gene_method, label)
            let mut sweep_configs: Vec<(CellClusterMethodArg, GeneReorderMethod, String)> =
                Vec::new();

            if args.cell_cluster_method_sweep_all {
                for &cm in CellClusterMethodArg::value_variants() {
                    let label = cell_cluster_method_name(cm);
                    sweep_configs.push((cm, args.gene_reorder_method, label));
                }
            } else {
                let label = cell_cluster_method_name(args.cell_cluster_method);
                sweep_configs.push((args.cell_cluster_method, args.gene_reorder_method, label));
            }

            for (cell_cluster_method, gene_reorder_method, config_label) in &sweep_configs {
                let is_joint = *cell_cluster_method == CellClusterMethodArg::SvdJoint;
                let gene_method_label = if is_joint {
                    config_label.clone()
                } else {
                    gene_reorder_method
                        .to_possible_value()
                        .map(|v| v.get_name().to_string())
                        .unwrap_or_else(|| format!("{:?}", gene_reorder_method).to_lowercase())
                };
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
                        fallback_ratio,
                        forest_cut_factor,
                        hnsw_build,
                        row_mst_neighbor_mode,
                        row_mst_window,
                        cluster_encoding,
                        args.column_template_count,
                        args.column_template_adaptive,
                        args.column_template_max,
                        args.row_template_adaptive,
                        args.row_template_max,
                        *cell_cluster_method,
                        spatial_graph,
                        tensor_grid,
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

                let actual_gzip_bytes = if args.no_output {
                    // Keep size metrics available while skipping disk writes.
                    let payload = serialize_payload(&selected.encoded_blocks, &selected.positions)?;
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
                    let payload = serialize_payload(&selected.encoded_blocks, &selected.positions)?;
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

                for (block_idx, block) in selected.encoded_blocks.iter().enumerate() {
                    let (p, ri, rv, i, dv, ng) = block.bytes_breakdown();
                    bytes_parent += p;
                    bytes_root_indices += ri;
                    bytes_root_vals += rv;
                    bytes_indices += i;
                    bytes_delta_vals += dv;
                    bytes_num_genes += ng;
                    match block {
                    EncodedClusterBlock::RowMst(_) => info!(
                        "  block[{}] type=row_mst: parent={} root_indices={} root_vals={} op_indices={} op_vals={} num_genes_meta={} total={}",
                        block_idx,
                        p,
                        ri,
                        rv,
                        i,
                        dv,
                        ng,
                        p + ri + rv + i + dv + ng
                    ),
                    EncodedClusterBlock::Column(_) => info!(
                        "  block[{}] type=column: local_to_global={} posting_count_streams={} posting_index_streams={} vals={} num_cells_genes_meta={} total={}",
                        block_idx,
                        ri,
                        rv,
                        i,
                        dv,
                        ng,
                        p + ri + rv + i + dv + ng
                    ),
                }
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
        Commands::RoundtripCheck(args) => {
            let (pos_x_col, pos_y_col) =
                resolve_position_columns(args.platform, args.pos_x_col, args.pos_y_col);

            let (encoded_blocks, _positions, row_order, gene_order) =
                decode_clustered_payload(&args.encoded, args.input_compression)?;
            info!(
                "Loaded payload: blocks={}, row_order_len={}, gene_order_len={}",
                encoded_blocks.len(),
                row_order.len(),
                gene_order.len()
            );
            if row_order.is_empty() {
                return Err(anyhow::anyhow!(
                    "Round-trip check requires row_order (and typically gene_order) in the payload; current builds omit these for a smaller archive. Use an older payload or a matching encoder/decoder pair."
                )
                .into());
            }

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
            HnswBuildConfig::default(),
            RowMstNeighborMode::Hnsw,
            8,
            ClusterEncodingArg::Hybrid,
            0,
            false,
            32,
            false,
            16,
            CellClusterMethodArg::Kmeans,
            SpatialGraphParams::default(),
            TensorGridParams::default(),
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
            HnswBuildConfig::default(),
            RowMstNeighborMode::Hnsw,
            8,
            ClusterEncodingArg::Hybrid,
            0,
            false,
            32,
            false,
            16,
            CellClusterMethodArg::SvdJoint,
            SpatialGraphParams::default(),
            TensorGridParams::default(),
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
