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
use nalgebra::linalg::SymmetricEigen;
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

/// How to partition cells into encoder clusters (before row-MST / column payloads).
#[derive(Debug, Clone, Copy, ValueEnum, Default, PartialEq, Eq)]
enum CellClusterMethodArg {
    /// K-means on `build_cell_features` (hashed ln1p buckets, L2-normalized per cell).
    #[default]
    Kmeans,
    /// Binary-threshold `build_cell_features` (`>0` => 1), then Lloyd-style clustering with L0/Hamming
    /// assignment and per-cluster majority-vote centers (k-modes-like on binary vectors).
    BinaryL0Kmeans,
    /// Double-L1 normalized `X`, **k-means on columns** via Gram `X^T X` (bucket × bucket),
    /// per-cell sums inside each column cluster, concatenated with truncated-SVD embedding
    /// `U Σ^{1/2}`; final **k-means on rows** (cells).
    Bicluster,
    /// Spectral bicluster variant with **L0/Hamming** clustering: binarize features (`>0`), double-L1 normalize,
    /// column grouping by binary L0 Lloyd on bucket patterns across cells, SVD row embedding, concatenate,
    /// binarize combined features and final binary L0 Lloyd on cells.
    BiclusterL0,
    /// Like `bicluster`, but **row k-means first** on `A`, then column k-means on Gram of **cell-cluster means** `R`,
    /// aggregates, SVD, concat **`[agg ∥ embed]`**, final row k-means (see `cluster_cells_spectral_bicluster_row_col_swapped`).
    BiclusterSwapped,
    /// Same preprocessing and SVD row embedding as `bicluster` (**double L1** on nonnegative `X`, then `U Σ^{1/2}`),
    /// but **no** Gram / column k-means / aggregates — final **k-means** only on the embedding.
    SvdKmeans,
    /// SVD seriation: same double-L1 preprocessing and SVD as `svd-kmeans`, but instead of
    /// k-means, **sort** cells by their left-singular-vector coordinates `(u₁, u₂, …)` and cut
    /// the sorted order into `k` contiguous blocks.  Guarantees cluster-contiguous cell ordering.
    SvdSeriation,
    /// **Column k-means only** (same as bicluster’s column step): k-means on Gram `X^T X` to group
    /// bucket columns, sum `X` within each column cluster per cell, L2-normalize rows, then
    /// k-means on cells (no SVD).
    ColumnKmeans,
    /// [SpectralCoclustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralCoclustering.html)-style
    /// (Dhillon 2001): `1/√(row sum)` × `1/√(col sum)` scaling, SVD, drop first factor, stack
    /// scaled `U` and `V`, one k-means on **rows ∪ columns**; **cell labels** are the first `n` assignments.
    SpectralCocluster,
    /// [FABIA](https://doi.org/10.1093/bioinformatics/btq227)-style sparse factor model `X ≈ Λ Z` (Hochreiter et al., 2010):
    /// row-standardize bucket×cell matrix, SVD warm start, ISTA with Laplace (`L1`) shrinkage on `Λ` and `Z`,
    /// then hard assignment (argmax `|Z|` per cell, or k-means on `Z` if `k` exceeds factor rank).
    Fabia,
    /// **Spatial graph + expression**: kNN in \((x,y)\) and kNN in feature space, combined symmetric affinity,
    /// normalized Laplacian embedding, then k-means (spectral clustering). Large `n` uses `[features \| scaled xy]` k-means.
    SpatialGraph,
    /// **Slide discretization** (grid or hex bins), then k-means on `[expression buckets \| tile embedding]`.
    /// Links cells to spatial tiles (tensor-style *bucket×cell×tile* linkage via tile assignment + features).
    TensorGrid,
    /// [cell-squeeze](https://github.com/maharshi95/cell-squeeze)-style Kluger (2003) SpectralBiclustering:
    /// bistochastic normalization, SVD, pick `n_best` singular vectors by piecewise-constant fit,
    /// project X @ V_best → row k-means, X^T @ U_best → column k-means.  Joint biclustering that
    /// simultaneously clusters cells **and** reorders genes.
    CellSqueeze,
    /// Joint SVD seriation: one TF-IDF-normalized SVD on the full sparse cell×gene matrix,
    /// **sort** cells by `(u₁, u₂, …)` into `k` contiguous blocks (like `svd-seriation`),
    /// **sort** genes by `(v₁, v₂)` (like `--gene-reorder-method svd`).  Pure permutation on
    /// both axes from the same decomposition — no k-means anywhere.
    SvdJoint,
}

/// Parameters for [`CellClusterMethodArg::SpatialGraph`] (ignored for other methods).
#[derive(Clone, Copy, Debug)]
struct SpatialGraphParams {
    spatial_knn: usize,
    expr_knn: usize,
    /// Weight on spatial kNN affinity (`1 - blend` on expression kNN).
    blend: f64,
}

impl Default for SpatialGraphParams {
    fn default() -> Self {
        Self {
            spatial_knn: 12,
            expr_knn: 12,
            blend: 0.45,
        }
    }
}

/// How to bin \((x,y)\) into tiles for [`CellClusterMethodArg::TensorGrid`].
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum TensorGridDisc {
    /// Axis-aligned grid of `nx × ny` tiles.
    #[default]
    Rect,
    /// Pointy-top hex bins (axial coordinates); `nx`,`ny` control approximate resolution via hex size.
    Hex,
}

/// Tile binning for [`CellClusterMethodArg::TensorGrid`].
#[derive(Clone, Copy, Debug)]
struct TensorGridParams {
    nx: usize,
    ny: usize,
    /// Scale for tile coordinates / one-hot vs expression features.
    tile_weight: f64,
    disc: TensorGridDisc,
    /// Use one-hot over tiles when the number of distinct tiles ≤ this; otherwise append 2D normalized tile centers.
    onehot_max_tiles: usize,
}

impl Default for TensorGridParams {
    fn default() -> Self {
        Self {
            nx: 8,
            ny: 8,
            tile_weight: 0.55,
            disc: TensorGridDisc::Rect,
            onehot_max_tiles: 64,
        }
    }
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
    cluster_seed: Option<u64>,
) -> anyhow::Result<Vec<usize>> {
    let nrows = features.nrows();
    if nrows == 0 {
        return Ok(Vec::new());
    }

    let k = num_clusters.max(1).min(nrows);
    let dataset = DatasetBase::from(features.clone());
    let model = if let Some(seed) = cluster_seed {
        KMeans::params_with_rng(k, Xoshiro256Plus::seed_from_u64(seed))
            .max_n_iterations(20)
            .fit(&dataset)
            .map_err(|e| anyhow::anyhow!("k-means clustering failed: {}", e))?
    } else {
        KMeans::params(k)
            .max_n_iterations(20)
            .fit(&dataset)
            .map_err(|e| anyhow::anyhow!("k-means clustering failed: {}", e))?
    };

    Ok(model.predict(&dataset).to_vec())
}

fn cluster_cells_binary_l0_kmeans(
    features: &Array2<f64>,
    num_clusters: usize,
) -> anyhow::Result<Vec<usize>> {
    let n = features.nrows();
    if n == 0 {
        return Ok(Vec::new());
    }
    let m = features.ncols();
    if m == 0 {
        anyhow::bail!("binary-l0-kmeans: zero feature columns");
    }
    let k = num_clusters.max(1).min(n);
    if k == 1 {
        return Ok(vec![0usize; n]);
    }

    let mut x = vec![0u8; n * m];
    for i in 0..n {
        for j in 0..m {
            x[i * m + j] = if features[(i, j)] > 0.0 { 1 } else { 0 };
        }
    }

    Ok(binary_l0_lloyd_assign(&x, n, m, k, 30))
}

fn binary_l0_lloyd_assign(x: &[u8], n: usize, m: usize, k: usize, max_iter: usize) -> Vec<usize> {
    let mut centers = vec![0u8; k * m];
    for c in 0..k {
        let src = c * n / k;
        for j in 0..m {
            centers[c * m + j] = x[src * m + j];
        }
    }

    let mut assignments = vec![0usize; n];
    let mut changed_any = true;

    for _ in 0..max_iter {
        if !changed_any {
            break;
        }
        changed_any = false;

        for i in 0..n {
            let mut best_c = 0usize;
            let mut best_d = usize::MAX;
            for c in 0..k {
                let mut d = 0usize;
                for j in 0..m {
                    if x[i * m + j] != centers[c * m + j] {
                        d += 1;
                    }
                }
                if d < best_d {
                    best_d = d;
                    best_c = c;
                }
            }
            if assignments[i] != best_c {
                assignments[i] = best_c;
                changed_any = true;
            }
        }

        let mut cluster_sizes = vec![0usize; k];
        let mut sums = vec![0usize; k * m];
        for i in 0..n {
            let c = assignments[i];
            cluster_sizes[c] += 1;
            for j in 0..m {
                sums[c * m + j] += x[i * m + j] as usize;
            }
        }

        for c in 0..k {
            if cluster_sizes[c] == 0 {
                let src = c * n / k;
                for j in 0..m {
                    centers[c * m + j] = x[src * m + j];
                }
                continue;
            }
            for j in 0..m {
                let ones = sums[c * m + j];
                let zeros = cluster_sizes[c] - ones;
                centers[c * m + j] = if ones >= zeros { 1 } else { 0 };
            }
        }
    }

    assignments
}

fn normalize_l1_rows_inplace(a: &mut Array2<f64>) {
    for mut row in a.rows_mut() {
        let s: f64 = row.sum();
        if s > 1e-15 {
            row.mapv_inplace(|x| x / s);
        }
    }
}

fn normalize_l1_cols_inplace(a: &mut Array2<f64>) {
    for j in 0..a.ncols() {
        let s: f64 = a.column(j).sum();
        if s > 1e-15 {
            let mut col = a.column_mut(j);
            col.mapv_inplace(|x| x / s);
        }
    }
}

/// sklearn `_scale_normalize`: nonnegative `X`, then `An[i,j] = X[i,j] / sqrt(row_sum[i] * col_sum[j])`.
fn scale_normalize_dhillon(x: &Array2<f64>) -> (Array2<f64>, Vec<f64>, Vec<f64>) {
    let n = x.nrows();
    let m = x.ncols();
    let mut row_sum = vec![0f64; n];
    let mut col_sum = vec![0f64; m];
    for i in 0..n {
        for j in 0..m {
            let v = x[(i, j)].max(0.0);
            row_sum[i] += v;
            col_sum[j] += v;
        }
    }
    let row_diag: Vec<f64> = row_sum
        .iter()
        .map(|&s| if s > 1e-18 { 1.0 / s.sqrt() } else { 0.0 })
        .collect();
    let col_diag: Vec<f64> = col_sum
        .iter()
        .map(|&s| if s > 1e-18 { 1.0 / s.sqrt() } else { 0.0 })
        .collect();
    let mut an = Array2::<f64>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            an[(i, j)] = row_diag[i] * x[(i, j)].max(0.0) * col_diag[j];
        }
    }
    (an, row_diag, col_diag)
}

/// `G = X^T X` with `X` shape `(n, m)` — `G` is `(m, m)` (inner products between bucket columns).
fn gram_xt_x(x: &Array2<f64>) -> Array2<f64> {
    let n = x.nrows();
    let m = x.ncols();
    let mut g = Array2::<f64>::zeros((m, m));
    for p in 0..m {
        for q in 0..=p {
            let mut s = 0.0f64;
            for i in 0..n {
                s += x[(i, p)] * x[(i, q)];
            }
            g[(p, q)] = s;
            g[(q, p)] = s;
        }
    }
    g
}

/// Sum `X[:, j]` over bucket columns `j` that share the same column-k-means label.
fn aggregate_rows_by_column_clusters(
    x: &Array2<f64>,
    col_assign: &[usize],
    k_col: usize,
) -> Array2<f64> {
    let n = x.nrows();
    let mut z = Array2::<f64>::zeros((n, k_col));
    for i in 0..n {
        for j in 0..x.ncols() {
            let c = col_assign[j];
            if c < k_col {
                z[(i, c)] += x[(i, j)];
            }
        }
    }
    z
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

/// Dhillon spectral **co-clustering** (sklearn `SpectralCoclustering`): joint k-means on embeddings of
/// row-nodes and column-nodes in the same space (`sklearn/cluster/_bicluster.py::_fit`).
fn cluster_cells_spectral_cocluster_dhillon(
    features: &Array2<f64>,
    num_clusters: usize,
    cluster_seed: Option<u64>,
) -> anyhow::Result<Vec<usize>> {
    let n = features.nrows();
    if n == 0 {
        return Ok(Vec::new());
    }
    let m = features.ncols();
    if m == 0 {
        anyhow::bail!("spectral cocluster: zero feature columns");
    }
    let k = num_clusters.max(1).min(n);
    if k == 1 {
        return Ok(vec![0; n]);
    }

    let x_nonneg = {
        let mut a = Array2::<f64>::zeros((n, m));
        for i in 0..n {
            for j in 0..m {
                a[(i, j)] = features[(i, j)].max(0.0);
            }
        }
        a
    };

    let (an, row_diag, col_diag) = scale_normalize_dhillon(&x_nonneg);
    let data: Vec<f64> = an.iter().copied().collect();
    let mat = DMatrix::from_row_slice(n, m, &data);
    let svd = mat.svd(true, true);
    let (u, v_t) = match (svd.u, svd.v_t) {
        (Some(u), Some(vt)) => (u, vt),
        _ => return cluster_cells_with_kmeans(features, k, cluster_seed),
    };

    let r = u.ncols();
    let n_discard = 1usize;
    let n_sv = (1usize + (k as f64).log2().ceil() as usize)
        .max(2)
        .min(n.min(m));
    let end_col = r.min(n_sv);
    let n_keep = end_col.saturating_sub(n_discard);
    if n_keep == 0 {
        return cluster_cells_with_kmeans(features, k, cluster_seed);
    }

    let mut z = Array2::<f64>::zeros((n + m, n_keep));
    for c in 0..n_keep {
        let sc = n_discard + c;
        debug_assert!(sc < r);
        for i in 0..n {
            z[(i, c)] = row_diag[i] * u[(i, sc)];
        }
        for j in 0..m {
            z[(n + j, c)] = col_diag[j] * v_t[(sc, j)];
        }
    }

    let dataset = DatasetBase::from(z);
    let model = if let Some(seed) = cluster_seed {
        KMeans::params_with_rng(k, Xoshiro256Plus::seed_from_u64(seed.wrapping_add(1001)))
            .max_n_iterations(20)
            .fit(&dataset)
            .map_err(|e| anyhow::anyhow!("spectral cocluster joint k-means failed: {}", e))?
    } else {
        KMeans::params(k)
            .max_n_iterations(20)
            .fit(&dataset)
            .map_err(|e| anyhow::anyhow!("spectral cocluster joint k-means failed: {}", e))?
    };
    let labels = model.predict(&dataset).to_vec();
    Ok(labels.into_iter().take(n).collect())
}

#[inline]
fn soft_threshold_scalar(x: f64, t: f64) -> f64 {
    if x > t {
        x - t
    } else if x < -t {
        x + t
    } else {
        0.0
    }
}

/// FABIA-style: `X` is `m×n` (bucket features × cells), row-z-score, `X ≈ Λ Z` with sparse `Λ`, `Z` via ISTA.
fn cluster_cells_fabia(
    features: &Array2<f64>,
    num_clusters: usize,
    cluster_seed: Option<u64>,
) -> anyhow::Result<Vec<usize>> {
    let n = features.nrows();
    if n == 0 {
        return Ok(Vec::new());
    }
    let m = features.ncols();
    if m == 0 {
        anyhow::bail!("fabia: zero feature columns");
    }
    let k_req = num_clusters.max(1).min(n);
    if k_req == 1 {
        return Ok(vec![0; n]);
    }

    let mut x = Array2::<f64>::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            x[(i, j)] = features[(j, i)];
        }
    }
    for i in 0..m {
        let mut mean = 0.0f64;
        for j in 0..n {
            mean += x[(i, j)];
        }
        mean /= n as f64;
        let mut var = 0.0f64;
        for j in 0..n {
            let d = x[(i, j)] - mean;
            var += d * d;
        }
        let sd = (var / n.max(1) as f64).max(1e-18).sqrt();
        for j in 0..n {
            x[(i, j)] = (x[(i, j)] - mean) / sd;
        }
    }

    let p = k_req.min(m).min(n);
    if p < 2 {
        return cluster_cells_with_kmeans(features, k_req, cluster_seed);
    }

    let data: Vec<f64> = x.iter().copied().collect();
    let mat = DMatrix::from_row_slice(m, n, &data);
    let svd = mat.svd(true, true);
    let (u, v_t) = match (svd.u, svd.v_t) {
        (Some(u), Some(vt)) => (u, vt),
        _ => return cluster_cells_with_kmeans(features, k_req, cluster_seed),
    };
    let sigma = svd.singular_values;

    let mut lambda = Array2::<f64>::zeros((m, p));
    let mut z = Array2::<f64>::zeros((p, n));
    for l in 0..p {
        let s = sigma[l].max(1e-18).sqrt();
        for i in 0..m {
            lambda[(i, l)] = u[(i, l)] * s;
        }
        for j in 0..n {
            z[(l, j)] = s * v_t[(l, j)];
        }
    }

    let mut abs_buf: Vec<f64> = Vec::with_capacity(m * n);
    for &v in x.iter() {
        abs_buf.push(v.abs());
    }
    abs_buf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = abs_buf[abs_buf.len() / 2].max(1e-12);
    let lambda_l1 = 0.08 * med;
    let z_l1 = 0.08 * med;

    let max_iter = 48usize;
    let mut residual = Array2::<f64>::zeros((m, n));
    let mut gz = Array2::<f64>::zeros((p, n));
    let mut gl = Array2::<f64>::zeros((m, p));

    for _ in 0..max_iter {
        // R = Λ Z - X
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f64;
                for l in 0..p {
                    s += lambda[(i, l)] * z[(l, j)];
                }
                residual[(i, j)] = s - x[(i, j)];
            }
        }

        let tr_zzt: f64 = (0..p)
            .map(|l| {
                (0..n)
                    .map(|j| z[(l, j)] * z[(l, j)])
                    .sum::<f64>()
            })
            .sum();
        let eta_l = 1.0 / tr_zzt.max(1e-18);

        for i in 0..m {
            for l in 0..p {
                let mut s = 0.0f64;
                for j in 0..n {
                    s += residual[(i, j)] * z[(l, j)];
                }
                gl[(i, l)] = s;
            }
        }
        for i in 0..m {
            for l in 0..p {
                let v = lambda[(i, l)] - eta_l * gl[(i, l)];
                lambda[(i, l)] = soft_threshold_scalar(v, eta_l * lambda_l1);
            }
        }

        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f64;
                for l in 0..p {
                    s += lambda[(i, l)] * z[(l, j)];
                }
                residual[(i, j)] = s - x[(i, j)];
            }
        }

        let mut tr_ll = 0.0f64;
        for i in 0..m {
            for l in 0..p {
                tr_ll += lambda[(i, l)] * lambda[(i, l)];
            }
        }
        let eta_z = 1.0 / tr_ll.max(1e-18);

        for l in 0..p {
            for j in 0..n {
                let mut s = 0.0f64;
                for i in 0..m {
                    s += lambda[(i, l)] * residual[(i, j)];
                }
                gz[(l, j)] = s;
            }
        }
        for l in 0..p {
            for j in 0..n {
                let v = z[(l, j)] - eta_z * gz[(l, j)];
                z[(l, j)] = soft_threshold_scalar(v, eta_z * z_l1);
            }
        }
    }

    if z.iter().any(|v| !v.is_finite()) || lambda.iter().any(|v| !v.is_finite()) {
        return cluster_cells_with_kmeans(features, k_req, cluster_seed);
    }

    if p >= k_req {
        let mut labels = vec![0usize; n];
        for j in 0..n {
            let mut best = 0usize;
            let mut best_abs = 0.0f64;
            for l in 0..k_req.min(p) {
                let a = z[(l, j)].abs();
                if a > best_abs {
                    best_abs = a;
                    best = l;
                }
            }
            labels[j] = best;
        }
        Ok(labels)
    } else {
        let mut z_sub = Array2::<f64>::zeros((n, p));
        for j in 0..n {
            for l in 0..p {
                z_sub[(j, l)] = z[(l, j)];
            }
        }
        let dataset = DatasetBase::from(z_sub);
        let model = if let Some(seed) = cluster_seed {
            KMeans::params_with_rng(k_req, Xoshiro256Plus::seed_from_u64(seed.wrapping_add(2001)))
                .max_n_iterations(20)
                .fit(&dataset)
                .map_err(|e| anyhow::anyhow!("fabia fallback k-means failed: {}", e))?
        } else {
            KMeans::params(k_req)
                .max_n_iterations(20)
                .fit(&dataset)
                .map_err(|e| anyhow::anyhow!("fabia fallback k-means failed: {}", e))?
        };
        Ok(model.predict(&dataset).to_vec())
    }
}

/// Spectral co-clustering style: double L1 normalization of nonnegative `X`; **k-means on columns**
/// (bucket × bucket Gram rows, cheap `m×m`); SVD row embedding; concatenate column-cluster
/// aggregates per cell with the spectral embedding; final **k-means on rows** (cells).
fn cluster_cells_spectral_bicluster(
    features: &Array2<f64>,
    num_clusters: usize,
    cluster_seed: Option<u64>,
) -> anyhow::Result<Vec<usize>> {
    let n = features.nrows();
    if n == 0 {
        return Ok(Vec::new());
    }
    let m = features.ncols();
    let k = num_clusters.max(1).min(n);

    let mut a = Array2::<f64>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            a[(i, j)] = features[(i, j)].max(0.0);
        }
    }
    normalize_l1_rows_inplace(&mut a);
    normalize_l1_cols_inplace(&mut a);

    // Column k-means: each bucket j is row j of G = X^T X (m points in R^m). No n-dim centroids.
    let g = gram_xt_x(&a);
    let k_col = k.min(m).max(1);
    let col_assign = match cluster_cells_with_kmeans(
        &g,
        k_col,
        cluster_seed.map(|s| s.wrapping_add(3001)),
    ) {
        Ok(a) => a,
        Err(_) => (0..m).map(|j| j % k_col).collect::<Vec<_>>(),
    };

    let agg = aggregate_rows_by_column_clusters(&a, &col_assign, k_col);

    let data: Vec<f64> = a.iter().copied().collect();
    let mat = DMatrix::from_row_slice(n, m, &data);
    let svd = mat.svd(true, false);
    let u = svd
        .u
        .ok_or_else(|| anyhow::anyhow!("SVD did not return U"))?;
    let sigma = svd.singular_values;

    let max_rank = n.min(m);
    let n_comp = k.min(max_rank).max(1);
    let mut embed = Array2::<f64>::zeros((n, n_comp));
    for i in 0..n {
        for j in 0..n_comp {
            let s_j = sigma[j].max(1e-18);
            embed[(i, j)] = u[(i, j)] * s_j.sqrt();
        }
    }

    let n_agg = agg.ncols();
    let mut combined = Array2::<f64>::zeros((n, n_comp + n_agg));
    for i in 0..n {
        for j in 0..n_comp {
            combined[(i, j)] = embed[(i, j)];
        }
        for j in 0..n_agg {
            combined[(i, n_comp + j)] = agg[(i, j)];
        }
    }
    l2_normalize_rows_inplace(&mut combined);

    let dataset = DatasetBase::from(combined);
    let model = if let Some(seed) = cluster_seed {
        KMeans::params_with_rng(k, Xoshiro256Plus::seed_from_u64(seed.wrapping_add(3002)))
            .max_n_iterations(20)
            .fit(&dataset)
            .map_err(|e| anyhow::anyhow!("k-means on spectral + column-cluster features failed: {}", e))?
    } else {
        KMeans::params(k)
            .max_n_iterations(20)
            .fit(&dataset)
            .map_err(|e| anyhow::anyhow!("k-means on spectral + column-cluster features failed: {}", e))?
    };

    Ok(model.predict(&dataset).to_vec())
}

fn cluster_cells_spectral_bicluster_l0(
    features: &Array2<f64>,
    num_clusters: usize,
) -> anyhow::Result<Vec<usize>> {
    let n = features.nrows();
    if n == 0 {
        return Ok(Vec::new());
    }
    let m = features.ncols();
    if m == 0 {
        anyhow::bail!("bicluster-l0: zero feature columns");
    }
    let k = num_clusters.max(1).min(n);
    if k == 1 {
        return Ok(vec![0usize; n]);
    }

    let mut b = vec![0u8; n * m];
    let mut a = Array2::<f64>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            let bit = if features[(i, j)] > 0.0 { 1u8 } else { 0u8 };
            b[i * m + j] = bit;
            a[(i, j)] = bit as f64;
        }
    }
    normalize_l1_rows_inplace(&mut a);
    normalize_l1_cols_inplace(&mut a);

    let k_col = k.min(m).max(1);
    let mut x_col = vec![0u8; m * n];
    for j in 0..m {
        for i in 0..n {
            x_col[j * n + i] = b[i * m + j];
        }
    }
    let col_assign = binary_l0_lloyd_assign(&x_col, m, n, k_col, 30);
    let agg = aggregate_rows_by_column_clusters(&a, &col_assign, k_col);

    let data: Vec<f64> = a.iter().copied().collect();
    let mat = DMatrix::from_row_slice(n, m, &data);
    let svd = mat.svd(true, false);
    let u = svd
        .u
        .ok_or_else(|| anyhow::anyhow!("SVD did not return U"))?;
    let sigma = svd.singular_values;
    let max_rank = n.min(m);
    let n_comp = k.min(max_rank).max(1);

    let mut embed = Array2::<f64>::zeros((n, n_comp));
    for i in 0..n {
        for j in 0..n_comp {
            let s_j = sigma[j].max(1e-18);
            embed[(i, j)] = u[(i, j)] * s_j.sqrt();
        }
    }

    let d = n_comp + k_col;
    let mut bin_combined = vec![0u8; n * d];
    for i in 0..n {
        for j in 0..n_comp {
            bin_combined[i * d + j] = if embed[(i, j)] > 0.0 { 1 } else { 0 };
        }
        for j in 0..k_col {
            bin_combined[i * d + n_comp + j] = if agg[(i, j)] > 0.0 { 1 } else { 0 };
        }
    }

    Ok(binary_l0_lloyd_assign(&bin_combined, n, d, k, 30))
}

/// **Swapped** vs `bicluster`: **row k-means first** on double-L1 `A` (cells in bucket space), build cluster-mean
/// matrix `R` (`k×m`), **column k-means** on `G2=R^T R` (not `A^T A`), aggregates on full `A`, **SVD** on full `A`,
/// then concat **`[agg ∥ embed]`** (aggregates before singular vectors) and final **row k-means**.
fn cluster_cells_spectral_bicluster_row_col_swapped(
    features: &Array2<f64>,
    num_clusters: usize,
    cluster_seed: Option<u64>,
) -> anyhow::Result<Vec<usize>> {
    let n = features.nrows();
    if n == 0 {
        return Ok(Vec::new());
    }
    let m = features.ncols();
    let k = num_clusters.max(1).min(n);

    let mut a = Array2::<f64>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            a[(i, j)] = features[(i, j)].max(0.0);
        }
    }
    normalize_l1_rows_inplace(&mut a);
    normalize_l1_cols_inplace(&mut a);

    let dataset_rows = DatasetBase::from(a.clone());
    let row_model = if let Some(seed) = cluster_seed {
        KMeans::params_with_rng(k, Xoshiro256Plus::seed_from_u64(seed.wrapping_add(4001)))
            .max_n_iterations(20)
            .fit(&dataset_rows)
            .map_err(|e| anyhow::anyhow!("swapped bicluster: initial row k-means failed: {}", e))?
    } else {
        KMeans::params(k)
            .max_n_iterations(20)
            .fit(&dataset_rows)
            .map_err(|e| anyhow::anyhow!("swapped bicluster: initial row k-means failed: {}", e))?
    };
    let cell_preassign = row_model.predict(&dataset_rows).to_vec();

    let mut counts = vec![0usize; k];
    let mut r_sum = Array2::<f64>::zeros((k, m));
    for i in 0..n {
        let c = cell_preassign[i];
        if c < k {
            counts[c] += 1;
            for j in 0..m {
                r_sum[(c, j)] += a[(i, j)];
            }
        }
    }
    let mut r = Array2::<f64>::zeros((k, m));
    for c in 0..k {
        let cnt = counts[c].max(1) as f64;
        for j in 0..m {
            r[(c, j)] = r_sum[(c, j)] / cnt;
        }
    }

    let g2 = gram_xt_x(&r);
    let k_col = k.min(m).max(1);
    let col_assign = match cluster_cells_with_kmeans(
        &g2,
        k_col,
        cluster_seed.map(|s| s.wrapping_add(4002)),
    ) {
        Ok(assign) => assign,
        Err(_) => (0..m).map(|j| j % k_col).collect::<Vec<_>>(),
    };

    let agg = aggregate_rows_by_column_clusters(&a, &col_assign, k_col);

    let data: Vec<f64> = a.iter().copied().collect();
    let mat = DMatrix::from_row_slice(n, m, &data);
    let svd = mat.svd(true, false);
    let u = svd
        .u
        .ok_or_else(|| anyhow::anyhow!("SVD did not return U"))?;
    let sigma = svd.singular_values;

    let max_rank = n.min(m);
    let n_comp = k.min(max_rank).max(1);
    let mut embed = Array2::<f64>::zeros((n, n_comp));
    for i in 0..n {
        for j in 0..n_comp {
            let s_j = sigma[j].max(1e-18);
            embed[(i, j)] = u[(i, j)] * s_j.sqrt();
        }
    }

    let n_agg = agg.ncols();
    let mut combined = Array2::<f64>::zeros((n, n_comp + n_agg));
    for i in 0..n {
        for j in 0..n_agg {
            combined[(i, j)] = agg[(i, j)];
        }
        for j in 0..n_comp {
            combined[(i, n_agg + j)] = embed[(i, j)];
        }
    }
    l2_normalize_rows_inplace(&mut combined);

    let dataset = DatasetBase::from(combined);
    let model = if let Some(seed) = cluster_seed {
        KMeans::params_with_rng(k, Xoshiro256Plus::seed_from_u64(seed.wrapping_add(4003)))
            .max_n_iterations(20)
            .fit(&dataset)
            .map_err(|e| anyhow::anyhow!("swapped bicluster: final row k-means failed: {}", e))?
    } else {
        KMeans::params(k)
            .max_n_iterations(20)
            .fit(&dataset)
            .map_err(|e| anyhow::anyhow!("swapped bicluster: final row k-means failed: {}", e))?
    };

    Ok(model.predict(&dataset).to_vec())
}

/// Double-L1 normalized nonnegative `X`, truncated SVD row embedding `U Σ^{1/2}`, L2-normalize rows, k-means only.
fn cluster_cells_svd_kmeans_only(
    features: &Array2<f64>,
    num_clusters: usize,
    cluster_seed: Option<u64>,
) -> anyhow::Result<Vec<usize>> {
    let n = features.nrows();
    if n == 0 {
        return Ok(Vec::new());
    }
    let m = features.ncols();
    let k = num_clusters.max(1).min(n);

    let mut a = Array2::<f64>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            a[(i, j)] = features[(i, j)].max(0.0);
        }
    }
    normalize_l1_rows_inplace(&mut a);
    normalize_l1_cols_inplace(&mut a);

    let data: Vec<f64> = a.iter().copied().collect();
    let mat = DMatrix::from_row_slice(n, m, &data);
    let svd = mat.svd(true, false);
    let u = svd
        .u
        .ok_or_else(|| anyhow::anyhow!("SVD did not return U"))?;
    let sigma = svd.singular_values;

    let max_rank = n.min(m);
    let n_comp = k.min(max_rank).max(1);
    let mut embed = Array2::<f64>::zeros((n, n_comp));
    for i in 0..n {
        for j in 0..n_comp {
            let s_j = sigma[j].max(1e-18);
            embed[(i, j)] = u[(i, j)] * s_j.sqrt();
        }
    }
    l2_normalize_rows_inplace(&mut embed);

    let dataset = DatasetBase::from(embed);
    let model = if let Some(seed) = cluster_seed {
        KMeans::params_with_rng(k, Xoshiro256Plus::seed_from_u64(seed.wrapping_add(5001)))
            .max_n_iterations(20)
            .fit(&dataset)
            .map_err(|e| anyhow::anyhow!("k-means on SVD embedding failed: {}", e))?
    } else {
        KMeans::params(k)
            .max_n_iterations(20)
            .fit(&dataset)
            .map_err(|e| anyhow::anyhow!("k-means on SVD embedding failed: {}", e))?
    };

    Ok(model.predict(&dataset).to_vec())
}

/// SVD seriation: double-L1 normalize nonneg `X`, truncated SVD to get left singular vectors,
/// sort cells by `(u₁, u₂, …)` coordinates, then cut the sorted order into `k` contiguous blocks.
/// Cells within each block are expression-neighbours in the SVD embedding, and the block
/// boundaries respect the continuous ordering — unlike k-means, which can assign spatially
/// interleaved labels.
fn cluster_cells_svd_seriation(
    features: &Array2<f64>,
    num_clusters: usize,
) -> anyhow::Result<Vec<usize>> {
    let n = features.nrows();
    if n == 0 {
        return Ok(Vec::new());
    }
    let m = features.ncols();
    let k = num_clusters.max(1).min(n);
    if k <= 1 {
        return Ok(vec![0; n]);
    }

    let mut a = Array2::<f64>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            a[(i, j)] = features[(i, j)].max(0.0);
        }
    }
    normalize_l1_rows_inplace(&mut a);
    normalize_l1_cols_inplace(&mut a);

    let data: Vec<f64> = a.iter().copied().collect();
    let mat = DMatrix::from_row_slice(n, m, &data);
    let svd = mat.svd(true, false);
    let u = match svd.u {
        Some(u) => u,
        None => return Ok((0..n).map(|i| i % k).collect()),
    };

    let max_rank = n.min(m);
    let n_comp = k.min(max_rank).max(1).min(u.ncols());

    // Build sortable key per cell: (u₁, u₂, …, original_index)
    let mut cell_keys: Vec<(usize, Vec<f64>)> = (0..n)
        .map(|i| {
            let scores: Vec<f64> = (0..n_comp).map(|j| u[(i, j)]).collect();
            (i, scores)
        })
        .collect();

    cell_keys.sort_unstable_by(|a, b| {
        for (va, vb) in a.1.iter().zip(b.1.iter()) {
            match va.total_cmp(vb) {
                std::cmp::Ordering::Equal => continue,
                other => return other,
            }
        }
        a.0.cmp(&b.0)
    });

    // Cut the sorted order into k contiguous blocks
    let mut labels = vec![0usize; n];
    let block_size = n / k;
    let remainder = n % k;
    let mut cursor = 0usize;
    for cluster_id in 0..k {
        let sz = block_size + if cluster_id < remainder { 1 } else { 0 };
        for offset in 0..sz {
            let original_idx = cell_keys[cursor + offset].0;
            labels[original_idx] = cluster_id;
        }
        cursor += sz;
    }

    Ok(labels)
}

/// Double-L1 normalized `X`; k-means on columns via Gram `X^T X`; per-cell column-cluster sums;
/// L2-normalize rows; k-means on cells only (no spectral term).
fn cluster_cells_simple_column_kmeans(
    features: &Array2<f64>,
    num_clusters: usize,
    cluster_seed: Option<u64>,
) -> anyhow::Result<Vec<usize>> {
    let n = features.nrows();
    if n == 0 {
        return Ok(Vec::new());
    }
    let m = features.ncols();
    let k = num_clusters.max(1).min(n);

    let mut a = Array2::<f64>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            a[(i, j)] = features[(i, j)].max(0.0);
        }
    }
    normalize_l1_rows_inplace(&mut a);
    normalize_l1_cols_inplace(&mut a);

    let g = gram_xt_x(&a);
    let k_col = k.min(m).max(1);
    let col_assign = match cluster_cells_with_kmeans(
        &g,
        k_col,
        cluster_seed.map(|s| s.wrapping_add(6001)),
    ) {
        Ok(a) => a,
        Err(_) => (0..m).map(|j| j % k_col).collect::<Vec<_>>(),
    };

    let mut agg = aggregate_rows_by_column_clusters(&a, &col_assign, k_col);
    l2_normalize_rows_inplace(&mut agg);

    let dataset = DatasetBase::from(agg);
    let model = if let Some(seed) = cluster_seed {
        KMeans::params_with_rng(k, Xoshiro256Plus::seed_from_u64(seed.wrapping_add(6002)))
            .max_n_iterations(20)
            .fit(&dataset)
            .map_err(|e| anyhow::anyhow!("k-means on column-cluster aggregates failed: {}", e))?
    } else {
        KMeans::params(k)
            .max_n_iterations(20)
            .fit(&dataset)
            .map_err(|e| anyhow::anyhow!("k-means on column-cluster aggregates failed: {}", e))?
    };

    Ok(model.predict(&dataset).to_vec())
}

fn median_f64(mut xs: Vec<f64>) -> f64 {
    if xs.is_empty() {
        return 1.0;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = xs.len() / 2;
    if xs.len() % 2 == 0 {
        (xs[mid - 1] + xs[mid]) * 0.5
    } else {
        xs[mid]
    }
}

/// k-means on concatenated L2 features + weighted z-scored `(x,y)` when spectral graph is too large.
fn cluster_cells_spatial_augmented_kmeans(
    points: &[Point],
    features: &Array2<f64>,
    num_clusters: usize,
    cluster_seed: Option<u64>,
) -> anyhow::Result<Vec<usize>> {
    let n = features.nrows();
    let m = features.ncols();
    let k = num_clusters.max(1).min(n);
    if n == 0 {
        return Ok(Vec::new());
    }
    if points.len() != n {
        anyhow::bail!("spatial augmented k-means: points/features length mismatch");
    }
    let mut mean_x = 0.0f64;
    let mut mean_y = 0.0f64;
    for p in points {
        mean_x += p.x;
        mean_y += p.y;
    }
    mean_x /= n as f64;
    mean_y /= n as f64;
    let mut vx = 0.0f64;
    let mut vy = 0.0f64;
    for p in points {
        vx += (p.x - mean_x).powi(2);
        vy += (p.y - mean_y).powi(2);
    }
    let sdx = (vx / n.max(1) as f64).max(1e-18).sqrt();
    let sdy = (vy / n.max(1) as f64).max(1e-18).sqrt();
    let w = 0.5f64;
    let mut aug = Array2::<f64>::zeros((n, m + 2));
    for i in 0..n {
        for j in 0..m {
            aug[(i, j)] = features[(i, j)];
        }
        aug[(i, m)] = w * (points[i].x - mean_x) / sdx;
        aug[(i, m + 1)] = w * (points[i].y - mean_y) / sdy;
    }
    cluster_cells_with_kmeans(&aug, k, cluster_seed)
}

/// Combined spatial + expression kNN affinity, normalized Laplacian spectral embedding, k-means (`n` small);
/// otherwise [`cluster_cells_spatial_augmented_kmeans`].
fn cluster_cells_spatial_graph(
    points: &[Point],
    features: &Array2<f64>,
    num_clusters: usize,
    params: SpatialGraphParams,
    cluster_seed: Option<u64>,
) -> anyhow::Result<Vec<usize>> {
    const DENSE_SPECTRAL_MAX_N: usize = 2048;

    let n = features.nrows();
    if n == 0 {
        return Ok(Vec::new());
    }
    if points.len() != n {
        anyhow::bail!("spatial graph: points/features length mismatch");
    }
    let k = num_clusters.max(1).min(n);
    if k == 1 {
        return Ok(vec![0; n]);
    }
    let m = features.ncols();

    if n > DENSE_SPECTRAL_MAX_N {
        warn!(
            "spatial-graph clustering: n={} > {} — using augmented (expression + xy) k-means",
            n, DENSE_SPECTRAL_MAX_N
        );
        return cluster_cells_spatial_augmented_kmeans(points, features, k, cluster_seed);
    }

    let ks = params.spatial_knn.max(1).min(n - 1);
    let ke = params.expr_knn.max(1).min(n - 1);
    let blend = params.blend.clamp(0.0, 1.0);

    let mut d2 = vec![0.0f64; n * n];
    let mut sim = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let dx = points[i].x - points[j].x;
            let dy = points[i].y - points[j].y;
            d2[i * n + j] = dx * dx + dy * dy;
            let mut s = 0.0f64;
            for t in 0..m {
                s += features[(i, t)] * features[(j, t)];
            }
            sim[i * n + j] = s.max(0.0);
        }
    }

    let mut spatial_d2_samples: Vec<f64> = Vec::new();
    let mut expr_one_minus_sim: Vec<f64> = Vec::new();

    for i in 0..n {
        let mut order: Vec<usize> = (0..n).filter(|&j| j != i).collect();
        order.sort_by(|&a, &b| {
            d2[i * n + a]
                .partial_cmp(&d2[i * n + b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for &j in order.iter().take(ks) {
            spatial_d2_samples.push(d2[i * n + j]);
        }

        order.sort_by(|&a, &b| {
            sim[i * n + b]
                .partial_cmp(&sim[i * n + a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for &j in order.iter().take(ke) {
            expr_one_minus_sim.push((1.0 - sim[i * n + j]).max(0.0));
        }
    }

    let sigma_s_sq = median_f64(spatial_d2_samples).max(1e-18);
    let tau_e = median_f64(expr_one_minus_sim).max(1e-6);

    let mut w = vec![0.0f64; n * n];
    for i in 0..n {
        let mut order: Vec<usize> = (0..n).filter(|&j| j != i).collect();
        order.sort_by(|&a, &b| {
            d2[i * n + a]
                .partial_cmp(&d2[i * n + b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for &j in order.iter().take(ks) {
            let ws = (-0.5 * d2[i * n + j] / sigma_s_sq).exp();
            w[i * n + j] += blend * ws;
        }

        order.sort_by(|&a, &b| {
            sim[i * n + b]
                .partial_cmp(&sim[i * n + a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for &j in order.iter().take(ke) {
            let we = (-(1.0 - sim[i * n + j]).max(0.0) / tau_e).exp();
            w[i * n + j] += (1.0 - blend) * we;
        }
    }

    for i in 0..n {
        for j in 0..n {
            if i != j {
                let v = 0.5 * (w[i * n + j] + w[j * n + i]);
                w[i * n + j] = v;
                w[j * n + i] = v;
            }
        }
    }

    let eps = 1e-9f64;
    for i in 0..n {
        w[i * n + i] = eps;
    }

    let mut deg = vec![0.0f64; n];
    for i in 0..n {
        for j in 0..n {
            deg[i] += w[i * n + j];
        }
        deg[i] = deg[i].max(1e-12);
    }

    let mut l_data = vec![0.0f64; n * n];
    for i in 0..n {
        let si = deg[i].sqrt();
        for j in 0..n {
            let sj = deg[j].sqrt();
            let val = if i == j {
                1.0 - w[i * n + j] / (si * sj)
            } else {
                -w[i * n + j] / (si * sj)
            };
            l_data[i * n + j] = val;
        }
    }

    let l_sym = DMatrix::from_row_slice(n, n, &l_data);
    let decomp = SymmetricEigen::new(l_sym);
    let evecs = decomp.eigenvectors;
    let n_ev = k.min(n - 1).max(1);
    let mut embed = Array2::<f64>::zeros((n, n_ev));
    for col in 0..n_ev {
        let src = col + 1;
        if src >= n {
            break;
        }
        for row in 0..n {
            embed[(row, col)] = evecs[(row, src)];
        }
    }

    let dataset = DatasetBase::from(embed);
    let model = if let Some(seed) = cluster_seed {
        KMeans::params_with_rng(k, Xoshiro256Plus::seed_from_u64(seed.wrapping_add(7001)))
            .max_n_iterations(30)
            .fit(&dataset)
            .map_err(|e| anyhow::anyhow!("spatial-graph spectral k-means failed: {}", e))?
    } else {
        KMeans::params(k)
            .max_n_iterations(30)
            .fit(&dataset)
            .map_err(|e| anyhow::anyhow!("spatial-graph spectral k-means failed: {}", e))?
    };
    Ok(model.predict(&dataset).to_vec())
}

fn bounds_xy(points: &[Point]) -> (f64, f64, f64, f64) {
    let mut xmin = f64::INFINITY;
    let mut xmax = f64::NEG_INFINITY;
    let mut ymin = f64::INFINITY;
    let mut ymax = f64::NEG_INFINITY;
    for p in points {
        xmin = xmin.min(p.x);
        xmax = xmax.max(p.x);
        ymin = ymin.min(p.y);
        ymax = ymax.max(p.y);
    }
    if !xmin.is_finite() || !xmax.is_finite() {
        (0.0, 1.0, 0.0, 1.0)
    } else {
        (xmin, xmax, ymin, ymax)
    }
}

/// Fractional axial coords, then cube rounding → integer axial (q, r).
fn axial_hex_round(q: f64, r: f64) -> (i32, i32) {
    let x = q;
    let y = r;
    let z = -q - r;
    let mut rx = x.round() as i32;
    let mut ry = y.round() as i32;
    let mut rz = z.round() as i32;
    let x_diff = (rx as f64 - x).abs();
    let y_diff = (ry as f64 - y).abs();
    let z_diff = (rz as f64 - z).abs();
    if x_diff > y_diff && x_diff > z_diff {
        rx = -ry - rz;
    } else if y_diff > z_diff {
        ry = -rx - rz;
    } else {
        let new_z = -rx - ry;
        rz = new_z;
    }
    debug_assert_eq!(rx + ry + rz, 0);
    let _ = rz;
    (rx, ry)
}

/// Pointy-top hex: pixel (x,y) relative to origin → fractional axial (q,r).
fn pixel_to_axial_hex(x: f64, y: f64, hex_size: f64) -> (f64, f64) {
    let s = hex_size.max(1e-18);
    let q = (3f64.sqrt() / 3.0 * x - 1.0 / 3.0 * y) / s;
    let r = (2.0 / 3.0 * y) / s;
    (q, r)
}

/// Pointy-top hex center (pixel space) for integer axial (q, r).
fn axial_hex_center_xy(q: i32, r: i32, hex_size: f64) -> (f64, f64) {
    let s = hex_size.max(1e-18);
    let q = q as f64;
    let r = r as f64;
    let x = s * (3f64.sqrt() * q + 3f64.sqrt() / 2.0 * r);
    let y = s * (3.0 / 2.0 * r);
    (x, y)
}

/// Discretize the slide into tiles, then k-means on `[per-cell buckets \| tile embedding]`.
fn cluster_cells_tensor_grid(
    points: &[Point],
    features: &Array2<f64>,
    num_clusters: usize,
    params: TensorGridParams,
    cluster_seed: Option<u64>,
) -> anyhow::Result<Vec<usize>> {
    let n = features.nrows();
    if n == 0 {
        return Ok(Vec::new());
    }
    if points.len() != n {
        anyhow::bail!("tensor-grid: points/features length mismatch");
    }
    let k = num_clusters.max(1).min(n);
    let m = features.ncols();
    let nx = params.nx.max(1);
    let ny = params.ny.max(1);
    let w_t = params.tile_weight.max(0.0);

    let (xmin, xmax, ymin, ymax) = bounds_xy(points);
    let span_x = (xmax - xmin).max(1e-18);
    let span_y = (ymax - ymin).max(1e-18);

    let mut tile_id: Vec<usize> = Vec::with_capacity(n);
    let mut center_tx: Vec<f64> = Vec::with_capacity(n);
    let mut center_ty: Vec<f64> = Vec::with_capacity(n);

    match params.disc {
        TensorGridDisc::Rect => {
            for i in 0..n {
                let x = points[i].x;
                let y = points[i].y;
                let fx = ((x - xmin) / span_x * nx as f64).floor() as i64;
                let fy = ((y - ymin) / span_y * ny as f64).floor() as i64;
                let ix = fx.clamp(0, nx as i64 - 1) as usize;
                let iy = fy.clamp(0, ny as i64 - 1) as usize;
                let tid = ix + iy * nx;
                tile_id.push(tid);
                center_tx.push((ix as f64 + 0.5) / nx as f64);
                center_ty.push((iy as f64 + 0.5) / ny as f64);
            }
        }
        TensorGridDisc::Hex => {
            let hex_size = (span_x / (nx as f64 * 1.25 + 0.25))
                .max(span_y / (ny as f64 * 1.25 + 0.25))
                .max(1e-18);
            let mut axial: Vec<(i32, i32)> = Vec::with_capacity(n);
            for i in 0..n {
                let x = points[i].x - xmin;
                let y = points[i].y - ymin;
                let (fq, fr) = pixel_to_axial_hex(x, y, hex_size);
                axial.push(axial_hex_round(fq, fr));
            }
            let min_q = axial.iter().map(|(q, _)| *q).min().unwrap_or(0);
            let min_r = axial.iter().map(|(_, r)| *r).min().unwrap_or(0);
            let max_r = axial.iter().map(|(_, r)| *r).max().unwrap_or(0);
            let nr = (max_r - min_r + 1).max(1) as usize;
            for i in 0..n {
                let (iq, ir) = axial[i];
                let tid = (iq - min_q) as usize * nr + (ir - min_r) as usize;
                tile_id.push(tid);
                let (cx, cy) = axial_hex_center_xy(iq, ir, hex_size);
                center_tx.push((cx / span_x).clamp(0.0, 1.0));
                center_ty.push((cy / span_y).clamp(0.0, 1.0));
            }
        }
    }

    let num_tiles = tile_id.iter().copied().max().unwrap_or(0) + 1;
    let use_onehot = num_tiles <= params.onehot_max_tiles.max(1) && num_tiles <= 4096;

    if use_onehot {
        let mut aug = Array2::<f64>::zeros((n, m + num_tiles));
        for i in 0..n {
            for j in 0..m {
                aug[(i, j)] = features[(i, j)];
            }
            aug[(i, m + tile_id[i])] = w_t;
        }
        cluster_cells_with_kmeans(&aug, k, cluster_seed)
    } else {
        let mut aug = Array2::<f64>::zeros((n, m + 2));
        for i in 0..n {
            for j in 0..m {
                aug[(i, j)] = features[(i, j)];
            }
            aug[(i, m)] = w_t * center_tx[i];
            aug[(i, m + 1)] = w_t * center_ty[i];
        }
        cluster_cells_with_kmeans(&aug, k, cluster_seed)
    }
}

fn cluster_cells(
    method: CellClusterMethodArg,
    features: &Array2<f64>,
    num_clusters: usize,
    points: &[Point],
    spatial: SpatialGraphParams,
    tensor_grid: TensorGridParams,
    cluster_seed: Option<u64>,
) -> anyhow::Result<Vec<usize>> {
    // Keep default behavior for most methods on continuous features, but preserve the
    // binary-feature variant for `spatial-graph` (requested for A/B testing).
    let mut features_bin = Array2::<f64>::zeros(features.raw_dim());
    for i in 0..features.nrows() {
        for j in 0..features.ncols() {
            features_bin[(i, j)] = if features[(i, j)] > 0.0 { 1.0 } else { 0.0 };
        }
    }
    let features_src = features;

    match method {
        CellClusterMethodArg::Kmeans => {
            cluster_cells_with_kmeans(features_src, num_clusters, cluster_seed)
        }
        CellClusterMethodArg::BinaryL0Kmeans => {
            cluster_cells_binary_l0_kmeans(features_src, num_clusters)
        }
        CellClusterMethodArg::Bicluster => {
            cluster_cells_spectral_bicluster(features_src, num_clusters, cluster_seed)
        }
        CellClusterMethodArg::BiclusterL0 => {
            cluster_cells_spectral_bicluster_l0(features_src, num_clusters)
        }
        CellClusterMethodArg::BiclusterSwapped => {
            cluster_cells_spectral_bicluster_row_col_swapped(features_src, num_clusters, cluster_seed)
        }
        CellClusterMethodArg::SvdKmeans => {
            cluster_cells_svd_kmeans_only(features_src, num_clusters, cluster_seed)
        }
        CellClusterMethodArg::SvdSeriation => {
            cluster_cells_svd_seriation(features_src, num_clusters)
        }
        CellClusterMethodArg::ColumnKmeans => {
            cluster_cells_simple_column_kmeans(features_src, num_clusters, cluster_seed)
        }
        CellClusterMethodArg::SpectralCocluster => {
            cluster_cells_spectral_cocluster_dhillon(features_src, num_clusters, cluster_seed)
        }
        CellClusterMethodArg::Fabia => cluster_cells_fabia(features_src, num_clusters, cluster_seed),
        CellClusterMethodArg::SpatialGraph => {
            cluster_cells_spatial_graph(points, &features_bin, num_clusters, spatial, cluster_seed)
        }
        CellClusterMethodArg::TensorGrid => {
            cluster_cells_tensor_grid(points, features_src, num_clusters, tensor_grid, cluster_seed)
        }
        CellClusterMethodArg::CellSqueeze | CellClusterMethodArg::SvdJoint => {
            // Normally handled by joint_bicluster_kluger / joint_svd_seriation in
            // run_clustered_compression; this branch is only reached as a fallback.
            cluster_cells_with_kmeans(features_src, num_clusters, cluster_seed)
        }
    }
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
    points: &[Point],
    initial_clusters: Vec<Vec<usize>>,
    features: &Array2<f64>,
    max_cluster_size: Option<usize>,
    cell_cluster_method: CellClusterMethodArg,
    spatial: SpatialGraphParams,
    tensor_grid: TensorGridParams,
    cluster_seed: Option<u64>,
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
        let subpoints: Vec<Point> = cluster.iter().map(|&i| points[i].clone()).collect();
        let sub_assignments = match cluster_cells(
            cell_cluster_method,
            &subfeatures,
            split_k,
            &subpoints,
            spatial,
            tensor_grid,
            cluster_seed.map(|s| s.wrapping_add(cluster.len() as u64)),
        ) {
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
        (0..max_svd_rows).map(|i| (i as f64 * step) as usize).collect()
    } else {
        (0..n_cells).collect()
    };

    let active_cols: Vec<usize> = if active_genes.len() > max_svd_cols {
        // Keep the most frequent genes
        let mut by_freq: Vec<(usize, u64)> = active_genes.iter().map(|&g| (g, col_nnz[g])).collect();
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
    let sampled_set: HashMap<usize, usize> = sampled_rows.iter().enumerate().map(|(i, &r)| (r, i)).collect();

    for &(stream_pos, gene_idx, _) in &triplets {
        if let Some(&local_row) = sampled_set.get(&stream_pos) {
            if let Some(&local_col) = col_local.get(&gene_idx) {
                dense[local_row * nc + local_col] = 1.0 / (col_nnz[gene_idx] as f64).sqrt().max(1e-12);
            }
        }
    }

    let mat = DMatrix::from_row_slice(nr, nc, &dense);
    let svd = mat.svd(false, true);
    let v_t = match svd.v_t {
        Some(vt) => vt,
        None => {
            info!("SVD gene reorder: V^T not available, falling back to projection method");
            return compute_gene_permutation_from_row_stream(points, data, row_stream_point_indices);
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
        (0..max_svd_rows).map(|i| (i as f64 * step) as usize).collect()
    } else {
        (0..n_cells).collect()
    };

    let active_cols: Vec<usize> = if active_genes.len() > max_svd_cols {
        let mut by_freq: Vec<(usize, u64)> = active_genes.iter().map(|&g| (g, col_nnz[g])).collect();
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
    let sampled_set: HashMap<usize, usize> = sampled_rows.iter().enumerate().map(|(i, &r)| (r, i)).collect();

    for &(stream_pos, gene_idx) in &triplets {
        if let Some(&local_row) = sampled_set.get(&stream_pos) {
            if let Some(&local_col) = col_local.get(&gene_idx) {
                dense[local_row * nc + local_col] = 1.0 / (col_nnz[gene_idx] as f64).sqrt().max(1e-12);
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
        KMeans::params_with_rng(gene_k, Xoshiro256Plus::seed_from_u64(seed.wrapping_add(9001)))
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
        let mean_a = if cluster_count[a] > 0 { cluster_v1_sum[a] / cluster_count[a] as f64 } else { f64::INFINITY };
        let mean_b = if cluster_count[b] > 0 { cluster_v1_sum[b] / cluster_count[b] as f64 } else { f64::INFINITY };
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

/// True joint biclustering: one SVD on the full sparse cell×gene matrix produces
/// **both** cell cluster assignments (from left singular vectors) **and** a gene
/// permutation (from right singular vectors).
///
/// Uses TF-IDF normalization on binary support, same as [`compute_gene_permutation_svd`],
/// and sorts cells into `k` contiguous blocks via seriation on left singular vectors,
/// same spirit as [`cluster_cells_svd_seriation`].
///
/// Returns `(cell_assignments, gene_new_to_old, gene_old_to_new)`.
fn joint_bicluster_svd(
    points: &[Point],
    data: &CsMat<u16>,
    num_clusters: usize,
    cluster_seed: Option<u64>,
) -> anyhow::Result<(Vec<usize>, Vec<u32>, Vec<u32>)> {
    let n_cells = points.len();
    let n_genes = data.cols();

    if n_cells == 0 {
        return Ok((Vec::new(), (0..n_genes as u32).collect(), (0..n_genes as u32).collect()));
    }

    let k = num_clusters.max(1).min(n_cells);

    // --- Build binary support triplets ---
    let mut col_nnz = vec![0u64; n_genes];
    let mut triplets: Vec<(usize, usize)> = Vec::new();
    for (ci, point) in points.iter().enumerate() {
        if let Some(row) = data.outer_view(point.row_index) {
            for (gene_idx, &value) in row.iter() {
                if value != 0 {
                    col_nnz[gene_idx] += 1;
                    triplets.push((ci, gene_idx));
                }
            }
        }
    }

    let active_genes: Vec<usize> = (0..n_genes).filter(|&g| col_nnz[g] > 0).collect();
    if active_genes.len() <= 2 {
        return Ok((
            (0..n_cells).map(|i| i % k).collect(),
            (0..n_genes as u32).collect(),
            (0..n_genes as u32).collect(),
        ));
    }

    // --- Subsample for dense SVD tractability ---
    let max_svd_rows = 4000usize;
    let max_svd_cols = 2000usize;

    let sampled_rows: Vec<usize> = if n_cells > max_svd_rows {
        let step = n_cells as f64 / max_svd_rows as f64;
        (0..max_svd_rows).map(|i| (i as f64 * step) as usize).collect()
    } else {
        (0..n_cells).collect()
    };

    let active_cols: Vec<usize> = if active_genes.len() > max_svd_cols {
        let mut by_freq: Vec<(usize, u64)> = active_genes.iter().map(|&g| (g, col_nnz[g])).collect();
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

    // --- Build dense matrix with TF-IDF normalization ---
    let mut dense = vec![0.0f64; nr * nc];
    let sampled_set: HashMap<usize, usize> =
        sampled_rows.iter().enumerate().map(|(i, &r)| (r, i)).collect();

    for &(ci, gene_idx) in &triplets {
        if let Some(&local_row) = sampled_set.get(&ci) {
            if let Some(&local_col) = col_local.get(&gene_idx) {
                dense[local_row * nc + local_col] = 1.0 / (col_nnz[gene_idx] as f64).sqrt().max(1e-12);
            }
        }
    }

    // --- SVD: need both U and V^T ---
    let mat = DMatrix::from_row_slice(nr, nc, &dense);
    let svd = mat.svd(true, true);
    let (u, v_t) = match (svd.u, svd.v_t) {
        (Some(u), Some(vt)) => (u, vt),
        _ => {
            return Ok((
                (0..n_cells).map(|i| i % k).collect(),
                (0..n_genes as u32).collect(),
                (0..n_genes as u32).collect(),
            ));
        }
    };

    // --- CELL SIDE: embed all cells via left singular vectors, then k-means ---
    let n_comp_cell = k.min(u.ncols()).max(1);

    // Sampled cells: direct from U
    let mut cell_embed = Array2::<f64>::zeros((n_cells, n_comp_cell));
    for (local_row, &ci) in sampled_rows.iter().enumerate() {
        for j in 0..n_comp_cell {
            cell_embed[(ci, j)] = u[(local_row, j)];
        }
    }

    // Unsampled cells: project via V^T (Nyström)
    let sigma = &svd.singular_values;
    for ci in 0..n_cells {
        if sampled_set.contains_key(&ci) {
            continue;
        }
        let point = &points[ci];
        if let Some(row) = data.outer_view(point.row_index) {
            for j in 0..n_comp_cell {
                let sig_inv = if sigma[j] > 1e-12 { 1.0 / sigma[j] } else { 0.0 };
                let mut dot = 0.0f64;
                for (g, &v) in row.iter() {
                    if v != 0 {
                        if let Some(&lc) = col_local.get(&g) {
                            let tfidf = 1.0 / (col_nnz[g] as f64).sqrt().max(1e-12);
                            dot += tfidf * v_t[(j, lc)];
                        }
                    }
                }
                cell_embed[(ci, j)] = dot * sig_inv;
            }
        }
    }

    l2_normalize_rows_inplace(&mut cell_embed);

    let dataset_cells = DatasetBase::from(cell_embed);
    let cell_model = if let Some(seed) = cluster_seed {
        KMeans::params_with_rng(k, Xoshiro256Plus::seed_from_u64(seed.wrapping_add(7001)))
            .max_n_iterations(30)
            .fit(&dataset_cells)
            .map_err(|e| anyhow::anyhow!("joint bicluster cell k-means failed: {}", e))?
    } else {
        KMeans::params(k)
            .max_n_iterations(30)
            .fit(&dataset_cells)
            .map_err(|e| anyhow::anyhow!("joint bicluster cell k-means failed: {}", e))?
    };
    let cell_labels: Vec<usize> = cell_model.predict(&dataset_cells).to_vec();

    // --- GENE SIDE: sort genes by (v₁, v₂) from the same SVD ---
    let n_sv = v_t.nrows().min(2);
    let mut scores: Vec<(usize, f64, f64)> = Vec::with_capacity(n_genes);
    for &global_gene in &active_cols {
        let local_col = col_local[&global_gene];
        let s1 = if n_sv >= 1 { v_t[(0, local_col)] } else { 0.0 };
        let s2 = if n_sv >= 2 { v_t[(1, local_col)] } else { 0.0 };
        scores.push((global_gene, s1, s2));
    }
    for g in 0..n_genes {
        if col_nnz[g] == 0 || !col_local.contains_key(&g) {
            scores.push((g, f64::INFINITY, g as f64));
        }
    }
    scores.sort_unstable_by(|a, b| {
        a.1.total_cmp(&b.1)
            .then_with(|| a.2.total_cmp(&b.2))
            .then_with(|| a.0.cmp(&b.0))
    });

    let mut gene_old_to_new = vec![0u32; n_genes];
    let mut gene_new_to_old = Vec::with_capacity(n_genes);
    for (new_idx, &(old_idx, _, _)) in scores.iter().enumerate() {
        gene_old_to_new[old_idx] = new_idx as u32;
        gene_new_to_old.push(old_idx as u32);
    }

    info!(
        "Joint bicluster SVD: {} cells -> {} clusters, {} genes reordered ({} active in SVD)",
        n_cells, k, n_genes, nc
    );

    Ok((cell_labels, gene_new_to_old, gene_old_to_new))
}

/// Joint SVD seriation: one SVD on TF-IDF-normalized cell×gene matrix, then pure sorting
/// on both axes — cells sorted by left singular vectors into `k` contiguous blocks,
/// genes sorted by right singular vectors.  No k-means anywhere.
fn joint_svd_seriation(
    points: &[Point],
    data: &CsMat<u16>,
    num_clusters: usize,
) -> anyhow::Result<(Vec<usize>, Vec<u32>, Vec<u32>)> {
    let n_cells = points.len();
    let n_genes = data.cols();

    if n_cells == 0 {
        return Ok((Vec::new(), (0..n_genes as u32).collect(), (0..n_genes as u32).collect()));
    }

    let k = num_clusters.max(1).min(n_cells);

    let mut col_nnz = vec![0u64; n_genes];
    let mut triplets: Vec<(usize, usize)> = Vec::new();
    for (ci, point) in points.iter().enumerate() {
        if let Some(row) = data.outer_view(point.row_index) {
            for (gene_idx, &value) in row.iter() {
                if value != 0 {
                    col_nnz[gene_idx] += 1;
                    triplets.push((ci, gene_idx));
                }
            }
        }
    }

    let active_genes: Vec<usize> = (0..n_genes).filter(|&g| col_nnz[g] > 0).collect();
    if active_genes.len() <= 2 {
        return Ok((
            (0..n_cells).map(|i| i % k).collect(),
            (0..n_genes as u32).collect(),
            (0..n_genes as u32).collect(),
        ));
    }

    let max_svd_rows = 4000usize;
    let max_svd_cols = 2000usize;

    let sampled_rows: Vec<usize> = if n_cells > max_svd_rows {
        let step = n_cells as f64 / max_svd_rows as f64;
        (0..max_svd_rows).map(|i| (i as f64 * step) as usize).collect()
    } else {
        (0..n_cells).collect()
    };

    let active_cols: Vec<usize> = if active_genes.len() > max_svd_cols {
        let mut by_freq: Vec<(usize, u64)> = active_genes.iter().map(|&g| (g, col_nnz[g])).collect();
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

    // TF-IDF: binary support / sqrt(doc-freq)
    let mut dense = vec![0.0f64; nr * nc];
    let sampled_set: HashMap<usize, usize> =
        sampled_rows.iter().enumerate().map(|(i, &r)| (r, i)).collect();

    for &(ci, gene_idx) in &triplets {
        if let Some(&local_row) = sampled_set.get(&ci) {
            if let Some(&local_col) = col_local.get(&gene_idx) {
                dense[local_row * nc + local_col] = 1.0 / (col_nnz[gene_idx] as f64).sqrt().max(1e-12);
            }
        }
    }

    let mat = DMatrix::from_row_slice(nr, nc, &dense);
    let svd = mat.svd(true, true);
    let (u, v_t) = match (svd.u, svd.v_t) {
        (Some(u), Some(vt)) => (u, vt),
        _ => {
            return Ok((
                (0..n_cells).map(|i| i % k).collect(),
                (0..n_genes as u32).collect(),
                (0..n_genes as u32).collect(),
            ));
        }
    };

    // --- CELL SIDE: embed all cells via U, Nyström for unsampled, then sort ---
    let n_comp = k.min(u.ncols()).max(1);
    let sigma = &svd.singular_values;

    let mut cell_coords: Vec<(usize, Vec<f64>)> = Vec::with_capacity(n_cells);
    for ci in 0..n_cells {
        let mut coords = vec![0.0f64; n_comp];
        if let Some(&lr) = sampled_set.get(&ci) {
            for j in 0..n_comp { coords[j] = u[(lr, j)]; }
        } else {
            let point = &points[ci];
            if let Some(row) = data.outer_view(point.row_index) {
                for j in 0..n_comp {
                    let sig_inv = if sigma[j] > 1e-12 { 1.0 / sigma[j] } else { 0.0 };
                    let mut dot = 0.0f64;
                    for (g, &v) in row.iter() {
                        if v != 0 {
                            if let Some(&lc) = col_local.get(&g) {
                                let tfidf = 1.0 / (col_nnz[g] as f64).sqrt().max(1e-12);
                                dot += tfidf * v_t[(j, lc)];
                            }
                        }
                    }
                    coords[j] = dot * sig_inv;
                }
            }
        }
        cell_coords.push((ci, coords));
    }

    // Sort cells by (u₁, u₂, …) lexicographically
    cell_coords.sort_unstable_by(|a, b| {
        for (ca, cb) in a.1.iter().zip(b.1.iter()) {
            match ca.total_cmp(cb) {
                std::cmp::Ordering::Equal => continue,
                ord => return ord,
            }
        }
        a.0.cmp(&b.0)
    });

    // Cut the sorted order into k contiguous blocks
    let mut cell_labels = vec![0usize; n_cells];
    let block_size = n_cells / k;
    let remainder = n_cells % k;
    let mut offset = 0usize;
    for cluster_id in 0..k {
        let sz = block_size + if cluster_id < remainder { 1 } else { 0 };
        for pos in offset..offset + sz {
            let ci = cell_coords[pos].0;
            cell_labels[ci] = cluster_id;
        }
        offset += sz;
    }

    // --- GENE SIDE: sort genes by (v₁, v₂) ---
    let n_sv = v_t.nrows().min(2);
    let mut scores: Vec<(usize, f64, f64)> = Vec::with_capacity(n_genes);
    for &global_gene in &active_cols {
        let lc = col_local[&global_gene];
        let s1 = if n_sv >= 1 { v_t[(0, lc)] } else { 0.0 };
        let s2 = if n_sv >= 2 { v_t[(1, lc)] } else { 0.0 };
        scores.push((global_gene, s1, s2));
    }
    for g in 0..n_genes {
        if col_nnz[g] == 0 || !col_local.contains_key(&g) {
            scores.push((g, f64::INFINITY, g as f64));
        }
    }
    scores.sort_unstable_by(|a, b| {
        a.1.total_cmp(&b.1)
            .then_with(|| a.2.total_cmp(&b.2))
            .then_with(|| a.0.cmp(&b.0))
    });

    let mut gene_old_to_new = vec![0u32; n_genes];
    let mut gene_new_to_old = Vec::with_capacity(n_genes);
    for (new_idx, &(old_idx, _, _)) in scores.iter().enumerate() {
        gene_old_to_new[old_idx] = new_idx as u32;
        gene_new_to_old.push(old_idx as u32);
    }

    info!(
        "Joint SVD seriation: {} cells -> {} contiguous blocks, {} genes reordered ({} active in SVD)",
        n_cells, k, n_genes, nc
    );

    Ok((cell_labels, gene_new_to_old, gene_old_to_new))
}

/// Generic post-hoc gene permutation derived from cell cluster assignments.
///
/// Given cell cluster labels (from any clustering method), build a `k × n_genes` centroid
/// matrix (mean `ln(1+count)` per cluster), SVD it, and sort genes by `(v₁, v₂)`.
/// This couples gene ordering to cell structure without requiring a method-specific joint
/// algorithm.
fn derive_gene_permutation_from_cell_clusters(
    points: &[Point],
    data: &CsMat<u16>,
    cell_labels: &[usize],
    num_clusters: usize,
) -> anyhow::Result<(Vec<u32>, Vec<u32>)> {
    let n_genes = data.cols();
    let k = num_clusters.max(1);

    let mut centroids = vec![0.0f64; k * n_genes];
    let mut cluster_sizes = vec![0u64; k];

    for (ci, point) in points.iter().enumerate() {
        let label = cell_labels[ci];
        if label >= k {
            continue;
        }
        cluster_sizes[label] += 1;
        if let Some(row) = data.outer_view(point.row_index) {
            for (gene_idx, &value) in row.iter() {
                if value != 0 {
                    centroids[label * n_genes + gene_idx] += (1.0 + value as f64).ln();
                }
            }
        }
    }

    for c in 0..k {
        let sz = cluster_sizes[c].max(1) as f64;
        for g in 0..n_genes {
            centroids[c * n_genes + g] /= sz;
        }
    }

    let mat = DMatrix::from_row_slice(k, n_genes, &centroids);
    let svd = mat.svd(false, true);
    let v_t = match svd.v_t {
        Some(vt) => vt,
        None => {
            return Ok(((0..n_genes as u32).collect(), (0..n_genes as u32).collect()));
        }
    };

    let n_sv = v_t.nrows().min(2);
    let mut scores: Vec<(usize, f64, f64)> = (0..n_genes)
        .map(|g| {
            let s1 = if n_sv >= 1 { v_t[(0, g)] } else { 0.0 };
            let s2 = if n_sv >= 2 { v_t[(1, g)] } else { 0.0 };
            (g, s1, s2)
        })
        .collect();

    scores.sort_unstable_by(|a, b| {
        a.1.total_cmp(&b.1)
            .then_with(|| a.2.total_cmp(&b.2))
            .then_with(|| a.0.cmp(&b.0))
    });

    let mut gene_old_to_new = vec![0u32; n_genes];
    let mut gene_new_to_old = Vec::with_capacity(n_genes);
    for (new_idx, &(old_idx, _, _)) in scores.iter().enumerate() {
        gene_old_to_new[old_idx] = new_idx as u32;
        gene_new_to_old.push(old_idx as u32);
    }

    info!(
        "Derived gene permutation from {} cell clusters, {} genes",
        k, n_genes
    );

    Ok((gene_new_to_old, gene_old_to_new))
}

/// Kluger (2003) spectral biclustering, as used in
/// [cell-squeeze](https://github.com/maharshi95/cell-squeeze).
///
/// 1. Bistochastic-normalize the sparse cell×gene matrix (iterative scale-normalize).
/// 2. SVD → U, V (first `n_components` singular vectors, discard DC).
/// 3. For each singular vector, fit 1-D k-means to find its best piecewise-constant
///    approximation; keep the `n_best` vectors with smallest residual.
/// 4. Row labels: project `X @ V_best`, k-means → `n_row_clusters`.
/// 5. Col labels: project `X^T @ U_best`, k-means → `n_col_clusters`.
/// 6. Gene permutation: `argsort(column_labels)`.
///
/// Returns `(cell_assignments, gene_new_to_old, gene_old_to_new)`.
fn joint_bicluster_kluger(
    points: &[Point],
    data: &CsMat<u16>,
    num_row_clusters: usize,
    num_col_clusters: usize,
    cluster_seed: Option<u64>,
) -> anyhow::Result<(Vec<usize>, Vec<u32>, Vec<u32>)> {
    let n_cells = points.len();
    let n_genes = data.cols();

    if n_cells == 0 {
        return Ok((Vec::new(), (0..n_genes as u32).collect(), (0..n_genes as u32).collect()));
    }

    let kr = num_row_clusters.max(1).min(n_cells);
    let kc = num_col_clusters.max(1).min(n_genes);

    // --- Subsample ---
    let max_rows = 4000usize;
    let max_cols = 2000usize;

    let mut col_nnz = vec![0u64; n_genes];
    let mut triplets: Vec<(usize, usize, f64)> = Vec::new();
    for (ci, point) in points.iter().enumerate() {
        if let Some(row) = data.outer_view(point.row_index) {
            for (gene_idx, &value) in row.iter() {
                if value != 0 {
                    col_nnz[gene_idx] += 1;
                    triplets.push((ci, gene_idx, (1.0 + value as f64).ln()));
                }
            }
        }
    }

    let active_genes: Vec<usize> = (0..n_genes).filter(|&g| col_nnz[g] > 0).collect();
    if active_genes.len() <= 2 || n_cells <= 2 {
        return Ok((
            (0..n_cells).map(|i| i % kr).collect(),
            (0..n_genes as u32).collect(),
            (0..n_genes as u32).collect(),
        ));
    }

    let sampled_rows: Vec<usize> = if n_cells > max_rows {
        let step = n_cells as f64 / max_rows as f64;
        (0..max_rows).map(|i| (i as f64 * step) as usize).collect()
    } else {
        (0..n_cells).collect()
    };

    let active_cols: Vec<usize> = if active_genes.len() > max_cols {
        let mut by_freq: Vec<(usize, u64)> = active_genes.iter().map(|&g| (g, col_nnz[g])).collect();
        by_freq.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        by_freq.truncate(max_cols);
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
    let sampled_set: HashMap<usize, usize> =
        sampled_rows.iter().enumerate().map(|(i, &r)| (r, i)).collect();

    // --- Build dense matrix with ln(1+count) ---
    let mut dense = vec![0.0f64; nr * nc];
    for &(ci, gene_idx, val) in &triplets {
        if let Some(&local_row) = sampled_set.get(&ci) {
            if let Some(&local_col) = col_local.get(&gene_idx) {
                dense[local_row * nc + local_col] = val;
            }
        }
    }

    // --- Bistochastic normalization (iterative scale-normalize) ---
    for _ in 0..100 {
        // Row normalize: each row sums to constant
        for i in 0..nr {
            let rsum: f64 = (0..nc).map(|j| dense[i * nc + j]).sum();
            if rsum > 1e-18 {
                let s = 1.0 / rsum.sqrt();
                for j in 0..nc { dense[i * nc + j] *= s; }
            }
        }
        // Column normalize: each column sums to constant
        for j in 0..nc {
            let csum: f64 = (0..nr).map(|i| dense[i * nc + j]).sum();
            if csum > 1e-18 {
                let s = 1.0 / csum.sqrt();
                for i in 0..nr { dense[i * nc + j] *= s; }
            }
        }
    }

    // --- SVD ---
    let n_components = 6usize.min(nr.min(nc).saturating_sub(1)).max(1);
    let n_best = 3usize.min(n_components);
    let n_discard = 1usize;
    let n_sv = n_components + n_discard;

    let mat = DMatrix::from_row_slice(nr, nc, &dense);
    let svd_result = mat.svd(true, true);
    let (u_full, v_t_full) = match (svd_result.u, svd_result.v_t) {
        (Some(u), Some(vt)) => (u, vt),
        _ => {
            return Ok((
                (0..n_cells).map(|i| i % kr).collect(),
                (0..n_genes as u32).collect(),
                (0..n_genes as u32).collect(),
            ));
        }
    };

    let r = u_full.ncols();
    let n_keep = n_sv.min(r).saturating_sub(n_discard);
    if n_keep == 0 {
        return Ok((
            (0..n_cells).map(|i| i % kr).collect(),
            (0..n_genes as u32).collect(),
            (0..n_genes as u32).collect(),
        ));
    }

    // Extract U[:,1..n_sv] and V[:,1..n_sv] (transposed from V^T)
    let mut ut_vecs = Vec::with_capacity(n_keep); // each is a row = one singular vector across cells
    let mut vt_vecs = Vec::with_capacity(n_keep); // each is a row = one singular vector across genes
    for k in 0..n_keep {
        let sc = n_discard + k;
        let u_vec: Vec<f64> = (0..nr).map(|i| u_full[(i, sc)]).collect();
        let v_vec: Vec<f64> = (0..nc).map(|j| v_t_full[(sc, j)]).collect();
        ut_vecs.push(u_vec);
        vt_vecs.push(v_vec);
    }

    // --- Find n_best vectors: for each, fit 1-D k-means, measure piecewise residual ---
    fn piecewise_residual(vec: &[f64], n_clusters: usize, seed: u64) -> f64 {
        let n = vec.len();
        if n == 0 { return 0.0; }
        let k = n_clusters.min(n).max(1);
        let mut embed = Array2::<f64>::zeros((n, 1));
        for i in 0..n { embed[(i, 0)] = vec[i]; }
        let dataset = DatasetBase::from(embed);
        let model = KMeans::params_with_rng(k, Xoshiro256Plus::seed_from_u64(seed))
            .max_n_iterations(20)
            .fit(&dataset);
        match model {
            Ok(m) => {
                let labels = m.predict(&dataset);
                let centroids = m.centroids();
                labels.iter().enumerate()
                    .map(|(i, &l)| {
                        let diff = vec[i] - centroids[(l, 0)];
                        diff * diff
                    })
                    .sum::<f64>()
                    .sqrt()
            }
            Err(_) => f64::INFINITY,
        }
    }

    let seed_base = cluster_seed.unwrap_or(42);

    // Score u-vectors for row selection
    let mut u_scores: Vec<(usize, f64)> = (0..n_keep)
        .map(|k| (k, piecewise_residual(&ut_vecs[k], kr, seed_base.wrapping_add(k as u64))))
        .collect();
    u_scores.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    let best_u_indices: Vec<usize> = u_scores.iter().take(n_best).map(|&(i, _)| i).collect();

    // Score v-vectors for column selection
    let mut v_scores: Vec<(usize, f64)> = (0..n_keep)
        .map(|k| (k, piecewise_residual(&vt_vecs[k], kc, seed_base.wrapping_add(100 + k as u64))))
        .collect();
    v_scores.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    let best_v_indices: Vec<usize> = v_scores.iter().take(n_best).map(|&(i, _)| i).collect();

    // --- Row labels: project X @ V_best, k-means ---
    // For sampled cells, use dense matrix; for unsampled, project from sparse
    let n_proj_cols = best_v_indices.len();
    let mut row_projected = Array2::<f64>::zeros((n_cells, n_proj_cols));

    for ci in 0..n_cells {
        let point = &points[ci];
        if let Some(row) = data.outer_view(point.row_index) {
            for (g, &v) in row.iter() {
                if v != 0 {
                    if let Some(&lc) = col_local.get(&g) {
                        let val = (1.0 + v as f64).ln();
                        for (pi, &vi) in best_v_indices.iter().enumerate() {
                            row_projected[(ci, pi)] += val * vt_vecs[vi][lc];
                        }
                    }
                }
            }
        }
    }

    let row_dataset = DatasetBase::from(row_projected);
    let row_model = if let Some(seed) = cluster_seed {
        KMeans::params_with_rng(kr, Xoshiro256Plus::seed_from_u64(seed.wrapping_add(2001)))
            .max_n_iterations(30)
            .fit(&row_dataset)
            .map_err(|e| anyhow::anyhow!("Kluger row k-means failed: {}", e))?
    } else {
        KMeans::params(kr)
            .max_n_iterations(30)
            .fit(&row_dataset)
            .map_err(|e| anyhow::anyhow!("Kluger row k-means failed: {}", e))?
    };
    let cell_labels: Vec<usize> = row_model.predict(&row_dataset).to_vec();

    // --- Column labels: project X^T @ U_best, k-means ---
    // Build U_best matrix (nr × n_best_u), then X^T @ U_best gives (n_genes × n_best_u)
    let n_proj_rows = best_u_indices.len();
    let mut col_projected = Array2::<f64>::zeros((n_genes, n_proj_rows));

    // For active genes in the SVD, compute the projection directly from the dense submatrix
    for &global_gene in &active_cols {
        let lc = col_local[&global_gene];
        for (pi, &ui) in best_u_indices.iter().enumerate() {
            let mut dot = 0.0f64;
            for lr in 0..nr {
                dot += dense[lr * nc + lc] * ut_vecs[ui][lr];
            }
            col_projected[(global_gene, pi)] = dot;
        }
    }

    let col_dataset = DatasetBase::from(col_projected);
    let col_model = if let Some(seed) = cluster_seed {
        KMeans::params_with_rng(kc, Xoshiro256Plus::seed_from_u64(seed.wrapping_add(3001)))
            .max_n_iterations(30)
            .fit(&col_dataset)
            .map_err(|e| anyhow::anyhow!("Kluger col k-means failed: {}", e))?
    } else {
        KMeans::params(kc)
            .max_n_iterations(30)
            .fit(&col_dataset)
            .map_err(|e| anyhow::anyhow!("Kluger col k-means failed: {}", e))?
    };
    let gene_cluster_labels: Vec<usize> = col_model.predict(&col_dataset).to_vec();

    // --- Gene permutation: sort by (cluster_label, first projection coordinate) ---
    let mut gene_scores: Vec<(usize, usize, f64)> = (0..n_genes)
        .map(|g| {
            let label = gene_cluster_labels[g];
            let proj0 = if !best_v_indices.is_empty() && col_local.contains_key(&g) {
                let lc = col_local[&g];
                vt_vecs[best_v_indices[0]][lc]
            } else {
                f64::INFINITY
            };
            (g, label, proj0)
        })
        .collect();

    gene_scores.sort_unstable_by(|a, b| {
        a.1.cmp(&b.1)
            .then_with(|| a.2.total_cmp(&b.2))
            .then_with(|| a.0.cmp(&b.0))
    });

    let mut gene_old_to_new = vec![0u32; n_genes];
    let mut gene_new_to_old = Vec::with_capacity(n_genes);
    for (new_idx, &(old_idx, _, _)) in gene_scores.iter().enumerate() {
        gene_old_to_new[old_idx] = new_idx as u32;
        gene_new_to_old.push(old_idx as u32);
    }

    info!(
        "Kluger SpectralBiclustering: {} cells -> {} row clusters, {} genes -> {} col clusters ({} active in SVD, n_best={})",
        n_cells, kr, n_genes, kc, nc, n_best
    );

    Ok((cell_labels, gene_new_to_old, gene_old_to_new))
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
    cell_cluster_method: CellClusterMethodArg,
    spatial_graph: SpatialGraphParams,
    tensor_grid: TensorGridParams,
    cluster_seed: Option<u64>,
    gene_reorder_method: GeneReorderMethod,
    bicluster_joint: bool,
    gene_blocks: usize,
) -> anyhow::Result<CompressionResult> {
    if points.is_empty() {
        anyhow::bail!("Cannot encode empty point set");
    }

    let requested_clusters = quantizers_requested.max(1).min(points.len());
    let features = build_cell_features(points, data, 24)?;

    // --- Joint bicluster path: one operation produces both cell assignments and gene ordering ---
    let joint_result = if cell_cluster_method == CellClusterMethodArg::CellSqueeze {
        let n_col_clusters = requested_clusters.min(10).max(2);
        match joint_bicluster_kluger(points, data, requested_clusters, n_col_clusters, cluster_seed) {
            Ok(result) => Some(result),
            Err(e) => {
                warn!("Kluger SpectralBiclustering failed ({}), falling back to separate cell+gene", e);
                None
            }
        }
    } else if cell_cluster_method == CellClusterMethodArg::SvdJoint {
        match joint_svd_seriation(points, data, requested_clusters) {
            Ok(result) => Some(result),
            Err(e) => {
                warn!("Joint SVD seriation failed ({}), falling back to separate cell+gene", e);
                None
            }
        }
    } else if bicluster_joint && cell_cluster_method == CellClusterMethodArg::Kmeans {
        match joint_bicluster_svd(points, data, requested_clusters, cluster_seed) {
            Ok(result) => Some(result),
            Err(e) => {
                warn!("Joint bicluster SVD failed ({}), falling back to separate cell+gene", e);
                None
            }
        }
    } else if bicluster_joint {
        // Generic bicluster: cluster cells with the specified method, then derive gene
        // ordering from the cell cluster centroids via SVD.
        let cell_assignments = match cluster_cells(
            cell_cluster_method,
            &features,
            requested_clusters,
            points,
            spatial_graph,
            tensor_grid,
            cluster_seed,
        ) {
            Ok(a) => a,
            Err(e) => {
                warn!("Cell clustering for bicluster failed ({}), using round-robin", e);
                (0..points.len()).map(|i| i % requested_clusters).collect()
            }
        };
        match derive_gene_permutation_from_cell_clusters(points, data, &cell_assignments, requested_clusters) {
            Ok((n2o, o2n)) => Some((cell_assignments, n2o, o2n)),
            Err(e) => {
                warn!("Gene permutation from clusters failed ({}), falling back to independent", e);
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

    let (gene_order, gene_old_to_new) = if let Some((_, ref new_to_old, ref old_to_new)) = joint_result {
        info!("Using gene ordering from joint bicluster SVD");
        (new_to_old.clone(), old_to_new.clone())
    } else {
        match gene_reorder_method {
            GeneReorderMethod::Projection => compute_gene_permutation_from_row_stream(
                points,
                &quantized_data,
                &row_stream_point_indices,
            )?,
            GeneReorderMethod::Svd => compute_gene_permutation_svd(
                points,
                &quantized_data,
                &row_stream_point_indices,
            )?,
            GeneReorderMethod::Kmeans => compute_gene_permutation_kmeans(
                points,
                &quantized_data,
                &row_stream_point_indices,
                cluster_seed,
            )?,
        }
    };

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
            gene_block_ranges.iter().map(|r| r.len()).collect::<Vec<_>>()
        );
    }

    let mut encoded_blocks = Vec::new();
    let mut positions = Vec::new();
    let mut row_order = Vec::new();
    let mut cluster_sizes = Vec::new();
    let mut total_mst_bytes = 0usize;

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
                if let Some((row_block, dfs_order)) = encode_subarray_mst_with_metric(
                    cluster_points,
                    &quantized_data,
                    knn_metric,
                    mst_weight_mode,
                    Some(block_gene_o2n),
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

            let mut tile_blocks = Vec::with_capacity(effective_gene_blocks);
            let mut total_bytes = 0usize;
            let mut first_order: Option<Vec<u32>> = None;
            let mut any_row_fallback = false;
            let mut any_col_fallback = false;

            for (gb_idx, _gene_block_range) in gene_block_ranges.iter().enumerate() {
                let (encoded, local_order, bytes, row_fb, col_fb) =
                    encode_tile(&cluster_points, &block_gene_o2ns[gb_idx], cluster_idx)?;
                total_bytes += bytes;
                if first_order.is_none() {
                    first_order = Some(local_order);
                }
                any_row_fallback |= row_fb;
                any_col_fallback |= col_fb;
                tile_blocks.push(encoded);
            }

            let local_order = first_order.unwrap_or_else(|| (0..cluster_points.len() as u32).collect());
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
                }
            }
        }
        total_mst_bytes += bytes;
        cluster_sizes.push(cluster_ncells);
        positions.extend(cluster_positions);
        row_order.extend(cluster_row_order);
        encoded_blocks.extend(tile_blocks);
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
    /// If set, do not write compressed payload to disk.
    #[arg(long = "no-output", default_value_t = false)]
    no_output: bool,
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
    /// How to cluster cells (`kmeans`, `bicluster`, `bicluster-swapped`, `svd-kmeans`, …).
    #[arg(long = "cell-cluster-method", value_enum, default_value_t = CellClusterMethodArg::Kmeans)]
    cell_cluster_method: CellClusterMethodArg,
    /// Optional RNG seed for k-means-based clustering stages. Same seed => reproducible clustering.
    #[arg(long = "cluster-seed")]
    cluster_seed: Option<u64>,
    /// If set, run all cell-cluster methods and append one stats row per method.
    #[arg(long = "cell-cluster-method-sweep-all", visible_alias = "cluster-all", default_value_t = false)]
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
    /// True joint biclustering: one SVD on the full sparse cell×gene matrix produces
    /// both cell cluster assignments and gene ordering simultaneously.  Overrides
    /// `--cell-cluster-method` and `--gene-reorder-method` when set.
    #[arg(long = "bicluster", default_value_t = false)]
    bicluster_joint: bool,
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
    #[arg(long = "gene-blocks", default_value_t = 1)]
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
        .unwrap_or("output.bin.gz");
    let parent = base.parent().unwrap_or_else(|| Path::new(""));

    if file_name.ends_with(".bin.gz") {
        let stem = file_name.trim_end_matches(".bin.gz");
        parent.join(format!("{}_{}.bin.gz", stem, label))
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
                "Encoding mode: {} | quantizers={} bins={} sweep={:?} max_cluster_size={:?} cell_cluster_method={:?} knn_metric={:?} mst_weight={:?} index_codec={:?} sorted_index_codec={:?} full_row_fallback={:?} forest_cut_factor={:?} cluster_encoding={:?} column_template_count={} column_template_adaptive={} column_template_max={} row_template_adaptive={} row_template_max={}",
                if quantize_values { "lossy" } else { "lossless" },
                target_quantizers,
                quantizer_bins,
                sweep_counts,
                args.max_cluster_size,
                args.cell_cluster_method,
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

            // Each sweep config: (cell_method, gene_method, bicluster_joint, label)
            let mut sweep_configs: Vec<(CellClusterMethodArg, GeneReorderMethod, bool, String)> = Vec::new();

            if args.cell_cluster_method_sweep_all {
                for &cm in CellClusterMethodArg::value_variants() {
                    let label = cell_cluster_method_name(cm);
                    sweep_configs.push((cm, args.gene_reorder_method, false, label));
                }
                // Add bicluster variants: each cell method + gene ordering derived from clusters.
                // Skip inherently-joint methods (CellSqueeze, SvdJoint) — they already
                // produce joint gene ordering in the non-bicluster entries above.
                for &cm in CellClusterMethodArg::value_variants() {
                    if cm == CellClusterMethodArg::CellSqueeze
                        || cm == CellClusterMethodArg::SvdJoint
                    {
                        continue;
                    }
                    let label = format!("{}-bicluster", cell_cluster_method_name(cm));
                    sweep_configs.push((cm, args.gene_reorder_method, true, label));
                }
            } else if args.bicluster_joint {
                let label = format!("{}-bicluster", cell_cluster_method_name(args.cell_cluster_method));
                sweep_configs.push((
                    args.cell_cluster_method,
                    args.gene_reorder_method,
                    true,
                    label,
                ));
            } else {
                let label = cell_cluster_method_name(args.cell_cluster_method);
                sweep_configs.push((args.cell_cluster_method, args.gene_reorder_method, false, label));
            }

            for (cell_cluster_method, gene_reorder_method, bicluster_joint, config_label) in &sweep_configs {
                let is_joint = *bicluster_joint
                    || *cell_cluster_method == CellClusterMethodArg::CellSqueeze
                    || *cell_cluster_method == CellClusterMethodArg::SvdJoint;
                let gene_method_label = if is_joint {
                    config_label.clone()
                } else {
                    gene_reorder_method
                        .to_possible_value()
                        .map(|v| v.get_name().to_string())
                        .unwrap_or_else(|| format!("{:?}", gene_reorder_method).to_lowercase())
                };
                info!(
                    "Running build: cell={}, gene={}, bicluster={}",
                    config_label, gene_method_label, bicluster_joint
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
                        *cell_cluster_method,
                        spatial_graph,
                        tensor_grid,
                        args.cluster_seed,
                        *gene_reorder_method,
                        *bicluster_joint,
                        args.gene_blocks,
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

            let config = bincode::config::standard()
                .with_little_endian()
                .with_fixed_int_encoding();
            let actual_gzip_bytes = if args.no_output {
                // Keep size metrics available while skipping disk writes.
                let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
                bincode::encode_into_std_write(&selected.encoded_blocks, &mut encoder, config)?;
                bincode::encode_into_std_write(&selected.positions, &mut encoder, config)?;
                bincode::encode_into_std_write(&selected.row_order, &mut encoder, config)?;
                bincode::encode_into_std_write(&selected.gene_order, &mut encoder, config)?;
                encoder.finish()?.len()
            } else {
                let mut output = args
                    .output
                    .clone()
                    .unwrap_or_else(|| PathBuf::from("output.bin.gz"));
                if sweep_configs.len() > 1 {
                    output = output_path_with_label_suffix(&output, config_label);
                }
                let file = File::create(&output)?;
                let writer = BufWriter::new(file);
                let mut encoder = GzEncoder::new(writer, Compression::default());
                bincode::encode_into_std_write(&selected.encoded_blocks, &mut encoder, config)?;
                bincode::encode_into_std_write(&selected.positions, &mut encoder, config)?;
                bincode::encode_into_std_write(&selected.row_order, &mut encoder, config)?;
                bincode::encode_into_std_write(&selected.gene_order, &mut encoder, config)?;
                let _ = encoder.finish()?;
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
            let gene_order_bytes = selected.gene_order.len() * std::mem::size_of::<u32>();
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

    fn assert_is_full_permutation(values: &[u32], expected_len: usize, label: &str) {
        assert_eq!(
            values.len(),
            expected_len,
            "{} length mismatch: got {}, expected {}",
            label,
            values.len(),
            expected_len
        );
        let mut sorted = values.to_vec();
        sorted.sort_unstable();
        let expected: Vec<u32> = (0..expected_len as u32).collect();
        assert_eq!(
            sorted, expected,
            "{} is not a full [0, n) permutation",
            label
        );
    }

    fn run_tiled_roundtrip_with_encoding(cluster_encoding: ClusterEncodingArg) {
        let n_cells = 18usize;
        let n_genes = 15usize;
        let mut tri = TriMatI::<u16, usize>::new((n_cells, n_genes));
        for cell in 0..n_cells {
            let cell_group = cell / 6;
            let base_gene = cell_group * 5;
            for offset in 0..5 {
                let gene = (base_gene + offset) % n_genes;
                let value = ((cell + 3 * offset + 1) % 11 + 1) as u16;
                tri.add_triplet(cell, gene, value);
            }
            // Add one cross-group signal so each cell has shared support too.
            tri.add_triplet(cell, (cell * 7 + 2) % n_genes, ((cell % 5) + 2) as u16);
        }
        let csr = tri.to_csr::<usize>();
        let points: Vec<Point> = (0..n_cells)
            .map(|i| Point::new((i % 6) as f64, (i / 6) as f64, i))
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
            cluster_encoding,
            0,
            false,
            32,
            false,
            16,
            CellClusterMethodArg::Kmeans,
            SpatialGraphParams::default(),
            TensorGridParams::default(),
            Some(7),
            GeneReorderMethod::Svd,
            false,
            3,
        )
        .unwrap_or_else(|e| panic!("tiled {:?} compression failed: {}", cluster_encoding, e));

        assert_eq!(result.cluster_sizes.iter().sum::<usize>(), n_cells);
        assert_eq!(result.positions.len(), n_cells);
        assert_is_full_permutation(&result.row_order, n_cells, "row_order");
        assert_is_full_permutation(&result.gene_order, n_genes, "gene_order");

        let reconstructed = reconstruct_csr_from_clustered_payload(
            &result.encoded_blocks,
            &result.row_order,
            &result.gene_order,
            n_cells,
            n_genes,
        )
        .expect("reconstruct tiled payload");
        compare_csr_exact(&csr, &reconstructed).expect("tiled payload must round-trip");
    }

    #[test]
    fn row_mst_gene_tiling_roundtrip_preserves_original_indices() {
        run_tiled_roundtrip_with_encoding(ClusterEncodingArg::Row);
    }

    #[test]
    fn column_gene_tiling_roundtrip_preserves_original_indices() {
        run_tiled_roundtrip_with_encoding(ClusterEncodingArg::Column);
    }

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
            CellClusterMethodArg::Kmeans,
            SpatialGraphParams::default(),
            TensorGridParams::default(),
            None,
            GeneReorderMethod::Projection,
            false,
            1,
        )
        .expect("lossy compression should succeed");

        assert_eq!(result.cluster_sizes.iter().sum::<usize>(), n_cells);
        assert!(result.cluster_sizes.iter().all(|&s| s <= max_cluster_size));
    }

    /// Disjoint nonzero patterns across `gene % 24` buckets: four blocks of cells use genes
    /// `[0..6]`, `[6..12]`, `[12..18]`, `[18..24]` only. Spectral biclustering should recover
    /// cleaner blocks than raw k-means on this toy, yielding a lower gzip-size proxy.
    #[test]
    fn spectral_bicluster_beats_kmeans_on_disjoint_bucket_blocks() {
        let n_cells = 32usize;
        let n_genes = 24usize;
        let mut tri = TriMatI::<u16, usize>::new((n_cells, n_genes));
        for cell in 0..n_cells {
            let block = cell / 8;
            let g0 = block * 6;
            for k in 0..6 {
                tri.add_triplet(cell, g0 + k, 80u16);
            }
        }
        let csr = tri.to_csr::<usize>();
        let points: Vec<Point> = (0..n_cells)
            .map(|i| Point::new(i as f64, 0.0, i))
            .collect();

        let run = |method: CellClusterMethodArg| {
            run_clustered_compression(
                &points,
                &csr,
                4,
                16,
                false,
                true,
                None,
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
                method,
                SpatialGraphParams::default(),
                TensorGridParams::default(),
                None,
                GeneReorderMethod::Projection,
                false,
                1,
            )
            .expect("compression")
        };

        let kmeans_gzip = run(CellClusterMethodArg::Kmeans).gzip_bytes_estimate;
        let bicluster_gzip = run(CellClusterMethodArg::Bicluster).gzip_bytes_estimate;

        assert!(
            bicluster_gzip <= kmeans_gzip,
            "expected bicluster gzip {} <= k-means gzip {} on disjoint-bucket toy data",
            bicluster_gzip,
            kmeans_gzip
        );
    }

    #[test]
    fn spectral_cocluster_runs_on_toy_matrix() {
        let n_cells = 8usize;
        let n_genes = 6usize;
        let mut tri = TriMatI::<u16, usize>::new((n_cells, n_genes));
        for cell in 0..n_cells {
            tri.add_triplet(cell, cell % n_genes, 10u16);
        }
        let csr = tri.to_csr::<usize>();
        let points: Vec<Point> = (0..n_cells)
            .map(|i| Point::new(i as f64, 0.0, i))
            .collect();
        let result = run_clustered_compression(
            &points,
            &csr,
            2,
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
            ClusterEncodingArg::Hybrid,
            0,
            false,
            32,
            false,
            16,
            CellClusterMethodArg::SpectralCocluster,
            SpatialGraphParams::default(),
            TensorGridParams::default(),
            None,
            GeneReorderMethod::Projection,
            false,
                    1,
        )
        .expect("spectral cocluster path");
        assert_eq!(result.cluster_sizes.iter().sum::<usize>(), n_cells);
    }

    #[test]
    fn svd_kmeans_runs_on_toy_matrix() {
        let n_cells = 12usize;
        let n_genes = 8usize;
        let mut tri = TriMatI::<u16, usize>::new((n_cells, n_genes));
        for cell in 0..n_cells {
            tri.add_triplet(cell, cell % n_genes, 12u16);
        }
        let csr = tri.to_csr::<usize>();
        let points: Vec<Point> = (0..n_cells)
            .map(|i| Point::new(i as f64, 0.0, i))
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
            ClusterEncodingArg::Hybrid,
            0,
            false,
            32,
            false,
            16,
            CellClusterMethodArg::SvdKmeans,
            SpatialGraphParams::default(),
            TensorGridParams::default(),
            None,
            GeneReorderMethod::Projection,
            false,
                    1,
        )
        .expect("svd-kmeans path");
        assert_eq!(result.cluster_sizes.iter().sum::<usize>(), n_cells);
    }

    #[test]
    fn binary_l0_kmeans_runs_on_toy_matrix() {
        let n_cells = 12usize;
        let n_genes = 8usize;
        let mut tri = TriMatI::<u16, usize>::new((n_cells, n_genes));
        for cell in 0..n_cells {
            tri.add_triplet(cell, cell % n_genes, 9u16);
        }
        let csr = tri.to_csr::<usize>();
        let points: Vec<Point> = (0..n_cells)
            .map(|i| Point::new(i as f64, 0.0, i))
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
            ClusterEncodingArg::Hybrid,
            0,
            false,
            32,
            false,
            16,
            CellClusterMethodArg::BinaryL0Kmeans,
            SpatialGraphParams::default(),
            TensorGridParams::default(),
            None,
            GeneReorderMethod::Projection,
            false,
                    1,
        )
        .expect("binary-l0-kmeans path");
        assert_eq!(result.cluster_sizes.iter().sum::<usize>(), n_cells);
    }

    #[test]
    fn bicluster_swapped_runs_on_toy_matrix() {
        let n_cells = 14usize;
        let n_genes = 8usize;
        let mut tri = TriMatI::<u16, usize>::new((n_cells, n_genes));
        for cell in 0..n_cells {
            tri.add_triplet(cell, cell % n_genes, 11u16);
        }
        let csr = tri.to_csr::<usize>();
        let points: Vec<Point> = (0..n_cells)
            .map(|i| Point::new(i as f64, 0.0, i))
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
            ClusterEncodingArg::Hybrid,
            0,
            false,
            32,
            false,
            16,
            CellClusterMethodArg::BiclusterSwapped,
            SpatialGraphParams::default(),
            TensorGridParams::default(),
            None,
            GeneReorderMethod::Projection,
            false,
                    1,
        )
        .expect("bicluster-swapped path");
        assert_eq!(result.cluster_sizes.iter().sum::<usize>(), n_cells);
    }

    #[test]
    fn bicluster_l0_runs_on_toy_matrix() {
        let n_cells = 14usize;
        let n_genes = 8usize;
        let mut tri = TriMatI::<u16, usize>::new((n_cells, n_genes));
        for cell in 0..n_cells {
            tri.add_triplet(cell, cell % n_genes, 7u16);
        }
        let csr = tri.to_csr::<usize>();
        let points: Vec<Point> = (0..n_cells)
            .map(|i| Point::new(i as f64, 0.0, i))
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
            ClusterEncodingArg::Hybrid,
            0,
            false,
            32,
            false,
            16,
            CellClusterMethodArg::BiclusterL0,
            SpatialGraphParams::default(),
            TensorGridParams::default(),
            None,
            GeneReorderMethod::Projection,
            false,
                    1,
        )
        .expect("bicluster-l0 path");
        assert_eq!(result.cluster_sizes.iter().sum::<usize>(), n_cells);
    }

    #[test]
    fn fabia_runs_on_toy_matrix() {
        let n_cells = 8usize;
        let n_genes = 6usize;
        let mut tri = TriMatI::<u16, usize>::new((n_cells, n_genes));
        for cell in 0..n_cells {
            tri.add_triplet(cell, cell % n_genes, 10u16);
        }
        let csr = tri.to_csr::<usize>();
        let points: Vec<Point> = (0..n_cells)
            .map(|i| Point::new(i as f64, 0.0, i))
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
            ClusterEncodingArg::Hybrid,
            0,
            false,
            32,
            false,
            16,
            CellClusterMethodArg::Fabia,
            SpatialGraphParams::default(),
            TensorGridParams::default(),
            None,
            GeneReorderMethod::Projection,
            false,
                    1,
        )
        .expect("fabia path");
        assert_eq!(result.cluster_sizes.iter().sum::<usize>(), n_cells);
    }

    #[test]
    fn spatial_graph_runs_on_toy_matrix() {
        let n_cells = 16usize;
        let n_genes = 8usize;
        let mut tri = TriMatI::<u16, usize>::new((n_cells, n_genes));
        for cell in 0..n_cells {
            tri.add_triplet(cell, cell % n_genes, 20u16);
        }
        let csr = tri.to_csr::<usize>();
        let points: Vec<Point> = (0..n_cells)
            .map(|i| Point::new((i % 4) as f64, (i / 4) as f64, i))
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
            ClusterEncodingArg::Hybrid,
            0,
            false,
            32,
            false,
            16,
            CellClusterMethodArg::SpatialGraph,
            SpatialGraphParams::default(),
            TensorGridParams::default(),
            None,
            GeneReorderMethod::Projection,
            false,
                    1,
        )
        .expect("spatial-graph path");
        assert_eq!(result.cluster_sizes.iter().sum::<usize>(), n_cells);
    }

    #[test]
    fn tensor_grid_rect_and_hex_run_on_toy_matrix() {
        let n_cells = 20usize;
        let n_genes = 8usize;
        let mut tri = TriMatI::<u16, usize>::new((n_cells, n_genes));
        for cell in 0..n_cells {
            tri.add_triplet(cell, cell % n_genes, 15u16);
        }
        let csr = tri.to_csr::<usize>();
        let points: Vec<Point> = (0..n_cells)
            .map(|i| Point::new((i % 5) as f64 * 10.0, (i / 5) as f64 * 10.0, i))
            .collect();

        for (disc, label) in [
            (TensorGridDisc::Rect, "rect"),
            (TensorGridDisc::Hex, "hex"),
        ] {
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
                ClusterEncodingArg::Hybrid,
                0,
                false,
                32,
                false,
                16,
                CellClusterMethodArg::TensorGrid,
                SpatialGraphParams::default(),
                TensorGridParams {
                    nx: 4,
                    ny: 4,
                    tile_weight: 0.5,
                    disc,
                    onehot_max_tiles: 64,
                },
                None,
                GeneReorderMethod::Projection,
                false,
                            1,
            )
            .unwrap_or_else(|e| panic!("tensor-grid {}: {}", label, e));
            assert_eq!(
                result.cluster_sizes.iter().sum::<usize>(),
                n_cells,
                "tensor-grid {}",
                label
            );
        }
    }

    #[test]
    fn bicluster_joint_runs_on_toy_matrix() {
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
            ClusterEncodingArg::Hybrid,
            0,
            false,
            32,
            false,
            16,
            CellClusterMethodArg::Kmeans,
            SpatialGraphParams::default(),
            TensorGridParams::default(),
            Some(42),
            GeneReorderMethod::Projection,
            true,
                    1,
        )
        .expect("bicluster-joint path");
        assert_eq!(result.cluster_sizes.iter().sum::<usize>(), n_cells);
    }

    #[test]
    fn cell_squeeze_kluger_runs_on_toy_matrix() {
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
            ClusterEncodingArg::Hybrid,
            0,
            false,
            32,
            false,
            16,
            CellClusterMethodArg::CellSqueeze,
            SpatialGraphParams::default(),
            TensorGridParams::default(),
            Some(42),
            GeneReorderMethod::Projection,
            false,
                    1,
        )
        .expect("cell-squeeze/Kluger path");
        assert_eq!(result.cluster_sizes.iter().sum::<usize>(), n_cells);
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
        )
        .expect("svd-joint seriation path");
        assert_eq!(result.cluster_sizes.iter().sum::<usize>(), n_cells);
    }

    #[test]
    fn generic_bicluster_wrapper_runs_for_multiple_methods() {
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

        for method in [
            CellClusterMethodArg::BiclusterL0,
            CellClusterMethodArg::BiclusterSwapped,
            CellClusterMethodArg::SpectralCocluster,
            CellClusterMethodArg::Fabia,
        ] {
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
                ClusterEncodingArg::Hybrid,
                0,
                false,
                32,
                false,
                16,
                method,
                SpatialGraphParams::default(),
                TensorGridParams::default(),
                Some(42),
                GeneReorderMethod::Projection,
                true,
                1,
            )
            .unwrap_or_else(|e| panic!("{:?}-bicluster failed: {}", method, e));
            assert_eq!(result.cluster_sizes.iter().sum::<usize>(), n_cells);
        }
    }
}
