use clap::ValueEnum;
use linfa::prelude::{Fit, Predict};
use linfa::DatasetBase;
use linfa_clustering::KMeans;
use nalgebra::DMatrix;
use ndarray::Array2;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use rayon::prelude::*;
use sprs::CsMat;
use std::collections::HashMap;
use tracing::{info, warn};

use crate::mst_codec::Point;

/// How to partition cells into encoder clusters (before row-MST / column payloads).
#[derive(Debug, Clone, Copy, ValueEnum, Default, PartialEq, Eq)]
pub(crate) enum CellClusterMethodArg {
    #[default]
    Kmeans,
    SvdJoint,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SpatialGraphParams {
    pub(crate) spatial_knn: usize,
    pub(crate) expr_knn: usize,
    pub(crate) blend: f64,
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

#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub(crate) enum TensorGridDisc {
    #[default]
    Rect,
    Hex,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct TensorGridParams {
    pub(crate) nx: usize,
    pub(crate) ny: usize,
    pub(crate) tile_weight: f64,
    pub(crate) disc: TensorGridDisc,
    pub(crate) onehot_max_tiles: usize,
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

pub(crate) fn build_cell_features(
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

fn lexicographic_order(cell_emb: &[Vec<f64>]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..cell_emb.len()).collect();
    order.sort_unstable_by(|&a, &b| {
        for (ca, cb) in cell_emb[a].iter().zip(cell_emb[b].iter()) {
            match ca.total_cmp(cb) {
                std::cmp::Ordering::Equal => continue,
                ord => return ord,
            }
        }
        a.cmp(&b)
    });
    order
}

fn labels_from_boundaries(order: &[usize], boundaries: &[usize], k: usize) -> Vec<usize> {
    let n = order.len();
    let mut labels = vec![0usize; n];
    let mut sorted_bounds = boundaries.to_vec();
    sorted_bounds.sort_unstable();
    sorted_bounds.dedup();
    let mut starts = Vec::with_capacity(k + 1);
    starts.push(0usize);
    starts.extend(sorted_bounds.into_iter().filter(|&b| b > 0 && b < n));
    starts.push(n);
    for (cluster_id, win) in starts.windows(2).enumerate() {
        for pos in win[0]..win[1] {
            labels[order[pos]] = cluster_id.min(k.saturating_sub(1));
        }
    }
    labels
}

fn equal_cut_labels(order: &[usize], k: usize) -> Vec<usize> {
    let n = order.len();
    let block_size = n / k;
    let remainder = n % k;
    let mut boundaries = Vec::with_capacity(k.saturating_sub(1));
    let mut offset = 0usize;
    for cluster_id in 0..k {
        let sz = block_size + if cluster_id < remainder { 1 } else { 0 };
        offset += sz;
        if cluster_id + 1 < k {
            boundaries.push(offset);
        }
    }
    labels_from_boundaries(order, &boundaries, k)
}

pub(crate) fn cluster_cells(
    method: CellClusterMethodArg,
    features: &Array2<f64>,
    num_clusters: usize,
    points: &[Point],
    spatial: SpatialGraphParams,
    tensor_grid: TensorGridParams,
    cluster_seed: Option<u64>,
) -> anyhow::Result<Vec<usize>> {
    let _ = points;
    let _ = spatial;
    let _ = tensor_grid;
    match method {
        CellClusterMethodArg::Kmeans => cluster_cells_with_kmeans(features, num_clusters, cluster_seed),
        CellClusterMethodArg::SvdJoint => cluster_cells_with_kmeans(features, num_clusters, cluster_seed),
    }
}

pub(crate) fn group_points_by_cluster(assignments: &[usize], num_clusters: usize) -> Vec<Vec<usize>> {
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

pub(crate) fn split_oversized_clusters(
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
            cluster.len(), split_k, max_cluster_size
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
                warn!("Sub-clustering failed ({}). Falling back to deterministic chunk split.", err);
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
            warn!("Sub-clustering made no progress (size={}): using deterministic chunk split.", cluster.len());
            let mut offset = 0usize;
            while offset < cluster.len() {
                let end = (offset + max_cluster_size).min(cluster.len());
                final_clusters.push(cluster[offset..end].to_vec());
                offset = end;
            }
            continue;
        }
        for local_cluster in local_clusters {
            let mapped: Vec<usize> = local_cluster.into_iter().map(|local_idx| cluster[local_idx]).collect();
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

pub(crate) fn projection_weight(col: usize, seed: u64) -> f64 {
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

pub(crate) fn reorder_rows_within_clusters(
    clusters: &[Vec<usize>],
    features: &Array2<f64>,
) -> Vec<Vec<usize>> {
    let ncols = features.ncols();
    let w1: Vec<f64> = (0..ncols).map(|c| projection_weight(c, 0x1234_5678_9ABC_DEF0)).collect();
    let w2: Vec<f64> = (0..ncols).map(|c| projection_weight(c, 0x0FED_CBA9_8765_4321)).collect();
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

/// Joint SVD on raw counts, equal-sized cuts along the lexicographic seriation of the cell and geneembedding.
pub(crate) fn joint_svd_seriation(
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
    let mut col_nnz = vec![0u64; n_genes];
    let mut triplets: Vec<(usize, usize, u16)> = Vec::new();
    for (ci, point) in points.iter().enumerate() {
        if let Some(row) = data.outer_view(point.row_index) {
            for (gene_idx, &value) in row.iter() {
                if value != 0 {
                    col_nnz[gene_idx] += 1;
                    triplets.push((ci, gene_idx, value));
                }
            }
        }
    }
    let active_genes: Vec<usize> = (0..n_genes).filter(|&g| col_nnz[g] > 0).collect();
    if active_genes.len() <= 2 {
        return Ok(((0..n_cells).map(|i| i % k).collect(), (0..n_genes as u32).collect(), (0..n_genes as u32).collect()));
    }
    let max_svd_rows = 4000usize;
    let max_svd_cols = 2000usize;
    let sampled_rows: Vec<usize> = if n_cells > max_svd_rows {
        if let Some(seed) = cluster_seed {
            let mut ranked_rows: Vec<(u64, usize)> = (0..n_cells)
                .map(|row| {
                    let mut x = (row as u64) ^ seed ^ 0x9E37_79B9_7F4A_7C15;
                    x ^= x >> 30;
                    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
                    x ^= x >> 27;
                    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
                    x ^= x >> 31;
                    (x, row)
                })
                .collect();
            ranked_rows.sort_unstable();
            let mut sample: Vec<usize> = ranked_rows.into_iter().take(max_svd_rows).map(|(_, row)| row).collect();
            sample.sort_unstable();
            sample
        } else {
            let step = n_cells as f64 / max_svd_rows as f64;
            (0..max_svd_rows).map(|i| (i as f64 * step) as usize).collect()
        }
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
    for &(ci, gene_idx, value) in &triplets {
        if let Some(&local_row) = sampled_set.get(&ci) {
            if let Some(&local_col) = col_local.get(&gene_idx) {
                dense[local_row * nc + local_col] = value as f64;
            }
        }
    }
    let mat = DMatrix::from_row_slice(nr, nc, &dense);
    let svd = mat.svd(true, true);
    let (u, v_t) = match (svd.u, svd.v_t) {
        (Some(u), Some(vt)) => (u, vt),
        _ => return Ok(((0..n_cells).map(|i| i % k).collect(), (0..n_genes as u32).collect(), (0..n_genes as u32).collect())),
    };
    let n_comp = k.min(u.ncols()).max(1);
    let sigma = &svd.singular_values;
    let mut cell_emb: Vec<Vec<f64>> = vec![vec![0.0f64; n_comp]; n_cells];
    for ci in 0..n_cells {
        if let Some(&lr) = sampled_set.get(&ci) {
            for j in 0..n_comp {
                cell_emb[ci][j] = u[(lr, j)];
            }
        } else {
            let point = &points[ci];
            if let Some(row) = data.outer_view(point.row_index) {
                for j in 0..n_comp {
                    let sig_inv = if sigma[j] > 1e-12 { 1.0 / sigma[j] } else { 0.0 };
                    let mut dot = 0.0f64;
                    for (g, &v) in row.iter() {
                        if v != 0 {
                            if let Some(&lc) = col_local.get(&g) {
                                let w = v as f64;
                                dot += w * v_t[(j, lc)];
                            }
                        }
                    }
                    cell_emb[ci][j] = dot * sig_inv;
                }
            }
        }
    }
    let order = lexicographic_order(&cell_emb);
    let cell_labels = equal_cut_labels(&order, k);
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
        "Joint SVD seriation: {} cells -> {} clusters (raw-count SVD, lexicographic seriation + equal cuts), {} genes reordered ({} active in SVD)",
        n_cells, k, n_genes, nc
    );
    Ok((cell_labels, gene_new_to_old, gene_old_to_new))
}
