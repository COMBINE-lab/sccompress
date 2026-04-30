use nalgebra::DMatrix;
use rand_xoshiro::rand_core::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256Plus;
use sprs::CsMat;
use std::collections::HashMap;
use tracing::{info, warn};

use crate::mst_codec::Point;

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

fn orthonormalize_columns(y: &DMatrix<f64>) -> DMatrix<f64> {
    let nr = y.nrows();
    let l = y.ncols();
    let mut q = DMatrix::zeros(nr, l);
    for j in 0..l {
        let mut v: Vec<f64> = (0..nr).map(|i| y[(i, j)]).collect();
        for prev in 0..j {
            let mut dot = 0.0;
            for i in 0..nr {
                dot += v[i] * q[(i, prev)];
            }
            for i in 0..nr {
                v[i] -= dot * q[(i, prev)];
            }
        }
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-14 {
            for i in 0..nr {
                q[(i, j)] = v[i] / norm;
            }
        }
    }
    q
}

struct JointSvdFactors {
    u: DMatrix<f64>,
    singular_values: Vec<f64>,
    v_t: DMatrix<f64>,
}

fn randomized_joint_svd_top_k(a: &DMatrix<f64>, k: usize, seed: u64) -> Option<JointSvdFactors> {
    let nr = a.nrows();
    let nc = a.ncols();
    if k == 0 || nr == 0 || nc == 0 {
        return None;
    }
    let oversample = 4usize;
    let l = (k + oversample).min(nr).min(nc).max(k.min(nr).min(nc));
    let mut rng = Xoshiro256Plus::seed_from_u64(seed);
    let omega = DMatrix::from_fn(nc, l, |_, _| {
        let u = rng.next_u64();
        (u as f64 / (u64::MAX as f64)) * 2.0 - 1.0
    });

    // One power iteration: A * (A^T * (A * Omega)).
    let y0 = a * &omega;
    let at_y0 = a.transpose() * &y0;
    let y = a * &at_y0;
    let q = orthonormalize_columns(&y);
    let w = q.transpose() * a;
    let svd = w.svd(true, true);
    let u_small = svd.u?;
    let v_t = svd.v_t?;
    let u = q * u_small;
    let nkeep = k
        .min(u.ncols())
        .min(v_t.nrows())
        .min(svd.singular_values.len());
    if nkeep == 0 {
        return None;
    }
    Some(JointSvdFactors {
        u: u.columns(0, nkeep).into_owned(),
        singular_values: svd.singular_values.rows(0, nkeep).iter().copied().collect(),
        v_t: v_t.rows(0, nkeep).into_owned(),
    })
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

pub(crate) fn group_points_by_cluster(
    assignments: &[usize],
    num_clusters: usize,
) -> Vec<Vec<usize>> {
    let mut clusters = vec![Vec::new(); num_clusters];
    for (point_idx, &cluster_id) in assignments.iter().enumerate() {
        if cluster_id < num_clusters {
            clusters[cluster_id].push(point_idx);
        }
    }
    clusters.retain(|cluster| !cluster.is_empty());
    clusters
}

pub(crate) fn split_oversized_clusters(
    initial_clusters: Vec<Vec<usize>>,
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
        info!(
            "Splitting oversized cluster deterministically: size={} max_cluster_size={}",
            cluster.len(),
            max_cluster_size
        );
        let mut offset = 0usize;
        while offset < cluster.len() {
            let end = (offset + max_cluster_size).min(cluster.len());
            final_clusters.push(cluster[offset..end].to_vec());
            offset = end;
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

/// Joint SVD on raw counts, equal-sized cuts along the lexicographic seriation of the cell and geneembedding.
pub(crate) fn joint_svd_seriation(
    points: &[Point],
    data: &CsMat<u16>,
    num_clusters: usize,
    cluster_seed: Option<u64>,
    joint_svd_fast: bool,
) -> anyhow::Result<(Vec<usize>, Vec<u32>, Vec<u32>)> {
    let n_cells = points.len();
    let n_genes = data.cols();
    if n_cells == 0 {
        return Ok((
            Vec::new(),
            (0..n_genes as u32).collect(),
            (0..n_genes as u32).collect(),
        ));
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
        return Ok((
            (0..n_cells).map(|i| i % k).collect(),
            (0..n_genes as u32).collect(),
            (0..n_genes as u32).collect(),
        ));
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
            let mut sample: Vec<usize> = ranked_rows
                .into_iter()
                .take(max_svd_rows)
                .map(|(_, row)| row)
                .collect();
            sample.sort_unstable();
            sample
        } else {
            let step = n_cells as f64 / max_svd_rows as f64;
            (0..max_svd_rows)
                .map(|i| (i as f64 * step) as usize)
                .collect()
        }
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
    for &(ci, gene_idx, value) in &triplets {
        if let Some(&local_row) = sampled_set.get(&ci) {
            if let Some(&local_col) = col_local.get(&gene_idx) {
                dense[local_row * nc + local_col] = value as f64;
            }
        }
    }
    let mat = DMatrix::from_row_slice(nr, nc, &dense);
    let svd_seed = cluster_seed
        .unwrap_or(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0x517C_C1B7_2722_0A95);
    let target_rank = k.min(nr).min(nc).max(1);
    let (u, sigma, v_t, svd_mode) = if joint_svd_fast {
        match randomized_joint_svd_top_k(&mat, target_rank, svd_seed) {
            Some(factors)
                if factors.u.iter().all(|x| x.is_finite())
                    && factors.v_t.iter().all(|x| x.is_finite()) =>
            {
                (
                    factors.u,
                    factors.singular_values,
                    factors.v_t,
                    "randomized",
                )
            }
            _ => {
                warn!("Fast joint SVD failed or produced non-finite values; using exact SVD");
                let svd = mat.svd(true, true);
                match (svd.u, svd.v_t) {
                    (Some(u), Some(vt)) => (
                        u,
                        svd.singular_values.iter().copied().collect(),
                        vt,
                        "exact-fallback",
                    ),
                    _ => {
                        return Ok((
                            (0..n_cells).map(|i| i % k).collect(),
                            (0..n_genes as u32).collect(),
                            (0..n_genes as u32).collect(),
                        ))
                    }
                }
            }
        }
    } else {
        let svd = mat.svd(true, true);
        match (svd.u, svd.v_t) {
            (Some(u), Some(vt)) => (
                u,
                svd.singular_values.iter().copied().collect(),
                vt,
                "exact",
            ),
            _ => {
                return Ok((
                    (0..n_cells).map(|i| i % k).collect(),
                    (0..n_genes as u32).collect(),
                    (0..n_genes as u32).collect(),
                ))
            }
        }
    };
    let n_comp = k.min(u.ncols()).max(1);
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
                    let sig_inv = if sigma[j] > 1e-12 {
                        1.0 / sigma[j]
                    } else {
                        0.0
                    };
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
        "Joint SVD seriation: {} cells -> {} clusters ({} raw-count SVD, rank={}, lexicographic seriation + equal cuts), {} genes reordered ({} active in SVD)",
        n_cells, k, svd_mode, n_comp, n_genes, nc
    );
    Ok((cell_labels, gene_new_to_old, gene_old_to_new))
}
