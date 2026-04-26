use anyhow::Result;
use clap::{ArgAction, Parser, ValueEnum};
use csv::ReaderBuilder;
use hdf5::types::FixedAscii;
use hdf5::File as Hdf5File;
use hnsw_rs::prelude::*;
use ndarray::{Array1, Array2, Axis};
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use rand::seq::IndexedRandom;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use tracing::info;
// use efficient_pca::PCA;
use hypors::common::TailType;
use hypors::mann_whitney;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input file path (CSV or HDF5)
    #[arg(short, long)]
    input: PathBuf,

    /// Input format
    #[arg(short, long, value_enum)]
    format: InputFormat,

    /// How to find neighbors: spatial (XY L2) or expression (L0 binary diff)
    #[arg(long, value_enum, default_value = "spatial")]
    neighbor_by: NeighborBy,

    /// Positions CSV file (required for HDF5)
    #[arg(short, long)]
    pos: Option<PathBuf>,

    /// Number of neighbors for analysis
    #[arg(short, default_value_t = 8)]
    k: usize,

    /// Number of permutations for anchor test
    #[arg(long, default_value_t = 100)]
    n_perm: usize,

    /// Column index for X coordinate in CSV (0-based)
    #[arg(long, default_value_t = 1)]
    x_col: usize,

    /// Column index for Y coordinate in CSV (0-based)
    #[arg(long, default_value_t = 2)]
    y_col: usize,

    /// Column index where gene expression starts in CSV (0-based)
    #[arg(long, default_value_t = 3)]
    gene_start: usize,

    /// Number of PCA components (0 = no PCA tests, only raw)
    #[arg(long, default_value_t = 0)]
    pca_dims: usize,

    /// Number of top variable genes to keep for PCA (0 = use all non-zero-variance genes)
    #[arg(long, default_value_t = 2000)]
    top_genes: usize,

    /// Output CSV file path
    #[arg(short, long, default_value = "statistics.csv")]
    output: PathBuf,

    // Accept an explicit boolean value, e.g. `--pca=false` or `--pca false`.
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    pca: bool,

    // test type: "permutation" or "mann-whitney"
    #[arg(long, value_enum, default_value_t = TestType::MeanWilcoxon)]
    test_type: TestType,

    /// Optional platform (Visium, Xenium, or single-cell). In single-cell mode, positions file is not required.
    #[arg(short = 'P', long = "platform", value_enum)]
    platform: Option<Platform>,

    /// After building neighbors, print cell 0's first N neighbors: matrix index, xy L2, expression L0 (omit to skip)
    #[arg(long, value_name = "N")]
    neighbor_preview: Option<usize>,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
enum TestType {
    Permutation,
    MeanWilcoxon,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
enum InputFormat {
    Csv,
    H5,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
enum Platform {
    Visium,
    Xenium,
    /// Single-cell matrices without spatial coordinates
    SingleCell,
}

/// How to define the k nearest neighbors
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
enum NeighborBy {
    /// KNN by L2 distance on XY coordinates
    Spatial,
    /// KNN by L2 on XY, test only L0 on expression
    SpatialL0,
    /// KNN by L2 distance on expression
    Expression,
    /// KNN by L0 binary diff (sparsity pattern) on expression
    ExpressionL0,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
enum Metric {
    Cosine,
    L2,
    L0,
}

/// Compute distances from an anchor point to each row of a matrix
fn compute_distances(anchor: &[f64], matrix: &Array2<f64>, metric: Metric) -> Vec<f64> {
    matrix
        .axis_iter(Axis(0))
        .map(|row| {
            let b = row.as_slice().unwrap();
            match metric {
                Metric::Cosine => {
                    let dot: f64 = anchor.iter().zip(b).map(|(a, b)| a * b).sum();
                    let norm_a: f64 = anchor.iter().map(|a| a * a).sum::<f64>().sqrt();
                    let norm_b: f64 = b.iter().map(|b| b * b).sum::<f64>().sqrt();
                    if norm_a == 0.0 || norm_b == 0.0 {
                        1.0
                    } else {
                        1.0 - (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
                    }
                }
                Metric::L2 => anchor
                    .iter()
                    .zip(b)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt(),
                Metric::L0 => anchor
                    .iter()
                    .zip(b)
                    .filter(|(&a, &b)| (a > 0.0) != (b > 0.0))
                    .count() as f64,
            }
        })
        .collect()
}

/// Print neighbor indices for `anchor`, up to `max_show`, with spatial xy L2 and raw expression L0 vs anchor.
fn print_neighbor_preview(
    anchor: usize,
    max_show: usize,
    coords: &Array2<f64>,
    data_raw: &Array2<f64>,
    neighbors: &[Vec<usize>],
) {
    if anchor >= neighbors.len() {
        println!("neighbor-preview: anchor {} has no neighbor row", anchor);
        return;
    }
    let nbrs = &neighbors[anchor];
    if nbrs.is_empty() {
        println!("neighbor-preview: cell {} has zero neighbors", anchor);
        return;
    }
    let take = max_show.min(nbrs.len());
    let anchor_c = coords.row(anchor).to_vec();
    let anchor_e = data_raw.row(anchor).to_vec();
    println!(
        "neighbor-preview: cell {} — first {} of {} neighbor indices | xy_L2 = Euclidean on coords, expr_L0 = binary mismatch count",
        anchor,
        take,
        nbrs.len()
    );
    println!("{:<8} {:>14} {:>18}", "nbr_idx", "xy_L2", "expr_L0");
    for &nid in nbrs.iter().take(take) {
        let cmat = coords.select(Axis(0), &[nid]);
        let xy_l2 = compute_distances(&anchor_c, &cmat, Metric::L2)[0];
        let emat = data_raw.select(Axis(0), &[nid]);
        let l0 = compute_distances(&anchor_e, &emat, Metric::L0)[0];
        println!("{:<8} {:>14.6} {:>18.0}", nid, xy_l2, l0);
    }
}

/// For a given anchor cell, compute distances to its neighbors and to k random non-neighbor cells.
/// Returns (observed_distances, random_distances).
fn random_cell_distance(
    anchor_idx: usize,
    data: &Array2<f64>,
    nbrs: &[usize],
    metric: Metric,
) -> (Vec<f64>, Vec<f64>) {
    let anchor = data.row(anchor_idx).to_vec();

    // Observed: distances to actual neighbors
    let nbr_matrix = data.select(Axis(0), nbrs);
    let obs_dists = compute_distances(&anchor, &nbr_matrix, metric);

    // Random: distances to random non-neighbor cells
    let pool: Vec<usize> = (0..data.nrows())
        .filter(|&i| i != anchor_idx && !nbrs.contains(&i))
        .collect();
    let n_to_pick = nbrs.len().min(pool.len());
    let mut rng = rand::rng();
    let rand_indices: Vec<usize> = pool.choose_multiple(&mut rng, n_to_pick).copied().collect();
    let rand_matrix = data.select(Axis(0), &rand_indices);
    let rand_dists = compute_distances(&anchor, &rand_matrix, metric);

    (obs_dists, rand_dists)
}

/// Perform a permutation test to compare the distances of an anchor cell to its true neighbors
/// with the distances to random (non-neighbor, non-self) cells. Returns:
///   (mean_observed_neighbor_distance, mean_random_distance, p_value)
fn permutation_test_cell_distance(
    anchor_idx: usize,
    data: &Array2<f64>,
    neighbor_indices: &[usize],
    metric: Metric,
    n_permutations: usize,
) -> (f64, Vec<f64>, f64) {
    let k = neighbor_indices.len();
    if k == 0 {
        return (0.0, vec![], 1.0); // No neighbors
    }

    // Observed: mean/vec of distances to true neighbors
    let anchor = data.row(anchor_idx).to_vec();
    let nbr_matrix = data.select(Axis(0), neighbor_indices);
    let obs_distances = compute_distances(&anchor, &nbr_matrix, metric);
    let mean_obs = obs_distances.iter().sum::<f64>() / k as f64;

    // Build pool of valid random candidates (not anchor/self, not neighbor)
    let all_candidates: Vec<usize> = (0..data.nrows())
        .filter(|&i| i != anchor_idx && !neighbor_indices.contains(&i))
        .collect();
    let n_random_choices = all_candidates.len().min(k);
    if n_random_choices == 0 {
        return (mean_obs, vec![], 1.0); // No randoms
    }

    // Compute mean distance for one shuffle (can be reused in perm loop)
    let mut rng = rand::rng();
    let mut random_means = Vec::with_capacity(n_permutations);

    for _ in 0..n_permutations {
        // Sample k random cells from pool, without replacement
        let idxs: Vec<usize> = all_candidates
            .choose_multiple(&mut rng, n_random_choices)
            .copied()
            .collect();
        let rand_matrix = data.select(Axis(0), &idxs);
        let rand_distances = compute_distances(&anchor, &rand_matrix, metric);
        let mean_rand = if !rand_distances.is_empty() {
            rand_distances.iter().sum::<f64>() / rand_distances.len() as f64
        } else {
            0.0
        };
        random_means.push(mean_rand);
    }

    // Empirical p-value: fraction of permuted means <= observed mean
    // (using one-sided test: are neighbors closer than random)
    let p_value = {
        let count = random_means
            .iter()
            .filter(|&&rm| rm <= mean_obs + 1e-12)
            .count();
        (count as f64 + 1.0) / (n_permutations as f64 + 1.0) // add-one smoothing
    };
    //println!("p_value: {}", p_value);

    /*
    // Also include the mean of all random means for interpretability
    let mean_rand_overall = if !random_means.is_empty() {
        random_means.iter().sum::<f64>() / random_means.len() as f64
    } else {
        0.0
    };
    */
    // println!("idx: {}, mean_obs: {}, p_value: {}", anchor_idx, mean_obs, p_value);
    (mean_obs, random_means, p_value)
}

/// Performs a (Mann-Whitney) Wilcoxon rank sum test between two samples and returns
/// (test_statistic, two_sided_p_value), using mann_whitney::u_test.
/// The inputs are two slices of f64 values. Returns the U statistic and p-value.
pub fn mann_whitney_wilcoxon(sample1: &[f64], sample2: &[f64]) -> Option<(f64, f64)> {
    if sample1.is_empty() || sample2.is_empty() {
        return None;
    }
    // Use mann_whitney::u_test from the mann_whitney crate
    // (assume import: use mann_whitney::u_test;)
    let result = mann_whitney::u_test(
        sample1.iter().copied(),
        sample2.iter().copied(),
        0.05,
        TailType::Left,
    )
    .unwrap();
    Some((result.test_statistic as f64, result.p_value))
}

fn load_from_csv(
    path: &Path,
    x_col: usize,
    y_col: usize,
    gene_start: usize,
) -> Result<(Array2<f64>, Array2<f64>)> {
    let file = File::open(path)?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);
    let mut expr_vec: Vec<Vec<f64>> = vec![];
    let mut coords_vec: Vec<[f64; 2]> = vec![];

    for result in rdr.records() {
        let record = result?;
        let x: f64 = record
            .get(x_col)
            .ok_or_else(|| anyhow::anyhow!("X column index {} out of bounds", x_col))?
            .parse()?;
        let y: f64 = record
            .get(y_col)
            .ok_or_else(|| anyhow::anyhow!("Y column index {} out of bounds", y_col))?
            .parse()?;
        coords_vec.push([x, y]);

        let genes: Vec<f64> = record
            .iter()
            .skip(gene_start)
            .map(|s| s.parse::<f64>().unwrap_or(0.0))
            .collect();
        expr_vec.push(genes);
    }

    let n_cells = expr_vec.len();
    let n_genes = expr_vec[0].len();
    let data =
        Array2::from_shape_vec((n_cells, n_genes), expr_vec.into_iter().flatten().collect())?;
    let coords = Array2::from_shape_vec((n_cells, 2), coords_vec.into_iter().flatten().collect())?;
    Ok((data, coords))
}

/// Robust helper to read fixed-length strings from HDF5 without truncation
fn read_strings_dynamic(file: &Hdf5File, path: &str) -> Result<Vec<String>> {
    let dataset = file.dataset(path)?;
    let dtype = dataset.dtype()?;
    let dsize = dtype.size();

    // Try Variable Length Unicode
    if let Ok(v) = dataset.read_1d::<hdf5::types::VarLenUnicode>() {
        return Ok(v.iter().map(|s| s.to_string()).collect());
    }

    // Use the exact size match for FixedAscii to prevent truncation
    // Supporting sizes commonly found in 10x/Visium HD files
    match dsize {
        6 => {
            if let Ok(v) = dataset.read_1d::<FixedAscii<6>>() {
                return Ok(v
                    .iter()
                    .map(|s| s.as_str().trim_end_matches('\0').to_string())
                    .collect());
            }
        }
        7 => {
            if let Ok(v) = dataset.read_1d::<FixedAscii<7>>() {
                return Ok(v
                    .iter()
                    .map(|s| s.as_str().trim_end_matches('\0').to_string())
                    .collect());
            }
        }
        10 => {
            if let Ok(v) = dataset.read_1d::<FixedAscii<10>>() {
                return Ok(v
                    .iter()
                    .map(|s| s.as_str().trim_end_matches('\0').to_string())
                    .collect());
            }
        }
        11 => {
            if let Ok(v) = dataset.read_1d::<FixedAscii<11>>() {
                return Ok(v
                    .iter()
                    .map(|s| s.as_str().trim_end_matches('\0').to_string())
                    .collect());
            }
        }
        15 => {
            if let Ok(v) = dataset.read_1d::<FixedAscii<15>>() {
                return Ok(v
                    .iter()
                    .map(|s| s.as_str().trim_end_matches('\0').to_string())
                    .collect());
            }
        }
        16 => {
            if let Ok(v) = dataset.read_1d::<FixedAscii<16>>() {
                return Ok(v
                    .iter()
                    .map(|s| s.as_str().trim_end_matches('\0').to_string())
                    .collect());
            }
        }
        18 => {
            if let Ok(v) = dataset.read_1d::<FixedAscii<18>>() {
                return Ok(v
                    .iter()
                    .map(|s| s.as_str().trim_end_matches('\0').to_string())
                    .collect());
            }
        }
        21 => {
            if let Ok(v) = dataset.read_1d::<FixedAscii<21>>() {
                return Ok(v
                    .iter()
                    .map(|s| s.as_str().trim_end_matches('\0').to_string())
                    .collect());
            }
        }
        23 => {
            if let Ok(v) = dataset.read_1d::<FixedAscii<23>>() {
                return Ok(v
                    .iter()
                    .map(|s| s.as_str().trim_end_matches('\0').to_string())
                    .collect());
            }
        }
        25 => {
            if let Ok(v) = dataset.read_1d::<FixedAscii<25>>() {
                return Ok(v
                    .iter()
                    .map(|s| s.as_str().trim_end_matches('\0').to_string())
                    .collect());
            }
        }
        32 => {
            if let Ok(v) = dataset.read_1d::<FixedAscii<32>>() {
                return Ok(v
                    .iter()
                    .map(|s| s.as_str().trim_end_matches('\0').to_string())
                    .collect());
            }
        }
        64 => {
            if let Ok(v) = dataset.read_1d::<FixedAscii<64>>() {
                return Ok(v
                    .iter()
                    .map(|s| s.as_str().trim_end_matches('\0').to_string())
                    .collect());
            }
        }
        _ => {}
    }

    // Fallback: Read as raw bytes and try to guess the length
    let bytes = dataset.read_1d::<u8>()?.to_vec();
    let shape = dataset.shape();
    let num_strings = if shape.is_empty() { 0 } else { shape[0] };
    if num_strings > 0 && bytes.len() % num_strings == 0 {
        let len = bytes.len() / num_strings;
        let mut strings = Vec::with_capacity(num_strings);
        for i in 0..num_strings {
            let start = i * len;
            let end = start + len;
            let s = String::from_utf8_lossy(&bytes[start..end])
                .trim_matches('\0')
                .to_string();
            strings.push(s);
        }
        return Ok(strings);
    }

    anyhow::bail!("Unsupported string format for dataset: {}", path)
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

fn load_positions_from_csv(
    pos_path: &Path,
    pos_x_col: usize,
    pos_y_col: usize,
) -> Result<HashMap<String, (f64, f64)>> {
    let mut pos_map = HashMap::new();
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(pos_path)?;
    for rec in rdr.records() {
        let rec = rec?;
        if rec.len() <= pos_y_col {
            continue;
        }
        let barcode = rec.get(0).unwrap_or_default().trim().trim_matches('"');
        if barcode.is_empty() {
            continue;
        }
        let x = match rec.get(pos_x_col).unwrap_or_default().trim().parse::<f64>() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let y = match rec.get(pos_y_col).unwrap_or_default().trim().parse::<f64>() {
            Ok(v) => v,
            Err(_) => continue,
        };
        pos_map.insert(barcode.to_string(), (x, y));
    }
    Ok(pos_map)
}

fn load_positions_from_parquet(
    pos_path: &Path,
    pos_x_col: usize,
    pos_y_col: usize,
) -> Result<HashMap<String, (f64, f64)>> {
    fn get_numeric_as_f64(row: &parquet::record::Row, idx: usize) -> Result<f64> {
        if let Ok(v) = row.get_double(idx) {
            return Ok(v);
        }
        if let Ok(v) = row.get_float(idx) {
            return Ok(v as f64);
        }
        if let Ok(v) = row.get_long(idx) {
            return Ok(v as f64);
        }
        if let Ok(v) = row.get_int(idx) {
            return Ok(v as f64);
        }
        if let Ok(v) = row.get_short(idx) {
            return Ok(v as f64);
        }
        if let Ok(v) = row.get_byte(idx) {
            return Ok(v as f64);
        }
        if let Ok(v) = row.get_ubyte(idx) {
            return Ok(v as f64);
        }
        if let Ok(v) = row.get_ushort(idx) {
            return Ok(v as f64);
        }
        if let Ok(v) = row.get_uint(idx) {
            return Ok(v as f64);
        }
        anyhow::bail!(
            "Parquet column {} is not a numeric type convertible to f64",
            idx
        );
    }

    let pos_file = std::fs::File::open(pos_path)?;
    let reader = SerializedFileReader::new(pos_file)?;
    let mut iter = reader.get_row_iter(None)?;
    let mut pos_map = HashMap::new();
    while let Some(Ok(row)) = iter.next() {
        let barcode = match row.get_string(0) {
            Ok(s) => s.to_string(),
            Err(_) => {
                // Some Xenium parquet files store `cell_id` as BLOB rather than UTF8.
                // Decode bytes lossily to match H5 barcode strings.
                let raw = row.get_bytes(0)?;
                String::from_utf8_lossy(raw.data()).to_string()
            }
        };
        let x = get_numeric_as_f64(&row, pos_x_col)?;
        let y = get_numeric_as_f64(&row, pos_y_col)?;
        pos_map.insert(barcode, (x, y));
    }
    Ok(pos_map)
}

/// Load 10x-style H5 with external positions file (CSV/Parquet), matching rows by barcode.
fn load_from_h5_with_pos(
    h5_path: &Path,
    pos_path: &Path,
    pos_x_col: usize,
    pos_y_col: usize,
) -> Result<(Array2<f64>, Array2<f64>)> {
    let file = Hdf5File::open(h5_path)?;
    let matrix_group = file.group("matrix")?;

    let shape: Vec<usize> = matrix_group.dataset("shape")?.read_1d::<usize>()?.to_vec();
    let num_features = shape[0];
    let num_cells = shape[1];

    let data_arr = matrix_group.dataset("data")?.read_1d::<u16>()?;
    let indices_arr = matrix_group.dataset("indices")?.read_1d::<usize>()?;
    let indptr_arr = matrix_group.dataset("indptr")?.read_1d::<usize>()?;

    let mut dense_data = Array2::<f64>::zeros((num_cells, num_features));
    for i in 0..num_cells {
        let start = indptr_arr[i];
        let end = indptr_arr[i + 1];
        for j in start..end {
            dense_data[[i, indices_arr[j]]] = data_arr[j] as f64;
        }
    }

    let barcodes = read_strings_dynamic(&file, "/matrix/barcodes")?;
    if barcodes.len() < num_cells {
        anyhow::bail!(
            "Not enough barcodes for selected rows: have {}, need {}",
            barcodes.len(),
            num_cells
        );
    }

    let pos_map = match pos_path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
    {
        Some(ext) if ext == "csv" => load_positions_from_csv(pos_path, pos_x_col, pos_y_col)?,
        _ => load_positions_from_parquet(pos_path, pos_x_col, pos_y_col)?,
    };

    let mut coords_vec = Vec::with_capacity(num_cells * 2);
    for row_idx in 0..num_cells {
        let barcode = &barcodes[row_idx];
        let (x, y) = pos_map.get(barcode).copied().ok_or_else(|| {
            anyhow::anyhow!(
                "Missing position for barcode '{}' at row {}",
                barcode,
                row_idx
            )
        })?;
        coords_vec.push(x);
        coords_vec.push(y);
    }
    let coords = Array2::from_shape_vec((num_cells, 2), coords_vec)?;

    Ok((dense_data, coords))
}

/// Load 10x-style H5 without a positions file (single-cell mode).
/// Returns dense expression data and dummy coordinates (0.0, 0.0) for each cell.
fn load_from_h5_no_pos(h5_path: &Path) -> Result<(Array2<f64>, Array2<f64>)> {
    let file = Hdf5File::open(h5_path)?;
    let matrix_group = file.group("matrix")?;

    let shape: Vec<usize> = matrix_group.dataset("shape")?.read_1d::<usize>()?.to_vec();
    let num_features = shape[0];
    let num_cells = shape[1];

    let data_arr = matrix_group.dataset("data")?.read_1d::<u16>()?;
    let indices_arr = matrix_group.dataset("indices")?.read_1d::<usize>()?;
    let indptr_arr = matrix_group.dataset("indptr")?.read_1d::<usize>()?;

    let mut dense_data = Array2::<f64>::zeros((num_cells, num_features));
    for i in 0..num_cells {
        let start = indptr_arr[i];
        let end = indptr_arr[i + 1];
        for j in start..end {
            dense_data[[i, indices_arr[j]]] = data_arr[j] as f64;
        }
    }

    // Dummy coords: all zeros
    let coords_vec = vec![0.0f64; num_cells * 2];
    let coords = Array2::from_shape_vec((num_cells, 2), coords_vec)?;

    Ok((dense_data, coords))
}

/// L0 binary distance: counts genes where sparsity pattern differs
/// (one is zero, the other is non-zero) — same metric used in MST encoding
#[derive(Clone, Copy)]
struct DistL0;

impl Distance<f64> for DistL0 {
    fn eval(&self, a: &[f64], b: &[f64]) -> f32 {
        a.iter()
            .zip(b.iter())
            .filter(|(&ai, &bi)| (ai > 0.0) != (bi > 0.0))
            .count() as f32
    }
}

fn find_neighbors(data: &Array2<f64>, k: usize, metric: Metric) -> Vec<Vec<usize>> {
    let n = data.nrows();

    // HNSW configuration
    let max_nb_connection = 16;
    let nb_elements = n;
    let nb_layers = 16;
    let ef_construction = 200;
    let ef_search = 100;

    match metric {
        Metric::Cosine => {
            let hnsw = Hnsw::<f64, DistCosine>::new(
                max_nb_connection,
                nb_elements,
                nb_layers,
                ef_construction,
                DistCosine,
            );
            // Parallel insertion
            data.axis_iter(Axis(0))
                .enumerate()
                .collect::<Vec<_>>()
                .par_iter()
                .for_each(|(i, row)| {
                    hnsw.insert((row.as_slice().unwrap(), *i));
                });

            (0..n)
                .into_par_iter()
                .map(|i| {
                    let row = data.row(i);
                    hnsw.search(row.as_slice().unwrap(), k, ef_search)
                        .into_iter()
                        .filter(|nb| nb.d_id != i) // Exclude self
                        .take(k)
                        .map(|nb| nb.d_id)
                        .collect()
                })
                .collect()
        }
        Metric::L2 => {
            let hnsw = Hnsw::<f64, DistL2>::new(
                max_nb_connection,
                nb_elements,
                nb_layers,
                ef_construction,
                DistL2,
            );
            data.axis_iter(Axis(0))
                .enumerate()
                .collect::<Vec<_>>()
                .par_iter()
                .for_each(|(i, row)| {
                    hnsw.insert((row.as_slice().unwrap(), *i));
                });

            (0..n)
                .into_par_iter()
                .map(|i| {
                    let row = data.row(i);
                    hnsw.search(row.as_slice().unwrap(), k, ef_search)
                        .into_iter()
                        .filter(|nb| nb.d_id != i)
                        .take(k)
                        .map(|nb| nb.d_id)
                        .collect()
                })
                .collect()
        }
        Metric::L0 => {
            let hnsw = Hnsw::<f64, DistL0>::new(
                max_nb_connection,
                nb_elements,
                nb_layers,
                ef_construction,
                DistL0,
            );
            data.axis_iter(Axis(0))
                .enumerate()
                .collect::<Vec<_>>()
                .par_iter()
                .for_each(|(i, row)| {
                    hnsw.insert((row.as_slice().unwrap(), *i));
                });

            (0..n)
                .into_par_iter()
                .map(|i| {
                    let row = data.row(i);
                    hnsw.search(row.as_slice().unwrap(), k, ef_search)
                        .into_iter()
                        .filter(|nb| nb.d_id != i)
                        .take(k)
                        .map(|nb| nb.d_id)
                        .collect()
                })
                .collect()
        }
    }
}

fn build_neighbors_for_mode(
    neighbor_by: NeighborBy,
    coords: &Array2<f64>,
    data_raw: &Array2<f64>,
    k: usize,
) -> Vec<Vec<usize>> {
    match neighbor_by {
        NeighborBy::Spatial | NeighborBy::SpatialL0 => {
            info!("Finding spatial neighbors (L2 on XY coords, k={})...", k);
            find_neighbors(coords, k, Metric::L2)
        }
        NeighborBy::Expression => {
            info!("Finding expression neighbors (L2 on raw, k={})...", k);
            find_neighbors(data_raw, k, Metric::L2)
        }
        NeighborBy::ExpressionL0 => {
            info!("Finding expression neighbors (L0 on raw, k={})...", k);
            find_neighbors(data_raw, k, Metric::L0)
        }
    }
}

/// Truncated PCA via power-iteration SVD.
/// Takes ownership of the input matrix to avoid cloning (~halves peak memory).
/// Input: centered+scaled matrix (n_cells × n_features).
/// Returns: projected matrix (n_cells × n_components).

fn pca_project(data: Array2<f64>, n_components: usize) -> Array2<f64> {
    let (n, p) = (data.nrows(), data.ncols());
    let k = n_components.min(n).min(p);
    let mut projected = Array2::<f64>::zeros((n, k));
    let mut residual = data; // take ownership, no clone

    for comp in 0..k {
        // Initialize v deterministically (avoid rand dependency for reproducibility)
        let mut v = Array1::<f64>::zeros(p);
        for j in 0..p {
            v[j] = ((j * 7 + 13) % 97) as f64;
        }
        let norm = v.dot(&v).sqrt();
        v /= norm;

        // Power iteration to find leading right singular vector
        for _ in 0..300 {
            let u = residual.dot(&v);
            let v_new = residual.t().dot(&u);
            let new_norm = v_new.dot(&v_new).sqrt();
            if new_norm < 1e-15 {
                break;
            }
            let v_next: Array1<f64> = &v_new / new_norm;
            let diff = (&v_next - &v).mapv(|x| x * x).sum().sqrt();
            v = v_next;
            if diff < 1e-10 {
                break;
            }
        }

        // Project data onto this component
        let scores = residual.dot(&v);
        projected.column_mut(comp).assign(&scores);

        // Deflate: remove this component from residual
        let scores_col: ndarray::ArrayView2<f64> = scores.view().insert_axis(Axis(1)); // (n, 1)
        let v_row: ndarray::ArrayView2<f64> = v.view().insert_axis(Axis(0)); // (1, p)
        residual -= &scores_col.dot(&v_row);
    }

    projected
}

/*
fn pca_project(data: Array2<f64>, n_components: usize) -> Array2<f64> {
    let mut pca = PCA::new();
    // rfit: randomized SVD, fast for large matrices
    // args: data, n_components, n_oversamples (0 = auto), seed, tolerance
    let scores = pca.rfit(data, n_components, 0, Some(42), None)
        .expect("PCA rfit failed");
    scores
}
 */

fn write_results_to_csv<P: AsRef<std::path::Path>>(
    filename: P,
    n_cells: usize,
    neighbor_label: &str,
    n_perm: usize,
    metric_results: &[(&str, &[(f64, Vec<f64>, f64)])],
) -> Result<()> {
    let file = File::create(filename)?;
    let buf = BufWriter::new(file);
    let mut wtr = csv::Writer::from_writer(buf);

    let mut header = vec!["cell".to_string(), "neighbor_by".to_string()];
    for (label, _) in metric_results {
        header.push(format!("mean_obs_{}", label));
        for j in 0..n_perm {
            header.push(format!("{}_val_{}", label, j));
        }
        header.push(format!("{}_p", label));
    }
    wtr.write_record(&header)?;

    for i in 0..n_cells {
        let mut row = vec![i.to_string(), neighbor_label.to_string()];
        for (_, res) in metric_results {
            row.push(res[i].0.to_string());
            let vals = &res[i].1;
            for j in 0..n_perm {
                row.push(vals.get(j).map(|v| v.to_string()).unwrap_or_default());
            }
            row.push(res[i].2.to_string());
        }
        wtr.write_record(&row)?;
    }
    wtr.flush()?;
    Ok(())
}

fn write_results_to_csv_utest<P: AsRef<std::path::Path>>(
    filename: P,
    n_cells: usize,
    neighbor_label: &str,
    metric_results: &[(&str, &[(Option<(f64, f64)>, Vec<f64>, Vec<f64>)])],
) -> Result<()> {
    let file = File::create(filename)?;
    let buf = BufWriter::new(file);
    let mut wtr = csv::Writer::from_writer(buf);

    // Per-metric max obs/rand lengths
    let metric_dims: Vec<(usize, usize)> = metric_results
        .iter()
        .map(|(_, res)| {
            let max_obs = res.iter().map(|(_, o, _)| o.len()).max().unwrap_or(0);
            let max_rand = res.iter().map(|(_, _, r)| r.len()).max().unwrap_or(0);
            (max_obs, max_rand)
        })
        .collect();

    let mut header = vec!["cell".to_string(), "neighbor_by".to_string()];
    for (idx, (label, _)) in metric_results.iter().enumerate() {
        header.push(format!("ustat_{}", label));
        header.push(format!("p_{}", label));
        let (max_obs, max_rand) = metric_dims[idx];
        for j in 0..max_obs {
            header.push(format!("{}_obs_{}", label, j));
        }
        for j in 0..max_rand {
            header.push(format!("{}_rand_{}", label, j));
        }
    }
    wtr.write_record(&header)?;

    for i in 0..n_cells {
        let mut row = vec![i.to_string(), neighbor_label.to_string()];
        for (idx, (_, res)) in metric_results.iter().enumerate() {
            let (u, p) = res[i].0.unwrap_or((f64::NAN, f64::NAN));
            row.push(u.to_string());
            row.push(p.to_string());
            let (max_obs, max_rand) = metric_dims[idx];
            for j in 0..max_obs {
                row.push(res[i].1.get(j).map(|v| v.to_string()).unwrap_or_default());
            }
            for j in 0..max_rand {
                row.push(res[i].2.get(j).map(|v| v.to_string()).unwrap_or_default());
            }
        }
        wtr.write_record(&row)?;
    }
    wtr.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    let test_type = args.test_type;
    let (data_raw, coords): (Array2<f64>, Array2<f64>) = match args.format {
        InputFormat::Csv => load_from_csv(&args.input, args.x_col, args.y_col, args.gene_start)?,
        InputFormat::H5 => match args.platform {
            Some(Platform::SingleCell) => load_from_h5_no_pos(&args.input)?,
            _ => {
                let pos_path = args.pos.as_ref().ok_or_else(|| {
                    anyhow::anyhow!(
                        "--pos is required for H5 unless --platform single-cell is used"
                    )
                })?;
                let (pos_x_col, pos_y_col) = resolve_position_columns(args.platform, None, None);
                load_from_h5_with_pos(&args.input, pos_path, pos_x_col, pos_y_col)?
            }
        },
    };

    let n_cells = data_raw.nrows();
    let n_genes = data_raw.ncols();
    println!("Loaded {} cells and {} genes", n_cells, n_genes);
    println!("Mode: --neighbor-by {:?}", args.neighbor_by);

    // ── Step 1: Find neighbors ──────────────────────────────────────
    let neighbors: Vec<Vec<usize>> =
        build_neighbors_for_mode(args.neighbor_by, &coords, &data_raw, args.k);
    if let Some(n) = args.neighbor_preview {
        if n > 0 {
            print_neighbor_preview(0, n, &coords, &data_raw, &neighbors);
        }
    }
    let nbr_label = if args.pca_dims > 0 && args.pca {
        match args.neighbor_by {
            NeighborBy::Spatial => match args.test_type {
                TestType::Permutation => "spatial_pca_perm",
                TestType::MeanWilcoxon => "spatial_pca_utest",
            },
            NeighborBy::SpatialL0 => match args.test_type {
                TestType::Permutation => "spatialL0_pca_perm",
                TestType::MeanWilcoxon => "spatialL0_pca_utest",
            },
            NeighborBy::Expression => match args.test_type {
                TestType::Permutation => "exprL2_pca_perm",
                TestType::MeanWilcoxon => "exprL2_pca_utest",
            },
            NeighborBy::ExpressionL0 => match args.test_type {
                TestType::Permutation => "exprL0_pca_perm",
                TestType::MeanWilcoxon => "exprL0_pca_utest",
            },
        }
    } else {
        match args.neighbor_by {
            NeighborBy::Spatial => match args.test_type {
                TestType::Permutation => "spatial_perm",
                TestType::MeanWilcoxon => "spatial_utest",
            },
            NeighborBy::SpatialL0 => match args.test_type {
                TestType::Permutation => "spatialL0_perm",
                TestType::MeanWilcoxon => "spatialL0_utest",
            },
            NeighborBy::Expression => match args.test_type {
                TestType::Permutation => "exprL2_perm",
                TestType::MeanWilcoxon => "exprL2_utest",
            },
            NeighborBy::ExpressionL0 => match args.test_type {
                TestType::Permutation => "exprL0_perm",
                TestType::MeanWilcoxon => "exprL0_utest",
            },
        }
    };

    // ── Raw mode: only run raw tests, no PCA ────────────────────────
    info!("Running per-cell tests (k={})...", args.k);

    match test_type {
        TestType::Permutation => {
            match args.neighbor_by {
                NeighborBy::SpatialL0 => {
                    println!("Running permutation test (L0 only, spatial neighbors)...");
                    let results_raw_l0: Vec<(f64, Vec<f64>, f64)> = (0..n_cells)
                        .into_par_iter()
                        .map(|i| {
                            permutation_test_cell_distance(
                                i,
                                &data_raw,
                                &neighbors[i],
                                Metric::L0,
                                args.n_perm,
                            )
                        })
                        .collect();
                    let metrics: Vec<(&str, &[(f64, Vec<f64>, f64)])> =
                        vec![("l0", &results_raw_l0)];
                    write_results_to_csv(&args.output, n_cells, nbr_label, args.n_perm, &metrics)?;
                }
                NeighborBy::ExpressionL0 => {
                    println!("Running permutation test (L0, expression neighbors)...");
                    let results_raw_l0: Vec<(f64, Vec<f64>, f64)> = (0..n_cells)
                        .into_par_iter()
                        .map(|i| {
                            permutation_test_cell_distance(
                                i,
                                &data_raw,
                                &neighbors[i],
                                Metric::L0,
                                args.n_perm,
                            )
                        })
                        .collect();
                    let metrics: Vec<(&str, &[(f64, Vec<f64>, f64)])> =
                        vec![("l0", &results_raw_l0)];
                    write_results_to_csv(&args.output, n_cells, nbr_label, args.n_perm, &metrics)?;
                }
                _ => {
                    println!("Running permutation test");
                    println!("  -> {} neighbors → raw cosine...", nbr_label);
                    let results_raw_cos: Vec<(f64, Vec<f64>, f64)> = (0..n_cells)
                        .into_par_iter()
                        .map(|i| {
                            permutation_test_cell_distance(
                                i,
                                &data_raw,
                                &neighbors[i],
                                Metric::Cosine,
                                args.n_perm,
                            )
                        })
                        .collect();
                    println!("  -> {} neighbors → raw L2...", nbr_label);
                    let results_raw_l2: Vec<(f64, Vec<f64>, f64)> = (0..n_cells)
                        .into_par_iter()
                        .map(|i| {
                            permutation_test_cell_distance(
                                i,
                                &data_raw,
                                &neighbors[i],
                                Metric::L2,
                                args.n_perm,
                            )
                        })
                        .collect();

                    let mut metrics: Vec<(&str, &[(f64, Vec<f64>, f64)])> =
                        vec![("cos", &results_raw_cos), ("l2", &results_raw_l2)];

                    // Spatial → also test L0 on expression; Expression → also test L2 on XY coords
                    let results_extra: Vec<(f64, Vec<f64>, f64)>;
                    let extra_label: &str;
                    match args.neighbor_by {
                        NeighborBy::Spatial => {
                            println!("  -> {} neighbors → raw L0...", nbr_label);
                            results_extra = (0..n_cells)
                                .into_par_iter()
                                .map(|i| {
                                    permutation_test_cell_distance(
                                        i,
                                        &data_raw,
                                        &neighbors[i],
                                        Metric::L0,
                                        args.n_perm,
                                    )
                                })
                                .collect();
                            extra_label = "l0";
                        }
                        NeighborBy::Expression => {
                            println!("  -> {} neighbors → XY L2...", nbr_label);
                            results_extra = (0..n_cells)
                                .into_par_iter()
                                .map(|i| {
                                    permutation_test_cell_distance(
                                        i,
                                        &coords,
                                        &neighbors[i],
                                        Metric::L2,
                                        args.n_perm,
                                    )
                                })
                                .collect();
                            extra_label = "xyL2";
                        }
                        NeighborBy::SpatialL0 | NeighborBy::ExpressionL0 => unreachable!(), // handled above
                    }
                    metrics.push((extra_label, &results_extra));

                    write_results_to_csv(&args.output, n_cells, nbr_label, args.n_perm, &metrics)?;
                }
            }
        }
        TestType::MeanWilcoxon => {
            match args.neighbor_by {
                NeighborBy::SpatialL0 => {
                    println!("Running mean Wilcoxon test (L0 only, spatial neighbors)...");
                    let l0_results: Vec<(Option<(f64, f64)>, Vec<f64>, Vec<f64>)> = (0..n_cells)
                        .into_par_iter()
                        .map(|i| {
                            let (obs, rand_d) =
                                random_cell_distance(i, &data_raw, &neighbors[i], Metric::L0);
                            let utest = mann_whitney_wilcoxon(&obs, &rand_d);
                            (utest, obs, rand_d)
                        })
                        .collect();
                    let metrics: Vec<(&str, &[(Option<(f64, f64)>, Vec<f64>, Vec<f64>)])> =
                        vec![("l0", &l0_results)];
                    write_results_to_csv_utest(&args.output, n_cells, nbr_label, &metrics)?;
                    println!(
                        "Results saved to {} ({} cells, raw only)",
                        args.output.display(),
                        n_cells
                    );
                }
                NeighborBy::ExpressionL0 => {
                    println!("Running mean Wilcoxon test (L0, expression neighbors)...");
                    let l0_results: Vec<(Option<(f64, f64)>, Vec<f64>, Vec<f64>)> = (0..n_cells)
                        .into_par_iter()
                        .map(|i| {
                            let (obs, rand_d) =
                                random_cell_distance(i, &data_raw, &neighbors[i], Metric::L0);
                            let utest = mann_whitney_wilcoxon(&obs, &rand_d);
                            (utest, obs, rand_d)
                        })
                        .collect();
                    let metrics: Vec<(&str, &[(Option<(f64, f64)>, Vec<f64>, Vec<f64>)])> =
                        vec![("l0", &l0_results)];
                    write_results_to_csv_utest(&args.output, n_cells, nbr_label, &metrics)?;
                    println!(
                        "Results saved to {} ({} cells, raw only)",
                        args.output.display(),
                        n_cells
                    );
                }
                _ => {
                    println!("Running mean Wilcoxon test");
                    let cos_results: Vec<(Option<(f64, f64)>, Vec<f64>, Vec<f64>)> = (0..n_cells)
                        .into_par_iter()
                        .map(|i| {
                            let (obs, rand_d) =
                                random_cell_distance(i, &data_raw, &neighbors[i], Metric::Cosine);
                            let utest = mann_whitney_wilcoxon(&obs, &rand_d);
                            (utest, obs, rand_d)
                        })
                        .collect();
                    let l2_results: Vec<(Option<(f64, f64)>, Vec<f64>, Vec<f64>)> = (0..n_cells)
                        .into_par_iter()
                        .map(|i| {
                            let (obs, rand_d) =
                                random_cell_distance(i, &data_raw, &neighbors[i], Metric::L2);
                            let utest = mann_whitney_wilcoxon(&obs, &rand_d);
                            (utest, obs, rand_d)
                        })
                        .collect();

                    let mut metrics: Vec<(&str, &[(Option<(f64, f64)>, Vec<f64>, Vec<f64>)])> =
                        vec![("cos", &cos_results), ("l2", &l2_results)];

                    let extra_results: Vec<(Option<(f64, f64)>, Vec<f64>, Vec<f64>)>;
                    let extra_label: &str;
                    match args.neighbor_by {
                        NeighborBy::Spatial => {
                            println!("  -> L0 on expression...");
                            extra_results = (0..n_cells)
                                .into_par_iter()
                                .map(|i| {
                                    let (obs, rand_d) = random_cell_distance(
                                        i,
                                        &data_raw,
                                        &neighbors[i],
                                        Metric::L0,
                                    );
                                    let utest = mann_whitney_wilcoxon(&obs, &rand_d);
                                    (utest, obs, rand_d)
                                })
                                .collect();
                            extra_label = "l0";
                        }
                        NeighborBy::Expression => {
                            println!("  -> L2 on XY coords...");
                            extra_results = (0..n_cells)
                                .into_par_iter()
                                .map(|i| {
                                    let (obs, rand_d) =
                                        random_cell_distance(i, &coords, &neighbors[i], Metric::L2);
                                    let utest = mann_whitney_wilcoxon(&obs, &rand_d);
                                    (utest, obs, rand_d)
                                })
                                .collect();
                            extra_label = "xyL2";
                        }
                        NeighborBy::SpatialL0 | NeighborBy::ExpressionL0 => unreachable!(), // handled above
                    }
                    metrics.push((extra_label, &extra_results));

                    write_results_to_csv_utest(&args.output, n_cells, nbr_label, &metrics)?;
                    println!(
                        "Results saved to {} ({} cells, raw only)",
                        args.output.display(),
                        n_cells
                    );
                }
            }
        }
    }
    println!(
        "Results saved to {} ({} cells, raw only)",
        args.output.display(),
        n_cells
    );

    if args.pca_dims > 0 && args.pca {
        // ── PCA mode: only run PCA tests, skip raw ──────────────────────
        println!("Performing PCA ({} dimensions)...", args.pca_dims);

        let n_f = n_cells as f64;
        let n_g = n_genes;

        // Compute per-gene mean on log1p data
        let mut gene_means = vec![0.0f64; n_g];
        for row in data_raw.axis_iter(Axis(0)) {
            for j in 0..n_g {
                gene_means[j] += (1.0 + row[j]).ln();
            }
        }
        for j in 0..n_g {
            gene_means[j] /= n_f;
        }

        // Compute per-gene variance on log1p data
        let mut gene_var = vec![0.0f64; n_g];
        for row in data_raw.axis_iter(Axis(0)) {
            for j in 0..n_g {
                let diff = (1.0 + row[j]).ln() - gene_means[j];
                gene_var[j] += diff * diff;
            }
        }
        for j in 0..n_g {
            gene_var[j] /= n_f - 1.0;
        }

        // Filter zero-variance genes
        let mut keep_cols: Vec<usize> = (0..n_g).filter(|&j| gene_var[j] > 1e-12).collect();
        println!("  {}/{} genes have non-zero variance", keep_cols.len(), n_g);

        // Keep only top variable genes (by variance) to reduce memory
        if args.top_genes > 0 && keep_cols.len() > args.top_genes {
            keep_cols.sort_by(|&a, &b| gene_var[b].partial_cmp(&gene_var[a]).unwrap());
            keep_cols.truncate(args.top_genes);
            keep_cols.sort(); // restore column order
        }
        let n_kept = keep_cols.len();
        println!(
            "  Using top {} genes for PCA ({:.1} MB matrix)",
            n_kept,
            (n_cells * n_kept * 8) as f64 / 1e6
        );

        // Build centered+scaled matrix
        let std_devs: Vec<f64> = keep_cols.iter().map(|&j| gene_var[j].sqrt()).collect();
        let kept_means: Vec<f64> = keep_cols.iter().map(|&j| gene_means[j]).collect();

        let mut data_scaled = Array2::<f64>::zeros((n_cells, n_kept));
        for (i, row) in data_raw.axis_iter(Axis(0)).enumerate() {
            for (jj, &orig_j) in keep_cols.iter().enumerate() {
                let log_val = (1.0 + row[orig_j]).ln();
                data_scaled[[i, jj]] = (log_val - kept_means[jj]) / std_devs[jj];
            }
        }

        // Drop data_raw BEFORE SVD to free memory
        let raw_bytes = n_cells * n_genes * 8;
        drop(data_raw);
        info!(
            "  Freed raw data ({:.1} MB) before SVD",
            raw_bytes as f64 / 1e6
        );

        // PCA via power-iteration SVD (data_scaled is already restricted to top HVGs)
        let pca = pca_project(data_scaled, args.pca_dims);
        println!(
            "  PCA output: {} cells × {} components",
            pca.nrows(),
            pca.ncols()
        );

        // Save per-cell results with "pca_" prefix in filename
        let orig_file_name = args
            .output
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("pca_results.csv");
        let parent = args
            .output
            .parent()
            .unwrap_or_else(|| std::path::Path::new(""));

        // Build new filename with "pca_" prefix
        let pca_file_name = format!("pca_{}", orig_file_name);
        let pca_full_path = parent.join(pca_file_name);

        match test_type {
            TestType::Permutation => {
                if args.neighbor_by == NeighborBy::SpatialL0 {
                    println!("Running PCA permutation test (L0 only, spatial neighbors)...");
                    let results_pca_l0: Vec<(f64, Vec<f64>, f64)> = (0..n_cells)
                        .into_par_iter()
                        .map(|i| {
                            permutation_test_cell_distance(
                                i,
                                &pca,
                                &neighbors[i],
                                Metric::L0,
                                args.n_perm,
                            )
                        })
                        .collect();
                    let metrics: Vec<(&str, &[(f64, Vec<f64>, f64)])> =
                        vec![("l0", &results_pca_l0)];
                    write_results_to_csv(
                        &pca_full_path,
                        n_cells,
                        nbr_label,
                        args.n_perm,
                        &metrics,
                    )?;
                } else {
                    println!("Running PCA permutation test");
                    println!("  -> {} neighbors → PCA cosine...", nbr_label);
                    let results_pca_cos: Vec<(f64, Vec<f64>, f64)> = (0..n_cells)
                        .into_par_iter()
                        .map(|i| {
                            permutation_test_cell_distance(
                                i,
                                &pca,
                                &neighbors[i],
                                Metric::Cosine,
                                args.n_perm,
                            )
                        })
                        .collect();
                    println!("  -> {} neighbors → PCA L2...", nbr_label);
                    let results_pca_l2: Vec<(f64, Vec<f64>, f64)> = (0..n_cells)
                        .into_par_iter()
                        .map(|i| {
                            permutation_test_cell_distance(
                                i,
                                &pca,
                                &neighbors[i],
                                Metric::L2,
                                args.n_perm,
                            )
                        })
                        .collect();

                    let mut metrics: Vec<(&str, &[(f64, Vec<f64>, f64)])> =
                        vec![("cos", &results_pca_cos), ("l2", &results_pca_l2)];

                    let results_extra: Vec<(f64, Vec<f64>, f64)>;
                    let extra_label: &str;
                    match args.neighbor_by {
                        NeighborBy::Spatial => {
                            println!("  -> PCA L0 on expression (not meaningful on PCA, skipping)");
                            results_extra = vec![];
                            extra_label = "";
                        }
                        NeighborBy::Expression | NeighborBy::ExpressionL0 => {
                            println!("  -> {} neighbors → XY L2...", nbr_label);
                            results_extra = (0..n_cells)
                                .into_par_iter()
                                .map(|i| {
                                    permutation_test_cell_distance(
                                        i,
                                        &coords,
                                        &neighbors[i],
                                        Metric::L2,
                                        args.n_perm,
                                    )
                                })
                                .collect();
                            extra_label = "xyL2";
                        }
                        NeighborBy::SpatialL0 => unreachable!(), // handled above
                    }
                    if !extra_label.is_empty() {
                        metrics.push((extra_label, &results_extra));
                    }

                    write_results_to_csv(
                        &pca_full_path,
                        n_cells,
                        nbr_label,
                        args.n_perm,
                        &metrics,
                    )?;
                }
                println!(
                    "Results saved to {} ({} cells, PCA)",
                    pca_full_path.display(),
                    n_cells
                );
            }
            TestType::MeanWilcoxon => {
                if args.neighbor_by == NeighborBy::SpatialL0 {
                    println!("Running PCA mean Wilcoxon test (L0 only, spatial neighbors)...");
                    let l0_results: Vec<(Option<(f64, f64)>, Vec<f64>, Vec<f64>)> = (0..n_cells)
                        .into_par_iter()
                        .map(|i| {
                            let (obs, rand_d) =
                                random_cell_distance(i, &pca, &neighbors[i], Metric::L0);
                            let utest = mann_whitney_wilcoxon(&obs, &rand_d);
                            (utest, obs, rand_d)
                        })
                        .collect();
                    let metrics: Vec<(&str, &[(Option<(f64, f64)>, Vec<f64>, Vec<f64>)])> =
                        vec![("l0", &l0_results)];
                    write_results_to_csv_utest(&pca_full_path, n_cells, nbr_label, &metrics)?;
                } else {
                    println!("Running PCA mean Wilcoxon test");
                    let cos_results: Vec<(Option<(f64, f64)>, Vec<f64>, Vec<f64>)> = (0..n_cells)
                        .into_par_iter()
                        .map(|i| {
                            let (obs_cos, rand_cos) =
                                random_cell_distance(i, &pca, &neighbors[i], Metric::Cosine);
                            let utest = mann_whitney_wilcoxon(&obs_cos, &rand_cos);
                            (utest, obs_cos, rand_cos)
                        })
                        .collect();
                    let l2_results: Vec<(Option<(f64, f64)>, Vec<f64>, Vec<f64>)> = (0..n_cells)
                        .into_par_iter()
                        .map(|i| {
                            let (obs_l2, rand_l2) =
                                random_cell_distance(i, &pca, &neighbors[i], Metric::L2);
                            let utest = mann_whitney_wilcoxon(&obs_l2, &rand_l2);
                            (utest, obs_l2, rand_l2)
                        })
                        .collect();

                    let mut metrics: Vec<(&str, &[(Option<(f64, f64)>, Vec<f64>, Vec<f64>)])> =
                        vec![("cos", &cos_results), ("l2", &l2_results)];

                    let extra_results: Vec<(Option<(f64, f64)>, Vec<f64>, Vec<f64>)>;
                    let extra_label: &str;
                    match args.neighbor_by {
                        NeighborBy::Spatial => {
                            extra_results = vec![];
                            extra_label = "";
                        }
                        NeighborBy::Expression | NeighborBy::ExpressionL0 => {
                            println!("  -> L2 on XY coords...");
                            extra_results = (0..n_cells)
                                .into_par_iter()
                                .map(|i| {
                                    let (obs, rand_d) =
                                        random_cell_distance(i, &coords, &neighbors[i], Metric::L2);
                                    let utest = mann_whitney_wilcoxon(&obs, &rand_d);
                                    (utest, obs, rand_d)
                                })
                                .collect();
                            extra_label = "xyL2";
                        }
                        NeighborBy::SpatialL0 => unreachable!(), // handled above
                    }
                    if !extra_label.is_empty() {
                        metrics.push((extra_label, &extra_results));
                    }

                    write_results_to_csv_utest(&pca_full_path, n_cells, nbr_label, &metrics)?;
                }
                println!(
                    "Results saved to {} ({} cells, PCA)",
                    pca_full_path.display(),
                    n_cells
                );
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use parquet::file::reader::FileReader;

    #[test]
    fn xenium_cells_parquet_schema_matches_resolve_position_columns() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test_data/Xenium_V1_FFPE_Human_Brain_Healthy_With_Addon_outs/cells.parquet");
        if !path.exists() {
            eprintln!(
                "skip xenium_cells_parquet_schema_matches_resolve_position_columns: missing {}",
                path.display()
            );
            return;
        }
        let f = File::open(&path).expect("open cells.parquet");
        let reader = SerializedFileReader::new(f).expect("parquet reader");
        let descr = reader.metadata().file_metadata().schema_descr();
        assert_eq!(descr.column(0).name(), "cell_id");
        assert_eq!(descr.column(1).name(), "x_centroid");
        assert_eq!(descr.column(2).name(), "y_centroid");

        let (x_col, y_col) = resolve_position_columns(Some(Platform::Xenium), None, None);
        assert_eq!(
            (x_col, y_col),
            (1, 2),
            "Xenium preset must index x_centroid/y_centroid"
        );
    }

    #[test]
    fn xenium_spatial_l0_uses_spatial_neighbors_and_xenium_cols() {
        let (x_col, y_col) = resolve_position_columns(Some(Platform::Xenium), None, None);
        assert_eq!((x_col, y_col), (1, 2));

        // Coords: cell 0 is closest to cell 2 in XY.
        let coords = Array2::from_shape_vec(
            (3, 2),
            vec![
                0.0, 0.0, // cell 0
                9.0, 0.0, // cell 1
                1.0, 0.0, // cell 2
            ],
        )
        .expect("valid coords");

        // Expression matrix intentionally makes cell 1 closest in expression space.
        // If SpatialL0 uses spatial KNN, neighbor for cell 0 should still be cell 2.
        let data_raw = Array2::from_shape_vec(
            (3, 2),
            vec![
                1.0, 1.0, // cell 0
                1.0, 1.0, // cell 1 (expr-identical to cell 0)
                0.0, 0.0, // cell 2
            ],
        )
        .expect("valid expression matrix");

        // Ask HNSW for 2 so that after self-filtering we still keep one neighbor.
        let neighbors = build_neighbors_for_mode(NeighborBy::SpatialL0, &coords, &data_raw, 2);
        assert_eq!(neighbors.len(), 3);
        assert_eq!(neighbors[0][0], 2);
    }
}
