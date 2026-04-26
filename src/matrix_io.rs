use crate::mst_codec::Point;
use clap::ValueEnum;
use hdf5::types::FixedAscii;
use hdf5::File as Hdf5File;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use sprs::CsMat;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum InputPosType {
    Csv,
    Parquet,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum Platform {
    Visium,
    Xenium,
    SingleCell,
}

fn read_strings_dynamic(file: &Hdf5File, path: &str) -> anyhow::Result<Vec<String>> {
    let dataset = file.dataset(path)?;

    if let Ok(v) = dataset.read_1d::<hdf5::types::VarLenUnicode>() {
        return Ok(v.iter().map(|s| s.to_string()).collect());
    }

    let dsize = dataset.dtype()?.size();
    macro_rules! read_fixed {
        ($n:expr) => {
            if dsize == $n {
                if let Ok(v) = dataset.read_1d::<FixedAscii<$n>>() {
                    return Ok(v
                        .iter()
                        .map(|s| s.as_str().trim_end_matches('\0').to_string())
                        .collect());
                }
            }
        };
    }

    read_fixed!(6);
    read_fixed!(7);
    read_fixed!(10);
    read_fixed!(11);
    read_fixed!(15);
    read_fixed!(16);
    read_fixed!(18);
    read_fixed!(21);
    read_fixed!(23);
    read_fixed!(25);
    read_fixed!(32);
    read_fixed!(64);

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

fn load_positions_from_csv(
    pos_path: &Path,
    pos_x_col: usize,
    pos_y_col: usize,
) -> anyhow::Result<HashMap<String, (f64, f64)>> {
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
) -> anyhow::Result<HashMap<String, (f64, f64)>> {
    let pos_file = std::fs::File::open(pos_path)?;
    let reader = SerializedFileReader::new(pos_file)?;
    let mut iter = reader.get_row_iter(None)?;
    let mut pos_map = HashMap::new();

    while let Some(Ok(row)) = iter.next() {
        let barcode = match row.get_string(0) {
            Ok(s) => s.to_string(),
            Err(_) => {
                let raw = row.get_bytes(0)?;
                String::from_utf8_lossy(raw.data()).to_string()
            }
        };
        let x = row.get_double(pos_x_col)?;
        let y = row.get_double(pos_y_col)?;
        pos_map.insert(barcode, (x, y));
    }

    Ok(pos_map)
}

pub fn load_10x_with_positions(
    h5_path: &Path,
    pos_path: &Path,
    pos_type: InputPosType,
    pos_x_col: usize,
    pos_y_col: usize,
    max_cells: Option<usize>,
) -> anyhow::Result<(CsMat<u16>, Vec<Point>)> {
    let file = Hdf5File::open(h5_path)?;
    let matrix = file.group("matrix")?;

    let shape = matrix.dataset("shape")?.read_1d::<usize>()?.to_vec();
    if shape.len() != 2 {
        anyhow::bail!("matrix/shape must be length 2, found {}", shape.len());
    }

    let num_features = shape[0];
    let num_cells_total = shape[1];
    let num_cells = max_cells
        .map(|m| m.min(num_cells_total))
        .unwrap_or(num_cells_total);

    let data_all = matrix.dataset("data")?.read_1d::<u16>()?.to_vec();
    let indices_all = matrix.dataset("indices")?.read_1d::<usize>()?.to_vec();
    let indptr_all = matrix.dataset("indptr")?.read_1d::<usize>()?.to_vec();

    let nnz_limit = *indptr_all
        .get(num_cells)
        .ok_or_else(|| anyhow::anyhow!("indptr missing upper bound for {} rows", num_cells))?;

    let indptr = indptr_all[..=num_cells].to_vec();
    let indices = indices_all[..nnz_limit].to_vec();
    let values = data_all[..nnz_limit].to_vec();
    let csr = CsMat::new((num_cells, num_features), indptr, indices, values);

    let barcodes = read_strings_dynamic(&file, "matrix/barcodes")?;
    if barcodes.len() < num_cells {
        anyhow::bail!(
            "Not enough barcodes for selected rows: have {}, need {}",
            barcodes.len(),
            num_cells
        );
    }

    let pos_map = match pos_type {
        InputPosType::Csv => load_positions_from_csv(pos_path, pos_x_col, pos_y_col)?,
        InputPosType::Parquet => load_positions_from_parquet(pos_path, pos_x_col, pos_y_col)?,
    };

    let mut points = Vec::with_capacity(num_cells);
    for row_idx in 0..num_cells {
        let barcode = &barcodes[row_idx];
        let (x, y) = pos_map.get(barcode).copied().ok_or_else(|| {
            anyhow::anyhow!(
                "Missing position for barcode '{}' at row {}",
                barcode,
                row_idx
            )
        })?;
        points.push(Point::new(x, y, row_idx));
    }

    Ok((csr, points))
}

/// Load 10x H5 matrix without any external positions.
/// Returns CSR expression matrix and dummy coordinates (0,0) for each retained cell.
pub fn load_10x_no_positions(
    h5_path: &Path,
    max_cells: Option<usize>,
) -> anyhow::Result<(CsMat<u16>, Vec<Point>)> {
    let file = Hdf5File::open(h5_path)?;
    let matrix = file.group("matrix")?;

    let shape = matrix.dataset("shape")?.read_1d::<usize>()?.to_vec();
    if shape.len() != 2 {
        anyhow::bail!("matrix/shape must be length 2, found {}", shape.len());
    }

    let num_features = shape[0];
    let num_cells_total = shape[1];
    let num_cells = max_cells
        .map(|m| m.min(num_cells_total))
        .unwrap_or(num_cells_total);

    let data_all = matrix.dataset("data")?.read_1d::<u16>()?.to_vec();
    let indices_all = matrix.dataset("indices")?.read_1d::<usize>()?.to_vec();
    let indptr_all = matrix.dataset("indptr")?.read_1d::<usize>()?.to_vec();

    let nnz_limit = *indptr_all
        .get(num_cells)
        .ok_or_else(|| anyhow::anyhow!("indptr missing upper bound for {} rows", num_cells))?;

    let indptr = indptr_all[..=num_cells].to_vec();
    let indices = indices_all[..nnz_limit].to_vec();
    let values = data_all[..nnz_limit].to_vec();
    let csr = CsMat::new((num_cells, num_features), indptr, indices, values);

    let mut points = Vec::with_capacity(num_cells);
    for row_idx in 0..num_cells {
        points.push(Point::new(0.0, 0.0, row_idx));
    }

    Ok((csr, points))
}
