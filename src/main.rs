use crate::quad_tree::tree::{ErrorMetric, Point, QuadTree, Rect};
pub mod bits;
pub mod quad_tree;
use clap::{Args, Parser, Subcommand, ValueEnum};
use csv::ReaderBuilder;
use flate2::write::GzEncoder;
use flate2::Compression;
use hdf5::types::FixedAscii;
use hdf5::File as Hdf5File;
use ndarray::{Array1, ArrayD};
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use quad_tree::tree::{BitFieldQuadTree, DatalessPoint, EncodedDiffs, PointLike};
use sprs::CsMat;
use std::error::Error;
use std::fmt::Write as FmtWrite;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::info;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::fmt;
use tracing_subscriber::prelude::*;
use tracing_subscriber::EnvFilter;
//use std::thread;
//use std::time::Duration;
use flate2::bufread::GzDecoder;
use std::io::BufReader;
// removed unused bincode::{Encode, Decode} import

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

// Simple ArrayData type to replace anndata::ArrayData
#[derive(Clone)]
pub struct ArrayData {
    pub data: ArrayD<u16>,
}

impl ArrayData {
    pub fn new(data: ArrayD<u16>) -> Self {
        Self { data }
    }

    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }
}

// Sparse gene expression data - much more memory efficient!
//#[derive(Clone)]
//pub struct SparseGeneData {
//    pub gene_indices: Vec<usize>,    // Which genes are expressed
//    pub expression_values: Vec<u16>,  // Expression counts
//    pub num_genes: usize,            // Total number of genes (for dense conversion if needed)
//}
//    pub expression_values: Vec<u16>,  // Expression counts
//    pub num_genes: usize,            // Total number of genes (for dense conversion if needed)
//}
//    pub num_genes: usize,            // Total number of genes (for dense conversion if needed)
//}

//impl SparseGeneData {
//    pub fn new(gene_indices: Vec<usize>, expression_values: Vec<u16>, num_genes: usize) -> Self {
//        Self { gene_indices, expression_values, num_genes }
//    }

// Convert to dense only when needed
//    pub fn to_dense(&self) -> Vec<u16> {
//        let mut dense = vec![0u16; self.num_genes];
//        for (idx, &value) in self.gene_indices.iter().zip(&self.expression_values) {
//            dense[*idx] = value;
//        }
//        dense
//    }
//    }

// Memory usage in bytes
//    pub fn memory_usage(&self) -> usize {
//        self.gene_indices.len() * std::mem::size_of::<usize>() +
//        self.expression_values.len() * std::mem::size_of::<u16>() +
//        std::mem::size_of::<usize>()
//    }

// Get expression for a specific gene
//    pub fn get_gene_expression(&self, gene_idx: usize) -> u16 {
//        if let Some(pos) = self.gene_indices.binary_search(&gene_idx).ok() {
//            self.expression_values[pos]
//        } else {
//            0
//        }
//    }
//    }
//}

// Helper function to extract numeric data from ArrayData
fn extract_numeric_data(data: &ArrayData) -> Vec<u16> {
    data.data.as_slice().unwrap().to_vec()
}

//use af_anndata::H5 as H52;

fn read_10x_features<T: AsRef<Path>>(h5_path: T) -> anyhow::Result<Vec<String>> {
    println!(
        "Reading features from 10x HDF5 file: {}",
        h5_path.as_ref().display()
    );

    let file = Hdf5File::open(h5_path.as_ref())?;
    println!("Successfully opened HDF5 file");

    let matrix_group = file.group("matrix")?;
    println!("Found matrix group");

    let features_group = matrix_group.group("features")?;
    println!("Found features group");

    // Try to read feature names
    match features_group.dataset("name") {
        Ok(name_dataset) => {
            println!("Found name dataset, attempting to read...");
            match name_dataset.read_1d::<u8>() {
                Ok(names_bytes) => {
                    println!(
                        "Successfully read {} bytes from name dataset",
                        names_bytes.len()
                    );

                    // Convert to Vec<u8> to avoid version conflicts
                    let names_vec: Vec<u8> = names_bytes.to_vec();

                    // Convert bytes to strings - each string is 23 bytes
                    let mut names = Vec::new();
                    for i in 0..541 {
                        let start = i * 23;
                        let end = start + 23;
                        if end <= names_vec.len() {
                            let string_bytes = &names_vec[start..end];
                            // Convert bytes to string, removing null padding
                            let name = String::from_utf8_lossy(string_bytes)
                                .trim_matches('\0')
                                .to_string();
                            names.push(name);
                        }
                    }

                    println!("Successfully read {} feature names from HDF5", names.len());
                    if names.len() > 0 {
                        println!("First 5 features: {:?}", &names[..names.len().min(5)]);
                    }

                    Ok(names)
                }
                Err(e) => {
                    println!("Failed to read name dataset: {:?}", e);
                    // Fallback to placeholder features
                    let names: Vec<String> = (0..541).map(|i| format!("Gene_{}", i)).collect();
                    println!("Using {} placeholder features", names.len());
                    Ok(names)
                }
            }
        }
        Err(e) => {
            println!("Failed to find name dataset: {:?}", e);
            // Fallback to placeholder features
            let names: Vec<String> = (0..541).map(|i| format!("Gene_{}", i)).collect();
            println!("Using {} placeholder features", names.len());
            Ok(names)
        }
    }
}

// Helper functions to read fixed-length strings from HDF5
fn read_strings_7(file: &Hdf5File, dataset_path: &str) -> anyhow::Result<Vec<String>> {
    let dataset = file.dataset(dataset_path)?;
    match dataset.read_1d::<FixedAscii<7>>() {
        Ok(strings_array) => {
            let strings: Vec<String> = strings_array
                .iter()
                .map(|s| s.as_str().trim_end_matches('\0').to_string())
                .collect();
            Ok(strings)
        }
        Err(_) => read_strings_as_bytes(file, dataset_path, 7, 541),
    }
}

fn read_strings_23(file: &Hdf5File, dataset_path: &str) -> anyhow::Result<Vec<String>> {
    let dataset = file.dataset(dataset_path)?;
    match dataset.read_1d::<FixedAscii<23>>() {
        Ok(strings_array) => {
            let strings: Vec<String> = strings_array
                .iter()
                .map(|s| s.as_str().trim_end_matches('\0').to_string())
                .collect();
            Ok(strings)
        }
        Err(_) => read_strings_as_bytes(file, dataset_path, 23, 541),
    }
}

fn read_strings_25(file: &Hdf5File, dataset_path: &str) -> anyhow::Result<Vec<String>> {
    let dataset = file.dataset(dataset_path)?;
    match dataset.read_1d::<FixedAscii<25>>() {
        Ok(strings_array) => {
            let strings: Vec<String> = strings_array
                .iter()
                .map(|s| s.as_str().trim_end_matches('\0').to_string())
                .collect();
            Ok(strings)
        }
        Err(_) => read_strings_as_bytes(file, dataset_path, 25, 541),
    }
}

fn read_strings_as_bytes(
    file: &Hdf5File,
    dataset_path: &str,
    string_length: usize,
    num_strings: usize,
) -> anyhow::Result<Vec<String>> {
    let dataset = file.dataset(dataset_path)?;
    let bytes = dataset.read_1d::<u8>()?;
    let bytes_vec: Vec<u8> = bytes.to_vec();

    let mut strings = Vec::new();
    for i in 0..num_strings {
        let start = i * string_length;
        let end = start + string_length;
        if end <= bytes_vec.len() {
            let string_bytes = &bytes_vec[start..end];
            let string = String::from_utf8_lossy(string_bytes)
                .trim_matches('\0')
                .to_string();
            strings.push(string);
        }
    }

    Ok(strings)
}

fn tree_from_csv<T: AsRef<Path>>(
    file_path: T,
    idx_x: usize,
    idx_y: usize,
    idx_gene_start: usize,
    idx_gene_end: Option<usize>,
    _method: ErrorMetric,
    _lossless: bool,
) -> anyhow::Result<QuadTree> {
    let mut coords = Vec::new();
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    //let mut mind = Vec::new();
    //let mut maxd = Vec::new();

    let mut rdr = ReaderBuilder::new()
        .has_headers(true) // Set to true to skip the header row
        .flexible(true) // Allow varying number of fields
        .from_path(file_path)
        .map_err(|e| anyhow::anyhow!("Failed to open file: {}", e))?;

    // Read all records into memory
    let records: Vec<_> = rdr
        .records()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| anyhow::anyhow!("Failed to read records: {}", e))?;

    println!("Total records read: {}", records.len());

    // Get the number of columns from the first record
    let num_columns = records[0].len();
    let idx_gene_end = idx_gene_end.unwrap_or(num_columns);
    println!("num_columns: {}", num_columns);
    // Process all records
    for record in records.iter() {
        //println!("i: {:?}", i);
        // Read coordinates
        let x: f64 = record[idx_x].parse().map_err(|e| {
            anyhow::anyhow!("Failed to parse x coordinate at column {}: {}", idx_x, e)
        })?;
        let y: f64 = record[idx_y].parse().map_err(|e| {
            anyhow::anyhow!("Failed to parse y coordinate at column {}: {}", idx_y, e)
        })?;
        xs.push(x);
        ys.push(y);
        //println!("x: {}, y: {}", x, y);

        // Read gene expression data
        let mut cells = Vec::new();
        for j in idx_gene_start..idx_gene_end {
            let value: u16 = match record[j].parse::<f64>() {
                Ok(v) => v as u16,
                Err(_e) => 0,
            };
            cells.push(value);
            // println!("cells: {:?}", cells);
            /*
            if mind.len() <= j - idx_gene_start {
                mind.push(value);
                maxd.push(value);
            } else {
                mind[j - idx_gene_start] = mind[j - idx_gene_start].min(value);
                maxd[j - idx_gene_start] = maxd[j - idx_gene_start].max(value);
            }
            */
        }
        let array_data = ArrayData::new(Array1::from_vec(cells).into_dyn());
        coords.push(Point::new(x, y, Arc::new(array_data)));
    }

    let minx = xs.iter().cloned().fold(f64::INFINITY, f64::min) - 1.0;
    let miny = ys.iter().cloned().fold(f64::INFINITY, f64::min) - 1.0;
    let maxx = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 1.0;
    let maxy = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 1.0;
    let w = maxx - minx;
    let h = maxy - miny;

    let domain = Rect::new(minx + w / 2.0_f64, miny + h / 2.0_f64, w, h);
    let mut qtree = QuadTree::new(domain, coords, 0);

    // Divide the quadtree and get cost log
    let division_cost_log = qtree.divide_recursive();
    //info!("Division cost log contains {} steps", division_cost_log.steps.len());
    //info!("Total nodes processed: {}", division_cost_log.total_nodes);
    //info!("Total cost: {}", division_cost_log.total_cost);

    // Serialize division cost log to file
    //    let division_cost_log_filename = PathBuf::from("division_costs.bin");
    //    let cost_config = bincode::config::standard()
    //        .with_little_endian()
    //        .with_fixed_int_encoding();
    //    let mut division_cost_file = File::create(&division_cost_log_filename)?;
    //    bincode::encode_into_std_write(&division_cost_log, &mut division_cost_file, cost_config)?;
    //    info!("Division cost log serialized to: {}", division_cost_log_filename.display());

    /*   // Optimize the quadtree and get cost log
    let optimization_cost_log = qtree.optimize_quadtree();
    info!("Optimization cost log contains {} steps", optimization_cost_log.steps.len());
    info!("Optimization total cost: {}", optimization_cost_log.total_cost);

    // Serialize optimization cost log to file
    let optimization_cost_log_filename = PathBuf::from("optimization_costs.bin");
    let mut optimization_cost_file = File::create(&optimization_cost_log_filename)?;
    bincode::encode_into_std_write(&optimization_cost_log, &mut optimization_cost_file, cost_config)?;
    info!("Optimization cost log serialized to: {}", optimization_cost_log_filename.display());
    */
    Ok(qtree)
}

/*
fn read_parquet_file(file_path: &str) -> Result<(), ParquetError> {
    let file = File::open(Path::new(file_path))?;
    let reader = SerializedFileReader::new(file)?;
    let data = reader.read_all()?;
    return reader;
}*/

fn tree_from_10X<T: AsRef<Path>>(
    h5_path: T,
    pos_path: T,
    pos_type: InputPosType,
    file_type: InputDataType,
    pos_x_col: usize,
    pos_y_col: usize,
    _method: ErrorMetric,
    _lossless: bool,
) -> anyhow::Result<QuadTree> {
    // Read features from 10x HDF5 file
    println!(
        "Reading features from HDF5 file: {}",
        h5_path.as_ref().display()
    );
    let features = read_10x_features(&h5_path)?;
    println!(
        "Found {} features: {:?}",
        features.len(),
        &features[..features.len().min(5)]
    ); // Show first 5 features

    // Read all feature information
    let file = Hdf5File::open(h5_path.as_ref())?;

    // Read gene names (23 chars each)
    let gene_names = read_strings_23(&file, "matrix/features/name")?;

    // Read gene IDs (23 chars each)
    let gene_ids = read_strings_23(&file, "matrix/features/id")?;

    // Read feature types (25 chars each)
    let feature_types = read_strings_25(&file, "matrix/features/feature_type")?;

    // Read genome references (7 chars each)
    let genomes = read_strings_7(&file, "matrix/features/genome")?;

    println!("Gene names: {:?}", &gene_names[..5]);
    println!("Gene IDs: {:?}", &gene_ids[..5]);
    println!("Feature types: {:?}", &feature_types[..5]);
    println!("Genomes: {:?}", &genomes[..5]);
    // Check if matrix group exists
    let matrix_group = match file.group("matrix") {
        Ok(g) => g,
        Err(_) => return Err(anyhow::anyhow!("No 'matrix' group; use molecule_info path")),
    };

    // Read the shape of the matrix
    let shape_dataset = matrix_group.dataset("shape")?;
    let shape_array = shape_dataset.read_1d::<usize>()?;
    let shape: Vec<usize> = shape_array.to_vec();
    let num_features = shape[0];
    let num_cells = shape[1];

    println!(
        "Matrix shape: {} cells x {} features",
        num_cells, num_features
    );

    //TODO: read from h5ad file need to be a function by itself
    // Read the sparse matrix components
    let data_dataset = matrix_group.dataset("data")?;
    let indices_dataset = matrix_group.dataset("indices")?;
    let indptr_dataset = matrix_group.dataset("indptr")?;
    let barcodes_dataset = matrix_group.dataset("barcodes")?;

    let data_array = data_dataset.read_1d::<u16>()?;
    let data_u16: Vec<u16> = data_array.to_vec();
    let data_i32: Vec<i32> = data_u16.iter().map(|&v| v as i32).collect();
    let indices_array = indices_dataset.read_1d::<usize>()?;
    let indices: Vec<usize> = indices_array.to_vec();
    let indptr_array = indptr_dataset.read_1d::<usize>()?;
    let indptr: Vec<usize> = indptr_array.to_vec();

    info!("Sparse matrix data: {} non-zero elements", data_u16.len());
    // Build CSR matrix
    let csr: CsMat<i32> = CsMat::new(
        (num_cells, num_features),
        indptr.clone(),
        indices.clone(),
        data_i32,
    );
    let pos_file = std::fs::File::open(pos_path.as_ref()).unwrap();
    let mut coords = Vec::new();
    let mut xs = Vec::new();
    let mut ys = Vec::new();

    // Pre-allocate vectors with capacity
    coords.reserve(num_cells);
    xs.reserve(num_cells);
    ys.reserve(num_cells);

    // Read all barcodes from HDF5 once
    let barcodes_arr = barcodes_dataset.read_1d::<FixedAscii<23>>()?;
    let barcodes: Vec<String> = barcodes_arr
        .iter()
        .map(|b| b.as_str().trim_end_matches('\0').to_string())
        .collect();

    // Build a map from barcode -> (x, y) from the positions file
    use std::collections::HashMap;
    let mut pos_map: HashMap<String, (f64, f64)> = HashMap::with_capacity(num_cells);

    match pos_type {
        InputPosType::Csv => {
            // Visium tissue_positions_list.csv has no header and columns:
            // [barcode, in_tissue, array_row, array_col, pxl_col_in_fullres, pxl_row_in_fullres]
            let mut rdr = csv::ReaderBuilder::new()
                .has_headers(true)
                .has_headers(false)
                .from_reader(pos_file);
            for rec in rdr.records() {
                let rec = rec?;
                if rec.len() <= pos_y_col {
                    return Err(anyhow::anyhow!(
                        "Invalid number of columns in positions file"
                    ));
                }
                let bc = rec.get(0).unwrap().to_string();
                let x_str = rec
                    .get(pos_x_col)
                    .ok_or_else(|| anyhow::anyhow!("Missing x column {}", pos_x_col))?;
                let y_str = rec
                    .get(pos_y_col)
                    .ok_or_else(|| anyhow::anyhow!("Missing y column {}", pos_y_col))?;
                let x: f64 = x_str.parse()?;
                let y: f64 = y_str.parse()?;
                pos_map.insert(bc, (x, y));
            }
        }
        InputPosType::Parquet => {
            let reader = SerializedFileReader::new(pos_file).unwrap();
            let mut iter = reader.get_row_iter(None).unwrap();
            while let Some(Ok(parquet_row)) = iter.next() {
                //println!("parquet_row: {:?}", parquet_row.get_string(0));
                let bc = parquet_row.get_string(0).unwrap().to_string();
                let x = parquet_row.get_double(pos_x_col).unwrap();
                let y = parquet_row.get_double(pos_y_col).unwrap();
                pos_map.insert(bc, (x, y));
            }
        }
    }

    // Walk HDF5 barcodes in order and look up positions by barcode
    for row_idx in 0..num_cells {
        let barcode = &barcodes[row_idx];
        let (x_coord, y_coord) = pos_map.get(barcode).copied().ok_or_else(|| {
            anyhow::anyhow!(
                "Missing position for HDF5 barcode '{}' at row {} in positions file",
                barcode,
                row_idx
            )
        })?;

        xs.push(x_coord);
        ys.push(y_coord);

        // Extract gene expression data for this cell from the sparse matrix
        let start_idx = indptr[row_idx];
        let end_idx = indptr[row_idx + 1];

        // Create sparse representation - only store non-zero values
        let mut gene_indices = Vec::new();
        let mut expression_values = Vec::new();
        for i in start_idx..end_idx {
            let gene_idx = indices[i];
            let expression_value = data_u16[i];
            if expression_value > 0 {
                gene_indices.push(gene_idx);
                expression_values.push(expression_value);
            }
        }

        // let sparse_data = SparseGeneData::new(gene_indices, expression_values, num_features);
        /*
        if row_idx < 5 {
            let dense_memory = num_features * std::mem::size_of::<u16>();
            let sparse_memory = sparse_data.memory_usage();
            println!(
                "Cell {}: Dense={} bytes, Sparse={} bytes, Savings={:.1}%",
                row_idx,
                dense_memory,
                sparse_memory,
                (1.0 - sparse_memory as f64 / dense_memory as f64) * 100.0
            );
        }
        */
        //  let array_data = ArrayData::new(csr.row(row_idx));
        let array_data = ArrayData::new(Array1::from_vec(expression_values).into_dyn());
        coords.push(Point::new(x_coord, y_coord, Arc::new(array_data)));
    }

    info!("num_rows: {}", num_cells);

    let minx = xs.iter().fold(f64::INFINITY, |a, &b| a.min(b)) - 1.0; // |a, &b| pattern matching, does not need clone()
    let miny = ys.iter().fold(f64::INFINITY, |a, &b| a.min(b)) - 1.0;
    let maxx = xs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)) + 1.0;
    let maxy = ys.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)) + 1.0;
    let w = maxx - minx;
    let h = maxy - miny;

    let domain = Rect::new(minx + w / 2.0_f64, miny + h / 2.0_f64, w, h);
    let mut qtree = QuadTree::new(domain, coords, 0);

    // Divide the quadtree and get cost log
    let division_cost_log = qtree.divide_recursive();
    /*
    info!("Division cost log contains {} steps", division_cost_log.steps.len());
    info!("Total nodes processed: {}", division_cost_log.total_nodes);
    info!("Total cost: {}", division_cost_log.total_cost);

    // Serialize division cost log to file
    let division_cost_log_filename = PathBuf::from("division_costs.bin");
    let cost_config = bincode::config::standard()
        .with_little_endian()
        .with_fixed_int_encoding();
    let mut division_cost_file = File::create(&division_cost_log_filename)?;
    bincode::encode_into_std_write(&division_cost_log, &mut division_cost_file, cost_config)?;
    info!("Division cost log serialized to: {}", division_cost_log_filename.display());

    // Optimize the quadtree and get cost log
    let optimization_cost_log = qtree.optimize_quadtree();
    info!("Optimization cost log contains {} steps", optimization_cost_log.steps.len());
    info!("Optimization total cost: {}", optimization_cost_log.total_cost);

    // Serialize optimization cost log to file
    let optimization_cost_log_filename = PathBuf::from("optimization_costs.bin");
    let mut optimization_cost_file = File::create(&optimization_cost_log_filename)?;
    bincode::encode_into_std_write(&optimization_cost_log, &mut optimization_cost_file, cost_config)?;
    info!("Optimization cost log serialized to: {}", optimization_cost_log_filename.display());
    */
    Ok(qtree)
}

#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    #[command(arg_required_else_help = true)]
    Build(BuildCommand),
    #[command(arg_required_else_help = true)]
    Dump(DumpCommand),
}

#[derive(Debug, Clone, ValueEnum)]
enum InputPosType {
    Csv,
    Parquet,
}

#[derive(Debug, Clone, ValueEnum)]
enum InputDataType {
    Csr,
    Csv,
    H5ad,
    Mtx,
    v2,
}

enum InputFileType {
    v2,
    HD,
}

#[derive(Debug, Clone, ValueEnum)]
enum Platform {
    Visium,
    Xenium,
}

/// Build a quadtree representation of spatial transcriptomics data
#[derive(Debug, Args)]
#[command(version, about, long_about = None)]
struct DumpCommand {
    /// the output serialized from build
    #[arg(short = 'i', long)]
    input: PathBuf,
    /// Output file
    #[arg(short = 'o', long)]
    output: PathBuf,
}

/// Build a quadtree representation of spatial transcriptomics data
#[derive(Debug, Args)]
#[command(version, about, long_about = None)]
struct BuildCommand {
    /// Input file (CSV or HDF5)
    #[arg(short = 'i', long)]
    input: PathBuf,
    /// Input CSV file for position data (only needed for CSV input)
    #[arg(short = 'p')]
    input_pos: Option<PathBuf>,
    /// Output file (default "output.bin.gz")
    #[arg(short = 'o', long)]
    output: Option<PathBuf>,
    /// Input file format (csv or csr, csc, h5ad, mtx)
    #[arg(short = 'f', long, value_enum, default_value_t = InputDataType::Csr)]
    format: InputDataType,
    /// Positions file x column index (0-based)
    #[arg(long = "pos-x-col")]
    pos_x_col: Option<usize>,
    /// Positions file y column index (0-based)
    #[arg(long = "pos-y-col")]
    pos_y_col: Option<usize>,
    // !!combine pos_x_col and pos_y_col into one argument!!
    /// Index of x coordinate, default 6
    #[arg(short = 'x', long, default_value_t = 6)]
    idx_x: usize,
    /// Index of y coordinate, default 7
    #[arg(short = 'y', long, default_value_t = 7)]
    idx_y: usize,
    /// Index of gene start, default 10
    #[arg(short = 's', long, default_value_t = 10)]
    idx_gene_start: usize,
    /// Index of gene end, default all remaining columns
    #[arg(short = 'e', long)]
    idx_gene_end: Option<usize>,
    /// Input position file format (csv or parquet)
    #[arg(short = 'F', long = "pos-format", value_enum, default_value_t = InputPosType::Parquet)]
    pos_format: InputPosType,
    #[arg(short = 'P', long = "platform", value_enum)]
    platform: Option<Platform>,
}

struct Data {
    pub data: Vec<EncodedDiffs>,
    pub pos: Vec<DatalessPoint>,
    // pub sep: Vec<usize>,
}

impl Data {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            pos: Vec::new(),
            //  sep: Vec::new(),
        }
    }
}
/*
fn main() -> Result<(), Box<dyn Error>> {
    // Check the `RUST_LOG` variable for the logger level and
    // respect the value found there. If this environment
    // variable is not set then set the logging level to
    // INFO.
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(
            EnvFilter::builder()
                .with_default_directive(LevelFilter::INFO.into())
                .from_env_lossy()
                // we don't want to hear anything below a warning from ureq
                .add_directive("ureq=warn".parse()?),
        )
        .init();

    // Test reading HDF5 file directly with hdf5 crate
    println!("Testing HDF5 file reading...");
    let file_path = "/Users/zhezhenwang/Documents/patro/data/Xenium_V1_hKidney_nondiseased_section_outs/cell_feature_matrix.h5";
    let parquet_path = "/Users/zhezhenwang/Documents/patro/data/Xenium_V1_hKidney_nondiseased_section_outs/cells.parquet";
    tree_from_10X(file_path, parquet_path, ErrorMetric::Mean, true)?;
    Ok(())
}
    */

fn main() -> Result<(), Box<dyn Error>> {
    // Check the `RUST_LOG` variable for the logger level and
    // respect the value found there. If this environment
    // variable is not set then set the logging level to
    // INFO.
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(
            EnvFilter::builder()
                .with_default_directive(LevelFilter::INFO.into())
                .from_env_lossy()
                // we don't want to hear anything below a warning from ureq
                .add_directive("ureq=warn".parse()?),
        )
        .init();

    let cli_args = Cli::parse();

    match cli_args.command {
        Commands::Build(args) => {
            //TODO: combine csv and 10x format function into one
            let (pos_x_col, pos_y_col) = match args.platform {
                Some(Platform::Visium) => (4, 5),
                Some(Platform::Xenium) => (1, 2),
                None => (args.idx_x, args.idx_y),
            };
            let qtree = match args.format {
                InputDataType::Csv => {
                    // let file_path_pos = args.input_pos.ok_or_else(|| {
                    //    anyhow::anyhow!("Position file required for CSV format")
                    //})?;
                    //println!("tree_from_csv");
                    tree_from_csv(
                        &args.input,
                        //file_path_pos,
                        args.idx_x,          // idx_x
                        args.idx_y,          // idx_y
                        args.idx_gene_start, // idx_gene_start
                        args.idx_gene_end,   // idx_gene_end (will use all remaining columns)
                        ErrorMetric::Mean,
                        true,
                    )?
                }
                InputDataType::Csr
                | InputDataType::H5ad
                | InputDataType::Mtx
                | InputDataType::v2 => {
                    let file_path_pos = args
                        .input_pos
                        .ok_or_else(|| anyhow::anyhow!("Position file required for HDF5 format"))?;
                    tree_from_10X(
                        &args.input,
                        &file_path_pos,
                        match args.pos_format {
                            InputPosType::Csv => InputPosType::Csv,
                            InputPosType::Parquet => InputPosType::Parquet,
                        },
                        match args.format {
                            InputDataType::Csr => InputDataType::Csr,
                            InputDataType::Csv => InputDataType::Csv,
                            InputDataType::H5ad => InputDataType::H5ad,
                            InputDataType::Mtx => InputDataType::Mtx,
                            InputDataType::v2 => InputDataType::v2,
                        },
                        pos_x_col,
                        pos_y_col,
                        ErrorMetric::Mean,
                        true,
                    )?
                }
            };

            // only serialize the the bit fields
            let config = bincode::config::standard()
                .with_little_endian()
                .with_fixed_int_encoding();
            let ofname = args.output.unwrap_or(PathBuf::from("output.bin.gz"));
            let file = File::create(ofname).unwrap();
            let writer = BufWriter::new(file);
            let mut encoder = GzEncoder::new(writer, Compression::default());
            let bit_field_tree = qtree.compute_quadtree_bit_fields();
            let mut d = Data::new();

            let mut collect_data = |n: &BitFieldQuadTree| {
                if n.encoded_diffs.bytes() > 0 {
                    d.data.push(n.encoded_diffs.clone());
                    d.pos.extend_from_slice(&n.positions);
                }
            };
            bit_field_tree.visit(&mut collect_data);
            info!(
                "QuadTree Blocks: (non-zero blocks: {})",
                qtree.non_zero_blocks()
            );
            info!("Collected Encoded Diffs : {}", d.data.len());
            bincode::encode_into_std_write(&d.data, &mut encoder, config).unwrap();
            bincode::encode_into_std_write(&d.pos, &mut encoder, config).unwrap();
        }
        Commands::Dump(args) => {
            info!("start dump");
            let ifile = std::fs::File::open(args.input)?;
            let ifile = std::io::BufReader::new(ifile);
            let gz = GzDecoder::new(ifile);
            let mut rdr = BufReader::new(gz);

            let config = bincode::config::standard()
                .with_little_endian()
                .with_fixed_int_encoding();

            let ofile = std::fs::File::create(args.output)?;
            let mut ofile = std::io::BufWriter::new(ofile);
            let mut d = Data::new();
            d.data = bincode::decode_from_std_read(&mut rdr, config)?;
            d.pos = bincode::decode_from_std_read(&mut rdr, config)?;
            let mut start = 0;

            info!("d.data.len(): {}", d.data.len());
            let mut str_out = String::new();
            for compressed_diffs in d.data.iter() {
                let n = compressed_diffs.num_cells();
                for (cell_id, loc) in d.pos.iter().skip(start).take(n).enumerate() {
                    str_out.clear();
                    for e in compressed_diffs
                        .expression_vec_iter()
                        .nth(cell_id)
                        .unwrap_or_default()
                        .iter()
                    {
                        let _ = write!(&mut str_out, "{e}, ");
                    }
                    str_out.pop();
                    str_out.pop();
                    ofile.write_all(
                        format!("{},{},{}\n", loc.xpos(), loc.ypos(), str_out).as_bytes(),
                    )?;
                    //writeln!(ofile, "{},{},{}", loc.xpos(), loc.ypos(), expression)?;
                }
                start += n;
            }
        }
    }
    Ok(())
}

// Function to explore HDF5 file structure
fn explore_hdf5_structure<T: AsRef<Path>>(h5_path: T) -> anyhow::Result<()> {
    let file = Hdf5File::open(h5_path.as_ref())?;

    println!("=== HDF5 File Structure ===");
    println!("File: {}", h5_path.as_ref().display());

    // Try to access common paths
    let common_paths = [
        "matrix/obs/name",
        "matrix/obs/_index",
        "obs/name",
        "matrix/features/name",
        "matrix/data",
        "matrix/indices",
        "matrix/indptr",
        "matrix/shape",
    ];

    for path in &common_paths {
        if let Ok(dataset) = file.dataset(path) {
            let shape = dataset.shape();
            let dtype = dataset.dtype()?;
            println!("✅ {} (shape: {:?}, dtype: {:?})", path, shape, dtype);
        } else {
            println!("❌ {} (not found)", path);
        }
    }

    println!("=== End Structure ===");

    Ok(())
}

// Simple function to demonstrate reading cell names from HDF5
fn demonstrate_cell_names<T: AsRef<Path>>(h5_path: T) -> anyhow::Result<()> {
    println!("\n=== Trying to read cell names from HDF5 ===");

    let file = Hdf5File::open(h5_path.as_ref())?;

    // Try common paths for cell names
    let cell_name_paths = [
        "matrix/obs/name",
        "matrix/obs/_index",
        "obs/name",
        "matrix/obs/id",
    ];

    for path in &cell_name_paths {
        match file.dataset(path) {
            Ok(dataset) => {
                let shape = dataset.shape();
                let dtype = dataset.dtype()?;
                println!(
                    "✅ Found cell names at '{}' (shape: {:?}, dtype: {:?})",
                    path, shape, dtype
                );

                // Try to read a few sample cell names
                if shape.len() == 1 && shape[0] > 0 {
                    let num_cells = shape[0];
                    let sample_size = std::cmp::min(5, num_cells);

                    // Try to read as strings
                    if let Ok(strings) = read_strings_as_bytes(&file, path, 23, sample_size) {
                        println!("   Sample cell names: {:?}", strings);
                    } else {
                        println!("   Could not read as strings");
                    }
                }
            }
            Err(_) => {
                println!("❌ No cell names found at '{}'", path);
            }
        }
    }

    println!("=== End cell names exploration ===\n");
    Ok(())
}
