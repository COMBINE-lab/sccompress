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
    _file_path: T,
    _idx_x: usize,
    _idx_y: usize,
    _idx_gene_start: usize,
    _idx_gene_end: Option<usize>,
    _method: ErrorMetric,
    _lossless: bool,
) -> anyhow::Result<()> {
    // CSV function needs to be implemented following the same pattern
    todo!("CSV function needs to be restructured to use SpatialData pattern")
}

/*
fn read_parquet_file(file_path: &str) -> Result<(), ParquetError> {
    let file = File::open(Path::new(file_path))?;
    let reader = SerializedFileReader::new(file)?;
    let data = reader.read_all()?;
    return reader;
}*/

// Function to read 10X data and return owned data
fn read_10x_data<T: AsRef<Path>>(
    h5_path: T,
    pos_path: T,
    pos_type: InputPosType,
    pos_x_col: usize,
    pos_y_col: usize,
) -> anyhow::Result<(CsMat<u16>, Vec<(f64, f64)>)> {
    // Read HDF5 and build CSR matrix
    let file = Hdf5File::open(h5_path.as_ref())?;
    let matrix_group = match file.group("matrix") {
        Ok(g) => g,
        Err(_) => return Err(anyhow::anyhow!("No 'matrix' group; use molecule_info path")),
    };
    
    let shape_dataset = matrix_group.dataset("shape")?;
    let shape_array = shape_dataset.read_1d::<usize>()?;
    let shape: Vec<usize> = shape_array.to_vec();
    let num_features = shape[0];
    let num_cells = shape[1];
    
    let data_dataset = matrix_group.dataset("data")?;
    let indices_dataset = matrix_group.dataset("indices")?;
    let indptr_dataset = matrix_group.dataset("indptr")?;
    let barcodes_dataset = matrix_group.dataset("barcodes")?;
    
    let data_array = data_dataset.read_1d::<u16>()?;
    let data_u16: Vec<u16> = data_array.to_vec();
    let indices_array = indices_dataset.read_1d::<usize>()?;
    let indices: Vec<usize> = indices_array.to_vec();
    let indptr_array = indptr_dataset.read_1d::<usize>()?;
    let indptr: Vec<usize> = indptr_array.to_vec();
    
    info!("Sparse matrix data: {} non-zero elements", data_u16.len());
    
    // Build CSR matrix
    let csr: CsMat<u16> = CsMat::new((num_cells, num_features), indptr, indices, data_u16);
    
    // Read barcodes
    let barcodes_arr = barcodes_dataset.read_1d::<FixedAscii<23>>()?;
    let barcodes: Vec<String> = barcodes_arr
        .iter()
        .map(|b| b.as_str().trim_end_matches('\0').to_string())
        .collect();
    
    // Read positions
    let pos_file = std::fs::File::open(pos_path.as_ref())?;
    let mut pos_map: std::collections::HashMap<String, (f64, f64)> = 
        std::collections::HashMap::with_capacity(num_cells);

    match pos_type {
        InputPosType::Csv => {
            let mut rdr = csv::ReaderBuilder::new()
                .has_headers(false)
                .from_reader(pos_file);
            for rec in rdr.records() {
                let rec = rec?;
                if rec.len() <= pos_y_col {
                    return Err(anyhow::anyhow!("Invalid number of columns in positions file"));
                }
                let bc = rec.get(0).unwrap().to_string();
                let x_str = rec.get(pos_x_col)
                    .ok_or_else(|| anyhow::anyhow!("Missing x column {}", pos_x_col))?;
                let y_str = rec.get(pos_y_col)
                    .ok_or_else(|| anyhow::anyhow!("Missing y column {}", pos_y_col))?;
                let x: f64 = x_str.parse()?;
                let y: f64 = y_str.parse()?;
                pos_map.insert(bc, (x, y));
            }
        }
        InputPosType::Parquet => {
            let reader = SerializedFileReader::new(pos_file)?;
            let mut iter = reader.get_row_iter(None)?;
            while let Some(Ok(parquet_row)) = iter.next() {
                let bc = parquet_row.get_string(0)?.to_string();
                let x = parquet_row.get_double(pos_x_col)?;
                let y = parquet_row.get_double(pos_y_col)?;
                pos_map.insert(bc, (x, y));
            }
        }
    }
    
    // Create positions vector in the same order as barcodes
    let mut positions = Vec::with_capacity(num_cells);
    for (row_idx, barcode) in barcodes.iter().enumerate() {
        let (x_coord, y_coord) = pos_map.get(barcode).copied().ok_or_else(|| {
            anyhow::anyhow!(
                "Missing position for HDF5 barcode '{}' at row {} in positions file",
                barcode,
                row_idx
            )
        })?;
        positions.push((x_coord, y_coord));
    }
    
    Ok((csr, positions))
}

// Function to build quadtree from CSR matrix and positions
fn build_quadtree_from_data<'a>(csr: &'a CsMat<u16>, positions: &[(f64, f64)]) -> anyhow::Result<QuadTree<'a>> {
    let mut coords = Vec::new();
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    
    let num_cells = csr.rows();
    coords.reserve(num_cells);
    xs.reserve(num_cells);
    ys.reserve(num_cells);
    
    // Create Points that borrow from the CSR matrix
    for (row_idx, &(x_coord, y_coord)) in positions.iter().enumerate() {
        xs.push(x_coord);
        ys.push(y_coord);
        
        if let Some(row_view) = csr.outer_view(row_idx) {
            coords.push(Point::new(x_coord, y_coord, row_view));
        }
    }
    
    info!("num_rows: {}", num_cells);

    let minx = xs.iter().fold(f64::INFINITY, |a, &b| a.min(b)) - 1.0;
    let miny = ys.iter().fold(f64::INFINITY, |a, &b| a.min(b)) - 1.0;
    let maxx = xs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)) + 1.0;
    let maxy = ys.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)) + 1.0;
    let w = maxx - minx;
    let h = maxy - miny;

    let domain = Rect::new(minx + w / 2.0_f64, miny + h / 2.0_f64, w, h);
    let mut qtree = QuadTree::new(domain, coords, 0);
    
    // Divide the quadtree
    qtree.divide_recursive();
    Ok(qtree)
}

// Function to serialize a quadtree to file
fn serialize_quadtree(bit_field_tree: BitFieldQuadTree, output_path: &Option<PathBuf>) -> anyhow::Result<()> {
    let config = bincode::config::standard()
        .with_little_endian()
        .with_fixed_int_encoding();
    let ofname = output_path.as_ref()
        .cloned()
        .unwrap_or_else(|| PathBuf::from("output.bin.gz"));
    let file = File::create(&ofname)?;
    let writer = BufWriter::new(file);
    let mut encoder = GzEncoder::new(writer, Compression::default());
    
    let mut d = Data::new();
    
    let mut collect_data = |n: &BitFieldQuadTree| {
        if n.encoded_diffs.bytes() > 0 {
            d.data.push(n.encoded_diffs.clone());
            d.pos.extend_from_slice(&n.positions);
        }
    };
    bit_field_tree.visit(&mut collect_data);
    
    info!("Collected Encoded Diffs : {}", d.data.len());
    bincode::encode_into_std_write(&d.data, &mut encoder, config)?;
    bincode::encode_into_std_write(&d.pos, &mut encoder, config)?;
    
    info!("Quadtree serialized to: {}", ofname.display());
    Ok(())
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
            let (pos_x_col, pos_y_col) = match args.platform {
                Some(Platform::Visium) => (4, 5),
                Some(Platform::Xenium) => (1, 2),
                None => (args.idx_x, args.idx_y),
            };
            
            // Process the quadtree building and serialization within the same scope
            // where the data lives, avoiding lifetime issues
            match args.format {
                InputDataType::Csv => {
                    tree_from_csv(
                        &args.input,
                        args.idx_x,          // idx_x
                        args.idx_y,          // idx_y
                        args.idx_gene_start, // idx_gene_start
                        args.idx_gene_end,   // idx_gene_end (will use all remaining columns)
                        ErrorMetric::Mean,
                        true,
                    )?;
                }
                InputDataType::Csr | InputDataType::H5ad | InputDataType::Mtx | InputDataType::v2 => {
                    let file_path_pos = args
                        .input_pos
                        .ok_or_else(|| anyhow::anyhow!("Position file required for HDF5 format"))?;
                    
                    // Read data and process quadtree immediately
                    let (csr, positions) = read_10x_data(
                        &args.input,
                        &file_path_pos,
                        match args.pos_format {
                            InputPosType::Csv => InputPosType::Csv,
                            InputPosType::Parquet => InputPosType::Parquet,
                        },
                        pos_x_col,
                        pos_y_col,
                    )?;
                    
                    // Build and serialize quadtree immediately while data is in scope
                    let qtree = build_quadtree_from_data(&csr, &positions)?;
                    let bit_field_tree = qtree.compute_quadtree_bit_fields();
                    serialize_quadtree(bit_field_tree, &args.output)?;
                }
            }
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


/* 
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
*/
