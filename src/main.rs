use crate::quad_tree::tree::{ErrorMetric, Point, QuadTree, Rect};
pub mod bits;
pub mod quad_tree;
use clap::{Args, Parser, Subcommand};
use csv::ReaderBuilder;
use quad_tree::tree::{DatalessPoint, EncodedDiffs};
use std::error::Error;
use std::path::{Path, PathBuf};
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::fmt;
use tracing_subscriber::prelude::*;
use tracing_subscriber::EnvFilter;
use std::sync::Arc;
use hdf5::{File as Hdf5File, Group, Dataset};
use ndarray::{Array1, ArrayD, s};

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

//use af_anndata::H5 as H52;
use std::fs::File;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;

fn read_10x_features<T: AsRef<Path>>(h5_path: T) -> anyhow::Result<Vec<String>> {
    println!("Reading features from 10x HDF5 file: {}", h5_path.as_ref().display());
    
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
                    println!("Successfully read {} bytes from name dataset", names_bytes.len());
                    
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
    let mut data_store = std::collections::HashMap::new();
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
        // Store data in data_store and use cell_id
        let cell_id = coords.len() as u32;
        let array_data = ArrayData::new(Array1::from_vec(cells).into_dyn());
        data_store.insert(cell_id, array_data);
        // Pass the data directly to Point
        coords.push(Point::new(x, y, Arc::new(data_store[&cell_id].clone()), 0..data_store[&cell_id].shape()[0]));
    }

    let minx = xs.iter().cloned().fold(f64::INFINITY, f64::min) - 1.0;
    let miny = ys.iter().cloned().fold(f64::INFINITY, f64::min) - 1.0;
    let maxx = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 1.0;
    let maxy = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 1.0;
    let w = maxx - minx;
    let h = maxy - miny;

    let domain = Rect::new(minx + w / 2.0_f64, miny + h / 2.0_f64, w, h);

    // No need to convert since we're already using u16
    let mut qtree = QuadTree::new(domain, coords, 0, data_store);
    qtree.divide();
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
    parquet_path: T,
    _method: ErrorMetric,
    _lossless: bool,
) -> anyhow::Result<QuadTree> {
    // Read features from 10x HDF5 file
    println!("Reading features from HDF5 file: {}", h5_path.as_ref().display());
    let features = read_10x_features(&h5_path)?;
    println!("Found {} features: {:?}", features.len(), &features[..features.len().min(5)]); // Show first 5 features
    
    // Read the sparse matrix data from HDF5
    let file = Hdf5File::open(h5_path.as_ref())?;
    let matrix_group = file.group("matrix")?;
    
    // Read the shape of the matrix
    let shape_dataset = matrix_group.dataset("shape")?;
    let shape_array = shape_dataset.read_1d::<usize>()?;
    let shape: Vec<usize> = shape_array.to_vec();
    let num_cells = shape[0];
    let num_features = shape[1];
    
    println!("Matrix shape: {} cells x {} features", num_cells, num_features);
    
    // Read the sparse matrix components
    let data_dataset = matrix_group.dataset("data")?;
    let indices_dataset = matrix_group.dataset("indices")?;
    let indptr_dataset = matrix_group.dataset("indptr")?;
    
    let data_array = data_dataset.read_1d::<u16>()?;
    let data: Vec<u16> = data_array.to_vec();
    let indices_array = indices_dataset.read_1d::<usize>()?;
    let indices: Vec<usize> = indices_array.to_vec();
    let indptr_array = indptr_dataset.read_1d::<usize>()?;
    let indptr: Vec<usize> = indptr_array.to_vec();
    
    println!("Sparse matrix data: {} non-zero elements", data.len());
    
    println!("Reading from parquet file: {}", parquet_path.as_ref().display());
    
    let coords = std::fs::File::open(parquet_path.as_ref()).unwrap();
    let reader = SerializedFileReader::new(coords).unwrap();
    let mut coords = Vec::new();
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut data_store = std::collections::HashMap::new();
    
    // Pre-allocate vectors with capacity
    coords.reserve(num_cells);
    xs.reserve(num_cells);
    ys.reserve(num_cells);
    
    // Iterate through parquet rows
    let mut iter = reader.get_row_iter(None).unwrap();
    let mut row_idx = 0;
    
    while let Some(Ok(parquet_row)) = iter.next() {
        // Extract spatial coordinates from parquet
        let x_coord = parquet_row.get_double(1).unwrap(); // Get x coordinate as f64
        let y_coord = parquet_row.get_double(2).unwrap(); // Get y coordinate as f64
        
        xs.push(x_coord);
        ys.push(y_coord);
        
        // Extract gene expression data for this cell from the sparse matrix
        let start_idx = indptr[row_idx];
        let end_idx = indptr[row_idx + 1];
        
        // Create a dense vector for this cell's gene expression
        let mut cell_expression = vec![0u16; num_features];
        
        for i in start_idx..end_idx {
            let gene_idx = indices[i];
            let expression_value = data[i];
            cell_expression[gene_idx] = expression_value;
        }
        
        // Create ArrayData for this cell
        let array_data = ArrayData::new(Array1::from_vec(cell_expression).into_dyn());
        
        // Store data in data_store and use cell_id
        let cell_id = row_idx as u32;
        data_store.insert(cell_id, array_data);
        // Pass the data directly to Point
        coords.push(Point::new(x_coord, y_coord, Arc::new(data_store[&cell_id].clone()), 0..data_store[&cell_id].shape()[0]));
        
        row_idx += 1;
    }
    
    println!("num_rows: {}", row_idx);

    let minx = xs.iter().fold(f64::INFINITY, |a, &b| a.min(b)) - 1.0; // |a, &b| pattern matching, does not need clone() 
    let miny = ys.iter().fold(f64::INFINITY, |a, &b| a.min(b)) - 1.0;
    let maxx = xs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)) + 1.0;
    let maxy = ys.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)) + 1.0;
    let w = maxx - minx;
    let h = maxy - miny;

    let domain = Rect::new(minx + w / 2.0_f64, miny + h / 2.0_f64, w, h);
    let mut qtree = QuadTree::new(domain, coords, 0, data_store);
    qtree.divide();
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
    #[arg(short = 'p', long)]
    input_pos: Option<PathBuf>,
    /// Output file (default "output.bin.gz")
    #[arg(short = 'o', long)]
    output: Option<PathBuf>,
    /// Input file format (csv or hdf5)
    #[arg(short = 'f', long, default_value = "csv")]
    format: String,
    /// Index of x coordinate, default 5
    #[arg(short = 'x', long, default_value_t = 6)]
    idx_x: usize,
    /// Index of y coordinate, default 6
    #[arg(short = 'y', long, default_value_t = 7)]
    idx_y: usize,
    /// Index of gene start, default 1
    #[arg(short = 's', long, default_value_t = 11)]
    idx_gene_start: usize,
    /// Index of gene end, default all remaining columns
    #[arg(short = 'e', long)]
    idx_gene_end: Option<usize>,
}

struct Data {
    pub data: Vec<EncodedDiffs>,
    pub pos: Vec<DatalessPoint>,
}

impl Data {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            pos: Vec::new(),
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

    // Test reading HDF5 file directly with hdf5 crate
    println!("Testing HDF5 file reading...");
    let file_path = "/Users/zhezhenwang/Documents/patro/data/Xenium_V1_hKidney_nondiseased_section_outs/cell_feature_matrix.h5";
    let parquet_path = "/Users/zhezhenwang/Documents/patro/data/Xenium_V1_hKidney_nondiseased_section_outs/cells.parquet";
    tree_from_10X(file_path, parquet_path, ErrorMetric::Mean, true)?;
    Ok(())
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

    let cli_args = Cli::parse();

    match cli_args.command {
        Commands::Build(args) => {
            let file_path = args.input;
            let qtree = match args.format.as_str() {
                "csv" => {
                    // let file_path_pos = args.input_pos.ok_or_else(|| {
                    //    anyhow::anyhow!("Position file required for CSV format")
                    //})?;
                    tree_from_csv(
                        file_path,
                        //file_path_pos,
                        args.idx_x,          // idx_x
                        args.idx_y,          // idx_y
                        args.idx_gene_start, // idx_gene_start
                        args.idx_gene_end,   // idx_gene_end (will use all remaining columns)
                        ErrorMetric::Mean,
                        true,
                    )?
                }
                // "hdf5" => {
                // let file_path_pos = args.input_pos.ok_or_else(|| {
                //   anyhow::anyhow!("Position file required for HDF5 format")
                //})?;
                //tree_from_h5(
                //  &args.input,
                //file_path_pos,
                //ErrorMetric::Mean,
                //true,
                //"10x"
                //)?
                //}
                _ => return Err(anyhow::anyhow!("Unsupported format: {}", args.format).into()),
            };

            // You can decide whether to serialize the original tree or the bit fields or both
            let config = bincode::config::standard()
                .with_little_endian()
                .with_fixed_int_encoding();
            let ofname = args.output.unwrap_or(PathBuf::from("output.bin.gz"));
            let mut file = StdFile::create(ofname).unwrap();
            //let mut encoder = GzEncoder::new(file, Compression::default());
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
            bincode::encode_into_std_write(&d.data, &mut file, config).unwrap();
            bincode::encode_into_std_write(&d.pos, &mut file, config).unwrap();
        }
        Commands::Dump(args) => {
            let ifile = std::fs::File::open(args.input)?;
            let mut ifile = std::io::BufReader::new(ifile);
            let config = bincode::config::standard()
                .with_little_endian()
                .with_fixed_int_encoding();

            let ofile = std::fs::File::create(args.output)?;
            let mut ofile = std::io::BufWriter::new(ofile);
            let mut d = Data::new();
            d.data = bincode::decode_from_std_read(&mut ifile, config)?;
            d.pos = bincode::decode_from_std_read(&mut ifile, config)?;
            let mut start = 0;
            for compressed_diffs in d.data.iter() {
                let n = compressed_diffs.num_cells();
                for (loc, expression_vec) in d
                    .pos
                    .iter()
                    .skip(start)
                    .take(n)
                    .zip(compressed_diffs.expression_vec_iter())
                {
                    let expression = expression_vec
                        .iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join(",");
                    writeln!(ofile, "{},{},{}", loc.xpos(), loc.ypos(), expression)?;
                }
                start += n;
            }
        }
    }
    Ok(())
}
*/