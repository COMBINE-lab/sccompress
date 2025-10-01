// Import shared modules
mod bits;
mod quad_tree;

use quad_tree::tree::{ErrorMetric, Point, QuadTree, Rect,BitFieldQuadTree, DatalessPoint, EncodedDiffs, PointLike};
use clap::{Args, Parser, Subcommand, ValueEnum};
use hdf5::types::FixedAscii;
use hdf5::File as Hdf5File;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
//use std::error::Error;
use std::fmt::Write as FmtWrite;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::io::Write;
use std::path::{Path, PathBuf};
use tracing::info;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::fmt;
use tracing_subscriber::prelude::*;
use tracing_subscriber::EnvFilter;
use flate2::Compression;
use flate2::write::GzEncoder;
use flate2::read::GzDecoder;
use bincode::{Decode, Encode};
use sprs::{CsMat, CsVecViewI};

// Cost tracking structures for serialization
#[derive(Clone, Encode, Decode)]
pub(crate) struct CostStep {
    pub depth: usize,
    pub points_count: usize,
    pub current_cost: usize,
    pub children_cost: usize,
    pub optimal_cost: usize,
    pub decision: String,  // "merge" or "divide"
    pub node_id: String,   // e.g., "root", "nw", "ne", "se", "sw"
}

#[derive(Clone, Encode, Decode)]
pub(crate) struct CostLog {
    //pub steps: Vec<CostStep>,
    pub total_nodes: usize,
    pub total_cost: usize,
}

impl CostLog {
    pub fn new() -> Self {
        Self {
            //steps: Vec::new(),
            total_nodes: 0,
            total_cost: 0,
        }
    }
    
    //pub fn add_step(&mut self, step: CostStep) {
    //    self.steps.push(step);
    //}
    
    pub fn update_totals(&mut self, nodes: usize, cost: usize) {
        self.total_nodes = nodes;
        self.total_cost = cost;
    }
}


fn tree_from_10x<T: AsRef<Path>>(
    h5_path: T,
    pos_path: T,
    pos_type: InputPosType,
    file_type: InputDataType,
    pos_x_col: usize,
    pos_y_col: usize,
    _method: ErrorMetric,
    _lossless: bool,
) -> anyhow::Result<(QuadTree, CsMat<u16>)> {
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

    // Check if matrix group exists
    let matrix_group = match file.group("matrix") {
        Ok(g) => g,
        Err(_) => return Err(anyhow::anyhow!("No 'matrix' group; use molecule_info path")),
    };

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
    // Use array directly instead of converting to Vec to avoid ownership issues
    let data_slice = data_array.as_slice().unwrap();
    let indices_array = indices_dataset.read_1d::<usize>()?;
    let indices: Vec<usize> = indices_array.to_vec();
    let indptr_array = indptr_dataset.read_1d::<usize>()?;
    let indptr: Vec<usize> = indptr_array.to_vec();

    info!("Sparse matrix data: {} non-zero elements", data_slice.len());
    // Build CSR matrix
    let csr: CsMat<u16> = CsMat::new(
        (num_cells, num_features),
        indptr.clone(),
        indices.clone(),
        data_slice.to_vec(),
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
                let bc = parquet_row.get_string(0).unwrap().to_string();
                //println!("x: {}", pos_x_col);
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
            let expression_value = data_slice[i];
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

        coords.push(Point::new(x_coord, y_coord, row_idx));
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
    let division_cost_log = qtree.divide_recursive(&csr);
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
    Ok((qtree, csr))
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
   // Mtx,
   // v2,
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
fn main() -> Result<(), Box<dyn std::error::Error>> {
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
            //let (qtree, csr) = match args.format {
               // InputDataType::Csr
                //| InputDataType::H5ad => {
                    let file_path_pos = args
                        .input_pos
                        .ok_or_else(|| anyhow::anyhow!("Position file required for HDF5 format"))?;
                    tree_from_10x(
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
                            //InputDataType::Mtx => InputDataType::Mtx,
                            //InputDataType::v2 => InputDataType::v2,
                        },
                        pos_x_col,
                        pos_y_col,
                        ErrorMetric::Mean,
                        true,
                    )?
                //}
            };

            // only serialize the the bit fields
            let config = bincode::config::standard()
                .with_little_endian()
                .with_fixed_int_encoding();
            let ofname = args.output.unwrap_or(PathBuf::from("output.bin.gz"));
            let file = File::create(ofname).unwrap();
            let writer = BufWriter::new(file);
            let mut encoder = GzEncoder::new(writer, Compression::default());
            let bit_field_tree = qtree.compute_quadtree_bit_fields(&csr);
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
                qtree.non_zero_blocks(&csr)
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
                    ofile.write_all(
                        format!("{},{},{}\n", loc.xpos(), loc.ypos(), str_out).as_bytes(),
                    )?;
                    //writeln!(ofile, "{},{},{}", loc.xpos(), loc.ypos(), expression)?;
                }
                start += n;
            }
        }
    }
*/
