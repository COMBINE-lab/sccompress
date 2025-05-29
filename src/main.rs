use crate::quad_tree::tree::{ErrorMetric, Point, QuadTree, Rect};
pub mod quad_tree;
use bincode::{BorrowDecode, Decode, Encode};
use clap::Parser;
use csv::ReaderBuilder;
use hdf5::types::FixedAscii;
use hdf5::File;
use ndarray::Array1;
use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::fs::File as StdFile;
use std::path::{Path, PathBuf};
use std::time::Instant;
use sux::prelude::BitFieldVec;
use sux::traits::BitFieldSliceMut;
use tracing::{info, warn};
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::fmt;
use tracing_subscriber::prelude::*;
use tracing_subscriber::EnvFilter;

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
    for (i, record) in records.iter().enumerate() {
        //println!("i: {:?}", i);
        // Read coordinates
        let x: f32 = record[idx_x].parse().map_err(|e| {
            anyhow::anyhow!("Failed to parse x coordinate at column {}: {}", idx_x, e)
        })?;
        let y: f32 = record[idx_y].parse().map_err(|e| {
            anyhow::anyhow!("Failed to parse y coordinate at column {}: {}", idx_y, e)
        })?;
        xs.push(x);
        ys.push(y);

        // Read gene expression data
        let mut cells = Vec::new();
        for j in idx_gene_start..idx_gene_end {
            let value: u16 = match record[j].parse::<f32>() {
                Ok(v) => v as u16,
                Err(e) => 0,
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
        coords.push(Point::new(x, y, cells));
    }

    let minx = xs.iter().cloned().fold(f32::INFINITY, f32::min) - 1.0;
    let miny = ys.iter().cloned().fold(f32::INFINITY, f32::min) - 1.0;
    let maxx = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max) + 1.0;
    let maxy = ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max) + 1.0;
    let w = maxx - minx;
    let h = maxy - miny;

    let domain = Rect::new(minx + w / 2.0, miny + h / 2.0, w, h);

    // No need to convert since we're already using u16
    let mut qtree = QuadTree::new(domain, coords, 0);
    qtree.divide();
    Ok(qtree)
}

fn tree_from_h5<T: AsRef<Path>>(
    h5_path: T,
    spatial_path: T,
    method: ErrorMetric,
    _lossless: bool,
    seq_type: &str,
) -> anyhow::Result<QuadTree> {
    // Read spatial coordinates from CSV
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .flexible(true) // Allow varying number of fields
        .from_path(spatial_path)
        .map_err(|e| anyhow::anyhow!("Failed to open spatial file: {}", e))?;

    let mut spatial_coords = HashMap::new();
    let mut spatial_count = 0;
    let mut line_number = 0;

    // Print header row for debugging
    if let Ok(header) = rdr.headers() {
        info!("CSV Headers: {:?}", header);
        info!("Number of columns in header: {}", header.len());
    }

    for result in rdr.records() {
        line_number += 1;
        let record = result
            .map_err(|e| anyhow::anyhow!("Failed to read record at line {}: {}", line_number, e))?;

        // Print first few records for debugging
        if line_number <= 5 {
            info!("Record {}: {:?}", line_number, record);
            info!("Number of fields in record: {}", record.len());
        }

        if record.len() < 3 {
            return Err(anyhow::anyhow!(
                "Record at line {} has insufficient fields (need 3, got {})",
                line_number,
                record.len()
            ));
        }

        let barcode = record[0].to_string(); // barcode is in first column

        // Debug print the raw values before parsing
        if line_number <= 5 {
            info!(
                "Raw values for barcode {}: x='{}', y='{}'",
                barcode,
                record[1].to_string(),
                record[2].to_string()
            );
        }

        // Try to parse with more detailed error handling
        let x_str = record[1].trim();
        let y_str = record[2].trim();

        if x_str.is_empty() || y_str.is_empty() {
            return Err(anyhow::anyhow!(
                "Empty coordinate value at line {} for barcode {}: x='{}', y='{}'",
                line_number,
                barcode,
                x_str,
                y_str
            ));
        }

        let x: f32 = x_str.parse().map_err(|e| {
            anyhow::anyhow!(
                "Failed to parse x coordinate at line {} for barcode {}: '{}' - {}",
                line_number,
                barcode,
                x_str,
                e
            )
        })?;

        let y: f32 = y_str.parse().map_err(|e| {
            anyhow::anyhow!(
                "Failed to parse y coordinate at line {} for barcode {}: '{}' - {}",
                line_number,
                barcode,
                y_str,
                e
            )
        })?;

        spatial_coords.insert(barcode.clone(), (x, y));
        spatial_count += 1;
        if spatial_count <= 5 {
            info!("Successfully parsed barcode: {} at ({}, {})", barcode, x, y);
        }
    }
    info!("Read {} spatial coordinates", spatial_count);

    if spatial_count == 0 {
        return Err(anyhow::anyhow!(
            "No valid coordinates found in the CSV file"
        ));
    }

    // Read H5AD file
    let file = File::open(h5_path)?;

    // Read barcodes based on seq_type
    let barcodes_dataset = if seq_type == "10x" {
        file.dataset("matrix/barcodes")?
    } else {
        file.dataset("obs/_index")?
    };
    let barcodes_array: Array1<FixedAscii<32>> = barcodes_dataset.read()?;
    let barcodes: Vec<String> = barcodes_array
        .iter()
        .map(|s| s.as_str().trim_end_matches('\0').to_string())
        .collect();
    info!("Read {} barcodes", barcodes.len());

    // Print sample barcodes from H5AD
    for (i, barcode) in barcodes.iter().take(5).enumerate() {
        info!("Sample H5AD barcode {}: {}", i, barcode);
    }

    // Read features based on seq_type
    let features_dataset = if seq_type == "10x" {
        file.dataset("matrix/features/name")?
    } else {
        file.dataset("var/gene_symbol")?
    };
    let features_array: Array1<FixedAscii<32>> = features_dataset.read()?;
    let features: Vec<String> = features_array
        .iter()
        .map(|s| s.as_str().trim_end_matches('\0').to_string())
        .collect();
    info!("Read {} features", features.len());

    // Read sparse matrix data based on seq_type
    let data_dataset = if seq_type == "10x" {
        file.dataset("matrix/data")?
    } else {
        file.dataset("X/data")?
    };
    let indices_dataset = if seq_type == "10x" {
        file.dataset("matrix/indices")?
    } else {
        file.dataset("X/indices")?
    };
    let indptr_dataset = if seq_type == "10x" {
        file.dataset("matrix/indptr")?
    } else {
        file.dataset("X/indptr")?
    };
    let shape_dataset = if seq_type == "10x" {
        file.dataset("matrix/shape")?
    } else {
        file.dataset("X/shape")?
    };

    let data_array: Array1<f32> = data_dataset.read()?;
    let indices_array: Array1<usize> = indices_dataset.read()?;
    let indptr_array: Array1<usize> = indptr_dataset.read()?;
    let shape_array: Array1<usize> = shape_dataset.read()?;

    let data: Vec<f32> = data_array.to_vec();
    let indices: Vec<usize> = indices_array.to_vec();
    let indptr: Vec<usize> = indptr_array.to_vec();
    let shape: Vec<usize> = shape_array.to_vec();

    info!("Matrix shape: {:?}", shape);
    info!("Number of non-zero elements: {}", data.len());

    // Create points with coordinates and gene expression
    let mut points = Vec::new();
    let num_genes = shape[1];
    let mut matched_count = 0;
    let mut unmatched_count = 0;

    for (cell_idx, barcode) in barcodes.iter().enumerate() {
        if let Some(&(x, y)) = spatial_coords.get(barcode) {
            // Initialize gene expression vector with zeros
            let mut gene_expr = vec![0 as u16; num_genes];

            // Fill in non-zero values from sparse matrix
            let start = indptr[cell_idx];
            let end = indptr[cell_idx + 1];
            for i in start..end {
                let gene_idx = indices[i];
                if gene_idx >= num_genes {
                    return Err(anyhow::anyhow!(
                        "Gene index {} out of bounds (max: {}) for cell {}",
                        gene_idx,
                        num_genes - 1,
                        cell_idx
                    ));
                }
                gene_expr[gene_idx] = data[i] as u16;
            }
            println!("gene_expr: {:?}", gene_expr);
            points.push(Point::new(x, y, gene_expr));
            matched_count += 1;
            if matched_count <= 5 {
                info!("Matched barcode: {} at position ({}, {})", barcode, x, y);
            }
        } else {
            unmatched_count += 1;
            if unmatched_count <= 5 {
                info!("Unmatched barcode: {}", barcode);
            }
        }
    }
    info!(
        "Matched {} barcodes with spatial coordinates",
        matched_count
    );
    info!("Unmatched {} barcodes", unmatched_count);
    info!("Created {} points", points.len());

    if points.is_empty() {
        return Err(anyhow::anyhow!(
            "No points were created - check if barcodes match between H5 and spatial files"
        ));
    }

    // Calculate boundaries
    let minx = points.iter().map(|p| p.x).fold(f32::INFINITY, f32::min) - 1.0;
    let miny = points.iter().map(|p| p.y).fold(f32::INFINITY, f32::min) - 1.0;
    let maxx = points.iter().map(|p| p.x).fold(f32::NEG_INFINITY, f32::max) + 1.0;
    let maxy = points.iter().map(|p| p.y).fold(f32::NEG_INFINITY, f32::max) + 1.0;
    let w = maxx - minx;
    let h = maxy - miny;

    info!(
        "Spatial boundaries: x=[{}, {}], y=[{}, {}]",
        minx, maxx, miny, maxy
    );

    // Create quadtree
    let domain = Rect::new(minx + w / 2.0, miny + h / 2.0, w, h);
    let mut qtree = QuadTree::new(domain, points.clone(), 1);
    qtree.divide();
    Ok(qtree)
}

/// Build a quadtree representation of spatial transcriptomics data
#[derive(Debug, Parser)]
#[command(version, about, long_about = None)]
struct CmdArgs {
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
    #[arg(short = 'x', long, default_value = "6")]
    idx_x: usize,
    /// Index of y coordinate, default 6
    #[arg(short = 'y', long, default_value = "7")]
    idx_y: usize,
    /// Index of gene start, default 1
    #[arg(short = 's', long, default_value = "11")]
    idx_gene_start: usize,
    /// Index of gene end, default all remaining columns
    #[arg(short = 'e', long)]
    idx_gene_end: Option<usize>,
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

    let args = CmdArgs::parse();
    let file_path = args.input;
    let qtree = match args.format.as_str() {
        "csv" => {
            // let file_path_pos = args.input_pos.ok_or_else(|| {
            //    anyhow::anyhow!("Position file required for CSV format")
            //})?;
            tree_from_csv(
                file_path,
                //file_path_pos,
                5,    // idx_x
                6,    // idx_y
                1,    // idx_gene_start
                None, // idx_gene_end (will use all remaining columns)
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
    bincode::encode_into_std_write(&bit_field_tree, &mut file, config).unwrap();
    info!(
        "QuadTree Blocks: (non-zero blocks: {})",
        //qtree.blocks(),
        qtree.non_zero_blocks()
    );
    Ok(())
}
