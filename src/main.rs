pub mod bits;
pub mod quad_tree;
//pub mod lossy_compression;
use crate::quad_tree::tree::{ErrorMetric, Point, QuadTree, Rect};
//use crate::lossy_compression::LloydMaxQuantizer;
use clap::{Args, Parser, Subcommand, ValueEnum};
use csv::ReaderBuilder;
use hdf5::types::FixedAscii;
use hdf5::File as Hdf5File;
use ndarray::ArrayD;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use quad_tree::tree::{BitFieldQuadTree, DatalessPoint, EncodedDiffsMST, PointLike};
//use std::error::Error;
use std::fmt::Write as FmtWrite;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::io::Write;
use std::path::{Path, PathBuf};
use tracing::{info, warn};
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::fmt;
use tracing_subscriber::prelude::*;
use tracing_subscriber::EnvFilter;
use flate2::Compression;
use flate2::write::GzEncoder;
use flate2::read::GzDecoder;
use sprs::CsMat;
// removed unused bincode::{Encode, Decode} import

use mimalloc::MiMalloc;
use std::collections::HashMap;
use std::io::BufRead;

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
) -> anyhow::Result<(QuadTree, CsMat<u16>)> {
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
    let num_genes = idx_gene_end - idx_gene_start;
    
    // Build dense matrix first, then convert to CSR
    use ndarray::Array2;
    let mut dense_data = Vec::new();
    
    for record in &records {
        for j in idx_gene_start..idx_gene_end {
            let value: i16 = match record[j].parse::<f64>() {
                Ok(v) => v as i16,
                Err(_) => 0,
            };
            dense_data.push(value);
        }
    }
    
    let dense = Array2::from_shape_vec((records.len(), num_genes), dense_data)?;
    let csr_i16 = CsMat::csr_from_dense(dense.view(), 0i16);
    
    // Convert i16 CSR to u16 CSR
    let (rows, cols) = csr_i16.shape();
    let (indptr, indices, data_i16) = csr_i16.into_raw_storage();
    let data_u16: Vec<u16> = data_i16.into_iter().map(|x| x as u16).collect();
    let csr = CsMat::new((rows, cols), indptr, indices, data_u16);
    // Process all records
    for row_idx in 0..records.len() {
        let record = &records[row_idx];
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
        coords.push(Point::new(x, y, row_idx));
    }

    let minx = xs.iter().cloned().fold(f64::INFINITY, f64::min) - 1.0;
    let miny = ys.iter().cloned().fold(f64::INFINITY, f64::min) - 1.0;
    let maxx = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 1.0;
    let maxy = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 1.0;
    let w = maxx - minx;
    let h = maxy - miny;

    let domain = Rect::new(minx + w / 2.0_f64, miny + h / 2.0_f64, w, h);
    let qtree = QuadTree::new(domain, coords, 0);

     // Divide the quadtree and get cost log
    //let division_cost_log = qtree.divide_recursive(&csr);
    
    Ok((qtree, csr))
}

/* 
fn read_parquet_file(file_path: &str) -> Result<(), ParquetError> { 
    let file = File::open(Path::new(file_path))?; 
    let reader = SerializedFileReader::new(file)?;  
    let data = reader.read_all()?;
    return reader;
}*/

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
    let qtree = QuadTree::new(domain, coords, 0);

    // Divide the quadtree and get cost log
    //let division_cost_log = qtree.divide_recursive(&csr);
    
    Ok((qtree, csr))
}

/// Read cluster assignments from a file where each line contains cell IDs separated by semicolons
fn read_cluster_file_semicolon<T: AsRef<Path>>(cluster_path: T) -> anyhow::Result<Vec<Vec<String>>> {
    let file = File::open(cluster_path)?;
    let reader = BufReader::new(file);
    
    let mut clusters = Vec::new();
    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        let cell_ids: Vec<String> = line
            .split(';')
            .map(|s| s.trim().trim_matches('"').trim().to_string())  // Strip quotes and whitespace
            .filter(|s| !s.is_empty())
            .collect();
        
        if !cell_ids.is_empty() {
            info!("Cluster {}: {} cells", line_num, cell_ids.len());
            clusters.push(cell_ids);
        }
    }
    
    info!("Total clusters loaded: {}", clusters.len());
    Ok(clusters)
}

/// Read cluster assignments from a two-column file (cell_id, cluster_id)
fn read_cluster_file_two_column<T: AsRef<Path>>(cluster_path: T) -> anyhow::Result<Vec<Vec<String>>> {
    let file = File::open(cluster_path)?;
    let reader = BufReader::new(file);
    
    // Use a HashMap to group cells by cluster ID
    let mut cluster_map: HashMap<String, Vec<String>> = HashMap::new();
    let mut first_line = true;
    
    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        let line = line.trim();
        
        // Skip empty lines
        if line.is_empty() {
            continue;
        }
        
        // Skip header line if it looks like a header
        if first_line {
            first_line = false;
            // Check if this looks like a header (contains "Barcode" or "Cluster" etc.)
            let lower = line.to_lowercase();
            if lower.contains("barcode") || lower.contains("cell") || 
               (lower.contains("cluster") && !line.chars().next().unwrap_or('0').is_numeric()) {
                info!("Skipping header line: {}", line);
                continue;
            }
        }
        
        // Try splitting by comma first (CSV format), then fall back to whitespace
        let parts: Vec<&str> = if line.contains(',') {
            line.split(',').collect()
        } else {
            line.split_whitespace().collect()
        };
        
        if parts.len() < 2 {
            warn!("Line {}: Expected 2 columns, found {}. Skipping.", line_num + 1, parts.len());
            continue;
        }
        
        // First column: cell ID, second column: cluster ID
        let cell_id = parts[0].trim().trim_matches('"').to_string();
        let cluster_id = parts[1].trim().trim_matches('"').to_string();
        
        cluster_map.entry(cluster_id).or_insert_with(Vec::new).push(cell_id);
    }
    
    // Convert HashMap to Vec<Vec<String>> sorted by cluster ID
    let mut cluster_ids: Vec<String> = cluster_map.keys().cloned().collect();
    cluster_ids.sort();
    
    let mut clusters = Vec::new();
    for cluster_id in cluster_ids {
        let cell_ids = cluster_map.remove(&cluster_id).unwrap();
        info!("Cluster '{}': {} cells", cluster_id, cell_ids.len());
        clusters.push(cell_ids);
    }
    
    info!("Total clusters loaded: {}", clusters.len());
    Ok(clusters)
}

/// Encode clusters from HDF5 data
fn encode_clusters_from_h5<T: AsRef<Path>>(
    h5_path: T,
    pos_path: T,
    cluster_path: T,
    pos_type: InputPosType,
    cluster_format: ClusterFormat,
    pos_x_col: usize,
    pos_y_col: usize,
) -> anyhow::Result<Vec<(EncodedDiffsMST, Vec<DatalessPoint>)>> {
    info!("Loading HDF5 data from: {}", h5_path.as_ref().display());
    
    // Read features from HDF5
    let file = Hdf5File::open(h5_path.as_ref())?;
    let matrix_group = file.group("matrix")?;
    
    // Read matrix shape
    let shape_dataset = matrix_group.dataset("shape")?;
    let shape_array = shape_dataset.read_1d::<usize>()?;
    let shape: Vec<usize> = shape_array.to_vec();
    let num_features = shape[0];
    let num_cells = shape[1];
    
    info!("Matrix shape: {} cells x {} features", num_cells, num_features);
    
    // Read sparse matrix components
    let data_dataset = matrix_group.dataset("data")?;
    let indices_dataset = matrix_group.dataset("indices")?;
    let indptr_dataset = matrix_group.dataset("indptr")?;
    let barcodes_dataset = matrix_group.dataset("barcodes")?;
    
    let data_array = data_dataset.read_1d::<u16>()?;
    let data_slice = data_array.as_slice().unwrap();
    let indices_array = indices_dataset.read_1d::<usize>()?;
    let indices: Vec<usize> = indices_array.to_vec();
    let indptr_array = indptr_dataset.read_1d::<usize>()?;
    let indptr: Vec<usize> = indptr_array.to_vec();
    
    // Build CSR matrix
    let csr: CsMat<u16> = CsMat::new(
        (num_cells, num_features),
        indptr.clone(),
        indices.clone(),
        data_slice.to_vec(),
    );
    
    // Read barcodes and create barcode -> row index map (needed for two-column format)
    let barcodes_arr = barcodes_dataset.read_1d::<FixedAscii<23>>()?;
    let barcodes: Vec<String> = barcodes_arr
        .iter()
        .map(|b| b.as_str().trim_end_matches('\0').to_string())
        .collect();
    
    let barcode_to_idx: HashMap<String, usize> = barcodes
        .iter()
        .enumerate()
        .map(|(idx, bc)| (bc.clone(), idx))
        .collect();
    
    // Read position data into a map indexed by row number
    let pos_file = File::open(pos_path.as_ref())?;
    let mut pos_map: HashMap<usize, (f64, f64)> = HashMap::with_capacity(num_cells);
    
    match pos_type {
        InputPosType::Csv => {
            let mut rdr = csv::ReaderBuilder::new()
                .has_headers(false)
                .from_reader(pos_file);
            for (row_idx, rec) in rdr.records().enumerate() {
                let rec = rec?;
                let x: f64 = rec.get(pos_x_col).ok_or_else(|| anyhow::anyhow!("Missing x column"))?.parse()?;
                let y: f64 = rec.get(pos_y_col).ok_or_else(|| anyhow::anyhow!("Missing y column"))?.parse()?;
                pos_map.insert(row_idx, (x, y));
            }
        }
        InputPosType::Parquet => {
            let reader = SerializedFileReader::new(pos_file)?;
            let mut iter = reader.get_row_iter(None)?;
            let mut row_idx = 0;
            while let Some(Ok(parquet_row)) = iter.next() {
                let x = parquet_row.get_double(pos_x_col)
                    .map_err(|e| anyhow::anyhow!("Failed to read x: {}", e))?;
                let y = parquet_row.get_double(pos_y_col)
                    .map_err(|e| anyhow::anyhow!("Failed to read y: {}", e))?;
                pos_map.insert(row_idx, (x, y));
                row_idx += 1;
            }
        }
    }
    
    // Read cluster assignments based on format
    let clusters = match cluster_format {
        ClusterFormat::Semicolon => read_cluster_file_semicolon(cluster_path)?,
        ClusterFormat::TwoColumn => read_cluster_file_two_column(cluster_path)?,
    };
    
    // Encode each cluster
    let mut encoded_clusters = Vec::new();
    
    for (cluster_idx, cell_ids) in clusters.iter().enumerate() {
        info!("Encoding cluster {} with {} cells", cluster_idx, cell_ids.len());
        if cell_ids.len() == 1 {
            info!("this cellid {} is a single cell cluster", cell_ids[0]);
        }
        
        // Build Point vector for this cluster
        let mut points = Vec::new();
        let mut positions = Vec::new();
        
        for cell_id_str in cell_ids {
            // For two-column format, try barcode lookup first, then fall back to row index
            // For semicolon format, always parse as row index
            let row_idx_opt = if matches!(cluster_format, ClusterFormat::TwoColumn) {
                // Try as barcode first
                barcode_to_idx.get(cell_id_str).copied()
                    .or_else(|| {
                        // Fall back to parsing as row index
                        cell_id_str.parse::<usize>().ok()
                    })
            } else {
                // Semicolon format: always parse as row index
                cell_id_str.parse::<usize>().ok()
            };
            
            match row_idx_opt {
                Some(row_idx) => {
                    // Check if row index is valid
                    if row_idx >= num_cells {
                        warn!("Cell '{}' maps to row index {} which is out of bounds (max: {})", 
                              cell_id_str, row_idx, num_cells - 1);
                        continue;
                    }
                    
                    // Look up position by row index
                    if let Some(&(x, y)) = pos_map.get(&row_idx) {
                        points.push(Point::new(x, y, row_idx));
                        positions.push(DatalessPoint::new(x, y));
                    } else {
                        warn!("Cell '{}' (row {}) not found in position data", cell_id_str, row_idx);
                    }
                }
                None => {
                    warn!("Cell ID '{}' not found as barcode and cannot be parsed as row index", cell_id_str);
                }
            }
        }
        
        // Encode this cluster
        if let Some((encoded, dfs_order)) = quad_tree::tree::encode_subarray_mst(&points, &csr) {
            info!("Cluster {} encoded: {} bytes", cluster_idx, encoded.bytes());
            // Reorder positions to match DFS order
            let reordered_positions: Vec<DatalessPoint> = dfs_order.iter()
                .map(|&idx| positions[idx as usize].clone())
                .collect();
            encoded_clusters.push((encoded, reordered_positions));
        } else {
            warn!("Failed to encode cluster {}", cluster_idx);
        }
    }
    
    Ok(encoded_clusters)
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
    #[command(arg_required_else_help = true)]
    EncodeClusters(EncodeClustersCommand),
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

#[derive(Debug, Clone, ValueEnum)]
enum ClusterFormat {
    /// Semicolon-separated format: each line is a cluster with cells separated by semicolons
    Semicolon,
    /// Two-column format: cell_id cluster_id (whitespace or tab separated)
    TwoColumn,
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

/// Encode clusters from a cluster assignment file
#[derive(Debug, Args)]
#[command(version, about, long_about = None)]
struct EncodeClustersCommand {
    /// HDF5 file with gene expression matrix
    #[arg(short = 'i', long)]
    input: PathBuf,
    /// Position file (CSV or Parquet)
    #[arg(short = 'p', long)]
    input_pos: PathBuf,
    /// Cluster assignment file
    #[arg(short = 'c', long)]
    clusters: PathBuf,
    /// Output file (default "clusters_encoded.bin.gz")
    #[arg(short = 'o', long)]
    output: Option<PathBuf>,
    /// Input position file format (csv or parquet)
    #[arg(short = 'F', long = "pos-format", value_enum, default_value_t = InputPosType::Parquet)]
    pos_format: InputPosType,
    /// Platform (visium or xenium)
    #[arg(short = 'P', long = "platform", value_enum)]
    platform: Option<Platform>,
    /// Cluster file format
    #[arg(short = 'f', long = "cluster-format", value_enum, default_value_t = ClusterFormat::Semicolon)]
    cluster_format: ClusterFormat,
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
    pub data: Vec<EncodedDiffsMST>,
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
            let (qtree, csr) = match args.format {
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
                | InputDataType::H5ad => {
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
            //bincode::encode_into_std_write(&d.data, &mut writer, config).unwrap();
            //bincode::encode_into_std_write(&d.pos, &mut writer, config).unwrap();
            bincode::encode_into_std_write(&d.data, &mut encoder, config).unwrap();
            bincode::encode_into_std_write(&d.pos, &mut encoder, config).unwrap();

        }
        Commands::Dump(args) => {
            info!("start dump");
            let ifile = std::fs::File::open(args.input)?;
            let mut rdr = std::io::BufReader::new(ifile);
            let gz = GzDecoder::new(&mut rdr);
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
        Commands::EncodeClusters(args) => {
            info!("Starting cluster encoding");
            
            // Determine position columns based on platform
            let (pos_x_col, pos_y_col) = match args.platform {
                Some(Platform::Visium) => (4, 5),
                Some(Platform::Xenium) => (1, 2),
                None => (1, 2), // Default to Xenium format
            };
            
            // Encode clusters
            let encoded_clusters = encode_clusters_from_h5(
                &args.input,
                &args.input_pos,
                &args.clusters,
                args.pos_format,
                args.cluster_format,
                pos_x_col,
                pos_y_col,
            )?;
            
            info!("Successfully encoded {} clusters", encoded_clusters.len());
            
            // Serialize to file
            let config = bincode::config::standard()
                .with_little_endian()
                .with_fixed_int_encoding();
            let ofname = args.output.unwrap_or(PathBuf::from("clusters_encoded.bin.gz"));
            let file = File::create(&ofname)?;
            let writer = BufWriter::new(file);
            let mut encoder = GzEncoder::new(writer, Compression::default());
            
            // Separate encoded diffs and positions
            let mut all_encoded_diffs = Vec::new();
            let mut all_positions = Vec::new();
            
            for (encoded, positions) in encoded_clusters {
                all_encoded_diffs.push(encoded);
                all_positions.extend(positions);
            }
            
            bincode::encode_into_std_write(&all_encoded_diffs, &mut encoder, config)?;
            bincode::encode_into_std_write(&all_positions, &mut encoder, config)?;
            
            info!("Encoded clusters saved to: {}", ofname.display());
            info!("Total encoded blocks: {}", all_encoded_diffs.len());
            info!("Total positions: {}", all_positions.len());
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
