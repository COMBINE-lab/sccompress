use hdf5::{File, Group};
use hdf5::types::FixedAscii;
use ndarray::Array1;
use std::path::Path;

/// Helper to read 10x fixed-length strings from HDF5
fn read_strings(ds: hdf5::Dataset) -> anyhow::Result<Vec<String>> {
    // Try common 10x fixed-length string sizes
    if let Ok(arr) = ds.read_1d::<FixedAscii<7>>() {
        return Ok(arr.iter().map(|s| s.as_str().trim_end_matches('\0').to_string()).collect());
    }
    if let Ok(arr) = ds.read_1d::<FixedAscii<23>>() {
        return Ok(arr.iter().map(|s| s.as_str().trim_end_matches('\0').to_string()).collect());
    }
    if let Ok(arr) = ds.read_1d::<FixedAscii<25>>() {
        return Ok(arr.iter().map(|s| s.as_str().trim_end_matches('\0').to_string()).collect());
    }
    if let Ok(arr) = ds.read_1d::<FixedAscii<64>>() {
        return Ok(arr.iter().map(|s| s.as_str().trim_end_matches('\0').to_string()).collect());
    }
    
    // Fallback to reading as standard strings if possible
    if let Ok(arr) = ds.read_1d::<hdf5::types::VarLenUnicode>() {
        return Ok(arr.iter().map(|s| s.to_string()).collect());
    }

    anyhow::bail!("Unsupported HDF5 string type for dataset: {}", ds.name());
}

/// Helper to write strings to HDF5 as fixed-length ASCII
fn write_strings_metadata(group: &Group, name: &str, strings: &[String]) -> anyhow::Result<()> {
    // 64 characters is enough for barcodes, gene names, and feature types in 10x files
    let data: Vec<FixedAscii<64>> = strings.iter()
        .map(|s| FixedAscii::<64>::from_ascii(s.as_bytes()).unwrap_or_default())
        .collect();
    group.new_dataset_builder()
        .with_data(&Array1::from_vec(data))
        .create(name)?;
    Ok(())
}

/// Subsets an existing 10x HDF5 file based on a list of cell indices.
pub fn subset_10x_h5<P: AsRef<Path>>(
    input_path: P,
    output_path: P,
    cell_indices: &[usize],
) -> anyhow::Result<()> {
    let input_file = File::open(input_path.as_ref())?;
    let matrix_group = input_file.group("matrix")?;

    // 1. Read original metadata
    let shape_arr = matrix_group.dataset("shape")?.read_1d()?;
    if shape_arr.len() < 2 {
        anyhow::bail!("Invalid shape dataset in HDF5 file");
    }
    let num_features = shape_arr[0];
    let original_num_cells = shape_arr[1];

    let barcodes = read_strings(matrix_group.dataset("barcodes")?)?;
    
    let features_group = matrix_group.group("features")?;
    let gene_names = read_strings(features_group.dataset("name")?)?;
    let gene_ids = read_strings(features_group.dataset("id")?)?;
    let feature_types = read_strings(features_group.dataset("feature_type")?)?;
    let genomes = read_strings(features_group.dataset("genome")?)?;

    // 2. Read sparse matrix components
    let original_data: Vec<u16> = matrix_group.dataset("data")?.read_1d()?.to_vec();
    let original_indices: Vec<usize> = matrix_group.dataset("indices")?.read_1d()?.to_vec();
    let original_indptr: Vec<usize> = matrix_group.dataset("indptr")?.read_1d()?.to_vec();

    // 3. Construct subset
    let mut subset_data = Vec::new();
    let mut subset_indices = Vec::new();
    let mut subset_indptr = Vec::with_capacity(cell_indices.len() + 1);
    let mut subset_barcodes = Vec::with_capacity(cell_indices.len());

    subset_indptr.push(0);
    let mut current_offset = 0;

    for &cell_idx in cell_indices {
        if cell_idx >= original_num_cells {
            anyhow::bail!("Cell index {} out of bounds (max {})", cell_idx, original_num_cells);
        }

        // Get barcode
        subset_barcodes.push(barcodes[cell_idx].clone());

        // Get data slice for this cell
        let start = original_indptr[cell_idx];
        let end = original_indptr[cell_idx + 1];
        
        let cell_data = &original_data[start..end];
        let cell_gene_indices = &original_indices[start..end];

        subset_data.extend_from_slice(cell_data);
        subset_indices.extend_from_slice(cell_gene_indices);

        current_offset += cell_data.len();
        subset_indptr.push(current_offset);
    }

    // 4. Create output file and write
    let output_file = File::create(output_path.as_ref())?;
    let out_matrix_group = output_file.create_group("matrix")?;

    // Write components
    out_matrix_group.new_dataset_builder().with_data(&Array1::from_vec(subset_data)).create("data")?;
    out_matrix_group.new_dataset_builder().with_data(&Array1::from_vec(subset_indices)).create("indices")?;
    out_matrix_group.new_dataset_builder().with_data(&Array1::from_vec(subset_indptr)).create("indptr")?;
    
    // Write shape [num_features, num_cells_subset]
    out_matrix_group.new_dataset_builder()
        .with_data(&Array1::from_vec(vec![num_features, cell_indices.len()]))
        .create("shape")?;

    // Write barcodes
    write_strings_metadata(&out_matrix_group, "barcodes", &subset_barcodes)?;

    // Write features group
    let out_features_group = out_matrix_group.create_group("features")?;
    write_strings_metadata(&out_features_group, "name", &gene_names)?;
    write_strings_metadata(&out_features_group, "id", &gene_ids)?;
    write_strings_metadata(&out_features_group, "feature_type", &feature_types)?;
    write_strings_metadata(&out_features_group, "genome", &genomes)?;

    Ok(())
}
