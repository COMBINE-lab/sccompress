use hdf5::{File, Group};
use hdf5::types::FixedAscii;
use ndarray::Array1;
use sprs::CsMat;
use std::path::Path;

/// Helper to read 10x fixed-length strings from HDF5
#[allow(dead_code)]
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
#[allow(dead_code)]
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

/// Write a decoded CSR sparse count matrix to a 10x Genomics-style HDF5 file.
///
/// The input `csr` is in `(n_cells × n_genes)` row-major layout — i.e. the
/// shape produced by `reconstruct_count_matrix_from_payload`. Because the
/// 10x on-disk format and a `(cells × genes)` CSR share the same
/// `(data, indices, indptr)` triple (the only difference is the `shape`
/// field, which is `[n_genes, n_cells]` on disk), we can write the CSR
/// arrays directly without any transposition.
///
/// The decoder cannot recover the original barcode / gene labels from the
/// payload (the row/column permutation back to the input H5 is not stored).
/// We therefore write **placeholder** metadata: barcodes are
/// `cell_0..cell_{n_cells-1}` and feature names/ids are
/// `gene_0..gene_{n_genes-1}`. Downstream tools that consume the matrix by
/// position (count totals, PCA, clustering on the decoded matrix) will work
/// fine; tools that rely on barcode-specific lookups will need a separate
/// mapping step.
pub fn write_csr_10x_h5<P: AsRef<Path>>(
    output_path: P,
    csr: &CsMat<u16>,
) -> anyhow::Result<()> {
    if !csr.is_csr() {
        anyhow::bail!("write_csr_10x_h5 requires a CSR matrix; got CSC");
    }
    let n_cells = csr.rows();
    let n_genes = csr.cols();

    let data: Vec<u16> = csr.data().to_vec();
    let indices: Vec<usize> = csr.indices().to_vec();
    let indptr: Vec<usize> = csr.indptr().raw_storage().to_vec();

    let output_file = File::create(output_path.as_ref())?;
    let matrix_group = output_file.create_group("matrix")?;

    matrix_group
        .new_dataset_builder()
        .with_data(&Array1::from_vec(data))
        .create("data")?;
    matrix_group
        .new_dataset_builder()
        .with_data(&Array1::from_vec(indices))
        .create("indices")?;
    matrix_group
        .new_dataset_builder()
        .with_data(&Array1::from_vec(indptr))
        .create("indptr")?;
    matrix_group
        .new_dataset_builder()
        .with_data(&Array1::from_vec(vec![n_genes, n_cells]))
        .create("shape")?;

    let barcodes: Vec<String> = (0..n_cells).map(|i| format!("cell_{}", i)).collect();
    write_strings_metadata(&matrix_group, "barcodes", &barcodes)?;

    let features_group = matrix_group.create_group("features")?;
    let gene_names: Vec<String> = (0..n_genes).map(|i| format!("gene_{}", i)).collect();
    let feature_types: Vec<String> = vec!["Gene Expression".to_string(); n_genes];
    let genomes: Vec<String> = vec!["decoded".to_string(); n_genes];
    write_strings_metadata(&features_group, "name", &gene_names)?;
    write_strings_metadata(&features_group, "id", &gene_names)?;
    write_strings_metadata(&features_group, "feature_type", &feature_types)?;
    write_strings_metadata(&features_group, "genome", &genomes)?;

    Ok(())
}
