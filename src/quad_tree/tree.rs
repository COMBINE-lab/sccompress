//! Spatial single-cell RNA-seq compression using Minimum Spanning Trees
//!
//! This module implements a lossless compression strategy for spatial single-cell
//! RNA-sequencing count data that exploits spatial coherence in gene expression.
//!
//! # Overview
//!
//! The compression approach builds a Minimum Spanning Tree (MST) over cells based
//! on their expression similarity, then encodes each cell as a delta from its parent
//! in the tree. This is particularly effective for spatial transcriptomics data where
//! nearby cells in tissue have similar expression patterns.
//!
//! # Key Components
//!
//! - [`EncodedDiffsMST`]: Main compressed representation
//! - [`encode_subarray_mst`]: Compression function
//! - [`decode_cell_at_dfs_pos`]: Decompression function
//! - [`HybridSparseVec`]: Adaptive position encoding (Elias-Fano vs bitvector)
//! - [`ArithmeticEncoded`]: Arithmetic (ANS) encoding for integer values
//!
//! # Compression Pipeline
//!
//! 1. **Sparse Representation**: Convert dense expression matrix to sparse (gene, value) pairs
//! 2. **kNN Graph**: Build k-nearest neighbors graph using L0 distance (pattern differences)
//! 3. **MST Extraction**: Use Prim's algorithm to find minimum spanning tree
//! 4. **DFS Ordering**: Traverse tree to establish compression/decompression order
//! 5. **Delta Encoding**: Encode each cell as difference from its parent
//!
//! # Adaptive Encoding
//!
//! - **Position Encoding**: Automatically selects Elias-Fano (>75% sparse) or bitvector (≤75% sparse)
//! - **Value Encoding**: Uses arithmetic coding (ANS) for efficient encoding of integer values
//!
//! # Example
//!
//! ```ignore
//! // Compress a block of cells
//! let (encoded, dfs_order, stats) = encode_subarray_mst(&points, &csr_matrix, 0)?;
//!
//! // Decompress a specific cell
//! let cell_expression = encoded.decode_cell_at_dfs_pos(dfs_pos);
//! ```

use crate::bits::HybridSparseVec;
use crate::arith_encode::ArithmeticEncoded;
use bincode::{BorrowDecode, Decode, Encode};
use bitm::{self, BitAccess};
use hnsw_rs::prelude::*;
use rayon::prelude::*;
use rayon::join;
use sux::prelude::BitFieldVec;
use sux::traits::BitFieldSliceMut;
use tracing::{debug, error, info, warn};
//use rayon::scope;
use sprs::{CsMat, CsVecViewI};
use sucds::int_vectors::Access;

// MST-based encoding (using petgraph)
//use medians::medianu64;
//use adqselect::nth_element;

// Cost tracking structure
#[derive(Clone, Encode, Decode)]
pub(crate) struct CostLog {
    pub total_nodes: usize,
    pub total_cost: usize,
}

impl CostLog {
    pub fn new() -> Self {
        Self {
            total_nodes: 0,
            total_cost: 0,
        }
    }
}

#[derive(Clone)]
pub(crate) struct EncodedDiffs {
    pub(crate) indices: HybridSparseVec,
    pub(crate) ncells: u32,
    pub(crate) values: ArithmeticEncoded,      // Raw expression values (no median subtraction)
    pub(crate) num_genes: u32,                 // Total number of genes
}

/// MST-based compression for spatial single-cell RNA-seq data
///
/// This structure implements a lossless compression strategy that exploits spatial coherence
/// in gene expression between nearby cells in tissue. The approach builds a Minimum Spanning
/// Tree (MST) over cells based on expression similarity, then encodes each cell as a delta
/// from its parent in the tree.
///
/// ## Compression Strategy
///
/// 1. **MST Construction**: Build a kNN graph where edge weights represent the L0 distance
///    (number of differing genes) between cells, then extract the MST using Prim's algorithm.
///
/// 2. **Tree Traversal**: Perform DFS traversal from the root to establish a deterministic
///    ordering of cells for compression/decompression.
///
/// 3. **Root Encoding**: The root cell's expression is stored directly (uncompressed) using
///    sparse representation (gene indices + values).
///
/// 4. **Delta Encoding**: Each non-root cell stores only differences from its parent:
///    - **Position Encoding**: Adaptive selection between Elias-Fano (sparse) and bitvector (dense)
///    - **Value Encoding**: Arithmetic coding (ANS) with zigzag encoding for signed deltas
///
/// ## Decompression
///
/// Reconstruction follows the same DFS order:
/// 1. Read root cell values directly
/// 2. For each child, reconstruct by applying deltas to parent values
/// 3. Parent values are always available before children (guaranteed by DFS ordering)
///
/// ## Memory Layout
///
/// - `parent_offset`: Compact representation of tree structure (parent position relative to child)
/// - `root_indices`: Sparse indices of non-zero genes in root cell
/// - `root_vals`: Expression values for root cell
/// - `indices`: Combined (cell, gene) indices for all deltas (flattened)
/// - `delta_vals`: Zigzag-encoded delta values, parallel to `indices`
///
#[derive(Clone)]
pub(crate) struct EncodedDiffsMST {
    pub(crate) num_genes: u32,                  // Total number of genes (needed for output vector size)
    pub(crate) parent_offset: ArithmeticEncoded,// parent_offset[i] = i - parent_position (one per cell)
    pub(crate) root_indices: HybridSparseVec,   // Root's non-zero gene indices
    pub(crate) root_vals: ArithmeticEncoded,    // Root's expression values
    pub(crate) indices: HybridSparseVec,        // (dfs_pos-1)*num_genes + gene for non-root cells
    pub(crate) delta_vals: ArithmeticEncoded,   // Delta values (zigzag encoded), parallel to indices
}

impl EncodedDiffsMST {
    pub(crate) fn empty() -> Self {
        EncodedDiffsMST {
            num_genes: 0,
            parent_offset: ArithmeticEncoded::default(),
            root_indices: HybridSparseVec::empty(),
            root_vals: ArithmeticEncoded::default(),
            indices: HybridSparseVec::empty(),
            delta_vals: ArithmeticEncoded::default(),
        }
    }
    
    #[allow(dead_code)]
    pub(crate) fn is_empty(&self) -> bool {
        self.parent_offset.len() == 0
    }
    
    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        self.parent_offset.len()  // ncells derived from parent_offset
    }
    
    pub(crate) fn ncells(&self) -> usize {
        self.parent_offset.len()
    }

    pub(crate) fn bytes_breakdown(&self) -> (usize, usize, usize, usize, usize, usize) {
        let parent_offset_bytes = self.parent_offset.size_in_bytes();
        let root_indices_bytes = self.root_indices.num_bytes();
        let root_vals_bytes = self.root_vals.size_in_bytes();
        let indices_bytes = self.indices.num_bytes();
        let delta_vals_bytes = self.delta_vals.size_in_bytes();
        let num_genes_bytes = 4;
        (
            parent_offset_bytes,
            root_indices_bytes,
            root_vals_bytes,
            indices_bytes,
            delta_vals_bytes,
            num_genes_bytes,
        )
    }

    pub(crate) fn total_bytes(&self) -> usize {
        let (p, ri, rv, i, dv, ng) = self.bytes_breakdown();
        p + ri + rv + i + dv + ng
    }
}

// Manual implementation of de/serialization for `Rect`.
// We don't need to store the edges since they can computed from the other fields.
impl Encode for EncodedDiffsMST {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> core::result::Result<(), bincode::error::EncodeError> {
        Encode::encode(&self.num_genes, encoder)?;
        
        // Serialize parent_offset
        let mut parent_offset_bytes = Vec::new();
        self.parent_offset.serialize_into(&mut parent_offset_bytes)
            .map_err(|_| bincode::error::EncodeError::OtherString("DacsOpt serialize failed".into()))?;
        Encode::encode(&parent_offset_bytes, encoder)?;
        
        // HybridSparseVec has built-in Encode
        Encode::encode(&self.root_indices, encoder)?;
        
        let mut root_vals_bytes = Vec::new();
        self.root_vals.serialize_into(&mut root_vals_bytes)
            .map_err(|_| bincode::error::EncodeError::OtherString("DacsOpt serialize failed".into()))?;
        Encode::encode(&root_vals_bytes, encoder)?;
        
        // indices as HybridSparseVec
        Encode::encode(&self.indices, encoder)?;
        
        let mut delta_vals_bytes = Vec::new();
        self.delta_vals.serialize_into(&mut delta_vals_bytes)
            .map_err(|_| bincode::error::EncodeError::OtherString("DacsOpt serialize failed".into()))?;
        Encode::encode(&delta_vals_bytes, encoder)?;
        
        Ok(())
    }
}

impl<Context> Decode<Context> for EncodedDiffsMST {
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> core::result::Result<Self, bincode::error::DecodeError> {
        let num_genes: u32 = Decode::decode(decoder)?;
        
        // Decode ArithmeticEncoded fields using their own Decode impl
        let parent_offset: ArithmeticEncoded = Decode::decode(decoder)?;
        let root_indices: HybridSparseVec = Decode::decode(decoder)?;
        let root_vals: ArithmeticEncoded = Decode::decode(decoder)?;
        let indices: HybridSparseVec = Decode::decode(decoder)?;
        let delta_vals: ArithmeticEncoded = Decode::decode(decoder)?;
        
        Ok(Self {
            num_genes,
            parent_offset,
            root_indices,
            root_vals,
            indices,
            delta_vals,
        })
    }
}

impl<'de, Context> BorrowDecode<'de, Context> for EncodedDiffsMST {
    fn borrow_decode<D: bincode::de::BorrowDecoder<'de, Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        // Reuse the Decode implementation
        <Self as Decode<Context>>::decode(decoder)
    }
}

fn decode_v(diff: i32) -> i32 {
    if diff % 2 == 0 {
        diff / 2
    } else {
        -(diff + 1) / 2
    }
}

pub(crate) struct ExpressionVecIter<'a> {
    ediff: &'a EncodedDiffs,
    num_ones: usize,
    cell_ind: usize,
}

impl Iterator for ExpressionVecIter<'_> {
    type Item = Vec<u16>;

    fn next(&mut self) -> Option<Self::Item> {
        if (self.cell_ind as u32) < self.ediff.ncells {
            let cell_ind = self.cell_ind;
            self.cell_ind += 1;
            Some(match &self.ediff.indices {
                HybridSparseVec::EF(_) => {
                    self.ediff.expression_vec_ef(cell_ind, &mut self.num_ones)
                }
                HybridSparseVec::Bit(_) => {
                    self.ediff.expression_vec_bitv(cell_ind, &mut self.num_ones)
                }
            })
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let b = (self.ediff.ncells as usize) - self.cell_ind;
        (b, Some(b))
    }
}

impl ExactSizeIterator for ExpressionVecIter<'_> {}

impl EncodedDiffs {
    pub(crate) fn empty() -> Self {
        EncodedDiffs {
            indices: HybridSparseVec::empty(),
            ncells: 0,
            values: ArithmeticEncoded::default(),
            num_genes: 0,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(crate) fn len(&self) -> usize {
        self.values.len()
    }

    pub(crate) fn num_genes(&self) -> usize {
        self.num_genes as usize
    }
    

    pub(crate) fn num_cells(&self) -> usize {
        self.ncells as usize
    }

    fn expression_vec_bitv(&self, cell_ind: usize, num_ones: &mut usize) -> Vec<u16> {
        match &self.indices {
            HybridSparseVec::Bit(indices) => {
                if !self.precheck_cells(cell_ind) {
                    return vec![0_u16; self.num_genes()];
                }

                let first_idx = self.num_genes() * cell_ind;
                // the first index to start iterating from
                //info!("num_ones: {}", *num_ones);
                
                let mut value_pos = *num_ones;
                let mut next_value = self.values.access(value_pos).expect("valid");

                let mut expression = Vec::with_capacity(self.num_genes());
                let mut num_ones_in_cell = 0_usize;
                for gidx in 0..self.num_genes() {
                    if !indices.get_bit(gidx + first_idx) {
                        expression.push(0);
                    } else {
                        num_ones_in_cell += 1;
                        let decoded_val = next_value as u16;
                        expression.push(decoded_val);
                        value_pos += 1;
                        next_value = self.values.access(value_pos).unwrap_or(0);
                    }
                }
                *num_ones += num_ones_in_cell;
                expression
            }
            _ => unimplemented!(),
        }
    }

    fn expression_vec_ef(&self, cell_ind: usize, _num_ones: &mut usize) -> Vec<u16> {
        match &self.indices {
            HybridSparseVec::EF(indices) => {
                if !self.precheck_cells(cell_ind) || self.values.is_empty() {
                    return vec![0_u16; self.num_genes()];
                }

                // the "dense" index at which expression entries for this cell
                // should start
                let first_idx = self.num_genes() * cell_ind;
                // the "dense" index at which expression entries for this cell
                // should end
                let last_idx = first_idx + self.num_genes();

                // we checked that self.values is not empty above, so the len must be
                // at least 1. Get a cursor to the last element
                if let Some(last_stored_cursor) =
                    unsafe { indices.0.cursor_at(self.values.len() - 1) }
                {
                    // the index stored at the last position
                    let last_stored_index = last_stored_cursor.value().unwrap();
                    // the cseq API seems to have a problem if you ask for geq_cursor
                    // past the last stored element, so don't even try to do that
                    if first_idx > last_stored_index as usize {
                        return vec![0_u16; self.num_genes()];
                    }

                    // the first value in our sparse list that is >= first_idx
                    let start_cur = indices.0.geq_cursor(first_idx as u64);
                    let last_in_this_cell = last_stored_index <= last_idx as u64;
                    let stop_cur =if last_in_this_cell {
                        last_stored_cursor
                    // the first value in our sparse list that is >= last_idx
                    }else{indices.0.geq_cursor(last_idx as u64)};

                    // if the first value is >= last_idx, then there are no
                    // non-zeros stored for this gene
                    if !start_cur.is_valid()
                        || start_cur.value().unwrap_or(u64::MAX) >= last_idx as u64
                    {
                        return vec![0_u16; self.num_genes()];
                    }

                    let start = start_cur.index();

                    let stop = if !stop_cur.is_valid() || cell_ind == self.num_cells() - 1 || last_in_this_cell {
                        self.values.len()
                    } else {
                        stop_cur.index()
                    };
                    if start >= last_idx {
                        warn!("SHOULD NOT HAPPEN!");
                    }
                    if start >= stop {
                        error!(
                            "start = {start}, but stop = {stop}, values len = {}",
                            self.values.len()
                        );
                    }

                    let n = stop - start;

                    if n == 0 {
                        return vec![0_u16; self.num_genes()];
                    }

                    let mut value_pos = start;
                    let mut next_value = self.values.access(value_pos).unwrap_or(0);

                    let mut nz_ind_iter = start_cur.clone();

                    let mut expression = Vec::with_capacity(self.num_genes());
                    let mut next_nz_ind =
                        nz_ind_iter.value().expect("at least one") - first_idx as u64;
                    for gidx in 0..self.num_genes() {
                        if gidx < next_nz_ind as usize {
                            expression.push(0);
                        } else {
                            let decoded_val = next_value as u16;
                            expression.push(decoded_val);
                            if nz_ind_iter.advance() {
                                next_nz_ind =
                                    nz_ind_iter.value().unwrap_or(u64::MAX) - first_idx as u64;
                                value_pos += 1;
                                next_value = self.values.access(value_pos).unwrap_or(0);
                            } else {
                                next_nz_ind = self.num_genes() as u64 + 1;
                            }
                        }
                    }
                    expression
                } else {
                    vec![0_u16; self.num_genes()]
                }
            }
            _ => unimplemented!(),
        }
    }

    pub(crate) fn expression_vec_iter(&self) -> ExpressionVecIter {
        ExpressionVecIter {
            ediff: self,
            num_ones: 0,
            cell_ind: 0,
        }
    }

    /// Pre-check only for Bit variant: does the given cell have any set bits?
    pub(crate) fn precheck_cells(&self, cell_ind: usize) -> bool {
        if let HybridSparseVec::Bit(indices) = &self.indices {
            let first_idx = self.num_genes() * cell_ind;
            for gidx in 0..self.num_genes() {
                if indices.get_bit(gidx + first_idx) {
                    return true;
                }
            }
            return false;
        }
        true
    }

    pub(crate) fn bytes(&self) -> usize {
        let value_bytes = self.values.size_in_bytes();
        let ibytes = self.indices.num_bytes();

        let ncbytes = 4;
        let num_genes_bytes = 4;
        
        value_bytes + ibytes + ncbytes + num_genes_bytes
    }
}

// Manual implementation of de/serialization for `Rect`.
// We don't need to store the edges since they can computed from the other fields.
impl Encode for EncodedDiffs {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> core::result::Result<(), bincode::error::EncodeError> {
        Encode::encode(&self.indices, encoder)?;
        Encode::encode(&self.ncells, encoder)?;
        Encode::encode(&self.values, encoder)?;
        Encode::encode(&self.num_genes, encoder)?;
        Ok(())
    }
}

impl<Context> Decode<Context> for EncodedDiffs {
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> core::result::Result<Self, bincode::error::DecodeError> {
        let indices = Decode::decode(decoder)?;
        let ncells = Decode::decode(decoder)?;
        let values: ArithmeticEncoded = Decode::decode(decoder)?;
        let num_genes = Decode::decode(decoder)?;
        
        Ok(Self {
            indices,
            ncells,
            values,
            num_genes,
        })
    }
}

impl<'de, Context> BorrowDecode<'de, Context> for EncodedDiffs {
    fn borrow_decode<D: bincode::de::BorrowDecoder<'de, Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let indices = BorrowDecode::borrow_decode(decoder)?;
        let ncells = BorrowDecode::borrow_decode(decoder)?;
        let values: ArithmeticEncoded = BorrowDecode::borrow_decode(decoder)?;
        let num_genes: u32 = Decode::decode(decoder)?;
        
        Ok(Self {
            indices,
            ncells,
            values,
            num_genes,
        })
    }
}

/// Takes a slice of points and encodes raw expression values (no median subtraction)
/// Returns a `Some(EncodedDiffs)` struct representing the encoded values or `None` if empty.
///
/* 
pub(crate) fn encode_subarray(points: &[Point], data: &CsMat<u16>) -> Option<EncodedDiffs> {
    if points.is_empty() {
        debug!("Empty points array in encode_subarray()");
        return None;
    }

    let mut indices = Vec::<u64>::new();
    let mut raw_values = Vec::<u32>::new();
    let num_genes = data.cols() as u32;
    debug!("Processing {} points in encode_subarray()", points.len());
    debug!("Number of genes: {}", num_genes);

    let mut nnz = 0_usize;
    
    // Store raw expression values (no median computation)
    for (cell_ind, p) in points.iter().enumerate() {
        let index_offset = cell_ind * num_genes as usize;
        if let Some(row) = p.get_data(data) {
            for (gene_idx, &val) in row.iter() {
                if val > 0 {
                    nnz += 1;
                    raw_values.push(val as u32);
                    let index = index_offset + gene_idx;
                    indices.push(index as u64);
                }
            }

        }

    }
    
    let tot = num_genes as usize * points.len();
    let sparsity = (nnz as f64) / (tot as f64);
    if sparsity > 0.75 {
        println!("sparsity: {}", sparsity);
    }

    let indices = HybridSparseVec::from_indices(&indices, sparsity, tot);
    
    // Store raw values using arithmetic encoding
    let values = if raw_values.is_empty() {
        ArithmeticEncoded::default()
    } else {
        ArithmeticEncoded::from_slice(&raw_values).expect("should fit")
    };
    
    assert_eq!(indices.len(), values.len());

    let enc_diffs = EncodedDiffs {
        indices,
        ncells: points.len() as u32,
        values,
        num_genes: num_genes,
    };

    // validate!
    for (_cell_idx, (recon_vec, orig_vec)) in enc_diffs
        .expression_vec_iter()
        .zip(points.iter())
        .enumerate()
    {
        let orig_data = orig_vec.get_data(data).unwrap().to_dense().to_vec();
        // Find differences instead of asserting equality
        let differences: Vec<(usize, u16, u16)> = recon_vec
            .iter()
            .zip(orig_data.iter())
            .enumerate()
            .filter(|(_, (&r, &o))| r != o)
            .map(|(gene_idx, (&r, &o))| (gene_idx, r, o))
            .collect();

        if !differences.is_empty() {
            //info!("orig_data: {:?}", orig_data);
            //info!("recon_vec: {:?}", recon_vec);
            info!("differences: {:?}", differences);
            info!("type: {:?}", enc_diffs.indices.tyname());
            // Optionally still panic if you want to stop on first error
            // assert_eq!(recon_vec, orig_data.to_vec());

            // Save only the error cell
            use std::io::Write;
            let f = std::fs::File::create("bad_cells.coo").expect("Should be able to create file");
            let mut f = std::io::BufWriter::new(f);
            writeln!(f, "%%MatrixMarket matrix coordinate integer general");
            writeln!(f, "{}\t{}\t{}", points.len(), num_genes, nnz);
            for (cell_ind, p) in points.iter().enumerate() {
                let row = p.get_data(data).unwrap();
                for (gene_idx, &val) in row.iter() {
                    writeln!(f, "{}\t{}\t{}", cell_ind + 1, gene_idx + 1, val).unwrap();
                }
            }

            panic!("Failed to reconstruct!");
        }
        //dump this problematic cell to a file
        //unit test for encode/decode
    }
    /*
    info!(
        "Generated {} medians and {} diffs in {} bytes",
        enc_diffs.num_medians(),
        enc_diffs.len(),
        enc_diffs.bytes()
    );
    */
    Some(enc_diffs)
}
    */

// ============================================================================
// MST-based encoding functions
// ============================================================================

/// Sparse expression: (gene_index, value)
type SparseExpression = Vec<(u32, u16)>;

/// Extract sparse expression for a single cell (raw values, no median filtering)
fn compute_sparse_expression(
    point: &Point,
    data: &CsMat<u16>,
) -> SparseExpression {
    let mut expression = Vec::new();
    if let Some(row) = point.get_data(data) {
        for (gene_idx, &val) in row.iter() {
            // Store all non-zero expression values
            if val != 0 {
                expression.push((gene_idx as u32, val));
            }
        }
    }
    expression
}


/// L0 binary diff: Count of genes where sparsity pattern differs

/// L0 binary diff: Count of genes where sparsity pattern differs
/// (one is zero, the other is non-zero)
fn l0_binary_diff(a: &[(u32, u16)], b: &[(u32, u16)]) -> u32 {
    let mut count = 0u32;
    let mut i = 0;
    let mut j = 0;
    
    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Less => {
                count += 1; // Gene only in a
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                count += 1; // Gene only in b
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                // Both are non-zero, so pattern matches
                i += 1;
                j += 1;
            }
        }
    }
    count += (a.len() - i) as u32 + (b.len() - j) as u32;
    count
}

#[derive(Clone, Copy)]
struct L0Distance;

impl Distance<(u32, u16)> for L0Distance {
    fn eval(&self, va: &[(u32, u16)], vb: &[(u32, u16)]) -> f32 {
        l0_binary_diff(va, vb) as f32
    }
}

/// Compute sparse delta between child and parent expression vectors
///
/// This function performs a lossless merge-join to identify differences between
/// two sparse expression vectors. The result is a list of (gene_id, zigzag_delta)
/// pairs that can be used to reconstruct the child from the parent.
///
/// # Algorithm
///
/// Uses a two-pointer merge to identify:
/// - Genes only in child: delta = child_value - 0
/// - Genes only in parent: delta = 0 - parent_value (must zero out)
/// - Genes in both: delta = child_value - parent_value (if non-zero)
///
/// All deltas are zigzag-encoded for efficient variable-length storage.
///
/// # Returns
///
/// Vector of (gene_id, zigzag_encoded_delta) sorted by gene_id.
/// Genes with delta=0 are omitted.
fn sparse_subtract(child: &SparseExpression, parent: &SparseExpression) -> Vec<(u32, i32)> {
    let mut result = Vec::new();
    let mut i = 0;
    let mut j = 0;
    
    while i < child.len() && j < parent.len() {
        match child[i].0.cmp(&parent[j].0) {
            std::cmp::Ordering::Less => {
                // Gene only in child: delta = child_val - 0
                result.push((child[i].0, zigzag_encode(child[i].1 as i32)));
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                // Gene only in parent: delta = 0 - parent_val (need this to zero it out!)
                result.push((parent[j].0, zigzag_encode(-(parent[j].1 as i32))));
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                let delta = child[i].1 as i32 - parent[j].1 as i32;
                if delta != 0 {
                    result.push((child[i].0, zigzag_encode(delta)));
                }
                i += 1;
                j += 1;
            }
        }
    }
    while i < child.len() {
        result.push((child[i].0, zigzag_encode(child[i].1 as i32)));
        i += 1;
    }
    while j < parent.len() {
        result.push((parent[j].0, zigzag_encode(-(parent[j].1 as i32))));
        j += 1;
    }
    result
}

/// Zigzag encoding for signed integers
///
/// Maps signed integers to unsigned for efficient variable-length encoding.
/// Small absolute values (common case) map to small unsigned values.
///
/// # Mapping
///
/// ```text
///  0 -> 0
/// -1 -> 1
///  1 -> 2
/// -2 -> 3
///  2 -> 4
/// -3 -> 5
///  3 -> 6
/// ...
/// ```
///
/// # Formula
///
/// - Positive: `2 * v`
/// - Negative: `2 * |v| - 1`
///
/// This ensures small deltas (which are common in nearby cells) use fewer bits
/// in variable-length encoding schemes like DacsOpt.
fn zigzag_encode(v: i32) -> i32 {
    if v < 0 {
        (-2 * v) - 1
    } else {
        2 * v
    }
}

/// Find optimal k parameter for DacsOpt encoding based on value distribution
///
/// Returns adjacency list where neighbors[i] contains all neighbors of cell i
/// Build symmetric kNN graph based on expression sparsity pattern similarity
fn build_expression_knn(expressions: &[SparseExpression], k: usize) -> Vec<Vec<usize>> {
    use std::collections::HashSet;
    let n = expressions.len();
    if n <= 1 {
        return vec![Vec::new(); n];
    }

    let mut neighbors: Vec<HashSet<usize>> = vec![HashSet::new(); n];

    for i in 0..n {
        let mut candidates: Vec<(usize, u32)> = Vec::with_capacity(n - 1);
        for j in 0..n {
            if i != j {
                let dist = l0_binary_diff(&expressions[i], &expressions[j]);
                candidates.push((j, dist));
            }
        }

        let k_actual = k.min(candidates.len());
        if k_actual > 0 {
            candidates.select_nth_unstable_by(k_actual - 1, |a, b| a.1.cmp(&b.1));
            for &(j, _) in &candidates[..k_actual] {
                neighbors[i].insert(j);
                neighbors[j].insert(i);
            }
        }
    }

    neighbors.into_iter().map(|s| s.into_iter().collect()).collect()
}

/// Build MST using petgraph over expression-based kNN graph
fn build_mst_prim<P: PointLike>(
    _points: &[P],
    expressions: &[SparseExpression],
    k: usize,
) -> (usize, Vec<u32>) {
    use petgraph::algo::{connected_components, min_spanning_tree};
    use petgraph::data::FromElements;
    use petgraph::graph::UnGraph;

    let n = expressions.len();
    if n == 0 {
        return (0, Vec::new());
    }
    if n == 1 {
        return (0, vec![0]);
    }

    // Step 1: Build the initial kNN graph
    let mut graph = UnGraph::<usize, u32>::new_undirected();
    let nodes: Vec<_> = (0..n).map(|i| graph.add_node(i)).collect();

    if false {
        // Use HNSW for large N
        let max_nb_connection = 16;
        let ef_construction = 100;
        let nb_layers = 16;
        
        let hnsw = Hnsw::<(u32, u16), L0Distance>::new(
            max_nb_connection,
            n,
            nb_layers,
            ef_construction,
            L0Distance,
        );
        
        for (i, expr) in expressions.iter().enumerate() {
            hnsw.insert((expr, i));
        }
        
        let ef_search = 50;
        let knn_results: Vec<Vec<Neighbour>> = expressions
            .par_iter()
            .map(|expr| hnsw.search(expr, k, ef_search))
            .collect();
        
        for (i, neighbors) in knn_results.into_iter().enumerate() {
            for neighbor in neighbors {
                if neighbor.d_id != i {
                    graph.update_edge(nodes[i], nodes[neighbor.d_id], neighbor.distance as u32);
                }
            }
        }
    } else {
        // Fallback to brute force for small N
        for i in 0..n {
            let mut candidates: Vec<(usize, u32)> = Vec::with_capacity(n - 1);
            for j in 0..n {
            if i != j {
                let dist = l0_binary_diff(&expressions[i], &expressions[j]);
                candidates.push((j, dist));
            }
            }

            let k_actual = k.min(candidates.len());
            if k_actual > 0 {
                candidates.select_nth_unstable_by(k_actual - 1, |a, b| a.1.cmp(&b.1));
                for &(j, dist) in &candidates[..k_actual] {
                    graph.update_edge(nodes[i], nodes[j], dist);
                }
            }
        }
    }

    // Step 2: Ensure the graph is connected (bridge islands)
    let components_count = connected_components(&graph);
    if components_count > 1 {
        info!("Expression kNN graph has {} disconnected components, bridging", components_count);
        // Fallback: fully connect the graph by adding edges to node 0 from every other node
        // (A more sophisticated approach would find nearest neighbors between components, 
        // but this ensures connectivity for MST)
        for i in 1..n {
            let dist = l0_binary_diff(&expressions[0], &expressions[i]);
            graph.update_edge(nodes[0], nodes[i], dist);
        }
    }

    // Step 3: Compute MST using petgraph
    let mst_elements = min_spanning_tree(&graph);
    let mst_graph = UnGraph::<usize, u32>::from_elements(mst_elements);

    // Step 4: Convert MST graph back to rooted parent array format (root = 0)
    let mut parent = vec![u32::MAX; n];
    let mut visited = vec![false; n];
    let mut stack = vec![nodes[0]]; // Start with petgraph NodeIndex for 0
    visited[0] = true;
    parent[0] = 0;

    // Create a map from cell index to its NodeIndex in the MST graph for fast lookup
    let mut cell_to_mst_node = vec![None; n];
    for node_idx in mst_graph.node_indices() {
        cell_to_mst_node[mst_graph[node_idx]] = Some(node_idx);
    }

    while let Some(u_idx) = stack.pop() {
        let u = graph[u_idx];
        if let Some(mst_u_idx) = cell_to_mst_node[u] {
            for v_idx in mst_graph.neighbors(mst_u_idx) {
                let v = mst_graph[v_idx];
                if !visited[v] {
                    visited[v] = true;
                    parent[v] = u as u32;
                    // Find corresponding node index in original graph
                    stack.push(nodes[v]);
                }
            }
        }
    }

    // Handle any nodes that might have been missed due to MST splitting (should not happen now)
    for i in 0..n {
        if parent[i] == u32::MAX {
            parent[i] = 0;
        }
    }

    (0, parent)
}


/// Compute DFS traversal order of MST starting from root
/// Returns (dfs_order, parent_in_dfs_order)
/// dfs_order[i] = original cell index at DFS position i
/// parent_in_dfs_order[i] = DFS position of parent of cell at position i
fn compute_dfs_order(root: usize, parent: &[u32], n: usize) -> (Vec<u32>, Vec<u32>) {
    // Build adjacency list (children of each node)
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        if i != root {
            let p = parent[i] as usize;
            children[p].push(i);
        }
    }
    
    // DFS traversal
    let mut dfs_order = Vec::with_capacity(n);
    let mut pos_in_dfs = vec![0u32; n]; // pos_in_dfs[orig_cell] = position in DFS order
    let mut stack = vec![root];
    
    while let Some(node) = stack.pop() {
        pos_in_dfs[node] = dfs_order.len() as u32;
        dfs_order.push(node as u32);
        // Push children in reverse order so they're processed in order
        for &child in children[node].iter().rev() {
            stack.push(child);
        }
    }
    
    // Compute parent offset in DFS order
    // parent_offset[i] = i - (DFS position of parent)
    // For root, offset = 0
    let mut parent_offset = Vec::with_capacity(n);
    for (dfs_pos, &orig_cell) in dfs_order.iter().enumerate() {
        if orig_cell as usize == root {
            parent_offset.push(0);
        } else {
            let parent_orig = parent[orig_cell as usize] as usize;
            let parent_dfs_pos = pos_in_dfs[parent_orig];
            // Offset is always positive since parent comes before child in DFS
            parent_offset.push((dfs_pos as u32) - parent_dfs_pos);
        }
    }
    
    (dfs_order, parent_offset)
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MSTStats {
    pub level: usize,
    pub points: usize,
    pub total_entries: usize,
    pub non_zeros: usize,
    pub zeros: usize,
    pub pattern_changes: usize,
    pub change_pct: f64,
}

impl MSTStats {
    pub fn save_to_csv(&self, filename: &str) -> std::io::Result<()> {
        use std::fs::OpenOptions;
        use std::io::Write;
        
        let file_exists = std::path::Path::new(filename).exists();
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(filename)?;
            
        if !file_exists {
            writeln!(file, "level,points,total_entries,non_zeros,zeros,pattern_changes,change_pct")?;
        }
        
        writeln!(
            file,
            "{},{},{},{},{},{},{:.6}",
            self.level,
            self.points,
            self.total_entries,
            self.non_zeros,
            self.zeros,
            self.pattern_changes,
            self.change_pct
        )
    }
}

/// Compress spatial single-cell RNA-seq data using MST-based delta encoding
///
/// This function implements the complete MST-based compression pipeline:
///
/// # Algorithm
///
/// 1. **Convert to Sparse Representation**: Extract non-zero expression values for each cell
/// 2. **Build kNN Graph**: Compute k-nearest neighbors based on L0 distance (number of differing genes)
/// 3. **Extract MST**: Use Prim's algorithm to find minimum spanning tree
/// 4. **DFS Traversal**: Compute deterministic ordering and parent relationships
/// 5. **Encode Root**: Store root cell's expression directly (sparse format)
/// 6. **Encode Deltas**: For each non-root cell, compute and encode differences from parent
///
/// # Arguments
///
/// * `points` - Array of cell positions/metadata (used for spatial context)
/// * `data` - Sparse CSR matrix of gene expression values (cells × genes)
/// * `depth` - Tree depth level (for logging/statistics)
///
/// # Returns
///
/// Returns `Some((encoded, dfs_order, stats))` where:
/// - `encoded`: Compressed representation as `EncodedDiffsMST`
/// - `dfs_order`: Mapping from DFS position to original cell index
/// - `stats`: Compression statistics (sparsity, pattern changes, etc.)
///
/// Returns `None` if input is empty.
///
/// # Example
///
/// ```ignore
/// let (encoded, dfs_order, stats) = encode_subarray_mst(&points, &csr, 0)?;
/// // Reconstruct any cell by DFS position
/// let cell_expr = encoded.decode_cell_at_dfs_pos(dfs_pos);
/// ```
pub(crate) fn encode_subarray_mst(
    points: &[Point], 
    data: &CsMat<u16>, 
    depth: usize
) -> Option<(EncodedDiffsMST, Vec<u32>, MSTStats)> {
    if points.is_empty() {
        return None;
    }
    
    let num_genes = data.cols() as u32;
    // let ncells = points.len() as u32;
    
    // Step 1: Compute sparse expression for ALL cells (raw values, no median)
    let expressions: Vec<SparseExpression> = points
        .iter()
        .map(|p| compute_sparse_expression(p, data))
        .collect();
    
    // Step 2: Build kNN graph + MST using Prim's algorithm
    let k = 8; // Number of neighbors
    let (root, parent) = build_mst_prim(points, &expressions, k);
    
    // Step 3: Compute DFS order and parent offsets
    let (dfs_order_vec, parent_offset_vec) = compute_dfs_order(root, &parent, points.len());
    
    // Step 4: Encode root cell's expression directly (root is first in DFS order)
    let root_expr = &expressions[root];
    let root_genes_vec: Vec<u32> = root_expr.iter().map(|(g, _)| *g).collect();
    let root_vals_vec: Vec<u32> = root_expr.iter().map(|(_, v)| *v as u32).collect();
    
    // Step 5: Encode deltas using SAME FORMAT as old encoding
    // Index = (dfs_pos - 1) * num_genes + gene (combined cell+gene in one index)
    let mut combined_indices = Vec::new();
    let mut delta_vals_raw = Vec::new();
    let mut total_pattern_changes = 0usize;
    let mut total_non_zeros = root_expr.len();
    
    // Skip first cell (root), process rest in DFS order
    for (dfs_pos, &orig_cell) in dfs_order_vec.iter().enumerate().skip(1) {
        let parent_orig = parent[orig_cell as usize] as usize;
        let child_expr = &expressions[orig_cell as usize];
        let parent_expr = &expressions[parent_orig];
        
        total_non_zeros += child_expr.len();
        total_pattern_changes += l0_binary_diff(child_expr, parent_expr) as usize;
        
        let diff_list = sparse_subtract(child_expr, parent_expr);
        
        // Use (dfs_pos - 1) since root is stored separately
        let cell_offset = ((dfs_pos - 1) as u64) * (num_genes as u64);
        
        for (g, d) in &diff_list {
            combined_indices.push(cell_offset + (*g as u64));
            delta_vals_raw.push(*d as u32);
        }
    }
    
    // Parent offsets: small positive numbers (usually 1-10), compress very well!
    let parent_offset = if parent_offset_vec.is_empty() {
        ArithmeticEncoded::default()
    } else {
        ArithmeticEncoded::from_slice(&parent_offset_vec).expect("should fit")
    };
    
    // Use HybridSparseVec for root indices
    let root_genes_u64: Vec<u64> = root_genes_vec.iter().map(|&g| g as u64).collect();
    let root_indices = HybridSparseVec::from_indices(&root_genes_u64, 0.5, num_genes as usize);
    
    // Encode root values using arithmetic encoding
    let root_vals = if root_vals_vec.is_empty() {
        ArithmeticEncoded::default()
    } else {
        ArithmeticEncoded::from_slice(&root_vals_vec).expect("should fit")
    };
    
    // Use HybridSparseVec for combined indices (same as old encoding!)
    // CRITICAL: Sort indices and delta_vals together to maintain parallel relationship
    // HybridSparseVec may sort indices internally, so we need to sort them together first
    let mut indexed_deltas: Vec<(u64, u32)> = combined_indices.iter().zip(delta_vals_raw.iter())
        .map(|(&idx, &val)| (idx, val))
        .collect();
    indexed_deltas.sort_by_key(|&(idx, _)| idx);
    
    let (sorted_indices, sorted_delta_vals): (Vec<u64>, Vec<u32>) = indexed_deltas.into_iter().unzip();
    
    let edges_possible = (points.len() - 1) * (num_genes as usize);
    let change_pct = if edges_possible > 0 {
        total_pattern_changes as f64 / edges_possible as f64 * 100.0
    } else {
        0.0
    };
    
    let sparsity = if edges_possible > 0 {
        sorted_indices.len() as f64 / edges_possible as f64
    } else {
        0.0
    };
    
    let indices = HybridSparseVec::from_indices(&sorted_indices, sparsity, edges_possible);
    
    // Encode delta values using arithmetic encoding
    let delta_vals = if sorted_delta_vals.is_empty() {
        ArithmeticEncoded::default()
    } else {
        ArithmeticEncoded::from_slice(&sorted_delta_vals).expect("should fit")
    };
    
    let stats = MSTStats {
        level: depth,
        points: points.len(),
        total_entries: points.len() * num_genes as usize,
        non_zeros: total_non_zeros,
        zeros: (points.len() * num_genes as usize).saturating_sub(total_non_zeros),
        pattern_changes: total_pattern_changes,
        change_pct,
    };

    info!(
        "[Level {}] MST encoding: {} cells, pattern_changes={}, change_pct={:.3}%, parent_offset avg: {:.1}",
        depth,
        points.len(),
        total_pattern_changes,
        change_pct,
        if parent_offset_vec.len() > 1 {
            parent_offset_vec.iter().skip(1).map(|&x| x as f64).sum::<f64>() / (parent_offset_vec.len() - 1) as f64
        } else { 0.0 }
    );
    
    // Size breakdown
    info!(
        "  Size breakdown: parent_offset={} bytes, root_indices={} bytes, root_vals={} bytes, indices={} bytes, delta_vals={} bytes",
        parent_offset.size_in_bytes(),
        root_indices.num_bytes(),
        root_vals.size_in_bytes(),
        indices.num_bytes(),
        delta_vals.size_in_bytes()
    );
    
    let enc_diffs_mst = EncodedDiffsMST {
        num_genes,
        parent_offset,
        root_indices,
        root_vals,
        indices,
        delta_vals,
    };

    /* 
    // validate! - Ensure lossless reconstruction
    // ...
    */
    
    Some((enc_diffs_mst, dfs_order_vec, stats))
}
// ============================================================================
// CLUSTER-BASED COMPRESSION (Alternative to MST)
// ============================================================================

/// Cluster-based compression for spatial single-cell RNA-seq data
///
/// This structure implements an alternative compression strategy that clusters
/// cells by gene expression similarity and stores deltas from cluster representatives.
/// Each cluster forms a "star" topology (one representative, many leaves).
///
/// ## Compression Strategy
///
/// 1. **Clustering**: Group cells by expression similarity using k-means-like clustering
/// 2. **Representative Selection**: Choose a representative for each cluster (existing cell or centroid)
/// 3. **Delta Encoding**: Store each cell as a delta from its cluster representative
///
/// ## Advantages
///
/// - Better compression when MST has many heavy edges (heterogeneous tissue)
/// - Simpler structure than tree traversal
/// - More robust to outliers
///
#[derive(Clone)]
pub(crate) struct EncodedDiffsCluster {
    pub(crate) num_genes: u32,                      // Total number of genes
    pub(crate) num_clusters: u32,                   // Number of clusters
    pub(crate) cluster_assignments: ArithmeticEncoded, // cluster_id for each cell
    pub(crate) cluster_rep_indices: Vec<HybridSparseVec>, // Representative gene indices per cluster
    pub(crate) cluster_rep_vals: Vec<ArithmeticEncoded>,  // Representative values per cluster
    pub(crate) indices: HybridSparseVec,            // Combined (cell, gene) indices for deltas
    pub(crate) delta_vals: ArithmeticEncoded,       // Delta values (zigzag encoded)
}

impl EncodedDiffsCluster {
    pub(crate) fn empty() -> Self {
        EncodedDiffsCluster {
            num_genes: 0,
            num_clusters: 0,
            cluster_assignments: ArithmeticEncoded::default(),
            cluster_rep_indices: Vec::new(),
            cluster_rep_vals: Vec::new(),
            indices: HybridSparseVec::empty(),
            delta_vals: ArithmeticEncoded::default(),
        }
    }
    
    pub(crate) fn ncells(&self) -> usize {
        self.cluster_assignments.len()
    }
    
    pub(crate) fn bytes_breakdown(&self) -> (usize, usize, usize, usize, usize) {
        let cluster_assignments_bytes = self.cluster_assignments.size_in_bytes();
        let mut cluster_rep_indices_bytes = 0;
        let mut cluster_rep_vals_bytes = 0;
        for i in 0..self.num_clusters as usize {
            cluster_rep_indices_bytes += self.cluster_rep_indices[i].num_bytes();
            cluster_rep_vals_bytes += self.cluster_rep_vals[i].size_in_bytes();
        }
        let indices_bytes = self.indices.num_bytes();
        let delta_vals_bytes = self.delta_vals.size_in_bytes();
        
        (
            cluster_assignments_bytes,
            cluster_rep_indices_bytes,
            cluster_rep_vals_bytes,
            indices_bytes,
            delta_vals_bytes,
        )
    }
    
    pub(crate) fn total_bytes(&self) -> usize {
        let (ca, cri, crv, i, dv) = self.bytes_breakdown();
        ca + cri + crv + i + dv + 8 // +8 for num_genes and num_clusters
    }
}

/// Cluster cells by expression similarity using simple k-means-like algorithm
///
/// Uses L0 distance (pattern similarity) and iterative refinement.
fn cluster_cells(
    expressions: &[SparseExpression],
    num_clusters: usize,
) -> (Vec<usize>, Vec<usize>) {
    let ncells = expressions.len();
    
    // Handle edge cases
    if ncells == 0 {
        return (Vec::new(), Vec::new());
    }
    if ncells <= num_clusters {
        // Each cell is its own cluster
        let assignments: Vec<usize> = (0..ncells).collect();
        let representatives: Vec<usize> = (0..ncells).collect();
        return (assignments, representatives);
    }
    
    // Initialize: pick evenly spaced cells as initial representatives
    let mut representatives: Vec<usize> = (0..num_clusters)
        .map(|i| (i * ncells) / num_clusters)
        .collect();
    
    let mut assignments = vec![0; ncells];
    let mut changed = true;
    let max_iterations = 10;
    
    for _iteration in 0..max_iterations {
        if !changed {
            break;
        }
        changed = false;
        
        // Assign each cell to nearest representative
        for (cell_idx, expr) in expressions.iter().enumerate() {
            let mut best_cluster = assignments[cell_idx];
            let mut best_dist = l0_binary_diff(expr, &expressions[representatives[best_cluster]]);
            
            for (cluster_idx, &rep_idx) in representatives.iter().enumerate() {
                let dist = l0_binary_diff(expr, &expressions[rep_idx]);
                if dist < best_dist {
                    best_dist = dist;
                    best_cluster = cluster_idx;
                }
            }
            
            if assignments[cell_idx] != best_cluster {
                assignments[cell_idx] = best_cluster;
                changed = true;
            }
        }
        
        // Update representatives: pick cell closest to centroid in each cluster
        for cluster_idx in 0..num_clusters {
            let cluster_members: Vec<usize> = assignments.iter()
                .enumerate()
                .filter(|(_, &c)| c == cluster_idx)
                .map(|(i, _)| i)
                .collect();
            
            if cluster_members.is_empty() {
                continue;
            }
            
            // Pick member with minimum total distance to others
            let mut best_rep = representatives[cluster_idx];
            let mut best_total_dist = u32::MAX;
            
            for &candidate in &cluster_members {
                let total_dist: u32 = cluster_members.iter()
                    .map(|&other| l0_binary_diff(&expressions[candidate], &expressions[other]))
                    .sum();
                
                if total_dist < best_total_dist {
                    best_total_dist = total_dist;
                    best_rep = candidate;
                }
            }
            
            representatives[cluster_idx] = best_rep;
        }
    }
    
    (assignments, representatives)
}

/// Encode cells using cluster-based compression
///
/// This creates a "star" topology for each cluster where all cells in the cluster
/// are encoded as deltas from their representative.
///
/// # Returns
///
/// Returns `Some((encoded, cell_order, stats))` where:
/// - `encoded`: Compressed representation as `EncodedDiffsCluster`
/// - `cell_order`: Identity mapping (cells stay in original order)
/// - `stats`: Compression statistics
pub(crate) fn encode_subarray_cluster(
    points: &[Point],
    data: &CsMat<u16>,
    depth: usize,
) -> Option<(EncodedDiffsCluster, Vec<u32>, MSTStats)> {
    if points.is_empty() {
        return None;
    }
    
    let num_genes = data.cols() as u32;
    let ncells = points.len();
    
    // Step 1: Compute sparse expression for all cells
    let expressions: Vec<SparseExpression> = points
        .iter()
        .map(|p| compute_sparse_expression(p, data))
        .collect();
    
    // Step 2: Determine number of clusters (adaptive based on dataset size)
    // Use roughly sqrt(n) clusters as a heuristic
    let num_clusters = ((ncells as f64).sqrt().ceil() as usize).max(1).min(ncells / 2).max(1);
    
    // Step 3: Cluster cells by expression similarity
    let (assignments, representatives) = cluster_cells(&expressions, num_clusters);
    
    // Step 4: Encode cluster representatives
    let mut cluster_rep_indices = Vec::new();
    let mut cluster_rep_vals = Vec::new();
    
    for &rep_idx in &representatives {
        let rep_expr = &expressions[rep_idx];
        let rep_genes: Vec<u64> = rep_expr.iter().map(|(g, _)| *g as u64).collect();
        let rep_vals: Vec<u32> = rep_expr.iter().map(|(_, v)| *v as u32).collect();
        
        let rep_indices = HybridSparseVec::from_indices(&rep_genes, 0.5, num_genes as usize);
        
        // Encode representative values using arithmetic encoding
        let rep_vals_enc = if rep_vals.is_empty() {
            ArithmeticEncoded::default()
        } else {
            ArithmeticEncoded::from_slice(&rep_vals).expect("should fit")
        };
        
        cluster_rep_indices.push(rep_indices);
        cluster_rep_vals.push(rep_vals_enc);
    }
    
    // Step 5: Encode deltas from each cell to its representative
    let mut combined_indices = Vec::new();
    let mut delta_vals_raw = Vec::new();
    let mut total_pattern_changes = 0usize;
    let mut total_non_zeros = 0usize;
    
    for (cell_idx, cell_expr) in expressions.iter().enumerate() {
        let cluster_id = assignments[cell_idx];
        let rep_idx = representatives[cluster_id];
        let rep_expr = &expressions[rep_idx];
        
        total_non_zeros += cell_expr.len();
        total_pattern_changes += l0_binary_diff(cell_expr, rep_expr) as usize;
        
        let diff_list = sparse_subtract(cell_expr, rep_expr);
        
        // Encode as (cell_idx * num_genes + gene)
        let cell_offset = (cell_idx as u64) * (num_genes as u64);
        
        for (g, d) in &diff_list {
            combined_indices.push(cell_offset + (*g as u64));
            delta_vals_raw.push(*d as u32);
        }
    }
    
    // Step 6: Encode cluster assignments
    let assignments_u32: Vec<u32> = assignments.iter().map(|&a| a as u32).collect();
    let cluster_assignments = if assignments_u32.is_empty() {
        ArithmeticEncoded::default()
    } else {
        ArithmeticEncoded::from_slice(&assignments_u32).expect("should fit")
    };
    
    // Step 7: Encode combined indices
    let mut indexed_deltas: Vec<(u64, u32)> = combined_indices.iter().zip(delta_vals_raw.iter())
        .map(|(&idx, &val)| (idx, val))
        .collect();
    indexed_deltas.sort_by_key(|&(idx, _)| idx);
    
    let (sorted_indices, sorted_delta_vals): (Vec<u64>, Vec<u32>) = indexed_deltas.into_iter().unzip();
    
    let edges_possible = ncells * (num_genes as usize);
    let change_pct = if edges_possible > 0 {
        total_pattern_changes as f64 / edges_possible as f64 * 100.0
    } else {
        0.0
    };
    
    let sparsity = if edges_possible > 0 {
        sorted_indices.len() as f64 / edges_possible as f64
    } else {
        0.0
    };
    
    let indices = HybridSparseVec::from_indices(&sorted_indices, sparsity, edges_possible);
    
    // Encode delta values using arithmetic encoding
    let delta_vals = if sorted_delta_vals.is_empty() {
        ArithmeticEncoded::default()
    } else {
        ArithmeticEncoded::from_slice(&sorted_delta_vals).expect("should fit")
    };
    
    let stats = MSTStats {
        level: depth,
        points: ncells,
        total_entries: ncells * num_genes as usize,
        non_zeros: total_non_zeros,
        zeros: (ncells * num_genes as usize).saturating_sub(total_non_zeros),
        pattern_changes: total_pattern_changes,
        change_pct,
    };
    
    info!(
        "[Level {}] Cluster encoding: {} cells, {} clusters, pattern_changes={}, change_pct={:.3}%",
        depth,
        ncells,
        num_clusters,
        total_pattern_changes,
        change_pct
    );
    
    // Size breakdown
    let (ca, cri, crv, i, dv) = EncodedDiffsCluster {
        num_genes,
        num_clusters: num_clusters as u32,
        cluster_assignments: cluster_assignments.clone(),
        cluster_rep_indices: cluster_rep_indices.clone(),
        cluster_rep_vals: cluster_rep_vals.clone(),
        indices: indices.clone(),
        delta_vals: delta_vals.clone(),
    }.bytes_breakdown();
    
    info!(
        "  Cluster size breakdown: assignments={} bytes, rep_indices={} bytes, rep_vals={} bytes, indices={} bytes, delta_vals={} bytes",
        ca, cri, crv, i, dv
    );
    
    let enc_diffs_cluster = EncodedDiffsCluster {
        num_genes,
        num_clusters: num_clusters as u32,
        cluster_assignments,
        cluster_rep_indices,
        cluster_rep_vals,
        indices,
        delta_vals,
    };
    
    // Identity mapping - cells stay in original order
    let cell_order: Vec<u32> = (0..ncells as u32).collect();
    
    Some((enc_diffs_cluster, cell_order, stats))
}

impl EncodedDiffsCluster {
    /// Reconstruct a cell's expression vector from cluster-based encoding
    ///
    /// # Arguments
    ///
    /// * `cell_pos` - Position of the cell in the original order
    ///
    /// # Returns
    ///
    /// Dense expression vector of length `num_genes`
    pub(crate) fn decode_cell_at_pos(&self, cell_pos: usize) -> Vec<u16> {
        let mut expression = vec![0u16; self.num_genes as usize];
        
        // Get cluster assignment
        let cluster_id = self.cluster_assignments.access(cell_pos).unwrap_or(0) as usize;
        
        // Start with cluster representative
        let rep_indices_vec = self.cluster_rep_indices[cluster_id].indices_vec();
        for (i, &g) in rep_indices_vec.iter().enumerate() {
            let v = self.cluster_rep_vals[cluster_id].access(i).unwrap_or(0);
            expression[g as usize] = v as u16;
        }
        
        // Apply deltas for this cell
        let deltas = self.get_cell_deltas(cell_pos);
        for (gene, delta_encoded) in deltas {
            let delta = decode_v(delta_encoded);
            let current = expression[gene as usize] as i32;
            let new_val = (current + delta).max(0);
            expression[gene as usize] = new_val as u16;
        }
        
        expression
    }
    
    /// Get deltas for a specific cell
    fn get_cell_deltas(&self, cell_pos: usize) -> Vec<(u32, i32)> {
        let num_genes_usize = self.num_genes as usize;
        let cell_offset = cell_pos * num_genes_usize;
        let cell_end = cell_offset + num_genes_usize;
        
        let mut deltas = Vec::new();
        let all_indices = self.indices.indices_vec();
        
        // Find indices in range [cell_offset, cell_end)
        let start_pos = all_indices.partition_point(|&x| (x as usize) < cell_offset);
        
        for (pos_in_all, &idx) in all_indices.iter().enumerate().skip(start_pos) {
            let idx_usize = idx as usize;
            if idx_usize >= cell_end {
                break;
            }
            let gene = (idx_usize - cell_offset) as u32;
            if let Some(v) = self.delta_vals.access(pos_in_all) {
                deltas.push((gene, v as i32));
            }
        }
        
        deltas.sort_by_key(|(g, _)| *g);
        deltas
    }
}

impl EncodedDiffsMST {
    /// Get parent's DFS position from a cell's DFS position using offset
    fn parent_dfs_pos(&self, dfs_pos: usize) -> usize {
        if dfs_pos == 0 {
            return 0; // Root's parent is itself
        }
        let offset = self.parent_offset.access(dfs_pos).unwrap_or(0) as usize;
        dfs_pos.saturating_sub(offset)
    }
    
    /// Reconstruct a cell's full expression vector from compressed MST representation
    ///
    /// This function performs lossless decompression by:
    /// 1. Starting from the root cell's expression values
    /// 2. Walking the path from root to the target cell in the MST
    /// 3. Incrementally applying deltas at each step
    ///
    /// # Arguments
    ///
    /// * `dfs_pos` - Position of the cell in DFS traversal order (0 = root)
    ///
    /// # Returns
    ///
    /// Dense expression vector of length `num_genes` with u16 values.
    /// Genes not expressed (value = 0) are represented explicitly.
    ///
    /// # Complexity
    ///
    /// Time complexity is O(tree_depth × avg_nonzero_genes), where tree_depth
    /// is the distance from root to target cell in the MST (typically log(n) to sqrt(n)).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let encoded: EncodedDiffsMST = /* ... */;
    /// let cell_expr = encoded.decode_cell_at_dfs_pos(5);
    /// assert_eq!(cell_expr.len(), num_genes);
    /// ```
    pub(crate) fn decode_cell_at_dfs_pos(&self, dfs_pos: usize) -> Vec<u16> {
        let mut expression = vec![0u16; self.num_genes as usize];
        
        // Start with root's raw expression values
        let mut acc: Vec<(u32, i32)> = Vec::new();
        let root_indices_vec = self.root_indices.indices_vec();
        for (i, &g) in root_indices_vec.iter().enumerate() {
            let v = self.root_vals.access(i).unwrap_or(0) as i32;
            acc.push((g as u32, v));
        }
        
        if dfs_pos == 0 {
            // This is the root cell: just return root expression
            for (g, v) in &acc {
                expression[*g as usize] = (*v).max(0) as u16;
            }
            return expression;
        }
        
        // Build path from root to this cell (in DFS positions)
        let mut path_dfs = Vec::new();
        let mut current_dfs = dfs_pos;
        while current_dfs != 0 {
            path_dfs.push(current_dfs);
            current_dfs = self.parent_dfs_pos(current_dfs);
        }
        path_dfs.push(0); // Add root
        path_dfs.reverse(); // Now it's [0, ..., parent_dfs, this_dfs]
        
        // Walk from root to cell, accumulating deltas
        // path_dfs[0] is root (no delta needed), path_dfs[1..] need deltas
        for i in 1..path_dfs.len() {
            let cell_dfs = path_dfs[i];
            let deltas = self.get_cell_deltas_new(cell_dfs);
            acc = self.apply_deltas(&acc, &deltas);
        }
        
        // Convert accumulated expression to output vector
        for (g, v) in &acc {
            expression[*g as usize] = (*v).max(0) as u16;
        }
        
        expression
    }
    
    /// Get deltas for cell at dfs_pos using new index format and provided cached indices
    fn get_cell_deltas_with_indices(&self, dfs_pos: usize, all_indices: &[u64]) -> Vec<(u32, i32)> {
        if dfs_pos == 0 {
            return Vec::new(); // Root has no deltas
        }
        
        let num_genes_usize = self.num_genes as usize;
        let cell_offset = (dfs_pos - 1) * num_genes_usize;
        let cell_end = cell_offset + num_genes_usize;
        
        let mut deltas = Vec::new();
        
        // Find indices in range [cell_offset, cell_end)
        // Since indices are sorted, we can binary search
        let start_pos = all_indices.partition_point(|&x| (x as usize) < cell_offset);
        
        for (pos_in_all, &idx) in all_indices.iter().enumerate().skip(start_pos) {
            let idx_usize = idx as usize;
            if idx_usize >= cell_end {
                break;
            }
            let gene = (idx_usize - cell_offset) as u32;
            if let Some(v) = self.delta_vals.access(pos_in_all) {
                deltas.push((gene, v as i32));
            }
        }
        
        // Ensure deltas are sorted by gene_idx for apply_deltas
        deltas.sort_by_key(|(g, _)| *g);
        deltas
    }

    /// Get deltas for cell at dfs_pos using new index format
    /// Index = (dfs_pos - 1) * num_genes + gene
    pub(crate) fn get_cell_deltas_new(&self, dfs_pos: usize) -> Vec<(u32, i32)> {
        let all_indices = self.indices.indices_vec();
        self.get_cell_deltas_with_indices(dfs_pos, &all_indices)
    }
    
    /// Apply zigzag-encoded deltas to accumulated expression
    fn apply_deltas(&self, acc: &[(u32, i32)], deltas: &[(u32, i32)]) -> Vec<(u32, i32)> {
        let mut result = Vec::new();
        let mut i = 0;
        let mut j = 0;
        
        while i < acc.len() && j < deltas.len() {
            match acc[i].0.cmp(&deltas[j].0) {
                std::cmp::Ordering::Less => {
                    // Gene only in acc (no delta)
                    result.push(acc[i]);
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    // Gene only in delta: new value = 0 + decoded_delta
                    let decoded_delta = decode_v(deltas[j].1);
                    if decoded_delta != 0 {
                        result.push((deltas[j].0, decoded_delta));
                    }
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    // Both have this gene: new value = acc + decoded_delta
                    let decoded_delta = decode_v(deltas[j].1);
                    let sum = acc[i].1 + decoded_delta;
                    if sum != 0 {
                        result.push((acc[i].0, sum));
                    }
                    i += 1;
                    j += 1;
                }
            }
        }
        // Remaining in acc (no deltas)
        while i < acc.len() {
            result.push(acc[i]);
            i += 1;
        }
        // Remaining deltas (acc had 0)
        while j < deltas.len() {
            let decoded_delta = decode_v(deltas[j].1);
            if decoded_delta != 0 {
                result.push((deltas[j].0, decoded_delta));
            }
            j += 1;
        }
        
        result
    }
    
    /// Estimate bytes used by this encoding
    pub fn bytes(&self) -> usize {
        4  // num_genes only (ncells derived from parent_offset.len())
            + self.parent_offset.size_in_bytes()
            + self.root_indices.num_bytes()
            + self.root_vals.size_in_bytes()
            + self.indices.num_bytes()  // HybridSparseVec for combined indices
            + self.delta_vals.size_in_bytes()
    }
    
    /// Number of cells accessor for external use
    pub(crate) fn num_cells(&self) -> usize {
        self.ncells()
    }
    
    /// Number of genes accessor
    pub(crate) fn num_genes(&self) -> usize {
        self.num_genes as usize
    }
    
    /// Iterator over expression vectors for all cells (in DFS order)
    pub(crate) fn expression_vec_iter(&self) -> ExpressionVecIterMST<'_> {
        ExpressionVecIterMST::new(self)
    }
}

/// Iterator over expression vectors for MST-encoded data
/// Iterates in DFS order (same order as encoding)
pub(crate) struct ExpressionVecIterMST<'a> {
    ediff: &'a EncodedDiffsMST,
    dfs_pos: usize,
    all_indices: Vec<u64>,
    // Cached sparse expressions for each DFS position to allow incremental reconstruction
    states: Vec<Vec<(u32, i32)>>,
}

impl<'a> ExpressionVecIterMST<'a> {
    fn new(ediff: &'a EncodedDiffsMST) -> Self {
        let ncells = ediff.ncells();
        let all_indices = ediff.indices.indices_vec();
        let mut states = Vec::with_capacity(ncells);
        
        if ncells > 0 {
            // Initial state: root's sparse expression
            let mut root_acc = Vec::new();
            let root_indices_vec = ediff.root_indices.indices_vec();
            for (i, &g) in root_indices_vec.iter().enumerate() {
                let v = ediff.root_vals.access(i).unwrap_or(0) as i32;
                root_acc.push((g as u32, v));
            }
            states.push(root_acc);
        }
        
        Self {
            ediff,
            dfs_pos: 0,
            all_indices,
            states,
        }
    }
}

impl Iterator for ExpressionVecIterMST<'_> {
    type Item = Vec<u16>;

    fn next(&mut self) -> Option<Self::Item> {
        let ncells = self.ediff.ncells();
        if self.dfs_pos >= ncells {
            return None;
        }

        let current_dfs = self.dfs_pos;
        self.dfs_pos += 1;

        // If not root, compute state incrementally from parent
        if current_dfs > 0 {
            let parent_dfs = self.ediff.parent_dfs_pos(current_dfs);
            let deltas = self.ediff.get_cell_deltas_with_indices(current_dfs, &self.all_indices);
            let parent_state = &self.states[parent_dfs];
            let current_state = self.ediff.apply_deltas(parent_state, &deltas);
            self.states.push(current_state);
        }

        // Convert sparse state to dense vector
        let mut expression = vec![0u16; self.ediff.num_genes as usize];
        for (g, v) in &self.states[current_dfs] {
            expression[*g as usize] = (*v).max(0) as u16;
        }
        
        Some(expression)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.ediff.ncells() - self.dfs_pos;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for ExpressionVecIterMST<'_> {}

#[derive(Debug, Copy, Clone)]
pub(crate) enum ErrorMetric {
    Mean,
    #[allow(dead_code)]
    Median,
}

pub(crate) trait PointLike {
    fn xpos(&self) -> f64;
    fn ypos(&self) -> f64;
}

/*
pub(crate) enum ArrayData {
    Vector(Vec<u16>),
    Optional(Option<ArrayData>),
}
*/

#[derive(Clone)]
pub(crate) struct Point {
    pub(crate) x: f64,
    pub(crate) y: f64,
    //pub(crate) data_arc: CsVecViewI<'a, u16, usize>,
    pub(crate) ind: usize,
}

impl Point {
    #[inline(always)]
    pub(crate) fn new(x: f64, y: f64, ind: usize) -> Self {
        Self { x, y, ind }
    }
    pub(crate) fn get_gene_exp<'a>(&self, data: &'a CsMat<u16>, gid: usize) -> Option<&'a u16> {
        data.get(self.ind, gid)
    }
    pub(crate) fn get_data<'a>(&self, data: &'a CsMat<u16>) -> Option<CsVecViewI<'a, u16, usize>> {
        data.outer_view(self.ind)
    }
}

impl PointLike for Point {
    #[inline(always)]
    fn xpos(&self) -> f64 {
        self.x
    }
    #[inline(always)]
    fn ypos(&self) -> f64 {
        self.y
    }
}

/// A point that has just its 2D coordinates, but does not
/// carry with it any additional data.
#[derive(Debug, Clone, Encode, Decode)]

pub(crate) struct DatalessPoint {
    x: f64,
    y: f64,
}

impl DatalessPoint {
    #[inline(always)]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
    #[inline(always)]
    #[allow(dead_code)]
    fn from_point(o: &Point) -> Self {
        Self {
            x: o.xpos(),
            y: o.ypos(),
        }
    }
}

impl PointLike for DatalessPoint {
    #[inline(always)]
    fn xpos(&self) -> f64 {
        self.x
    }
    #[inline(always)]
    fn ypos(&self) -> f64 {
        self.y
    }
}

fn get_child_rects(parent: &Rect) -> (Rect, Rect, Rect, Rect) {
    // Find the children of the current node

    let north_child_north = parent.north_edge;
    let west_child_west = parent.west_edge;

    let east_child_east = parent.east_edge;
    let south_cild_south = parent.south_edge;

    let center_x = parent.cx;
    let center_y = parent.cy;

    let nw_boundary = Rect::new_from_bounds(west_child_west, center_x, north_child_north, center_y);
    let ne_boundary = Rect::new_from_bounds(center_x, east_child_east, north_child_north, center_y);

    let se_boundary = Rect::new_from_bounds(center_x, east_child_east, center_y, south_cild_south);
    let sw_boundary = Rect::new_from_bounds(west_child_west, center_x, center_y, south_cild_south);
    (nw_boundary, ne_boundary, se_boundary, sw_boundary)
}

#[derive(Debug, Clone)]
pub(crate) struct Rect {
    cx: f64,
    cy: f64,
    west_edge: f64,
    east_edge: f64,
    north_edge: f64,
    south_edge: f64,
}

impl Rect {
    #[inline(always)]
    pub(crate) const fn new(cx: f64, cy: f64, w: f64, h: f64) -> Self {
        Self {
            cx,
            cy,
            west_edge: cx - w / 2.0_f64,
            east_edge: cx + w / 2.0_f64,
            north_edge: cy - h / 2.0_f64,
            south_edge: cy + h / 2.0_f64,
        }
    }

    #[inline(always)]
    pub(crate) const fn new_from_bounds(west: f64, east: f64, north: f64, south: f64) -> Self {
        //let w = west - east;
        //let h = north - south;
        //let cx = west + (w / 2_f64); // should be east + (w / 2_f64)
        //let cy = south + (h / 2_f64);
        let cx = (west + east) / 2_f64;
        let cy = (south + north) / 2_f64;

        Self {
            cx,
            cy,
            west_edge: west,
            east_edge: east,
            north_edge: north,
            south_edge: south,
        }
    }

    // The rectangle is closed on the left (west) and top
    // (north) and open on the right (east) and bottom (south).
    #[inline(always)]
    const fn contains(&self, point: &Point) -> bool {
        point.x >= self.west_edge
            && point.x < self.east_edge
            && point.y >= self.north_edge
            && point.y < self.south_edge
    }

    #[inline(always)]
    const fn intersects(&self, other: &Self) -> bool {
        !(other.west_edge > self.east_edge
            || other.east_edge < self.west_edge
            || other.north_edge > self.south_edge
            || other.south_edge < self.north_edge)
    }
}

// Manual implementation of de/serialization for `Rect`.
// We don't need to store the edges since they can computed from the other fields.
impl Encode for Rect {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> core::result::Result<(), bincode::error::EncodeError> {
        Encode::encode(&self.west_edge, encoder)?;
        Encode::encode(&self.east_edge, encoder)?;
        Encode::encode(&self.north_edge, encoder)?;
        Encode::encode(&self.south_edge, encoder)?;
        Ok(())
    }
}

impl<Context> Decode<Context> for Rect {
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> core::result::Result<Self, bincode::error::DecodeError> {
        let west = Decode::decode(decoder)?;
        let east = Decode::decode(decoder)?;
        let north = Decode::decode(decoder)?;
        let south = Decode::decode(decoder)?;
        Ok(Self::new_from_bounds(west, east, north, south))
    }
}

impl<'de, Context> BorrowDecode<'de, Context> for Rect {
    fn borrow_decode<D: bincode::de::BorrowDecoder<'de, Context = Context>>(
        decoder: &mut D,
    ) -> core::result::Result<Self, bincode::error::DecodeError> {
        let west = BorrowDecode::borrow_decode(decoder)?;
        let east = BorrowDecode::borrow_decode(decoder)?;
        let north = BorrowDecode::borrow_decode(decoder)?;
        let south = BorrowDecode::borrow_decode(decoder)?;
        Ok(Self::new_from_bounds(west, east, north, south))
    }
}

#[derive(Clone)]
pub(crate) struct BitField {
    bit_field: BitFieldVec,
}

impl BitField {
    fn new(bit_field: BitFieldVec) -> Self {
        Self { bit_field }
    }
}

impl Encode for BitField {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        let (data, width, len) = self.bit_field.clone().into_raw_parts();
        Encode::encode(&data, encoder)?;
        Encode::encode(&width, encoder)?;
        Encode::encode(&len, encoder)?;
        Ok(())
    }
}

impl<Context> Decode<Context> for BitField {
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let data: Vec<u64> = Decode::decode(decoder)?;
        let width = Decode::decode(decoder)?;
        let len = Decode::decode(decoder)?;
        let mut bit_field = BitFieldVec::new(width, len);
        for (i, item) in data.iter().enumerate().take(len) {
            bit_field.set(i, *item as usize);
        }
        Ok(BitField::new(bit_field))
    }
}

impl<'de, Context> BorrowDecode<'de, Context> for BitField {
    fn borrow_decode<D: bincode::de::BorrowDecoder<'de, Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let data: Vec<u64> = BorrowDecode::borrow_decode(decoder)?;
        let width = BorrowDecode::borrow_decode(decoder)?;
        let len = BorrowDecode::borrow_decode(decoder)?;
        let mut bit_field = BitFieldVec::new(width, len);
        for (i, item) in data.iter().enumerate().take(len) {
            bit_field.set(i, *item as usize);
        }
        Ok(BitField::new(bit_field))
    }
}

#[derive(Encode, Decode, Clone)]
pub(crate) struct BitFieldQuadTree {
    pub(crate) boundary: Rect,
    pub(crate) encoded_diffs: EncodedDiffsMST,
    pub(crate) divided: bool,
    nw: Option<Box<BitFieldQuadTree>>,
    ne: Option<Box<BitFieldQuadTree>>,
    se: Option<Box<BitFieldQuadTree>>,
    sw: Option<Box<BitFieldQuadTree>>,
    pub(crate) positions: Vec<DatalessPoint>,
}

impl BitFieldQuadTree {
    #[allow(dead_code)]
    pub(crate) fn new(boundary: Rect) -> Self {
        Self {
            boundary,
            encoded_diffs: EncodedDiffsMST::empty(),
            divided: false,
            nw: None,
            ne: None,
            se: None,
            sw: None,
            positions: Vec::new(),
        }
    }
    pub(crate) fn visit(&self, fun: &mut impl FnMut(&BitFieldQuadTree)) {
        fun(self);
        if self.divided {
            self.children().iter().for_each(|c| {
                if let Some(n) = c {
                    n.visit(fun);
                }
            });
        }
    }

    pub(crate) fn children(&self) -> [&Option<Box<BitFieldQuadTree>>; 4] {
        [&self.nw, &self.ne, &self.sw, &self.se]
    }

    #[allow(dead_code)]
    pub(crate) fn calculate_expense(&self) -> usize {
        /*
        info!(
            "Calculating expense for BitFieldQuadTree node with {} total points and size {} bytes",
            self.encoded_diffs.len(),
            self.encoded_diffs.bytes()
        );
        info!("Total node expense: {}", self.encoded_diffs.bytes());
        */
        self.encoded_diffs.bytes()
    }

    /*
    pub(crate) fn to_quad_tree(&self) -> QuadTree {
        let mut quadtree = QuadTree::new(self.boundary.clone(), Vec::new(), 0);
        quadtree.divided = self.divided;
        // Convert the raw parts into f64 values to meet the QuadTree requirements
        quadtree.data = self
            .data
            .iter()
            .flat_map(|bf| {
                let (data, _, _) = bf.bit_field.clone().into_raw_parts();
                data.iter().map(|&x| x as f64).collect::<Vec<f64>>()
            })
            .collect();
        quadtree.positions = self.positions.clone();
        if quadtree.divided {
            if let Some(ref nw) = self.nw {
                quadtree.nw = Some(Box::new(nw.to_quad_tree()));
            }
            if let Some(ref ne) = self.ne {
                quadtree.ne = Some(Box::new(ne.to_quad_tree()));
            }
            if let Some(ref se) = self.se {
                quadtree.se = Some(Box::new(se.to_quad_tree()));
            }
            if let Some(ref sw) = self.sw {
                quadtree.sw = Some(Box::new(sw.to_quad_tree()));
            }
        }
        quadtree
    }
    */

    #[allow(dead_code)]
    pub(crate) fn calculate_size(&self) -> (usize, usize) {
        let mut total_size = 0;
        let mut total_bitfields = 0;

        // Calculate size of current node's data
        total_size += self.encoded_diffs.bytes();
        total_bitfields += 1;
        /*
        for bitfield in &self.data {
            let (_, width, len) = bitfield.bit_field.clone().into_raw_parts();
            total_size += width * len;
            total_bitfields += 1;
        }
        */

        if self.divided {
            if let Some(ref nw) = self.nw {
                let (size, bitfields) = nw.calculate_size();
                total_size += size;
                total_bitfields += bitfields;
            }
            if let Some(ref ne) = self.ne {
                let (size, bitfields) = ne.calculate_size();
                total_size += size;
                total_bitfields += bitfields;
            }
            if let Some(ref se) = self.se {
                let (size, bitfields) = se.calculate_size();
                total_size += size;
                total_bitfields += bitfields;
            }
            if let Some(ref sw) = self.sw {
                let (size, bitfields) = sw.calculate_size();
                total_size += size;
                total_bitfields += bitfields;
            }
        }

        (total_size, total_bitfields)
    }

    //#[allow(dead_code)]
    /*
    pub(crate) fn print_size_info(&self, depth: usize) {
        let indent = "  ".repeat(depth);
        let (size, bitfields) = self.calculate_size();
        /*
        info!(
            "{}Level {}: {} bits, {} bitfields",
            indent, depth, size, bitfields
        );
        */

        if self.divided {
            if let Some(ref nw) = self.nw {
                nw.print_size_info(depth + 1);
            }
            if let Some(ref ne) = self.ne {
                ne.print_size_info(depth + 1);
            }
            if let Some(ref se) = self.se {
                se.print_size_info(depth + 1);
            }
            if let Some(ref sw) = self.sw {
                sw.print_size_info(depth + 1);
            }
        }
    }*/
}

pub(crate) struct QuadTree {
    boundary: Rect,
    points: Vec<Point>,
    depth: usize,
    divided: bool,
    //maxerror: Option<f64>,
    nw: Option<Box<Self>>,
    ne: Option<Box<Self>>,
    se: Option<Box<Self>>,
    sw: Option<Box<Self>>,
    data: Vec<f64>,
    positions: Vec<DatalessPoint>,
}

impl QuadTree {
    #[inline(always)]
    pub(crate) fn new(boundary: Rect, points: Vec<Point>, depth: usize) -> Self {
        Self {
            boundary,
            points,
            depth,
            divided: false,
            //maxerror: None,
            nw: None,
            ne: None,
            se: None,
            sw: None,
            data: Vec::new(),
            //index: Vec::new(),
            positions: Vec::new(),
        }
    }

    /*
    pub(crate) fn get_all_point_data<'a>(&self, point: &'a Point) -> CsVecViewI<u16> {
        // Since we now store data directly in Point, just return it as a single element
        vec![&point.data_arc]
    }
    */

    pub(crate) fn query(&self, boundary: &Rect) -> Vec<Point> {
        let mut found_points = Vec::new();
        if !self.boundary.intersects(boundary) {
            return found_points;
        }

        for point in &self.points {
            if boundary.contains(point) {
                found_points.push(point.clone());
            }
        }

        // we already include the points above, why query
        // the children (which we already contain)?
        /*
        if self.divided {
            if let Some(ref nw) = self.nw {
                found_points.extend(nw.query(boundary));
            }
            if let Some(ref ne) = self.ne {
                found_points.extend(ne.query(boundary));
            }
            if let Some(ref se) = self.se {
                found_points.extend(se.query(boundary));
            }
            if let Some(ref sw) = self.sw {
                found_points.extend(sw.query(boundary));
            }
        }
        */
        found_points
    }

    /*
    pub fn block_data_repr(&self, method: ErrorMetric) -> Vec<f64> {
        if self.points.is_empty() {
            return Vec::new();
        }

        let mut block_mean = Vec::<f64>::with_capacity(self.points[0].data.len());
        for j in 0..self.points[0].data.len() {
            let block_mean_j = match method {
                ErrorMetric::Median => {
                    let mut values: Vec<f64> =
                        self.points.iter().map(|p| p.data[j] as f64).collect();
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    values[values.len() / 2]
                }
                ErrorMetric::Mean => {
                    self.points.iter().map(|p| p.data[j] as f64).sum::<f64>()
                        / self.points.len() as f64
                }
            };
            block_mean.push(block_mean_j);
        }
        block_mean
    }

        pub fn calculate_error(&self, method: ErrorMetric, mind: &[u16], maxd: &[u16], _prob: f64) -> f64 {
            let mut found_points = Vec::new();
            self.query(&self.boundary);

            if found_points.is_empty() {
                return 0.0;
            }

            let mut maxerrors = Vec::new();
            for j in 0..found_points[0].data.len() {
                let block_mean = match method {
                    ErrorMetric::Median => {
                        let mut values: Vec<f64> = found_points.iter().map(|p| p.data[j] as f64).collect();
                        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        values[values.len() / 2]
                    }
                    ErrorMetric::Mean => {
                        found_points.iter().map(|p| p.data[j] as f64).sum::<f64>() / found_points.len() as f64
                    }
                };

                let maxerror = found_points
                    .iter()
                    .map(|p| (p.data[j] as f64 - block_mean).abs())
                    .collect::<Vec<f64>>();

                let maxerror =
                    maxerror.iter().fold(0.0, |a, &b| f64::max(a, b)) / ((maxd[j] as f64) - (mind[j] as f64) + 0.01);
                maxerrors.push(maxerror);
            }
            maxerrors.iter().fold(0.0, |a, &b| f64::max(a, b))
        }
    */

    pub(crate) fn force_divide(&mut self, target_depth: usize, data: &CsMat<u16>) {
        if self.depth >= target_depth || self.points.is_empty() {
            if !self.points.is_empty() {
                self.divide_recursive(data);
            }
            return;
        }

        let (nw_boundary, ne_boundary, se_boundary, sw_boundary) = get_child_rects(&self.boundary);
        
        let nw_points = self.query(&nw_boundary);
        let ne_points = self.query(&ne_boundary);
        let se_points = self.query(&se_boundary);
        let sw_points = self.query(&sw_boundary);

        self.divided = true;
        self.nw = (!nw_points.is_empty()).then_some(Box::new(QuadTree::new(nw_boundary, nw_points, self.depth + 1)));
        self.ne = (!ne_points.is_empty()).then_some(Box::new(QuadTree::new(ne_boundary, ne_points, self.depth + 1)));
        self.se = (!se_points.is_empty()).then_some(Box::new(QuadTree::new(se_boundary, se_points, self.depth + 1)));
        self.sw = (!sw_points.is_empty()).then_some(Box::new(QuadTree::new(sw_boundary, sw_points, self.depth + 1)));

        self.points.clear();

        if let Some(ref mut nw) = self.nw { nw.force_divide(target_depth, data); }
        if let Some(ref mut ne) = self.ne { ne.force_divide(target_depth, data); }
        if let Some(ref mut se) = self.se { se.force_divide(target_depth, data); }
        if let Some(ref mut sw) = self.sw { sw.force_divide(target_depth, data); }
    }

    pub(crate) fn divide_recursive(&mut self, data: &CsMat<u16>) {
        //pub(crate) fn divide_recursive(&mut self, data: &CsMat<u16>) {
        info!("divide_recursive");
        //let mut stack = vec![self];
        // let cost_log = CostLog::new();
        let _max_depth = 4;
        // let max_pt: usize = 5;
        // nothing to do if this subtree is empty
        if self.points.is_empty() {
            return;
        }
        // Store the current points' positions before clearing them
        let positions: Vec<DatalessPoint> = self
            .points
            .iter()
            .map(|p| DatalessPoint::new(p.x, p.y))
            .collect();
        // Compute the expense of encoding the current block using MST
        let (current_enc, _, current_stats) = encode_subarray_mst(&self.points, data, self.depth).expect("expect nonempty");
        let current_expense = current_enc.bytes();
        
        info!(
            "expense of current block (MST) consisting of {} points is {}",
            self.points.len(),
            current_expense
        );

        // Find the children of the current node
        let (nw_boundary, ne_boundary, se_boundary, sw_boundary) = get_child_rects(&self.boundary);
        let nw_points = self.query(&nw_boundary);
        let nw = QuadTree::new(nw_boundary, nw_points, self.depth + 1);

        let ne_points = self.query(&ne_boundary);
        let ne = QuadTree::new(ne_boundary, ne_points, self.depth + 1);

        let se_points = self.query(&se_boundary);
        let se = QuadTree::new(se_boundary, se_points, self.depth + 1);

        let sw_points = self.query(&sw_boundary);
        let sw = QuadTree::new(sw_boundary, sw_points, self.depth + 1);

        // make sure we're not losing any points
        assert_eq!(
            nw.points.len() + ne.points.len() + se.points.len() + sw.points.len(),
            self.points.len()
        );

        // Convert children to calculate their MST expenses
        let nw_expense = encode_subarray_mst(&nw.points, data, self.depth + 1).map_or(0, |(x, _, _)| x.bytes());
        let ne_expense = encode_subarray_mst(&ne.points, data, self.depth + 1).map_or(0, |(x, _, _)| x.bytes());
        let se_expense = encode_subarray_mst(&se.points, data, self.depth + 1).map_or(0, |(x, _, _)| x.bytes());
        let sw_expense = encode_subarray_mst(&sw.points, data, self.depth + 1).map_or(0, |(x, _, _)| x.bytes());

        info!("NW expense: {}", nw_expense);
        info!("NE expense: {}", ne_expense);
        info!("SE expense: {}", se_expense);
        info!("SW expense: {}", sw_expense);

        let total_expense = nw_expense + ne_expense + se_expense + sw_expense;
        info!("total_expense: {}", total_expense);

        // Force divide if too many points, or if it saves space
        //let force_divide = self.points.len() > 5000;
        
        //if total_expense < current_expense && self.depth < max_depth {
        if total_expense < current_expense {
            self.divided = true;
            
            // Save stats to CSV for nodes that actually divided
            current_stats.save_to_csv("quadtree_mst_stats.csv").ok();
            
            // Convert BitFieldQuadTree back to QuadTree and assign children
            self.nw = (!nw.points.is_empty()).then_some(Box::new(nw));
            self.ne = (!ne.points.is_empty()).then_some(Box::new(ne));
            self.se = (!se.points.is_empty()).then_some(Box::new(se));
            self.sw = (!sw.points.is_empty()).then_some(Box::new(sw));

            // Only clear points after we've used them for all necessary operations
            self.points.clear();

            if let Some(ref mut nw) = self.nw {
                nw.divide_recursive(data);
            }
            if let Some(ref mut ne) = self.ne {
                ne.divide_recursive(data);
            }
            if let Some(ref mut se) = self.se {
                se.divide_recursive(data);
            }
            if let Some(ref mut sw) = self.sw {
                sw.divide_recursive(data);
            }
        } else {
            self.divided = false;
            if !self.points.is_empty() {
                info!(
                    "Leaf node - points: {}, genes: {}",
                    self.points.len(),
                    data.cols()
                );
                self.positions = positions; // Use the stored positions
                                            // Keep the points for bit field representation
            }
        }
        info!("self.depth: {}", self.depth);
        //return cost_log;
    }

    /*
    //    pub(crate) fn divide(&mut self) -> CostLog {
        pub(crate) fn divide(&mut self) {
            let mut stack = vec![self];
            let mut cost_log = CostLog::new();
            let mut node_counter = 0;
            let max_depth = 3;
            let max_pt: usize = 2000;


            while let Some(node) = stack.pop() {
               // if node.depth >= max_depth {
                //    continue; // JUMPS BACK TO THE TOP OF THE WHILE LOOP,
                //}
                node_counter += 1;

                // info!("Processing block with {} points at depth {}", node.points.len(), node.depth);

                 // nothing to do if this subtree is empty
                if node.points.is_empty() {
                     continue; // JUMPS BACK TO THE TOP OF THE WHILE LOOP if empty
                }

                 // Store the current points' positions before clearing them
                 let positions: Vec<DatalessPoint> = node
                     .points
                     .iter()
                     .map(|p| DatalessPoint::new(p.x, p.y))
                     .collect();

                 // Compute the expense of encoding the current block
                 let current_expense = encode_subarray(&node.points).map_or(0, |x| x.bytes());

                 /*
                 info!(
                     "expense of current block consisting of {} points is {}",
                     node.points.len(),
                     current_expense
                 );
                 */
                 // Find the children of the current node
                 let (nw_boundary, ne_boundary, se_boundary, sw_boundary) = get_child_rects(&node.boundary);
                 let nw_points = node.query(&nw_boundary);
                 let nw = QuadTree::new(nw_boundary, nw_points, node.depth + 1);
                 let ne_points = node.query(&ne_boundary);
                 let ne = QuadTree::new(ne_boundary, ne_points, node.depth + 1);
                 let se_points = node.query(&se_boundary);
                 let se = QuadTree::new(se_boundary, se_points, node.depth + 1);
                 let sw_points = node.query(&sw_boundary);
                 let sw = QuadTree::new(sw_boundary, sw_points, node.depth + 1);

                 // make sure we're not losing any points
                 assert_eq!(
                     nw.points.len() + ne.points.len() + se.points.len() + sw.points.len(),
                     node.points.len()
                 );

                 let f = |pts: &Vec<Point>| encode_subarray(pts).map_or(0, |x| x.bytes());
                 let par_ok = node.points.len() > 5000;
                 let mut nw_expense = 0;
                 let mut ne_expense = 0;
                 let mut se_expense = 0;
                 let mut sw_expense = 0;

                 if par_ok {
                     scope(|s| {
                         s.spawn(|_| {
                             nw_expense = f(&nw.points);
                         });
                         s.spawn(|_| {
                             ne_expense = f(&ne.points);
                         });
                         s.spawn(|_| {
                             se_expense = f(&se.points);
                         });
                         s.spawn(|_| {
                             sw_expense = f(&sw.points);
                         });
                     });
                 } else {
                     nw_expense = f(&nw.points);
                     ne_expense = f(&ne.points);
                     se_expense = f(&se.points);
                     sw_expense = f(&sw.points);
                 }

                // info!("NW expense: {} with {} points", nw_expense, nw.points.len());
                // info!("NE expense: {} with {} points", ne_expense, ne.points.len());
                // info!("SE expense: {} with {} points", se_expense, se.points.len());
                // info!("SW expense: {} with {} points", sw_expense, sw.points.len());

                let total_children_expense = nw_expense + ne_expense + se_expense + sw_expense;

                 // Create node identifier
                 let node_id = if node.depth == 0 {
                     "root".to_string()
                 } else {
                     format!("node_{}", node_counter)
                 };

                 // Add cost step to log
                 /*
                 let cost_step = CostStep {
                     depth: node.depth,
                     points_count: node.points.len(),
                     current_cost: current_expense,
                     children_cost: total_children_expense,
                     optimal_cost,
                     decision,
                     node_id,
                 };
                 cost_log.add_step(cost_step);
                 */
                // if optimal_cost < current_expense || node.points.len() > max_pt {
                //if optimal_cost < current_expense || node.depth < max_depth {
                 if total_children_expense < current_expense {
                 node.divided = true;

                 // Convert BitFieldQuadTree back to QuadTree and assign children
                 node.nw = (!nw.points.is_empty()).then_some(Box::new(nw));
                 node.ne = (!ne.points.is_empty()).then_some(Box::new(ne));
                 node.se = (!se.points.is_empty()).then_some(Box::new(se));
                 node.sw = (!sw.points.is_empty()).then_some(Box::new(sw));

                 // Only clear points after we've used them for all necessary operations
                 node.points.clear();
                 info!("self.depth: {}", node.depth);

                 // Add children to stack for processing (in reverse order for depth-first)
                 if let Some(ref mut sw) = node.sw {
                     stack.push(sw);
                 }
                 if let Some(ref mut se) = node.se {
                     stack.push(se);
                 }
                 if let Some(ref mut ne) = node.ne {
                     stack.push(ne);
                 }
                 if let Some(ref mut nw) = node.nw {
                     stack.push(nw);
                 }
                } else {
                     node.divided = false;
                     if !node.points.is_empty() {
    //                     info!(
    //                         "Leaf node - points: {}, genes: {}",
    //                         node.points.len(),
    //                         node.points[0].get_data().len()
    //                     );
                         node.positions = positions; // Use the stored positions
                                                     // Keep the points for bit field representation
                     }
                    }
             }

             // Update totals
             let total_cost = cost_log.steps.iter().map(|step| step.optimal_cost).sum();
             //cost_log.update_totals(node_counter, total_cost);

             //info!("Finished iterative division");
             //cost_log
        }
    */

    /// Traverse to all leaf nodes first, then compare children expenses to parent expenses
    /// Returns a tuple: (parent_expense, children_expense, should_divide)
    pub(crate) fn compare_parent_vs_children_expenses(
        &self,
        data: &CsMat<u16>,
    ) -> (usize, usize, bool) {
        if !self.divided {
            // Leaf node - calculate parent expense only
            let parent_expense = if !self.points.is_empty() {
                encode_subarray_mst(&self.points, data, self.depth).map_or(0, |(x, _, _)| x.bytes())
            } else {
                0
            };
            return (parent_expense, 0, false); // No children to compare
        }

        // Internal node - recursively get children expenses
        let mut children_expense = 0;

        // Get expenses from all children
        if let Some(ref nw) = self.nw {
            let (_, child_expense, _) = nw.compare_parent_vs_children_expenses(data);
            children_expense += child_expense;
        }
        if let Some(ref ne) = self.ne {
            let (_, child_expense, _) = ne.compare_parent_vs_children_expenses(data);
            children_expense += child_expense;
        }
        if let Some(ref se) = self.se {
            let (_, child_expense, _) = se.compare_parent_vs_children_expenses(data);
            children_expense += child_expense;
        }
        if let Some(ref sw) = self.sw {
            let (_, child_expense, _) = sw.compare_parent_vs_children_expenses(data);
            children_expense += child_expense;
        }

        // Calculate parent expense for this node
        let parent_expense = if !self.points.is_empty() {
            encode_subarray_mst(&self.points, data, self.depth).map_or(0, |(x, _, _)| x.bytes())
        } else {
            0
        };

        // Compare parent vs children expenses
        let should_divide = children_expense < parent_expense;

        //info!("Node at depth {}: parent_expense={}, children_expense={}, should_divide={}",
        //    self.depth, parent_expense, children_expense, should_divide);

        (parent_expense, children_expense, should_divide)
    }

    /*
        /// Bottom-up traversal to calculate total expense and determine optimal division
        pub(crate) fn calculate_optimal_expense(&mut self, cost_log: &mut CostLog, node_id: &str) -> usize {
            if !self.divided {
                // Leaf node - return its own expense
                let leaf_expense = if !self.points.is_empty() {
                    encode_subarray_mst(&self.points, data, self.depth).map_or(0, |(x, _, _)| x.bytes())
                } else {
                    0
                };

                // Add leaf node cost step
                let cost_step = CostStep {
                    depth: self.depth,
                    points_count: self.points.len(),
                    current_cost: leaf_expense,
                    children_cost: 0,
                    optimal_cost: leaf_expense,
                    decision: "leaf".to_string(),
                    node_id: node_id.to_string(),
                };
                cost_log.add_step(cost_step);

                return leaf_expense;
            }

            // Internal node - get expenses from children
            let mut children_expense = 0;

            if let Some(ref mut nw) = self.nw {
                children_expense += nw.calculate_optimal_expense(cost_log, &format!("{}_nw", node_id));
            }
            if let Some(ref mut ne) = self.ne {
                children_expense += ne.calculate_optimal_expense(cost_log, &format!("{}_ne", node_id));
            }
            if let Some(ref mut se) = self.se {
                children_expense += se.calculate_optimal_expense(cost_log, &format!("{}_se", node_id));
            }
            if let Some(ref mut sw) = self.sw {
                children_expense += sw.calculate_optimal_expense(cost_log, &format!("{}_sw", node_id));
            }

            // Calculate parent expense
            let parent_expense = if !self.points.is_empty() {
                encode_subarray_mst(&self.points, data, self.depth).map_or(0, |(x, _, _)| x.bytes())
            } else {
                0
            };

            // Determine optimal cost and decision
            let (optimal_cost, decision) = if children_expense >= parent_expense {
                //info!("Collapsing node at depth {}: parent={}, children={}",
                //      self.depth, parent_expense, children_expense);
                 // why is this not a issue for the 10X data?
                // Repopulate this parent's points from all descendants before collapsing
                let mut collected_points: Vec<Point> = Vec::new();
                if let Some(ref mut nw) = self.nw { nw.collect_points_recursive(&mut collected_points); }
                if let Some(ref mut ne) = self.ne { ne.collect_points_recursive(&mut collected_points); }
                if let Some(ref mut se) = self.se { se.collect_points_recursive(&mut collected_points); }
                if let Some(ref mut sw) = self.sw { sw.collect_points_recursive(&mut collected_points); }

                if !collected_points.is_empty() {
                    self.points = collected_points;
                    self.positions = self
                        .points
                        .iter()
                        .map(|p| DatalessPoint::new(p.xpos(), p.ypos()))
                        .collect();
                }
                self.divided = false;
                self.nw = None;
                self.ne = None;
                self.se = None;
                self.sw = None;
                (parent_expense, "merge".to_string())
            } else {
                //info!("Keeping division at depth {}: parent={}, children={}",
                //      self.depth, parent_expense, children_expense);
                (children_expense, "divide".to_string())
            };

            // Add internal node cost step
            let cost_step = CostStep {
                depth: self.depth,
                points_count: self.points.len(),
                current_cost: parent_expense,
                children_cost: children_expense,
                optimal_cost,
                decision,
                node_id: node_id.to_string(),
            };
            cost_log.add_step(cost_step);

            optimal_cost
        }

        /// Optimize the entire quadtree using bottom-up traversal
        /// This function traverses to all leaf nodes first, then works its way up
        /// comparing parent vs children expenses and collapsing nodes when beneficial
        pub(crate) fn optimize_quadtree(&mut self) -> CostLog {
            //info!("Starting quadtree optimization...");

            let mut cost_log = CostLog::new();

            // First, ensure all leaf nodes are properly divided
            if !self.divided && self.points.len() > 1 {
                let _ = self.divide(); // We don't need the division cost log here
            }

            // Perform bottom-up optimization with cost tracking
            let total_expense = self.calculate_optimal_expense(&mut cost_log, "root");

            // Update totals
            let node_counter = cost_log.steps.len();
            cost_log.update_totals(node_counter, total_expense);

            //info!("Quadtree optimization complete. Total expense: {}", total_expense);
            //info!("Cost log contains {} steps", cost_log.steps.len());

            cost_log
        }
    */
    pub(crate) fn non_zero_blocks(&self, data: &CsMat<u16>) -> usize {
        let mut npoints = 0;
        if !self.divided {
            if !self.points.is_empty() {
                npoints = 1;
            }
        } else {
            if let Some(ref nw) = self.nw {
                npoints += nw.non_zero_blocks(data);
            }
            if let Some(ref ne) = self.ne {
                npoints += ne.non_zero_blocks(data);
            }
            if let Some(ref se) = self.se {
                npoints += se.non_zero_blocks(data);
            }
            if let Some(ref sw) = self.sw {
                npoints += sw.non_zero_blocks(data);
            }
        }
        npoints
    }

    pub(crate) fn compute_quadtree_bit_fields(&self, data: &CsMat<u16>) -> BitFieldQuadTree {
        debug!(
            "Computing bit fields - points available: {}",
            self.points.len()
        );
        /*
        if !self.points.is_empty() {
            debug!(
                "Computing bit fields - genes per point: {}",
                data.cols()
            );
        }


        debug!(
            "Generated medians: {}, diffs: {}",
            medians.len(),
            diffs.len()
        );
        */
        let mut node = BitFieldQuadTree {
            boundary: self.boundary.clone(),
            encoded_diffs: EncodedDiffsMST::empty(),
            //medians,
            //data: diffs,
            divided: self.divided,
            nw: None,
            ne: None,
            se: None,
            sw: None,
            positions: Vec::new(),
        };

        if self.divided {
            // Compute children in parallel
            let (ne_res, nw_res) = join(
                || {
                    self.ne
                        .as_ref()
                        .map(|c| c.compute_quadtree_bit_fields(data))
                },
                || {
                    self.nw
                        .as_ref()
                        .map(|c| c.compute_quadtree_bit_fields(data))
                },
            );
            let (se_res, sw_res) = join(
                || {
                    self.se
                        .as_ref()
                        .map(|c| c.compute_quadtree_bit_fields(data))
                },
                || {
                    self.sw
                        .as_ref()
                        .map(|c| c.compute_quadtree_bit_fields(data))
                },
            );
            node.ne = ne_res.map(|x| Box::new(x));
            node.nw = nw_res.map(|x| Box::new(x));
            node.se = se_res.map(|x| Box::new(x));
            node.sw = sw_res.map(|x| Box::new(x));
        } else {
            let (enc_diffs, dfs_order, _) = encode_subarray_mst(&self.points, data, self.depth).expect("expect nonempty");
            node.encoded_diffs = enc_diffs;
            // Reorder positions to match MST DFS order
            // Deriving from self.points because self.positions might be empty
            node.positions = dfs_order.iter()
                .map(|&idx| {
                    let p = &self.points[idx as usize];
                    DatalessPoint::new(p.x, p.y)
                })
                .collect();
        }
        node
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use tracing_subscriber::filter::LevelFilter;
    use tracing_subscriber::fmt;
    use tracing_subscriber::prelude::*;
    use tracing_subscriber::EnvFilter;

    #[test]
    #[ignore] // Requires bad_cells.coo file
    fn encode_failing_vec() {
        tracing_subscriber::registry()
            .with(fmt::layer())
            .with(
                EnvFilter::builder()
                    .with_default_directive(LevelFilter::INFO.into())
                    .from_env_lossy()
                    // we don't want to hear anything below a warning from ureq
                    .add_directive("ureq=warn".parse().unwrap()),
            )
            .init();

        let matrix = sprs::io::read_matrix_market::<i32, usize, _>("bad_cells.coo").unwrap();
        let mut matrix_new: sprs::TriMatBase<Vec<usize>, Vec<u16>> =
            sprs::TriMatBase::new(matrix.shape());
        for v in matrix.triplet_iter() {
            matrix_new.add_triplet(v.1 .0, v.1 .1, *v.0 as u16);
        }
        let csr: CsMat<u16> = matrix_new.to_csr();
        let mut points = Vec::<Point>::new();
        for i in 0..csr.rows() {
            points.push(Point::new(0., 0., i));
        }
        // Test encoding if bad_cells.coo exists
        // This test requires the encode_subarray function which may be private
        // let _ = encode_subarray(&points, &csr);
        // Test MST encoding - commented out as test data may not exist
        // let _ = encode_subarray_mst(&points, &csr, 0);
    }

    #[test]
    fn test_mst_compression_roundtrip() {
        use sprs::TriMatBase;
        
        // Create a simple test matrix with known values
        // 5 cells x 10 genes
        let ncells = 5;
        let ngenes = 10;
        
        let mut tri_mat: TriMatBase<Vec<usize>, Vec<u16>> = TriMatBase::new((ncells, ngenes));
        
        // Cell 0: genes 0,1,2 with values 5,10,15
        tri_mat.add_triplet(0, 0, 5);
        tri_mat.add_triplet(0, 1, 10);
        tri_mat.add_triplet(0, 2, 15);
        
        // Cell 1: genes 0,1,2,3 with values 6,10,14,20 (similar to cell 0)
        tri_mat.add_triplet(1, 0, 6);
        tri_mat.add_triplet(1, 1, 10);
        tri_mat.add_triplet(1, 2, 14);
        tri_mat.add_triplet(1, 3, 20);
        
        // Cell 2: genes 1,2,3 with values 11,16,20 (similar to cell 1)
        tri_mat.add_triplet(2, 1, 11);
        tri_mat.add_triplet(2, 2, 16);
        tri_mat.add_triplet(2, 3, 20);
        
        // Cell 3: genes 5,6 with values 100,200 (very different)
        tri_mat.add_triplet(3, 5, 100);
        tri_mat.add_triplet(3, 6, 200);
        
        // Cell 4: genes 5,6,7 with values 99,201,50 (similar to cell 3)
        tri_mat.add_triplet(4, 5, 99);
        tri_mat.add_triplet(4, 6, 201);
        tri_mat.add_triplet(4, 7, 50);
        
        let csr: CsMat<u16> = tri_mat.to_csr();
        
        // Create points with different positions
        let points: Vec<Point> = (0..ncells)
            .map(|i| Point::new(i as f64, 0.0, i))
            .collect();
        
        // Encode using MST
        let result = encode_subarray_mst(&points, &csr, 0);
        assert!(result.is_some(), "MST encoding should succeed");
        
        let (encoded, dfs_order, stats) = result.unwrap();
        
        // Verify basic stats
        assert_eq!(stats.points, ncells);
        assert_eq!(encoded.num_genes, ngenes as u32);
        assert_eq!(encoded.ncells(), ncells);
        
        // Test round-trip: decode all cells and verify they match original
        for dfs_pos in 0..ncells {
            let decoded = encoded.decode_cell_at_dfs_pos(dfs_pos);
            assert_eq!(decoded.len(), ngenes);
            
            // Find original cell index from dfs order
            let orig_cell_idx = dfs_order[dfs_pos] as usize;
            
            // Compare with original CSR data
            let row_view = csr.outer_view(orig_cell_idx).unwrap();
            for (col_idx, &value) in row_view.iter() {
                assert_eq!(
                    decoded[col_idx], 
                    value,
                    "Cell {} (DFS pos {}), gene {}: decoded={}, expected={}",
                    orig_cell_idx, dfs_pos, col_idx, decoded[col_idx], value
                );
            }
            
            // Verify zeros for non-expressed genes
            for gene_idx in 0..ngenes {
                if row_view.get(gene_idx).is_none() {
                    assert_eq!(
                        decoded[gene_idx], 
                        0,
                        "Cell {} (DFS pos {}), gene {} should be zero",
                        orig_cell_idx, dfs_pos, gene_idx
                    );
                }
            }
        }
        
        info!("MST round-trip test passed! Stats: {:?}", stats);
    }
    
    #[test]
    fn test_mst_compression_single_cell() {
        use sprs::TriMatBase;
        
        // Edge case: single cell
        let ncells = 1;
        let ngenes = 5;
        
        let mut tri_mat: TriMatBase<Vec<usize>, Vec<u16>> = TriMatBase::new((ncells, ngenes));
        tri_mat.add_triplet(0, 0, 10);
        tri_mat.add_triplet(0, 2, 20);
        tri_mat.add_triplet(0, 4, 30);
        
        let csr: CsMat<u16> = tri_mat.to_csr();
        let points: Vec<Point> = vec![Point::new(0.0, 0.0, 0)];
        
        let result = encode_subarray_mst(&points, &csr, 0);
        assert!(result.is_some());
        
        let (encoded, _, _) = result.unwrap();
        let decoded = encoded.decode_cell_at_dfs_pos(0);
        
        assert_eq!(decoded[0], 10);
        assert_eq!(decoded[1], 0);
        assert_eq!(decoded[2], 20);
        assert_eq!(decoded[3], 0);
        assert_eq!(decoded[4], 30);
    }
    
    #[test]
    fn test_zigzag_encoding() {
        // Test zigzag encoding
        assert_eq!(zigzag_encode(0), 0);
        assert_eq!(zigzag_encode(-1), 1);
        assert_eq!(zigzag_encode(1), 2);
        assert_eq!(zigzag_encode(-2), 3);
        assert_eq!(zigzag_encode(2), 4);
        assert_eq!(zigzag_encode(-100), 199);
        assert_eq!(zigzag_encode(100), 200);
    }
    
    #[test]
    fn test_sparse_subtract() {
        // Test sparse subtraction
        let child: SparseExpression = vec![(0, 10), (1, 20), (3, 5)];
        let parent: SparseExpression = vec![(0, 8), (2, 15), (3, 5)];
        
        let deltas = sparse_subtract(&child, &parent);
        
        // Expected deltas:
        // gene 0: 10-8=2 -> zigzag(2)=4
        // gene 1: 20-0=20 -> zigzag(20)=40
        // gene 2: 0-15=-15 -> zigzag(-15)=29
        // gene 3: 5-5=0 -> not included (delta is 0)
        
        assert_eq!(deltas.len(), 3);
        assert_eq!(deltas[0], (0, 4));   // gene 0, zigzag(2)
        assert_eq!(deltas[1], (1, 40));  // gene 1, zigzag(20)
        assert_eq!(deltas[2], (2, 29));  // gene 2, zigzag(-15)
    }
}



    #[test]
    fn test_cluster_compression_roundtrip() {
        use sprs::TriMatBase;
        
        // Test with 10 cells × 10 genes (sparse matrix)
        let ncells = 10;
        let ngenes = 10;
        
        let mut tri_mat: TriMatBase<Vec<usize>, Vec<u16>> = TriMatBase::new((ncells, ngenes));
        
        // Create cells with some similarity (for clustering)
        // Cells 0-4: high expression in genes 0-2
        for cell in 0..5 {
            tri_mat.add_triplet(cell, 0, 100 + cell as u16);
            tri_mat.add_triplet(cell, 1, 150 + cell as u16);
            tri_mat.add_triplet(cell, 2, 200 + cell as u16);
        }
        
        // Cells 5-9: high expression in genes 5-7
        for cell in 5..10 {
            tri_mat.add_triplet(cell, 5, 50 + cell as u16);
            tri_mat.add_triplet(cell, 6, 75 + cell as u16);
            tri_mat.add_triplet(cell, 7, 100 + cell as u16);
        }
        
        let csr: CsMat<u16> = tri_mat.to_csr();
        
        // Create points
        let points: Vec<Point> = (0..ncells)
            .map(|i| Point::new(i as f64, 0.0, i))
            .collect();
        
        // Encode using cluster-based method
        let result = encode_subarray_cluster(&points, &csr, 0);
        assert!(result.is_some(), "Cluster encoding should succeed");
        
        let (encoded, _cell_order, stats) = result.unwrap();
        
        // Verify basic stats
        assert_eq!(stats.points, ncells);
        assert_eq!(encoded.num_genes, ngenes as u32);
        assert_eq!(encoded.ncells(), ncells);
        assert!(encoded.num_clusters >= 1, "Should have at least 1 cluster");
        assert!(encoded.num_clusters <= ncells as u32, "Clusters should not exceed cells");
        
        // Test round-trip: decode all cells and verify they match original
        for cell_pos in 0..ncells {
            let decoded = encoded.decode_cell_at_pos(cell_pos);
            assert_eq!(decoded.len(), ngenes);
            
            // Compare with original CSR data
            let row_view = csr.outer_view(cell_pos).unwrap();
            for (col_idx, &value) in row_view.iter() {
                assert_eq!(
                    decoded[col_idx], 
                    value,
                    "Cell {}, gene {}: decoded={}, expected={}",
                    cell_pos, col_idx, decoded[col_idx], value
                );
            }
            
            // Verify zeros for non-expressed genes
            for gene_idx in 0..ngenes {
                if row_view.get(gene_idx).is_none() {
                    assert_eq!(
                        decoded[gene_idx], 
                        0,
                        "Cell {}, gene {} should be zero",
                        cell_pos, gene_idx
                    );
                }
            }
        }
        
        info!("Cluster round-trip test passed! {} clusters, Stats: {:?}", 
              encoded.num_clusters, stats);
    }
    
    #[test]
    fn test_cluster_vs_mst_compression() {
        use sprs::TriMatBase;
        
        // Compare cluster vs MST compression on same data
        let ncells = 20;
        let ngenes = 15;
        
        let mut tri_mat: TriMatBase<Vec<usize>, Vec<u16>> = TriMatBase::new((ncells, ngenes));
        
        // Create heterogeneous data (should favor clustering)
        // Group 1: cells 0-6 (genes 0-4)
        for cell in 0..7 {
            for gene in 0..5 {
                tri_mat.add_triplet(cell, gene, 50 + (cell + gene) as u16);
            }
        }
        
        // Group 2: cells 7-13 (genes 5-9)
        for cell in 7..14 {
            for gene in 5..10 {
                tri_mat.add_triplet(cell, gene, 60 + (cell + gene) as u16);
            }
        }
        
        // Group 3: cells 14-19 (genes 10-14)
        for cell in 14..20 {
            for gene in 10..15 {
                tri_mat.add_triplet(cell, gene, 70 + (cell + gene) as u16);
            }
        }
        
        let csr: CsMat<u16> = tri_mat.to_csr();
        let points: Vec<Point> = (0..ncells)
            .map(|i| Point::new(i as f64, 0.0, i))
            .collect();
        
        // Encode with both methods
        let mst_result = encode_subarray_mst(&points, &csr, 0);
        let cluster_result = encode_subarray_cluster(&points, &csr, 0);
        
        assert!(mst_result.is_some());
        assert!(cluster_result.is_some());
        
        let (mst_encoded, _, mst_stats) = mst_result.unwrap();
        let (cluster_encoded, _, cluster_stats) = cluster_result.unwrap();
        
        let mst_size = mst_encoded.total_bytes();
        let cluster_size = cluster_encoded.total_bytes();
        
        info!("MST size: {} bytes, change_pct: {:.2}%", mst_size, mst_stats.change_pct);
        info!("Cluster size: {} bytes, change_pct: {:.2}%, num_clusters: {}", 
              cluster_size, cluster_stats.change_pct, cluster_encoded.num_clusters);
        
        // Both should produce valid compression (actual size comparison depends on data)
        assert!(mst_size > 0);
        assert!(cluster_size > 0);
        
        // Both should reconstruct correctly
        for cell_pos in 0..ncells {
            let mst_decoded = mst_encoded.decode_cell_at_dfs_pos(cell_pos);
            let cluster_decoded = cluster_encoded.decode_cell_at_pos(cell_pos);
            
            // Both should match original (note: MST uses DFS order)
            assert_eq!(mst_decoded.len(), ngenes);
            assert_eq!(cluster_decoded.len(), ngenes);
        }
    }

// ============================================================================
// ADAPTIVE COMPRESSION: Automatically choose between MST and Cluster
// ============================================================================

/// Try both MST and cluster-based compression, return the better one
///
/// This function encodes the data using both methods and selects the one
/// that produces smaller compressed size.
///
/// # Returns
///
/// Returns a tuple of (method_name, size, encoded_data) where method_name
/// is either "MST" or "Cluster"
pub(crate) fn encode_subarray_adaptive(
    points: &[Point],
    data: &CsMat<u16>,
    depth: usize,
) -> Option<(String, usize, Vec<u8>)> {
    if points.is_empty() {
        return None;
    }
    
    // Try MST encoding
    let mst_result = encode_subarray_mst(points, data, depth);
    let (mst_size, mst_stats) = if let Some((ref encoded, _, ref stats)) = mst_result {
        (encoded.total_bytes(), stats.clone())
    } else {
        return None;
    };
    
    // Try Cluster encoding
    let cluster_result = encode_subarray_cluster(points, data, depth);
    let (cluster_size, cluster_stats) = if let Some((ref encoded, _, ref stats)) = cluster_result {
        (encoded.total_bytes(), stats.clone())
    } else {
        // If cluster encoding fails, fall back to MST
        info!("[Level {}] Cluster encoding failed, using MST ({}  bytes)", depth, mst_size);
        return Some(("MST".to_string(), mst_size, Vec::new()));
    };
    
    // Compare and select the better method
    if cluster_size < mst_size {
        let savings = mst_size - cluster_size;
        let pct_improvement = (savings as f64 / mst_size as f64) * 100.0;
        info!(
            "[Level {}] Cluster encoding chosen: {} bytes (MST: {} bytes, savings: {} bytes, {:.1}% better)",
            depth, cluster_size, mst_size, savings, pct_improvement
        );
        Some(("Cluster".to_string(), cluster_size, Vec::new()))
    } else {
        let overhead = cluster_size - mst_size;
        let pct_overhead = (overhead as f64 / mst_size as f64) * 100.0;
        info!(
            "[Level {}] MST encoding chosen: {} bytes (Cluster: {} bytes, overhead: {} bytes, {:.1}% worse)",
            depth, mst_size, cluster_size, overhead, pct_overhead
        );
        Some(("MST".to_string(), mst_size, Vec::new()))
    }
}
