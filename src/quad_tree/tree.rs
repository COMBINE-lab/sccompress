use crate::bits::HybridSparseVec;
use bincode::{BorrowDecode, Decode, Encode};
use bitm::{self, BitAccess};
use rayon::join;
use sux::prelude::{BitFieldSlice, BitFieldVec};
use sux::traits::BitFieldSliceCore;
use sux::traits::BitFieldSliceMut;
use tracing::{debug, error, info, warn};
//use rayon::scope;
use sprs::{CsMat, CsVecViewI};
use sucds::int_vectors::DacsOpt;
use sucds::int_vectors::Access;
use sucds::Serializable;

// MST-based encoding (using Prim's algorithm - petgraph no longer needed for MST)
use sucds::bit_vectors::Rank9Sel;
use sucds::bit_vectors::prelude::*;
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
    pub(crate) diffs: DacsOpt,
    pub(crate) medians: BitFieldVec,
}

/// MST-based delta encoding for a block of cells
/// Stores deltas along MST edges instead of per-cell residuals
#[derive(Clone)]
pub(crate) struct EncodedDiffsMST {
    pub(crate) ncells: u32,
    pub(crate) num_genes: u32,
    pub(crate) medians: BitFieldVec,           // per-gene medians
    pub(crate) root: u32,                       // root cell index
    pub(crate) parent: Vec<u32>,                // parent[c] = parent of cell c in MST (root has self)
    pub(crate) root_residual_genes: DacsOpt,    // sparse gene indices for root
    pub(crate) root_residual_vals: DacsOpt,     // sparse values for root (zigzag encoded)
    pub(crate) delta_genes: DacsOpt,            // concatenated gene indices for all deltas
    pub(crate) delta_vals: DacsOpt,             // concatenated delta values (zigzag encoded)
    pub(crate) boundary: Rank9Sel,              // bitvector: 1 marks start of each cell's deltas
    pub(crate) order: Vec<u32>,                 // traversal order (non-root cells)
}

fn decode_v(diff: i32) -> i32 {
    if diff % 2 == 0 {
        diff / 2
    } else {
        -((diff - 1) / 2)
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
            diffs: DacsOpt::default(),
            medians: BitFieldVec::with_capacity(0, 0),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(crate) fn num_medians(&self) -> usize {
        self.medians.len()
    }

    pub(crate) fn len(&self) -> usize {
        self.diffs.len()
    }

    pub(crate) fn num_genes(&self) -> usize {
        self.medians.len()
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
                
                //let mut diff_iter = self.diffs.iter().skip(*num_ones);
                let mut diff_pos = *num_ones;
                let mut next_diff = self.diffs.access(diff_pos).expect("valid");

                let mut expression = Vec::with_capacity(self.num_genes());
                let mut num_ones_in_cell = 0_usize;
                for gidx in 0..self.num_genes() {
                    if !indices.get_bit(gidx + first_idx) {
                        expression.push(0);
                    } else {
                        num_ones_in_cell += 1;
                        let ddiff = decode_v(next_diff as i32);
                        let decoded_val = (ddiff + self.medians.get(gidx) as i32) as u16;
                        expression.push(decoded_val);
                        diff_pos += 1;
                        next_diff = self.diffs.access(diff_pos).unwrap_or(usize::MAX);
                        //next_diff = diff_iter.next().unwrap_or(usize::MAX);
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
                if !self.precheck_cells(cell_ind) || self.diffs.is_empty() {
                    return vec![0_u16; self.num_genes()];
                }

                // the "dense" index at which expression entries for this cell
                // should start
                let first_idx = self.num_genes() * cell_ind;
                // the "dense" index at which expression entries for this cell
                // should end
                let last_idx = first_idx + self.num_genes();

                // we checked that self.diffs is not empty above, so the len must be
                // at least 1. Get a cursor to the last element
                if let Some(last_stored_cursor) =
                    unsafe { indices.0.cursor_at(self.diffs.len() - 1) }
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
                        //warn!("no values stored for cell {}", cell_ind);
                        //warn!("start_cur.value() =  {:?}", start_cur.value());
                        return vec![0_u16; self.num_genes()];
                    }

                    let start = start_cur.index();

                    let stop = if !stop_cur.is_valid() || cell_ind == self.num_cells() - 1 || last_in_this_cell {
                        self.diffs.len()
                    } else {
                        stop_cur.index()
                    };
                    if start >= last_idx {
                        warn!("SHOULD NOT HAPPEN!");
                    }
                    if start >= stop {
                        error!(
                            "start = {start}, but stop = {stop}, diffs len = {}",
                            self.diffs.len()
                        );
                    }

                    let n = stop - start;

                    if n == 0 {
                        return vec![0_u16; self.num_genes()];
                    }

                    //info!("start: {}", start);
                    // too slow
                    //let mut diff_iter = self.diffs.iter().skip(start);
                    let mut diff_pos = start;
                    let mut next_diff = self.diffs.access(diff_pos).unwrap_or(usize::MAX);

                    let mut nz_ind_iter = start_cur.clone();

                    let mut expression = Vec::with_capacity(self.num_genes());
                    let mut next_nz_ind =
                        nz_ind_iter.value().expect("at least one") - first_idx as u64;
                    for gidx in 0..self.num_genes() {
                        if gidx < next_nz_ind as usize {
                            expression.push(0);
                        } else {
                            let ddiff = decode_v(next_diff as i32);
                            let decoded_val = (ddiff + self.medians.get(gidx) as i32) as u16;
                            //println!("pushing {decoded_val} at index {gidx}");
                            expression.push(decoded_val);
                            if nz_ind_iter.advance() {
                                next_nz_ind =
                                    nz_ind_iter.value().unwrap_or(u64::MAX) - first_idx as u64;
                                diff_pos += 1;
                                next_diff = self.diffs.access(diff_pos).unwrap_or(usize::MAX);
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
        let diff_bits = self.diffs.size_in_bytes();
        let ibits = self.indices.num_bits(); //0.write_bytes() * 8;

        let ncbits = 32;
        let m_bits = self.medians.len() * self.medians.bit_width();
        (4 * 24) + (diff_bits + ibits + ncbits + m_bits) / 8
    }
}

// Manual implementation of de/serialization for `Rect`.
// We don't need to store the edges since they can computed from the other fields.
impl Encode for EncodedDiffs {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> core::result::Result<(), bincode::error::EncodeError> {
        //Encode::encode(&self.indices, encoder)?;
        Encode::encode(&self.indices, encoder)?;
        Encode::encode(&self.ncells, encoder)?;

        let mut diffs_bytes = Vec::new();
        self.diffs.serialize_into(&mut diffs_bytes)
            .map_err(|_| bincode::error::EncodeError::OtherString("DacsOpt serialize failed".into()))?;
        Encode::encode(&diffs_bytes, encoder)?;

        let (b, w, l) = self.medians.clone().into_raw_parts();
        Encode::encode(&b, encoder)?;
        Encode::encode(&w, encoder)?;
        Encode::encode(&l, encoder)?;
        Ok(())
    }
}

impl<Context> Decode<Context> for EncodedDiffs {
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> core::result::Result<Self, bincode::error::DecodeError> {
        let indices = Decode::decode(decoder)?;

        let ncells = Decode::decode(decoder)?;

        let data = Decode::decode(decoder)?;
        let w = Decode::decode(decoder)?;
        let l = Decode::decode(decoder)?;

        let diffs_bytes: Vec<u8> = Decode::decode(decoder)?;
        let diffs = DacsOpt::deserialize_from(&diffs_bytes[..])
            .map_err(|_| bincode::error::DecodeError::OtherString("DacsOpt deserialize failed".into()))?;

        let data = Decode::decode(decoder)?;
        let w = Decode::decode(decoder)?;
        let l = Decode::decode(decoder)?;
        let medians = unsafe { BitFieldVec::from_raw_parts(data, w, l) };
        Ok(Self {
            indices,
            ncells,
            diffs,
            medians,
        })
    }
}

impl<'de, Context> BorrowDecode<'de, Context> for EncodedDiffs {
    fn borrow_decode<D: bincode::de::BorrowDecoder<'de, Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let indices = BorrowDecode::borrow_decode(decoder)?;
        let ncells = BorrowDecode::borrow_decode(decoder)?;

        let data = BorrowDecode::borrow_decode(decoder)?;
        let width = BorrowDecode::borrow_decode(decoder)?;
        let len = BorrowDecode::borrow_decode(decoder)?;
        let diffs_bytes: Vec<u8> = Decode::decode(decoder)?;
        let diffs = DacsOpt::deserialize_from(&diffs_bytes[..])
            .map_err(|_| bincode::error::DecodeError::OtherString("DacsOpt deserialize failed".into()))?;


        let data = BorrowDecode::borrow_decode(decoder)?;
        let width = BorrowDecode::borrow_decode(decoder)?;
        let len = BorrowDecode::borrow_decode(decoder)?;
        let medians = unsafe { BitFieldVec::from_raw_parts(data, width, len) };
        Ok(Self {
            indices,
            ncells,
            diffs,
            medians,
        })
    }
}

/// Takes a slice of points and applies delta encoding from the median value
/// along the "gene" axis.  That is, each coordinate of the point is encoded
/// with respect to the difference from the median of points amongst all points
/// along that axis/coordinate.
/// Returns a `Some(EncodedDiff)` struct representing the encoded differences or `None`
/// if the slice is empty.
///
pub fn encode_subarray(points: &[Point], data: &CsMat<u16>) -> Option<EncodedDiffs> {
    if points.is_empty() {
        debug!("Empty points array in encode_subarray()");
        return None;
    }

    let mut indices = Vec::<u64>::new();
    let mut median_vec = Vec::<u16>::new();
    //let mut mean = vec![0_f64; data.cols()];
    //let mut n = 0_u16;
    let mut raw_diffs = Vec::<u32>::new();
    let mut max_diff = 0_i32;
    let num_genes = data.cols() as u32;
    debug!("Processing {} points in encode_subarray()", points.len());
    debug!("Number of genes: {}", num_genes);

    let mut nnz = 0_usize;
    
    // Collect non-zero values per gene in a single pass through sparse rows
    // This is more efficient than iterating gene-by-gene for CSR matrices
    let mut gene_values: Vec<Vec<u16>> = vec![Vec::new(); data.cols()];
    
    // Single pass: collect all non-zero values per gene from sparse rows
    for p in points.iter() {
        if let Some(row) = p.get_data(data) {
            // Iterate only over non-zero entries in this sparse row
            for (gene_idx, &val) in row.iter() {
                nnz += 1;
                gene_values[gene_idx].push(val);
            }
        }
    }

    // Calculate median for each gene from collected values
    for j in 0..data.cols() {
        let nz_values = &mut gene_values[j];
        
        // Calculate median using select_nth_unstable (O(n) average)
        let median = if !nz_values.is_empty() {
            let mid = nz_values.len() / 2;
            *nz_values.select_nth_unstable(mid).1
        } else {
            0
        };
        
        if j == 20000 {
            debug!("j: {}, median: {}", j, median);
        }
        median_vec.push(median);
    }
    
    let tot = num_genes as usize * points.len();
    let sparsity = (nnz as f64) / (tot as f64);
    if sparsity > 0.75 {
        println!("sparsity: {}", sparsity);
    }
    /* 
    let mut mean_u16 = vec![0_u16; data.cols()];

    for gene_idx in 0..data.cols() {
        let mean_val = if gene_counts[gene_idx] > 0 {
            (mean[gene_idx] / (gene_counts[gene_idx] as f64)).clamp(0.0, 65535.0) as u16
        } else {
            0_u16
        };
        mean_u16[gene_idx] = mean_val;
    }*/
    // for each cell
    //for (gene_idx, &val) in row.iter() {

    for (cell_ind, p) in points.iter().enumerate() {
        let index_offset = cell_ind * num_genes as usize;
        // cell_indices.push(gene_indices.len() as u32);
        // for each gene in this cell
        if let Some(row) = p.get_data(data) {
            let mut recorded_values_for_cell = 0;
            for (gene_idx, &val) in row.iter() {
                //let raw_mean = mean[gene_idx];
                //let mean_val = mean_u16[gene_idx];
                let median_val = median_vec[gene_idx];

                // no observation for this gene
                if median_val == 0 || val == 0 {
                    continue;
                } else {
                    let diff = val as i32 - median_val as i32;
                    let diff = if diff < 0 {
                        (-2_i32 * diff) + 1
                    } else {
                        2_i32 * diff
                    };

                    assert_eq!(
                        val as u16,
                        (decode_v(diff) + median_val as i32) as u16
                    );
                    raw_diffs.push(diff as u32);
                    let index = index_offset + gene_idx;
                    indices.push(index as u64);
                    max_diff = max_diff.max(diff);
                    recorded_values_for_cell += 1;
                }
            }
        }
    }

    let indices = HybridSparseVec::from_indices(&indices, sparsity, tot); //InnerEFVector::with_items_from_slice_s(&indices);
    let medians = BitFieldVec::<usize>::from_slice(&median_vec).expect("should fit");
    // changed to DacsOpts Directly Addressable Codes variable length encoding to save space that is affected by large outliers
    //let diffs = BitFieldVec::<usize>::from_slice(&raw_diffs).expect("should fit");
    // max_levels max_levels: Maximum number of levels. 
    //     The resulting number of levels is related to the access time. The smaller this value is, the faster operations can be, 
    //     but the larger the memory can be. If None, it computes configuration without limitation in the number of levels.
    let diffs = DacsOpt::from_slice(&raw_diffs, Some(3)).expect("should fit");
    //info!("sparsity : {}", sparsity);
    assert_eq!(indices.len(), diffs.len());

    let enc_diffs = EncodedDiffs {
        indices,
        ncells: points.len() as u32,
        diffs,
        medians,
    };

    // validate!
    for (cell_idx, (recon_vec, orig_vec)) in enc_diffs
        .expression_vec_iter()
        .zip(points.iter())
        .enumerate()
    {
        let orig_data = orig_vec.get_data(data).unwrap().to_dense().to_vec();
        // Find differences instead of asserting equality
        let differences: Vec<(usize, u16, u16, u16)> = recon_vec
            .iter()
            .zip(orig_data.iter())
            .enumerate()
            .filter(|(_, (&r, &o))| r != o)
            .map(|(gene_idx, (&r, &o))| (gene_idx, r, o, median_vec[gene_idx]))
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
                let index_offset = cell_ind * num_genes as usize;
                let row = p.get_data(data).unwrap();
                for (gene_idx, &val) in row.iter() {
                    writeln!(f, "{}\t{}\t{}", cell_ind + 1, gene_idx + 1, val);
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

// ============================================================================
// MST-based encoding functions
// ============================================================================

/// Sparse residual: (gene_index, zigzag-encoded value)
type SparseResidual = Vec<(u32, i32)>;

/// Compute sparse residual for a single cell: R[c][g] = X[c][g] - median[g]
/// Only stores non-zero residuals where both val != 0 and median != 0
fn compute_sparse_residual(
    point: &Point,
    data: &CsMat<u16>,
    median_vec: &[u16],
) -> SparseResidual {
    let mut residual = Vec::new();
    if let Some(row) = point.get_data(data) {
        for (gene_idx, &val) in row.iter() {
            let median = median_vec[gene_idx];
            // Skip if either is zero (same logic as original encode_subarray)
            if median == 0 || val == 0 {
                continue;
            }
            let diff = val as i32 - median as i32;
            // Zigzag encode
            let encoded = if diff < 0 {
                (-2 * diff) + 1
            } else {
                2 * diff
            };
            residual.push((gene_idx as u32, encoded));
        }
    }
    residual
}

/// L0 diff: count genes where values differ between two sparse residuals
fn l0_diff(a: &SparseResidual, b: &SparseResidual) -> u32 {
    let mut count = 0u32;
    let mut i = 0;
    let mut j = 0;
    
    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Less => {
                count += 1; // gene in a but not b
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                count += 1; // gene in b but not a
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                if a[i].1 != b[j].1 {
                    count += 1; // same gene, different value
                }
                i += 1;
                j += 1;
            }
        }
    }
    count += (a.len() - i) as u32 + (b.len() - j) as u32;
    count
}

/// Sparse subtract: compute diff_list = R[child] - R[parent]
/// Returns list of (gene, delta) where delta != 0
fn sparse_subtract(child: &SparseResidual, parent: &SparseResidual) -> Vec<(u32, i32)> {
    let mut result = Vec::new();
    let mut i = 0;
    let mut j = 0;
    
    while i < child.len() && j < parent.len() {
        match child[i].0.cmp(&parent[j].0) {
            std::cmp::Ordering::Less => {
                // Gene only in child
                result.push((child[i].0, child[i].1));
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                // Gene only in parent: delta = 0 - parent_val = -parent_val
                // But we need to negate the zigzag value properly
                // For simplicity, store the negated zigzag (will decode correctly)
                result.push((parent[j].0, negate_zigzag(parent[j].1)));
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                let delta = child[i].1 - parent[j].1;
                if delta != 0 {
                    result.push((child[i].0, delta));
                }
                i += 1;
                j += 1;
            }
        }
    }
    // Remaining in child
    while i < child.len() {
        result.push((child[i].0, child[i].1));
        i += 1;
    }
    // Remaining in parent (negated)
    while j < parent.len() {
        result.push((parent[j].0, negate_zigzag(parent[j].1)));
        j += 1;
    }
    result
}

/// Negate a zigzag-encoded value
fn negate_zigzag(v: i32) -> i32 {
    // Decode, negate, re-encode
    let decoded = decode_v(v);
    let negated = -decoded;
    if negated < 0 {
        (-2 * negated) + 1
    } else {
        2 * negated
    }
}

/// Find k nearest spatial neighbors of point i (excluding self)
fn k_nearest_neighbors<P: PointLike>(points: &[P], i: usize, k: usize) -> Vec<usize> {
    if points.len() <= 1 {
        return Vec::new();
    }
    
    let mut distances: Vec<(usize, f64)> = points
        .iter()
        .enumerate()
        .filter(|(j, _)| *j != i)
        .map(|(j, p)| {
            let dx = p.xpos() - points[i].xpos();
            let dy = p.ypos() - points[i].ypos();
            (j, dx * dx + dy * dy)
        })
        .collect();
    
    let k_actual = k.min(distances.len());
    if k_actual == 0 {
        return Vec::new();
    }
    
    // Partial sort to get k smallest
    distances.select_nth_unstable_by(k_actual - 1, |a, b| {
        a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    
    distances[..k_actual].iter().map(|(j, _)| *j).collect()
}

/// Build MST using Prim's algorithm over kNN neighbor graph
/// This only considers edges to k-nearest spatial neighbors, matching the pseudocode
fn build_mst_prim<P: PointLike>(
    points: &[P],
    residuals: &[SparseResidual],
    k: usize,
) -> (usize, Vec<u32>) {
    use std::collections::BinaryHeap;
    use std::cmp::Reverse;
    
    let n = points.len();
    if n == 0 {
        return (0, Vec::new());
    }
    if n == 1 {
        return (0, vec![0]);
    }
    
    // Precompute kNN neighbors for each cell
    let neighbors: Vec<Vec<usize>> = (0..n)
        .map(|i| k_nearest_neighbors(points, i, k))
        .collect();
    
    // Prim's algorithm
    let root = 0; // Could choose cell with smallest total residual
    let mut parent = vec![u32::MAX; n];
    let mut key = vec![u32::MAX; n];  // Minimum edge weight to reach this node
    let mut in_mst = vec![false; n];
    
    parent[root] = root as u32;
    key[root] = 0;
    
    // Min-heap: (weight, node)
    let mut heap = BinaryHeap::new();
    heap.push(Reverse((0u32, root)));
    
    while let Some(Reverse((_dist, u))) = heap.pop() {
        if in_mst[u] {
            continue; // Already processed
        }
        
        in_mst[u] = true;
        
        // Check all kNN neighbors of u
        for &v in &neighbors[u] {
            if !in_mst[v] {
                let weight = l0_diff(&residuals[u], &residuals[v]);
                if weight < key[v] {
                    key[v] = weight;
                    parent[v] = u as u32;
                    heap.push(Reverse((weight, v)));
                }
            }
        }
        
        // Also check neighbors where u is in their kNN (reverse edges)
        // This ensures the graph is effectively undirected
        for i in 0..n {
            if !in_mst[i] && neighbors[i].contains(&u) {
                let weight = l0_diff(&residuals[u], &residuals[i]);
                if weight < key[i] {
                    key[i] = weight;
                    parent[i] = u as u32;
                    heap.push(Reverse((weight, i)));
                }
            }
        }
    }
    
    // Handle any disconnected nodes (shouldn't happen with proper kNN, but be safe)
    for i in 0..n {
        if parent[i] == u32::MAX {
            warn!("Node {} was disconnected, connecting to root", i);
            parent[i] = root as u32;
        }
    }
    
    (root, parent)
}

/// MST-based encoding of a subarray (block of cells)
pub fn encode_subarray_mst(points: &[Point], data: &CsMat<u16>) -> Option<EncodedDiffsMST> {
    if points.is_empty() {
        return None;
    }
    
    let num_genes = data.cols() as u32;
    let ncells = points.len() as u32;
    
    // Step 1: Compute per-gene medians (same as original)
    let mut gene_values: Vec<Vec<u16>> = vec![Vec::new(); data.cols()];
    for p in points.iter() {
        if let Some(row) = p.get_data(data) {
            for (gene_idx, &val) in row.iter() {
                gene_values[gene_idx].push(val);
            }
        }
    }
    
    let mut median_vec = Vec::with_capacity(data.cols());
    for j in 0..data.cols() {
        let nz_values = &mut gene_values[j];
        let median = if !nz_values.is_empty() {
            let mid = nz_values.len() / 2;
            *nz_values.select_nth_unstable(mid).1
        } else {
            0
        };
        median_vec.push(median);
    }
    
    // Step 2: Compute sparse residuals for ALL cells
    let residuals: Vec<SparseResidual> = points
        .iter()
        .map(|p| compute_sparse_residual(p, data, &median_vec))
        .collect();
    
    // Step 3 & 4: Build kNN graph + MST using petgraph
    let k = 8; // Number of neighbors
    let (root, parent) = build_mst_prim(points, &residuals, k);
    
    // Step 5: Encode deltas along MST edges
    let root_residual = &residuals[root];
    
    // Separate root residual into genes and values
    let root_genes: Vec<u32> = root_residual.iter().map(|(g, _)| *g).collect();
    let root_vals: Vec<u32> = root_residual.iter().map(|(_, v)| *v as u32).collect();
    
    // Build order (BFS order excluding root)
    let order: Vec<u32> = (0..points.len())
        .filter(|&c| c != root)
        .map(|c| c as u32)
        .collect();
    
    // Encode deltas for each non-root cell
    let mut delta_genes_raw = Vec::new();
    let mut delta_vals_raw = Vec::new();
    let mut boundary_bits = Vec::new();
    
    for &c in &order {
        let p = parent[c as usize] as usize;
        let diff_list = sparse_subtract(&residuals[c as usize], &residuals[p]);
        
        // Mark start of this cell's deltas
        boundary_bits.push(true);
        
        // Append gene indices and values
        for (g, d) in &diff_list {
            delta_genes_raw.push(*g);
            delta_vals_raw.push(*d as u32);
            boundary_bits.push(false); // zeros after the 1
        }
    }
    
    // Convert to compressed structures
    let medians = BitFieldVec::<usize>::from_slice(&median_vec).expect("should fit");
    
    let root_residual_genes = if root_genes.is_empty() {
        DacsOpt::default()
    } else {
        DacsOpt::from_slice(&root_genes, Some(3)).expect("should fit")
    };
    
    let root_residual_vals = if root_vals.is_empty() {
        DacsOpt::default()
    } else {
        DacsOpt::from_slice(&root_vals, Some(3)).expect("should fit")
    };
    
    let delta_genes = if delta_genes_raw.is_empty() {
        DacsOpt::default()
    } else {
        DacsOpt::from_slice(&delta_genes_raw, Some(3)).expect("should fit")
    };
    
    let delta_vals = if delta_vals_raw.is_empty() {
        DacsOpt::default()
    } else {
        DacsOpt::from_slice(&delta_vals_raw, Some(3)).expect("should fit")
    };
    
    // Build boundary bitvector with rank/select support
    let boundary = Rank9Sel::from_bits(boundary_bits);
    
    info!(
        "MST encoding: {} cells, root residual {} entries, {} total delta entries",
        ncells,
        root_genes.len(),
        delta_genes_raw.len()
    );
    
    Some(EncodedDiffsMST {
        ncells,
        num_genes,
        medians,
        root: root as u32,
        parent,
        root_residual_genes,
        root_residual_vals,
        delta_genes,
        delta_vals,
        boundary,
        order,
    })
}

impl EncodedDiffsMST {
    /// Decode the expression vector for a single cell
    pub fn decode_cell(&self, cell_idx: usize) -> Vec<u16> {
        let mut expression = vec![0u16; self.num_genes as usize];
        
        // Start with root residual
        let mut acc: Vec<(u32, i32)> = Vec::new();
        for i in 0..self.root_residual_genes.len() {
            let g = self.root_residual_genes.access(i).unwrap() as u32;
            let v = self.root_residual_vals.access(i).unwrap() as i32;
            acc.push((g, v));
        }
        
        if cell_idx == self.root as usize {
            // Root cell: just apply root residual
            for (g, v) in &acc {
                let median = self.medians.get(*g as usize) as i32;
                let decoded = decode_v(*v);
                expression[*g as usize] = (decoded + median).max(0) as u16;
            }
            return expression;
        }
        
        // Find path from cell to root
        let path = self.path_to_root(cell_idx);
        
        // Walk from root to cell, accumulating deltas
        // Path is [cell, parent, grandparent, ..., root], so we reverse it
        for i in (1..path.len()).rev() {
            let child = path[i - 1];
            // Find child's position in order
            if let Some(order_idx) = self.order.iter().position(|&c| c == child as u32) {
                let deltas = self.get_cell_deltas(order_idx);
                acc = self.apply_deltas(&acc, &deltas);
            }
        }
        
        // Convert accumulated residual to expression
        for (g, v) in &acc {
            let median = self.medians.get(*g as usize) as i32;
            let decoded = decode_v(*v);
            expression[*g as usize] = (decoded + median).max(0) as u16;
        }
        
        expression
    }
    
    /// Get path from cell to root: [cell, parent, grandparent, ..., root]
    fn path_to_root(&self, cell_idx: usize) -> Vec<usize> {
        let mut path = Vec::new();
        let mut current = cell_idx;
        
        while current != self.root as usize {
            path.push(current);
            current = self.parent[current] as usize;
            // Safety: prevent infinite loop
            if path.len() > self.ncells as usize {
                break;
            }
        }
        path.push(self.root as usize);
        path
    }
    
    /// Get the delta list for cell at order_idx (0-based among non-root cells)
    fn get_cell_deltas(&self, order_idx: usize) -> Vec<(u32, i32)> {
        // Use select to find boundaries
        // boundary has pattern: 1 0 0 0 1 0 0 1 0 ...
        //                       ^       ^     ^
        //                       cell0   cell1 cell2
        
        // Find position of (order_idx + 1)-th 1-bit
        let bit_start = if order_idx == 0 {
            0
        } else {
            // select1 returns position of the k-th 1-bit (0-indexed)
            match self.boundary.select1(order_idx - 1) {
                Some(pos) => pos + 1,
                None => return Vec::new(),
            }
        };
        
        let bit_end = match self.boundary.select1(order_idx) {
            Some(pos) => pos,
            None => return Vec::new(),
        };
        
        // Number of deltas = number of 0-bits between bit_start and bit_end
        // Actually: deltas start at delta_start in delta_genes/delta_vals
        let delta_start = if order_idx == 0 {
            0
        } else {
            // Count 0-bits before this position
            bit_start - order_idx
        };
        
        let num_deltas = bit_end - bit_start;
        
        let mut deltas = Vec::with_capacity(num_deltas);
        for i in 0..num_deltas {
            if let (Some(g), Some(v)) = (
                self.delta_genes.access(delta_start + i),
                self.delta_vals.access(delta_start + i),
            ) {
                deltas.push((g as u32, v as i32));
            }
        }
        
        deltas
    }
    
    /// Apply deltas to accumulated residual: acc = acc + deltas
    fn apply_deltas(&self, acc: &[(u32, i32)], deltas: &[(u32, i32)]) -> Vec<(u32, i32)> {
        let mut result = Vec::new();
        let mut i = 0;
        let mut j = 0;
        
        while i < acc.len() && j < deltas.len() {
            match acc[i].0.cmp(&deltas[j].0) {
                std::cmp::Ordering::Less => {
                    result.push(acc[i]);
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    result.push(deltas[j]);
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    let sum = acc[i].1 + deltas[j].1;
                    if sum != 0 {
                        result.push((acc[i].0, sum));
                    }
                    i += 1;
                    j += 1;
                }
            }
        }
        while i < acc.len() {
            result.push(acc[i]);
            i += 1;
        }
        while j < deltas.len() {
            result.push(deltas[j]);
            j += 1;
        }
        
        result
    }
    
    /// Estimate bytes used by this encoding
    pub fn bytes(&self) -> usize {
        self.medians.len() * 2  // roughly
            + self.parent.len() * 4
            + self.root_residual_genes.size_in_bytes()
            + self.root_residual_vals.size_in_bytes()
            + self.delta_genes.size_in_bytes()
            + self.delta_vals.size_in_bytes()
            + self.boundary.size_in_bytes()
            + self.order.len() * 4
    }
}

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
    pub(crate) encoded_diffs: EncodedDiffs,
    divided: bool,
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
            encoded_diffs: EncodedDiffs::empty(),
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

    pub(crate) fn divide_recursive(&mut self, data: &CsMat<u16>) {
        //pub(crate) fn divide_recursive(&mut self, data: &CsMat<u16>) {
        info!("divide_recursive");
        //let mut stack = vec![self];
        let cost_log = CostLog::new();
        //let max_depth = 3;
        let max_pt: usize = 5;
        // nothing to do if this subtree is empty
        if self.points.is_empty() {
            //return cost_log;
            info!("points is empty");
            //return CostLog::new();
        }
        // Store the current points' positions before clearing them
        let positions: Vec<DatalessPoint> = self
            .points
            .iter()
            .map(|p| DatalessPoint::new(p.x, p.y))
            .collect();
        // Compute the expense of encoding the current block
        //info!("encoding current block");
        let current_expense = encode_subarray(&self.points, data).map_or(0, |x| x.bytes());
        info!(
            "expense of current block consisting of {} points is {}",
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

        // Convert children to BitFieldQuadTree to calculate their expenses
        let nw_expense = encode_subarray(&nw.points, data).map_or(0, |x| x.bytes());
        let ne_expense = encode_subarray(&ne.points, data).map_or(0, |x| x.bytes());
        let se_expense = encode_subarray(&se.points, data).map_or(0, |x| x.bytes());
        let sw_expense = encode_subarray(&sw.points, data).map_or(0, |x| x.bytes());

        info!("NW expense: {}", nw_expense);
        info!("NE expense: {}", ne_expense);
        info!("SE expense: {}", se_expense);
        info!("SW expense: {}", sw_expense);

        let total_expense = nw_expense + ne_expense + se_expense + sw_expense;
        info!("total_expense: {}", total_expense);

        //if self.points.len() > 1 {
        if total_expense < current_expense {
            self.divided = true;
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
                encode_subarray(&self.points, data).map_or(0, |x| x.bytes())
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
            encode_subarray(&self.points, data).map_or(0, |x| x.bytes())
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
                    encode_subarray(&self.points).map_or(0, |x| x.bytes())
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
                encode_subarray(&self.points).map_or(0, |x| x.bytes())
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
            encoded_diffs: EncodedDiffs::empty(),
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
            node.encoded_diffs = encode_subarray(&self.points, data).expect("expect nonempty");
            node.positions = self.positions.clone();
        }
        node
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use tracing::info;
    use tracing_subscriber::filter::LevelFilter;
    use tracing_subscriber::fmt;
    use tracing_subscriber::prelude::*;
    use tracing_subscriber::EnvFilter;

    #[test]
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
        let csr = matrix_new.to_csr();
        let mut points = Vec::<Point>::new();
        for i in 0..csr.rows() {
            points.push(Point::new(0., 0., i));
        }
        encode_subarray(&points, &csr);
    }
}
