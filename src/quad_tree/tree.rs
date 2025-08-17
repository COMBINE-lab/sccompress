
use crate::bits::HybridSparseVec;
use bincode::{BorrowDecode, Decode, Encode};
use bitm::{self, BitAccess};
use sux::prelude::{BitFieldSlice, BitFieldVec};
use sux::traits::BitFieldSliceCore;
use sux::traits::BitFieldSliceMut;
use tracing::{debug, info, warn};
use crate::ArrayData;
use std::sync::Arc;

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
    pub steps: Vec<CostStep>,
    pub total_nodes: usize,
    pub total_cost: usize,
}

impl CostLog {
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            total_nodes: 0,
            total_cost: 0,
        }
    }
    
    pub fn add_step(&mut self, step: CostStep) {
        self.steps.push(step);
    }
    
    pub fn update_totals(&mut self, nodes: usize, cost: usize) {
        self.total_nodes = nodes;
        self.total_cost = cost;
    }
}


// Helper function to extract numeric data from ArrayData
fn extract_numeric_data(data: &ArrayData) -> Vec<u16> {
    data.data.as_slice().unwrap().to_vec()
}


#[derive(Clone)]
pub(crate) struct EncodedDiffs {
    pub(crate) indices: HybridSparseVec,
    pub(crate) ncells: u32,
    pub(crate) diffs: BitFieldVec,
    pub(crate) medians: BitFieldVec,
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
            diffs: BitFieldVec::with_capacity(0, 0),
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
                let mut diff_iter = self.diffs.iter_from(*num_ones);

                let mut expression = Vec::with_capacity(self.num_genes());
                let mut next_diff = diff_iter.next().expect("at least one");
                let mut num_ones_in_cell = 0_usize;
                for gidx in 0..self.num_genes() {
                    if !indices.get_bit(gidx + first_idx) {
                        expression.push(0);
                    } else {
                        num_ones_in_cell += 1;
                        let ddiff = decode_v(next_diff as i32);
                        let decoded_val = (ddiff + self.medians.get(gidx) as i32) as u16;
                        expression.push(decoded_val);
                        next_diff = diff_iter.next().unwrap_or(usize::MAX);
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
                if !self.precheck_cells(cell_ind) {
                    return vec![0_u16; self.num_genes()];
                }

                let first_idx = self.num_genes() * cell_ind;
                let last_idx = first_idx + self.num_genes();

                let start_cur = indices.0.geq_cursor(first_idx as u64);
                let stop_cur = indices.0.geq_cursor(last_idx as u64);

                let start = if !start_cur.is_valid() {
                    // No valid starting position found for this cell
                    return vec![0_u16; self.num_genes()];
                } else {
                    start_cur.index()
                };

                let stop = if !stop_cur.is_valid() {
                    self.diffs.len()
                } else {
                    stop_cur.index()
                };
                if start >= last_idx {
                    warn!("SHOULD NOT HAPPEN!");
                }

                let n = stop - start;
                
                if n == 0 {
                    return vec![0_u16; self.num_genes()];
                }

                let mut diff_iter = self.diffs.iter_from(start);
                let mut nz_ind_iter = start_cur.clone();

                let mut expression = Vec::with_capacity(self.num_genes());
                let mut next_diff = diff_iter.next().expect("at least one");
                let mut next_nz_ind = nz_ind_iter.value().expect("at least one") - first_idx as u64;
                for gidx in 0..self.num_genes() {
                    if gidx < next_nz_ind as usize {
                        expression.push(0);
                    } else {
                        let ddiff = decode_v(next_diff as i32);
                        let decoded_val = (ddiff + self.medians.get(gidx) as i32) as u16;
                        expression.push(decoded_val);
                        let _more = nz_ind_iter.advance();
                        next_nz_ind = nz_ind_iter.value().unwrap_or(u64::MAX) - first_idx as u64;
                        next_diff = diff_iter.next().unwrap_or(usize::MAX);
                    }
                }
                expression
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

    pub(crate) fn is_diff_iter_empty(&self, start_pos: usize) -> bool {
        self.diffs.iter_from(start_pos).next().is_none()
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
        let diff_bits = self.diffs.len() * self.diffs.bit_width();
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

        let (b, w, l) = self.diffs.clone().into_raw_parts();
        Encode::encode(&b, encoder)?;
        Encode::encode(&w, encoder)?;
        Encode::encode(&l, encoder)?;

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
        let diffs = unsafe { BitFieldVec::from_raw_parts(data, w, l) };

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
        let diffs = unsafe { BitFieldVec::from_raw_parts(data, width, len) };

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
pub fn encode_subarray(points: &[Point]) -> Option<EncodedDiffs> {
    if points.is_empty() {
        debug!("Empty points array in encode_subarray()");
        return None;
    }

    let mut indices = Vec::<u64>::new();
    let mut medians = Vec::<u16>::new();
    let mut raw_diffs = Vec::<u32>::new();
    let mut max_diff = 0_i32;
    let num_genes = points[0].get_data().len() as u32;
    debug!("Processing {} points in encode_subarray()", points.len());
    debug!("Number of genes: {}", num_genes);

    let mut nnz = 0_usize;
    // we'll make 2 passes, because we want to store the final results in
    // "cell-major" order (i.e. all values for one cell first, then the next, etc.)
    for j in 0..points[0].get_data().len() {
        // get the non-zero values and the non-zero indices
        let nz_values: Vec<u16> = points
            .iter()
            .filter_map(|p| {
                let data = p.get_data();
                if data[j] > 0 { Some(data[j]) } else { None }
            })
            .collect();

        // nonzero median values
        let median = if !nz_values.is_empty() {
            nnz += nz_values.len();
            let mut sorted_values = nz_values.clone();
            sorted_values.sort_unstable();
            sorted_values[sorted_values.len() / 2] //median of the expressed values
        } else {
            0
        };
        medians.push(median);
    }

    let tot = num_genes as usize * points.len();
    let sparsity = (nnz as f64) / (tot as f64);

    // for each cell
    for (cell_ind, gene_exp) in points.iter().enumerate() {
        let index_offset = cell_ind * num_genes as usize;
        // cell_indices.push(gene_indices.len() as u32);
        // for each gene in this cell
        let gene_data = gene_exp.get_data();
        for ((gene_ind, val), med_val) in gene_data.iter().enumerate().zip(&medians) {
            // the median of **non-zero** expression values was zero, this
            // is recorded iff there were no cells in the current block expressing
            // this gene. Otherwise, if the original value itself is zero, then
            // don't record anything
            if *med_val == 0 || *val == 0 {
                continue;
            } else {
                // we have a non-zero median value
                let diff = *val as i32 - *med_val as i32;
                let diff = if diff < 0 {
                    (-2_i32 * diff) + 1
                } else {
                    2_i32 * diff
                };
                assert_eq!(*val as i32, decode_v(diff) + *med_val as i32);
                raw_diffs.push(diff as u32);
                let index = index_offset + gene_ind;
                indices.push(index as u64);
                max_diff = max_diff.max(diff);
            }
        }
    }

    let indices = HybridSparseVec::from_indices(&indices, sparsity, tot); //InnerEFVector::with_items_from_slice_s(&indices);
    let medians = BitFieldVec::<usize>::from_slice(&medians).expect("should fit");
    let diffs = BitFieldVec::<usize>::from_slice(&raw_diffs).expect("should fit");
    //info!("sparsity : {}", sparsity);
    assert_eq!(indices.len(), diffs.len());

    let enc_diffs = EncodedDiffs {
        indices,
        ncells: points.len() as u32,
        diffs,
        medians,
    };

    // validate!
    for (recon_vec, orig_vec) in enc_diffs.expression_vec_iter().zip(points.iter()) {
        let orig_data = orig_vec.get_data();
        assert_eq!(recon_vec, orig_data);
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
    pub(crate) data_arc: Arc<ArrayData>,
}

impl Point {
    #[inline(always)]
    pub(crate) fn new(x: f64, y: f64, data_arc: Arc<ArrayData>) -> Self {
        Self { x, y, data_arc }
    }

    pub(crate) fn get_data(&self) -> Vec<u16> {
        extract_numeric_data(&self.data_arc)
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
    const fn new(x: f64, y: f64) -> Self {
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
        let w = west - east;
        let h = north - south;
        let cx = west + (w / 2_f64);
        let cy = south + (h / 2_f64);
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

    #[allow(dead_code)]
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
    }
}

pub(crate) struct QuadTree {
    boundary: Rect,
    points: Vec<Point>,
    depth: usize,
    divided: bool,
    maxerror: Option<f64>,
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
            maxerror: None,
            nw: None,
            ne: None,
            se: None,
            sw: None,
            data: Vec::new(),
            //index: Vec::new(),
            positions: Vec::new(),
        }
    }


    /// Get the expression data for a point
    pub(crate) fn get_point_data<'a>(&self, point: &'a Point) -> Option<&'a ArrayData> {
        Some(&point.data_arc)
    }

    /// Get all expression data for a point
    pub(crate) fn get_all_point_data<'a>(&self, point: &'a Point) -> Vec<&'a ArrayData> {
        // Since we now store data directly in Point, just return it as a single element
        vec![&point.data_arc]
    }

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

    pub(crate) fn divide(&mut self) -> CostLog {
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
 
             // Convert children to BitFieldQuadTree to calculate their expenses
             let nw_expense = encode_subarray(&nw.points).map_or(0, |x| x.bytes());
             let ne_expense = encode_subarray(&ne.points).map_or(0, |x| x.bytes());
             let se_expense = encode_subarray(&se.points).map_or(0, |x| x.bytes());
             let sw_expense = encode_subarray(&sw.points).map_or(0, |x| x.bytes());
 
            // info!("NW expense: {} with {} points", nw_expense, nw.points.len());
            // info!("NE expense: {} with {} points", ne_expense, ne.points.len());
            // info!("SE expense: {} with {} points", se_expense, se.points.len());
            // info!("SW expense: {} with {} points", sw_expense, sw.points.len());
 
            let total_children_expense = nw_expense + ne_expense + se_expense + sw_expense;

             // Determine optimal cost and decision
             let optimal_cost = if current_expense < total_children_expense {
                 current_expense
             } else {
                 total_children_expense
             };
             
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
             if optimal_cost < current_expense {
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
                     info!(
                         "Leaf node - points: {}, genes: {}",
                         node.points.len(),
                         node.points[0].get_data().len()
                     );
                     node.positions = positions; // Use the stored positions
                                                 // Keep the points for bit field representation
                 }
             }
         }
         
         // Update totals
         let total_cost = cost_log.steps.iter().map(|step| step.optimal_cost).sum();
         cost_log.update_totals(node_counter, total_cost);
         
         //info!("Finished iterative division");
         cost_log
     }


    /// Traverse to all leaf nodes first, then compare children expenses to parent expenses
    /// Returns a tuple: (parent_expense, children_expense, should_divide)
    pub(crate) fn compare_parent_vs_children_expenses(&self) -> (usize, usize, bool) {
        if !self.divided {
            // Leaf node - calculate parent expense only
            let parent_expense = if !self.points.is_empty() {
                encode_subarray(&self.points).map_or(0, |x| x.bytes())
            } else {
                0
            };
            return (parent_expense, 0, false); // No children to compare
        }

        // Internal node - recursively get children expenses
        let mut children_expense = 0;
        
        // Get expenses from all children
        if let Some(ref nw) = self.nw {
            let (_, child_expense, _) = nw.compare_parent_vs_children_expenses();
            children_expense += child_expense;
        }
        if let Some(ref ne) = self.ne {
            let (_, child_expense, _) = ne.compare_parent_vs_children_expenses();
            children_expense += child_expense;
        }
        if let Some(ref se) = self.se {
            let (_, child_expense, _) = se.compare_parent_vs_children_expenses();
            children_expense += child_expense;
        }
        if let Some(ref sw) = self.sw {
            let (_, child_expense, _) = sw.compare_parent_vs_children_expenses();
            children_expense += child_expense;
        }

        // Calculate parent expense for this node
        let parent_expense = if !self.points.is_empty() {
            encode_subarray(&self.points).map_or(0, |x| x.bytes())
        } else {
            0
        };

        // Compare parent vs children expenses
        let should_divide = children_expense < parent_expense;
        
        //info!("Node at depth {}: parent_expense={}, children_expense={}, should_divide={}", 
          //    self.depth, parent_expense, children_expense, should_divide);

        (parent_expense, children_expense, should_divide)
    }

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

    pub(crate) fn non_zero_blocks(&self) -> usize {
        let mut npoints = 0;
        if !self.divided {
            if !self.points.is_empty() {
                npoints = 1;
            }
        } else {
            if let Some(ref nw) = self.nw {
                npoints += nw.non_zero_blocks();
            }
            if let Some(ref ne) = self.ne {
                npoints += ne.non_zero_blocks();
            }
            if let Some(ref se) = self.se {
                npoints += se.non_zero_blocks();
            }
            if let Some(ref sw) = self.sw {
                npoints += sw.non_zero_blocks();
            }
        }
        npoints
    }

    pub(crate) fn compute_quadtree_bit_fields(&self) -> BitFieldQuadTree {
        debug!(
            "Computing bit fields - points available: {}",
            self.points.len()
        );
        if !self.points.is_empty() {
            debug!(
                "Computing bit fields - genes per point: {}",
                self.points[0].get_data().len()
            );
        }

        /*
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
            //info!("Node is divided, computing children bit fields");
            if let Some(ref ne) = self.ne {
                node.ne = Some(Box::new(ne.compute_quadtree_bit_fields()));
            }
            if let Some(ref nw) = self.nw {
                node.nw = Some(Box::new(nw.compute_quadtree_bit_fields()));
            }
            if let Some(ref se) = self.se {
                node.se = Some(Box::new(se.compute_quadtree_bit_fields()));
            }
            if let Some(ref sw) = self.sw {
                node.sw = Some(Box::new(sw.compute_quadtree_bit_fields()));
            }
        } else {
            node.encoded_diffs = encode_subarray(&self.points).expect("expect nonempty") ;
            node.positions = self.positions.clone();
        }
        node
    }
}

