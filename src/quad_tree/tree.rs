use std::io::Cursor;

use bincode::{BorrowDecode, Decode, Encode};
use bitm;
use sux::prelude::{BitFieldSlice, BitFieldVec};
use sux::traits::BitFieldSliceCore;
use sux::traits::BitFieldSliceMut;
use tracing::{debug, info, warn};

type InnerEFVector = cseq::elias_fano::Sequence<bitm::BinaryRankSearch, bitm::BinaryRankSearch>;
pub struct EFVector(InnerEFVector);

impl EFVector {
    pub fn empty() -> Self {
        let e = Vec::<u64>::new();
        Self(InnerEFVector::with_items_from_slice_s(&e))
    }
}

impl Clone for EFVector {
    // Required method
    fn clone(&self) -> Self {
        //if !self.0.is_empty() {
        let mut c = Cursor::new(Vec::new());
        self.0.write(&mut c);
        //let v = self.0.iter().collect::<Vec<u64>>();
        //let seq = InnerEFVector::with_items_from_slice_s(&v);
        c.set_position(0);
        let seq = InnerEFVector::read_s(&mut c).expect("works");
        Self(seq)
        //} else {
        //let v = Vec::<u64>::new();
        //Self(InnerEFVector::with_items_from_slice_s(&v))
        //}
    }
}

impl Encode for EFVector {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> core::result::Result<(), bincode::error::EncodeError> {
        let b = self.0.write_bytes();
        let mut d = Cursor::new(Vec::<u8>::with_capacity(b));
        self.0.write(&mut d).expect("can write to cursor");
        d.set_position(0);
        Encode::encode(&d.into_inner(), encoder)
    }
}

impl<Context> Decode<Context> for EFVector {
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> core::result::Result<Self, bincode::error::DecodeError> {
        let d: Vec<u8> = Decode::decode(decoder)?;
        let mut c = Cursor::new(d);
        c.set_position(0);
        let seq = InnerEFVector::read_s(&mut c).expect("valid EF sequence");
        Ok(Self(seq))
    }
}

impl<'de, Context> BorrowDecode<'de, Context> for EFVector {
    fn borrow_decode<D: bincode::de::BorrowDecoder<'de, Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let d: Vec<u8> = BorrowDecode::borrow_decode(decoder)?;
        let mut c = Cursor::new(d);
        c.set_position(0);
        let indices = InnerEFVector::read_s(&mut c).expect("valid EF sequence");
        Ok(Self(indices))
    }
}

#[derive(Clone)]
pub struct EncodedDiffs {
    pub indices: EFVector,
    pub ncells: u32,
    pub diffs: BitFieldVec,
    pub medians: BitFieldVec,
    pub sparse_type: u8,
}

fn decode_v(diff: i32) -> i32 {
    if diff % 2 == 0 {
        diff / 2
    } else {
        -((diff - 1) / 2)
    }
}

fn encode_v(val: i32, med_val: i32) -> i32 {
    let diff = val as i32 - med_val as i32;
    let diff = if diff <= 0 {
        (-2_i32 * diff) + 1
    } else {
        2_i32 * diff
    };
    diff
}

impl EncodedDiffs {
    pub fn empty() -> Self {
        EncodedDiffs {
            indices: EFVector::empty(),
            ncells: 0,
            diffs: BitFieldVec::with_capacity(0, 0),
            medians: BitFieldVec::with_capacity(0, 0),
            sparse_type: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn num_medians(&self) -> usize {
        self.medians.len()
    }

    pub fn len(&self) -> usize {
        self.diffs.len()
    }

    pub fn num_genes(&self) -> usize {
        self.medians.len()
    }

    pub fn num_cells(&self) -> usize {
        self.ncells as usize
    }

    pub fn expression_vec(&self, cell_ind: usize) -> Vec<u16> {
        let mut expression = Vec::with_capacity(self.num_genes());
        let first_idx = self.num_genes() * cell_ind;
        let last_idx = first_idx + self.num_genes();
        if self.sparse_type != 1{
            let start_cur = self.indices.0.geq_cursor(first_idx as u64);
            let stop_cur = self.indices.0.geq_cursor(last_idx as u64);

            let start = if !start_cur.is_valid() {
                usize::MAX
            } else {
                start_cur.index()
            };

            let stop = if !stop_cur.is_valid() {
                self.diffs.len()
            } else {
                stop_cur.index()
            };
            if start == usize::MAX {
                warn!("SHOULD NOT HAPPEN!");
            }
            if start >= last_idx {
                warn!("SHOULD NOT HAPPEN 2!");
            }

            let n = stop - start;
            if n == 0 {
                return vec![0_u16; self.num_genes()];
            }

            let mut diff_iter = self.diffs.iter_from(start);
            let mut nz_ind_iter = start_cur.clone();

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
        }
        else{
            let mut diff_iter = self.diffs.iter_from(first_idx as usize);
            let mut next_diff = diff_iter.next().expect("at least one");
            for gidx in 0..self.num_genes() {
                let median = self.medians.get(gidx);
                let decoded_val = if next_diff == 0{
                    0_i32
                }else{
                    decode_v(next_diff as i32)+ median as i32
                };
                next_diff = diff_iter.next().unwrap_or(usize::MAX);
                expression.push(decoded_val as u16);
            }
            
        }

        expression
    }

    pub fn bytes(&self) -> usize {
        let diff_bits = self.diffs.len() * self.diffs.bit_width();
        let ibits = self.indices.0.write_bytes() * 8;
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

        Encode::encode(&self.sparse_type, encoder)?;
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

        let sparse_type = Decode::decode(decoder)?;

        Ok(Self {
            indices,
            ncells,
            diffs,
            medians,
            sparse_type,
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

        let sparse_type = BorrowDecode::borrow_decode(decoder)?;

        Ok(Self {
            indices,
            ncells,
            diffs,
            medians,
            sparse_type,
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

    let mut indices = Vec::<u32>::new();
    let mut medians = Vec::<u16>::new();
    let mut raw_diffs = Vec::<u32>::new();
    //let mut max_diff = 0_i32;
    let num_genes = points[0].data.len() as u32;
    let mut sparse_type: u8 = 0;
    let mut nz_len = 0_u32;
    debug!("Processing {} points in encode_subarray()", points.len());
    debug!("Number of genes: {}", num_genes);

    // we'll make 2 passes, because we want to store the final results in
    // "cell-major" order (i.e. all values for one cell first, then the next, etc.)
    for j in 0..points[0].data.len() {
        // get the non-zero values and the non-zero indices
        let nz_values: Vec<u16> = points
            .iter()
            .filter_map(|p| if p.data[j] > 0 { Some(p.data[j]) } else { None })
            .collect();
        nz_len += nz_values.len() as u32;

        // nonzero median values
        let median = if !nz_values.is_empty() {
            let mut sorted_values = nz_values.clone();
            sorted_values.sort_unstable();
            sorted_values[sorted_values.len() / 2] //median of the expressed values
        } else {
            0
        };
        medians.push(median);
    }

    let total_len = num_genes as u32 * points.len() as u32;
    let sparsity = nz_len as f32 / total_len as f32;

    // for each cell
    for (cell_ind, gene_exp) in points.iter().enumerate() {
        let index_offset = cell_ind * num_genes as usize;
        // cell_indices.push(gene_indices.len() as u32);
        // for each gene in this cell
        for ((gene_ind, val), med_val) in gene_exp.data.iter().enumerate().zip(&medians) {
            // the median of **non-zero** expression values was zero, this
            // is recorded iff there were no cells in the current block expressing
            // this gene. Otherwise, if the original value itself is zero, then
            // don't record anything
            if *med_val == 0 || *val == 0 {
                if sparsity >= 0.75 {
                    let index = index_offset + gene_ind;
                    indices.push(index as u32);
                }else if sparsity >= 0.25 && sparsity < 0.75 {
                    if *val == 0 {
                        raw_diffs.push(0_u32);
                    }else{
                        let diff = encode_v(*val as i32, *med_val as i32);
                        raw_diffs.push(diff as u32);
                    }
                }
            } else {
                // we have a non-zero median value
                let diff = encode_v(*val as i32, *med_val as i32);
                assert_eq!(*val as i32, decode_v(diff) + *med_val as i32);
                raw_diffs.push(diff as u32);
                let index = index_offset + gene_ind;
                if sparsity < 0.25 {
                indices.push(index as u32);
                //gene_indices.push(gene_ind as u32);
                //max_diff = max_diff.max(diff);
                }
            }
        }
    } 
    if sparsity >= 0.25 && sparsity < 0.75 {
        sparse_type = 1;
    }else if sparsity >= 0.75 {
        sparse_type = 2;
    }

    //let gene_indices = BitFieldVec::<usize>::from_slice(&gene_indices).expect("should fit");
    //let cell_indices = BitFieldVec::<usize>::from_slice(&cell_indices).expect("should fit");
    let indices = InnerEFVector::with_items_from_slice_s(&indices);
    let medians = BitFieldVec::<usize>::from_slice(&medians).expect("should fit");
    let diffs = BitFieldVec::<usize>::from_slice(&raw_diffs).expect("should fit");
    if sparsity < 0.25{assert_eq!(indices.len(), diffs.len());}

    let enc_diffs = EncodedDiffs {
        indices: EFVector(indices),
        ncells: points.len() as u32,
        diffs,
        medians,
        sparse_type,
    };

    // validate!
    for (cell_ind, gene_exp) in points.iter().enumerate() {
        assert_eq!(enc_diffs.expression_vec(cell_ind), gene_exp.data);
    }

    info!(
        "Generated {} medians and {} diffs in {} bytes",
        enc_diffs.num_medians(),
        enc_diffs.len(),
        enc_diffs.bytes()
    );
    Some(enc_diffs)
}

#[derive(Debug, Copy, Clone)]
pub enum ErrorMetric {
    Mean,
    Median,
}

pub trait PointLike {
    fn xpos(&self) -> f64;
    fn ypos(&self) -> f64;
}

#[derive(Debug, Clone, Encode, Decode)]
pub struct Point {
    pub x: f64,
    pub y: f64,
    pub data: Vec<u16>,
}

impl Point {
    #[inline(always)]
    pub const fn new(x: f64, y: f64, data: Vec<u16>) -> Self {
        Self { x, y, data }
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
pub struct DatalessPoint {
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

    // pub const fn new_from_bounds(west: f64, east: f64, north: f64, south: f64)
    let nw_boundary = Rect::new_from_bounds(west_child_west, center_x, north_child_north, center_y);
    let ne_boundary = Rect::new_from_bounds(center_x, east_child_east, north_child_north, center_y);

    let se_boundary = Rect::new_from_bounds(center_x, east_child_east, center_y, south_cild_south);
    let sw_boundary = Rect::new_from_bounds(west_child_west, center_x, center_y, south_cild_south);
    (nw_boundary, ne_boundary, se_boundary, sw_boundary)
}

#[derive(Debug, Clone)]
pub struct Rect {
    cx: f64,
    cy: f64,
    west_edge: f64,
    east_edge: f64,
    north_edge: f64,
    south_edge: f64,
}

impl Rect {
    #[inline(always)]
    pub const fn new(cx: f64, cy: f64, w: f64, h: f64) -> Self {
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
    pub const fn new_from_bounds(west: f64, east: f64, north: f64, south: f64) -> Self {
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
pub struct BitField {
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
pub struct BitFieldQuadTree {
    pub boundary: Rect,
    pub encoded_diffs: EncodedDiffs,
    divided: bool,
    nw: Option<Box<BitFieldQuadTree>>,
    ne: Option<Box<BitFieldQuadTree>>,
    se: Option<Box<BitFieldQuadTree>>,
    sw: Option<Box<BitFieldQuadTree>>,
    pub positions: Vec<DatalessPoint>,
}

impl BitFieldQuadTree {
    pub fn new(boundary: Rect) -> Self {
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

    pub fn visit(&self, fun: &mut impl FnMut(&BitFieldQuadTree)) {
        fun(self);
        if self.divided {
            self.children().iter().for_each(|c| {
                if let Some(n) = c {
                    n.visit(fun);
                }
            });
        }
    }

    pub fn children(&self) -> [&Option<Box<BitFieldQuadTree>>; 4] {
        [&self.nw, &self.ne, &self.sw, &self.se]
    }

    pub fn calculate_expense(&self) -> usize {
        info!(
            "Calculating expense for BitFieldQuadTree node with {} total points and size {} bytes",
            self.encoded_diffs.len(),
            self.encoded_diffs.bytes()
        );
        info!("Total node expense: {}", self.encoded_diffs.bytes());
        self.encoded_diffs.bytes()
    }

    /*
    pub fn to_quad_tree(&self) -> QuadTree {
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

    pub fn calculate_size(&self) -> (usize, usize) {
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

    pub fn print_size_info(&self, depth: usize) {
        let indent = "  ".repeat(depth);
        let (size, bitfields) = self.calculate_size();
        info!(
            "{}Level {}: {} bits, {} bitfields",
            indent, depth, size, bitfields
        );

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

#[derive(Debug, Encode, Decode)]
pub struct QuadTree {
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
    pub const fn new(boundary: Rect, points: Vec<Point>, depth: usize) -> Self {
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
            positions: Vec::new(),
        }
    }

    pub fn query(&self, boundary: &Rect) -> Vec<Point> {
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
    /*
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

    pub fn divide(&mut self) {
        info!("Processing block with {} points", self.points.len());

        debug!("self.points.len(): {}", self.points.len());
        if !self.points.is_empty() {
            println!("Initial number of genes: {}", self.points[0].data.len());
        }

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

        // Compute the expense of encoding the current block
        let current_expense = encode_subarray(&self.points).map_or(0, |x| x.bytes());
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
        let nw_expense = encode_subarray(&nw.points).map_or(0, |x| x.bytes());
        let ne_expense = encode_subarray(&ne.points).map_or(0, |x| x.bytes());
        let se_expense = encode_subarray(&se.points).map_or(0, |x| x.bytes());
        let sw_expense = encode_subarray(&sw.points).map_or(0, |x| x.bytes());

        info!("NW expense: {}", nw_expense);
        info!("NE expense: {}", ne_expense);
        info!("SE expense: {}", se_expense);
        info!("SW expense: {}", sw_expense);

        let total_expense = nw_expense + ne_expense + se_expense + sw_expense;

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
                nw.divide();
            }
            if let Some(ref mut ne) = self.ne {
                ne.divide();
            }
            if let Some(ref mut se) = self.se {
                se.divide();
            }
            if let Some(ref mut sw) = self.sw {
                sw.divide();
            }
        } else {
            self.divided = false;
            if !self.points.is_empty() {
                info!(
                    "Leaf node - points: {}, genes: {}",
                    self.points.len(),
                    self.points[0].data.len()
                );
                self.positions = positions; // Use the stored positions
                                            // Keep the points for bit field representation
            }
        }
        info!("self.depth: {}", self.depth);
    }

    pub fn non_zero_blocks(&self) -> usize {
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

    /* This is simply a bit vector across genes for each block */
    pub fn block_data_to_sarray(&self, _sparse: bool) -> (Vec<u16>, Vec<BitField>) {
        let mut sarray = Vec::new();
        let mut diffs = Vec::new();

        if self.points.is_empty() {
            debug!("Empty points array in block_data_to_sarray");
            return (sarray, diffs);
        }

        debug!(
            "Processing {} points in block_data_to_sarray",
            self.points.len()
        );
        debug!("Number of genes: {}", self.points[0].data.len());

        for j in 0..self.points[0].data.len() {
            // for each gene
            let nz_values: Vec<u16> = self
                .points
                .iter()
                .filter_map(|p| if p.data[j] > 0 { Some(p.data[j]) } else { None })
                .collect(); // Keep only non-zero values

            let median = if !nz_values.is_empty() {
                let mut sorted_values = nz_values.clone();
                sorted_values.sort_unstable();
                sorted_values[sorted_values.len() / 2] //median of the expressed values
            } else {
                0
            };

            if median != 0 {
                sarray.push(median); // Use push instead of append
                let mut max_diff = 0;
                let mut min_diff = 0;

                // Find min and max diffs
                for &value in &nz_values {
                    let diff = value.wrapping_sub(median);
                    min_diff = min_diff.min(diff as usize);
                    max_diff = max_diff.max(diff as usize);
                }

                let bit_width = (max_diff as u16).ilog2() as usize + 1;
                //println!("Gene {}: median={}, bit_width={}, min_diff={}, max_diff={}",
                //    j, median, bit_width, min_diff, max_diff);

                let mut bit_field = BitFieldVec::new(bit_width, nz_values.len());
                // Calculate and store differences
                for (i, &value) in nz_values.iter().enumerate() {
                    let diff = value.wrapping_sub(median);
                    bit_field.set(i, diff as usize);
                }

                diffs.push(BitField::new(bit_field));
            }
        }

        info!(
            "Generated {} medians and {} diffs",
            sarray.len(),
            diffs.len()
        );
        (sarray, diffs)
    }

    pub fn compute_quadtree_bit_fields(&self) -> BitFieldQuadTree {
        debug!(
            "Computing bit fields - points available: {}",
            self.points.len()
        );
        if !self.points.is_empty() {
            debug!(
                "Computing bit fields - genes per point: {}",
                self.points[0].data.len()
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
            info!("Node is divided, computing children bit fields");
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
            node.encoded_diffs = encode_subarray(&self.points).expect("nonempty");
            node.positions = self.positions.clone();
        }
        node
    }
}
