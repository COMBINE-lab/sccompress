use bincode::{BorrowDecode, Decode, Encode};
use sux::prelude::BitFieldVec;
use sux::traits::BitFieldSliceMut;
use tracing::info;

#[derive(Debug, Copy, Clone)]
pub enum ErrorMetric {
    Mean,
    Median,
}

pub trait PointLike {
    fn xpos(&self) -> f32;
    fn ypos(&self) -> f32;
}

#[derive(Debug, Clone, Encode, Decode)]
pub struct Point {
    pub x: f32,
    pub y: f32,
    pub data: Vec<u16>,
}

impl Point {
    #[inline(always)]
    pub const fn new(x: f32, y: f32, data: Vec<u16>) -> Self {
        Self { x, y, data }
    }
}

impl PointLike for Point {
    #[inline(always)]
    fn xpos(&self) -> f32 {
        self.x
    }
    #[inline(always)]
    fn ypos(&self) -> f32 {
        self.y
    }
}

#[derive(Debug, Clone, Encode, Decode)]
pub struct DatalessPoint {
    x: f32,
    y: f32,
}

impl DatalessPoint {
    #[inline(always)]
    const fn new(x: f32, y: f32) -> Self {
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
    fn xpos(&self) -> f32 {
        self.x
    }
    #[inline(always)]
    fn ypos(&self) -> f32 {
        self.y
    }
}

#[derive(Debug, Clone)]
pub struct Rect {
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
    west_edge: f32,
    east_edge: f32,
    north_edge: f32,
    south_edge: f32,
}

impl Rect {
    #[inline(always)]
    pub const fn new(cx: f32, cy: f32, w: f32, h: f32) -> Self {
        Self {
            cx,
            cy,
            w,
            h,
            west_edge: cx - w / 2.0,
            east_edge: cx + w / 2.0,
            north_edge: cy - h / 2.0,
            south_edge: cy + h / 2.0,
        }
    }

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
        Encode::encode(&self.cx, encoder)?;
        Encode::encode(&self.cy, encoder)?;
        Encode::encode(&self.w, encoder)?;
        Encode::encode(&self.h, encoder)?;
        Ok(())
    }
}

impl<Context> Decode<Context> for Rect {
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> core::result::Result<Self, bincode::error::DecodeError> {
        let cx = Decode::decode(decoder)?;
        let cy = Decode::decode(decoder)?;
        let w = Decode::decode(decoder)?;
        let h = Decode::decode(decoder)?;
        Ok(Self::new(cx, cy, w, h))
    }
}

impl<'de, Context> BorrowDecode<'de, Context> for Rect {
    fn borrow_decode<D: bincode::de::BorrowDecoder<'de, Context = Context>>(
        decoder: &mut D,
    ) -> core::result::Result<Self, bincode::error::DecodeError> {
        let cx = BorrowDecode::borrow_decode(decoder)?;
        let cy = BorrowDecode::borrow_decode(decoder)?;
        let w = BorrowDecode::borrow_decode(decoder)?;
        let h = BorrowDecode::borrow_decode(decoder)?;
        Ok(Self::new(cx, cy, w, h))
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
    boundary: Rect,
    medians: Vec<u16>,
    data: Vec<BitField>,
    divided: bool,
    nw: Option<Box<BitFieldQuadTree>>,
    ne: Option<Box<BitFieldQuadTree>>,
    se: Option<Box<BitFieldQuadTree>>,
    sw: Option<Box<BitFieldQuadTree>>,
    positions: Vec<DatalessPoint>,
}

impl BitFieldQuadTree {
    pub fn new(boundary: Rect) -> Self {
        Self {
            boundary,
            medians: Vec::new(),
            data: Vec::new(),
            divided: false,
            nw: None,
            ne: None,
            se: None,
            sw: None,
            positions: Vec::new(),
        }
    }

    pub fn calculate_expense(&self) -> u32 {
        let mut expense: u32 = 0;
        info!(
            "Calculating expense for BitFieldQuadTree node with {} medians and {} bitfields",
            self.medians.len(),
            self.data.len()
        );

        for (i, bitfield) in self.data.iter().enumerate() {
            let (_, width, len) = bitfield.bit_field.clone().into_raw_parts();
            let bit_expense = width as u32 * len as u32;
            expense += bit_expense;
            info!(
                "Bitfield {}: width={}, len={}, expense={}",
                i, width, len, bit_expense
            );
        }

        info!("Total node expense: {}", expense);
        expense
    }

    pub fn to_quad_tree(&self) -> QuadTree {
        let mut quadtree = QuadTree::new(self.boundary.clone(), Vec::new(), 0);
        quadtree.divided = self.divided;
        // Convert the raw parts into f32 values to meet the QuadTree requirements
        quadtree.data = self
            .data
            .iter()
            .flat_map(|bf| {
                let (data, _, _) = bf.bit_field.clone().into_raw_parts();
                data.iter().map(|&x| x as f32).collect::<Vec<f32>>()
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

    pub fn calculate_size(&self) -> (usize, usize) {
        let mut total_size = 0;
        let mut total_bitfields = 0;

        // Calculate size of current node's data
        for bitfield in &self.data {
            let (_, width, len) = bitfield.bit_field.clone().into_raw_parts();
            total_size += width * len;
            total_bitfields += 1;
        }

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
    maxerror: Option<f32>,
    nw: Option<Box<Self>>,
    ne: Option<Box<Self>>,
    se: Option<Box<Self>>,
    sw: Option<Box<Self>>,
    data: Vec<f32>,
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
        found_points
    }

    pub fn block_data_repr(&self, method: ErrorMetric) -> Vec<f32> {
        if self.points.is_empty() {
            return Vec::new();
        }

        let mut block_mean = Vec::<f32>::with_capacity(self.points[0].data.len());
        for j in 0..self.points[0].data.len() {
            let block_mean_j = match method {
                ErrorMetric::Median => {
                    let mut values: Vec<f32> =
                        self.points.iter().map(|p| p.data[j] as f32).collect();
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    values[values.len() / 2]
                }
                ErrorMetric::Mean => {
                    self.points.iter().map(|p| p.data[j] as f32).sum::<f32>()
                        / self.points.len() as f32
                }
            };
            block_mean.push(block_mean_j);
        }
        block_mean
    }
    /*
        pub fn calculate_error(&self, method: ErrorMetric, mind: &[u16], maxd: &[u16], _prob: f32) -> f32 {
            let mut found_points = Vec::new();
            self.query(&self.boundary);

            if found_points.is_empty() {
                return 0.0;
            }

            let mut maxerrors = Vec::new();
            for j in 0..found_points[0].data.len() {
                let block_mean = match method {
                    ErrorMetric::Median => {
                        let mut values: Vec<f32> = found_points.iter().map(|p| p.data[j] as f32).collect();
                        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        values[values.len() / 2]
                    }
                    ErrorMetric::Mean => {
                        found_points.iter().map(|p| p.data[j] as f32).sum::<f32>() / found_points.len() as f32
                    }
                };

                let maxerror = found_points
                    .iter()
                    .map(|p| (p.data[j] as f32 - block_mean).abs())
                    .collect::<Vec<f32>>();

                let maxerror =
                    maxerror.iter().fold(0.0, |a, &b| f32::max(a, b)) / ((maxd[j] as f32) - (mind[j] as f32) + 0.01);
                maxerrors.push(maxerror);
            }
            maxerrors.iter().fold(0.0, |a, &b| f32::max(a, b))
        }
    */

    pub fn divide(&mut self) {
        info!("Processing node with {} points", self.points.len());
        println!("self.points.len(): {}", self.points.len());
        if !self.points.is_empty() {
            println!("Initial number of genes: {}", self.points[0].data.len());
        }

        // Store the current points' positions before clearing them
        let positions: Vec<DatalessPoint> = self
            .points
            .iter()
            .map(|p| DatalessPoint::new(p.x, p.y))
            .collect();

        // Convert current node to BitFieldQuadTree to calculate expense
        let current_bit_tree = self.compute_quadtree_bit_fields();
        println!(
            "current_bit_tree.medians.len(): {}",
            current_bit_tree.medians.len()
        );
        let current_expense = current_bit_tree.calculate_expense();

        // Find the children of the current node
        let cx = self.boundary.cx;
        let cy = self.boundary.cy;
        let w = self.boundary.w / 2.0;
        let h = self.boundary.h / 2.0;

        let nw_boundary = Rect::new(cx - w / 2.0, cy - h / 2.0, w, h);
        let nw_points = self.query(&nw_boundary);
        println!(
            "NW points: {}, genes per point: {}",
            nw_points.len(),
            if !nw_points.is_empty() {
                nw_points[0].data.len()
            } else {
                0
            }
        );
        let nw = QuadTree::new(nw_boundary, nw_points, self.depth + 1);

        let ne_boundary = Rect::new(cx + w / 2.0, cy - h / 2.0, w, h);
        let ne_points = self.query(&ne_boundary);
        println!(
            "NE points: {}, genes per point: {}",
            ne_points.len(),
            if !ne_points.is_empty() {
                ne_points[0].data.len()
            } else {
                0
            }
        );
        let ne = QuadTree::new(ne_boundary, ne_points, self.depth + 1);

        let se_boundary = Rect::new(cx + w / 2.0, cy + h / 2.0, w, h);
        let se_points = self.query(&se_boundary);
        println!(
            "SE points: {}, genes per point: {}",
            se_points.len(),
            if !se_points.is_empty() {
                se_points[0].data.len()
            } else {
                0
            }
        );
        let se = QuadTree::new(se_boundary, se_points, self.depth + 1);

        let sw_boundary = Rect::new(cx - w / 2.0, cy + h / 2.0, w, h);
        let sw_points = self.query(&sw_boundary);
        println!(
            "SW points: {}, genes per point: {}",
            sw_points.len(),
            if !sw_points.is_empty() {
                sw_points[0].data.len()
            } else {
                0
            }
        );
        let sw = QuadTree::new(sw_boundary, sw_points, self.depth + 1);

        // Convert children to BitFieldQuadTree to calculate their expenses
        let nw_bit_tree = nw.compute_quadtree_bit_fields();
        let ne_bit_tree = ne.compute_quadtree_bit_fields();
        let se_bit_tree = se.compute_quadtree_bit_fields();
        let sw_bit_tree = sw.compute_quadtree_bit_fields();

        let nw_expense = nw_bit_tree.calculate_expense();
        println!("NW expense: {}", nw_expense);
        let ne_expense = ne_bit_tree.calculate_expense();
        println!("NE expense: {}", ne_expense);
        let se_expense = se_bit_tree.calculate_expense();
        println!("SE expense: {}", se_expense);
        let sw_expense = sw_bit_tree.calculate_expense();
        println!("SW expense: {}", sw_expense);

        let total_expense = nw_expense + ne_expense + se_expense + sw_expense;

        if total_expense < current_expense {
            self.divided = true;
            // Convert BitFieldQuadTree back to QuadTree and assign children
            self.nw = Some(Box::new(nw_bit_tree.to_quad_tree()));
            self.ne = Some(Box::new(ne_bit_tree.to_quad_tree()));
            self.se = Some(Box::new(se_bit_tree.to_quad_tree()));
            self.sw = Some(Box::new(sw_bit_tree.to_quad_tree()));

            // Only clear points after we've used them for all necessary operations
            self.points = Vec::new();

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
                info!("Leaf node with {} points", self.points.len());
                println!(
                    "Leaf node - points: {}, genes: {}",
                    self.points.len(),
                    self.points[0].data.len()
                );
                self.positions = positions; // Use the stored positions
                                            // Keep the points for bit field representation
            }
        }
        println!("self.depth: {}", self.depth);
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
            println!("Empty points array in block_data_to_sarray");
            return (sarray, diffs);
        }

        println!(
            "Processing {} points in block_data_to_sarray",
            self.points.len()
        );
        println!("Number of genes: {}", self.points[0].data.len());

        for j in 0..self.points[0].data.len() {
            // for each gene
            let values: Vec<u16> = self
                .points
                .iter()
                .map(|p| p.data[j])
                .filter(|&v| v != 0)
                .collect(); // Keep only non-zero values

            let median = if !values.is_empty() {
                let mut sorted_values = values.clone();
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
                for &value in &values {
                    let diff = value.wrapping_sub(median);
                    min_diff = min_diff.min(diff as usize);
                    max_diff = max_diff.max(diff as usize);
                }

                let bit_width = (max_diff as u16).ilog2() as usize + 1;
                //println!("Gene {}: median={}, bit_width={}, min_diff={}, max_diff={}",
                //    j, median, bit_width, min_diff, max_diff);

                let mut bit_field = BitFieldVec::new(bit_width, values.len());
                // Calculate and store differences
                for (i, &value) in values.iter().enumerate() {
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
        println!(
            "Computing bit fields - points available: {}",
            self.points.len()
        );
        if !self.points.is_empty() {
            println!(
                "Computing bit fields - genes per point: {}",
                self.points[0].data.len()
            );
        }

        let (medians, diffs) = self.block_data_to_sarray(true);
        println!(
            "Generated medians: {}, diffs: {}",
            medians.len(),
            diffs.len()
        );

        let mut node = BitFieldQuadTree {
            boundary: self.boundary.clone(),
            medians,
            data: diffs,
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
        }
        node
    }
}
