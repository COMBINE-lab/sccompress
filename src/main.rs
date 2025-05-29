use bincode::{BorrowDecode, Decode, Encode};
use clap::Parser;
use csv::ReaderBuilder;
use hdf5::File;
use hdf5::types::FixedAscii;
use ndarray::Array1;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::{info, warn};
use sux::prelude::BitFieldVec;
use sux::traits::BitFieldSliceMut;
use std::error::Error;
use std::env;
use std::time::Instant;
use tracing_subscriber::fmt;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::filter::LevelFilter;
use std::fs::File as StdFile;
use tracing_subscriber::prelude::*;

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
struct Point {
    x: f32,
    y: f32,
    data: Vec<u16>,
}

impl Point {
    #[inline(always)]
    const fn new(x: f32, y: f32, data: Vec<u16>) -> Self {
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
struct DatalessPoint {
    x: f32,
    y: f32,
}

impl DatalessPoint {
    #[inline(always)]
    const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
    #[inline(always)]
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
struct Rect {
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
    const fn new(cx: f32, cy: f32, w: f32, h: f32) -> Self {
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
struct BitField {
    bit_field: BitFieldVec,
}

impl BitField {
    fn new(bit_field: BitFieldVec) -> Self {
        Self { bit_field }
    }
}

impl Encode for BitField {
    fn encode<E: bincode::enc::Encoder>(&self, encoder: &mut E) -> Result<(), bincode::error::EncodeError> {
        let (data, width, len) = self.bit_field.clone().into_raw_parts();
        Encode::encode(&data, encoder)?;
        Encode::encode(&width, encoder)?;
        Encode::encode(&len, encoder)?;
        Ok(())
    }
}

impl<Context> Decode<Context> for BitField {
    fn decode<D: bincode::de::Decoder<Context = Context>>(decoder: &mut D) -> Result<Self, bincode::error::DecodeError> {
        let data: Vec<u64> = Decode::decode(decoder)?;
        let width = Decode::decode(decoder)?;
        let len = Decode::decode(decoder)?;
        let mut bit_field = BitFieldVec::new(width, len);
        for i in 0..len {
            bit_field.set(i, data[i] as usize);
        }
        Ok(BitField::new(bit_field))
    }
}

impl<'de, Context> BorrowDecode<'de, Context> for BitField {
    fn borrow_decode<D: bincode::de::BorrowDecoder<'de, Context = Context>>(decoder: &mut D) -> Result<Self, bincode::error::DecodeError> {
        let data: Vec<u64> = BorrowDecode::borrow_decode(decoder)?;
        let width = BorrowDecode::borrow_decode(decoder)?;
        let len = BorrowDecode::borrow_decode(decoder)?;
        let mut bit_field = BitFieldVec::new(width, len);
        for i in 0..len {
            bit_field.set(i, data[i] as usize);
        }
        Ok(BitField::new(bit_field))
    }
}

#[derive(Encode, Decode, Clone)]
struct BitFieldQuadTree {
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
    fn new(boundary: Rect) -> Self {
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

    fn calculate_expense(&self) -> u32 {
        let mut expense: u32 = 0;
        info!("Calculating expense for BitFieldQuadTree node with {} medians and {} bitfields", 
            self.medians.len(), self.data.len());
        
        for (i, bitfield) in self.data.iter().enumerate() {    
            let (_, width, len) = bitfield.bit_field.clone().into_raw_parts();
            let bit_expense = width as u32 * len as u32;
            expense += bit_expense;
            info!("Bitfield {}: width={}, len={}, expense={}", 
                i, width, len, bit_expense);
        }
        
        info!("Total node expense: {}", expense);
        expense
    }

    fn to_quad_tree(&self) -> QuadTree {
        let mut quadtree = QuadTree::new(self.boundary.clone(), Vec::new(), 0);
        quadtree.divided = self.divided;
        // Convert the raw parts into f32 values to meet the QuadTree requirements
        quadtree.data = self.data.iter().map(|bf| {
            let (data, _, _) = bf.bit_field.clone().into_raw_parts();
            data.iter().map(|&x| x as f32).collect::<Vec<f32>>()
        }).flatten().collect();
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

    fn calculate_size(&self) -> (usize, usize) {
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

    fn print_size_info(&self, depth: usize) {
        let indent = "  ".repeat(depth);
        let (size, bitfields) = self.calculate_size();
        info!("{}Level {}: {} bits, {} bitfields", indent, depth, size, bitfields);
        
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
struct QuadTree {
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
    const fn new(boundary: Rect, points: Vec<Point>, depth: usize) -> Self {
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

    fn query(&self, boundary: &Rect) -> Vec<Point> {
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

    fn block_data_repr(&self, method: ErrorMetric) -> Vec<f32> {
        if self.points.is_empty() {
            return Vec::new();
        }

        let mut block_mean = Vec::<f32>::with_capacity(self.points[0].data.len());
        for j in 0..self.points[0].data.len() {
            let block_mean_j = match method {
                ErrorMetric::Median => {
                    let mut values: Vec<f32> = self.points.iter().map(|p| p.data[j] as f32).collect();
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    values[values.len() / 2]
                }
                ErrorMetric::Mean => {
                    self.points.iter().map(|p| p.data[j] as f32).sum::<f32>() / self.points.len() as f32
                }
            };
            block_mean.push(block_mean_j);
        }
        block_mean
    }
/* 
    fn calculate_error(&self, method: ErrorMetric, mind: &[u16], maxd: &[u16], _prob: f32) -> f32 {
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

    fn divide(&mut self) {
        info!("Processing node with {} points", self.points.len());
        println!("self.points.len(): {}", self.points.len());
        if !self.points.is_empty() {
            println!("Initial number of genes: {}", self.points[0].data.len());
        }
        
        // Store the current points' positions before clearing them
        let positions: Vec<DatalessPoint> = self.points.iter()
            .map(|p| DatalessPoint::new(p.x, p.y))
            .collect();

        // Convert current node to BitFieldQuadTree to calculate expense
        let current_bit_tree = self.compute_quadtree_bit_fields();
        println!("current_bit_tree.medians.len(): {}", current_bit_tree.medians.len());
        let current_expense = current_bit_tree.calculate_expense();

        // Find the children of the current node
        let cx = self.boundary.cx;
        let cy = self.boundary.cy;
        let w = self.boundary.w / 2.0;
        let h = self.boundary.h / 2.0;

        let nw_boundary = Rect::new(cx - w / 2.0, cy - h / 2.0, w, h);
        let nw_points = self.query(&nw_boundary);
        println!("NW points: {}, genes per point: {}", 
            nw_points.len(), 
            if !nw_points.is_empty() { nw_points[0].data.len() } else { 0 });
        let nw = QuadTree::new(nw_boundary, nw_points, self.depth + 1);

        let ne_boundary = Rect::new(cx + w / 2.0, cy - h / 2.0, w, h);
        let ne_points = self.query(&ne_boundary);
        println!("NE points: {}, genes per point: {}", 
            ne_points.len(), 
            if !ne_points.is_empty() { ne_points[0].data.len() } else { 0 });
        let ne = QuadTree::new(ne_boundary, ne_points, self.depth + 1);

        let se_boundary = Rect::new(cx + w / 2.0, cy + h / 2.0, w, h);
        let se_points = self.query(&se_boundary);
        println!("SE points: {}, genes per point: {}", 
            se_points.len(), 
            if !se_points.is_empty() { se_points[0].data.len() } else { 0 });
        let se = QuadTree::new(se_boundary, se_points, self.depth + 1);

        let sw_boundary = Rect::new(cx - w / 2.0, cy + h / 2.0, w, h);
        let sw_points = self.query(&sw_boundary);
        println!("SW points: {}, genes per point: {}", 
            sw_points.len(), 
            if !sw_points.is_empty() { sw_points[0].data.len() } else { 0 });
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
                println!("Leaf node - points: {}, genes: {}", 
                    self.points.len(), 
                    self.points[0].data.len());
                self.positions = positions; // Use the stored positions
                // Keep the points for bit field representation
            }
        }    
        println!("self.depth: {}", self.depth);
    }

    fn non_zero_blocks(&self) -> usize {
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
    fn block_data_to_sarray(&self, _sparse: bool) -> (Vec<u16>, Vec<BitField>) {
        let mut sarray = Vec::new();
        let mut diffs = Vec::new();

        if self.points.is_empty() {
            println!("Empty points array in block_data_to_sarray");
            return (sarray, diffs);
        }

        println!("Processing {} points in block_data_to_sarray", self.points.len());
        println!("Number of genes: {}", self.points[0].data.len());

        for j in 0..self.points[0].data.len() { // for each gene
            let values: Vec<u16> = self.points.iter()
                .filter(|&v| v != 0)
                .map(|p| p.data[j])
                .collect(); // Keep only non-zero values
            if !values.is_empty() {
                let mut sorted_values = values.clone();
                sorted_values.sort_unstable();
                let median = sorted_values[sorted_values.len() / 2]; //median of the expressed values
                
                if median != 0 {
                    sarray.push(median);  // Use push instead of append
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
            }else{
                median = 0;
            }
        }

        info!("Generated {} medians and {} diffs", sarray.len(), diffs.len());
        (sarray, diffs)
    }

    fn compute_quadtree_bit_fields(&self) -> BitFieldQuadTree {
        println!("Computing bit fields - points available: {}", self.points.len());
        if !self.points.is_empty() {
            println!("Computing bit fields - genes per point: {}", self.points[0].data.len());
        }
        
        let (medians, diffs) = self.block_data_to_sarray(true);
        println!("Generated medians: {}, diffs: {}", medians.len(), diffs.len());
        
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
        .has_headers(true)  // Set to true to skip the header row
        .flexible(true)  // Allow varying number of fields
        .from_path(file_path)
        .map_err(|e| anyhow::anyhow!("Failed to open file: {}", e))?;


    // Read all records into memory
    let records: Vec<_> = rdr.records().collect::<Result<Vec<_>, _>>()
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
                Err(e) => {
                    0
                }
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
        .flexible(true)  // Allow varying number of fields
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
        let record = result.map_err(|e| {
            anyhow::anyhow!("Failed to read record at line {}: {}", line_number, e)
        })?;
        
        // Print first few records for debugging
        if line_number <= 5 {
            info!("Record {}: {:?}", line_number, record);
            info!("Number of fields in record: {}", record.len());
        }
        
        if record.len() < 3 {
            return Err(anyhow::anyhow!(
                "Record at line {} has insufficient fields (need 3, got {})", 
                line_number, record.len()
            ));
        }
        
        let barcode = record[0].to_string();  // barcode is in first column
        
        // Debug print the raw values before parsing
        if line_number <= 5 {
            info!("Raw values for barcode {}: x='{}', y='{}'", 
                barcode, record[1].to_string(), record[2].to_string());
        }
        
        // Try to parse with more detailed error handling
        let x_str = record[1].trim();
        let y_str = record[2].trim();
        
        if x_str.is_empty() || y_str.is_empty() {
            return Err(anyhow::anyhow!(
                "Empty coordinate value at line {} for barcode {}: x='{}', y='{}'",
                line_number, barcode, x_str, y_str
            ));
        }
        
        let x: f32 = x_str.parse().map_err(|e| {
            anyhow::anyhow!(
                "Failed to parse x coordinate at line {} for barcode {}: '{}' - {}",
                line_number, barcode, x_str, e
            )
        })?;
        
        let y: f32 = y_str.parse().map_err(|e| {
            anyhow::anyhow!(
                "Failed to parse y coordinate at line {} for barcode {}: '{}' - {}",
                line_number, barcode, y_str, e
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
        return Err(anyhow::anyhow!("No valid coordinates found in the CSV file"));
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
    let barcodes: Vec<String> = barcodes_array.iter()
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
    let features: Vec<String> = features_array.iter()
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
                        gene_idx, num_genes - 1, cell_idx
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
    info!("Matched {} barcodes with spatial coordinates", matched_count);
    info!("Unmatched {} barcodes", unmatched_count);
    info!("Created {} points", points.len());

    if points.is_empty() {
        return Err(anyhow::anyhow!("No points were created - check if barcodes match between H5 and spatial files"));
    }

    // Calculate boundaries
    let minx = points.iter().map(|p| p.x).fold(f32::INFINITY, f32::min) - 1.0;
    let miny = points.iter().map(|p| p.y).fold(f32::INFINITY, f32::min) - 1.0;
    let maxx = points.iter().map(|p| p.x).fold(f32::NEG_INFINITY, f32::max) + 1.0;
    let maxy = points.iter().map(|p| p.y).fold(f32::NEG_INFINITY, f32::max) + 1.0;
    let w = maxx - minx;
    let h = maxy - miny;

    info!("Spatial boundaries: x=[{}, {}], y=[{}, {}]", minx, maxx, miny, maxy);

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
                5,  // idx_x
                6,  // idx_y
                1,  // idx_gene_start
                None,  // idx_gene_end (will use all remaining columns)
                ErrorMetric::Mean,
                true
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
