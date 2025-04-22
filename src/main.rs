use bincode::{BorrowDecode, Decode, Encode};
use clap::Parser;
use csv::ReaderBuilder;
//use flate2::{Compression, write::GzEncoder};
use std::fs::File;
use std::path::{Path, PathBuf};
use tracing::info;
use tracing_subscriber::{EnvFilter, filter::LevelFilter, fmt, prelude::*};
use bvrs::SparseArray;
use sux::prelude::BitFieldVec;

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
    data: Vec<f32>,
}

impl Point {
    #[inline(always)]
    const fn new(x: f32, y: f32, data: Vec<f32>) -> Self {
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

#[derive(Debug, Encode, Decode)]
struct BitFieldQuadTree {
    boundary: Rect,
    sarray: SparseArray,
    bit_field: BitFieldVec,
    divided: bool,
    nw: Option<Box<BitFieldQuadTree>>,
    ne: Option<Box<BitFieldQuadTree>>,
    se: Option<Box<BitFieldQuadTree>>,
    sw: Option<Box<BitFieldQuadTree>>,
    positions: Vec<DatalessPoint>,
}

impl BitFieldQuadTree {
    // Implementation methods here
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

    fn query(&self, boundary: &Rect, found_points: &mut Vec<Point>) {
        if !self.boundary.intersects(boundary) {
            return;
        }

        for point in &self.points {
            if boundary.contains(point) {
                found_points.push(point.clone());
            }
        }

        if self.divided {
            if let Some(ref nw) = self.nw {
                nw.query(boundary, found_points);
            }
            if let Some(ref ne) = self.ne {
                ne.query(boundary, found_points);
            }
            if let Some(ref se) = self.se {
                se.query(boundary, found_points);
            }
            if let Some(ref sw) = self.sw {
                sw.query(boundary, found_points);
            }
        }
    }
/*
    fn block_data_repr(&self, method: ErrorMetric) -> Vec<f32> {
        let mut block_mean = Vec::<f32>::with_capacity(self.points[0].data.len());
        for j in 0..self.points[0].data.len() {
            let block_mean_j = match method {
                ErrorMetric::Median => {
                    let mut values: Vec<f32> = self.points.iter().map(|p| p.data[j]).collect();
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    values[values.len() / 2]
                }
                ErrorMetric::Mean => {
                    self.points.iter().map(|p| p.data[j]).sum::<f32>() / self.points.len() as f32
                }
            };
            block_mean.push(block_mean_j);
        }
        block_mean
    }
*/

    fn calculate_error(&self, method: ErrorMetric, mind: &[f32], maxd: &[f32], _prob: f32) -> f32 {
        let mut found_points = Vec::new();
        self.query(&self.boundary, &mut found_points);

        if found_points.is_empty() {
            return 0.0;
        }

        let mut maxerrors = Vec::new();
        for j in 0..found_points[0].data.len() {
            let block_mean = match method {
                ErrorMetric::Median => {
                    let mut values: Vec<f32> = found_points.iter().map(|p| p.data[j]).collect();
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    values[values.len() / 2]
                }
                ErrorMetric::Mean => {
                    found_points.iter().map(|p| p.data[j]).sum::<f32>() / found_points.len() as f32
                }
            };

            let maxerror = found_points
                .iter()
                .map(|p| (p.data[j] - block_mean).abs())
                .collect::<Vec<f32>>();

            let maxerror =
                maxerror.iter().fold(0.0, |a, &b| f32::max(a, b)) / (maxd[j] - mind[j] + 0.01);
            maxerrors.push(maxerror);
        }

        maxerrors.iter().fold(0.0, |a, &b| f32::max(a, b))
    }

    fn divide(
        &mut self,
        threshold: f32,
        method: ErrorMetric,
        mind: &[f32],
        maxd: &[f32],
        maxerrors: &mut Vec<f32>,
    ) -> f32 {
        let maxerror = self.calculate_error(method, mind, maxd, 1.0);
        maxerrors.push(maxerror);

        if maxerror > threshold && self.points.len() > 1 {
            let cx = self.boundary.cx;
            let cy = self.boundary.cy;
            let w = self.boundary.w / 2.0;
            let h = self.boundary.h / 2.0;

            let nw_boundary = Rect::new(cx - w / 2.0, cy - h / 2.0, w, h);
            let mut nw_points = Vec::new();
            self.query(&nw_boundary, &mut nw_points);
            self.nw = Some(Box::new(Self::new(nw_boundary, nw_points, self.depth + 1)));

            let ne_boundary = Rect::new(cx + w / 2.0, cy - h / 2.0, w, h);
            let mut ne_points = Vec::new();
            self.query(&ne_boundary, &mut ne_points);
            self.ne = Some(Box::new(Self::new(ne_boundary, ne_points, self.depth + 1)));

            let se_boundary = Rect::new(cx + w / 2.0, cy + h / 2.0, w, h);
            let mut se_points = Vec::new();
            self.query(&se_boundary, &mut se_points);
            self.se = Some(Box::new(Self::new(se_boundary, se_points, self.depth + 1)));

            let sw_boundary = Rect::new(cx - w / 2.0, cy + h / 2.0, w, h);
            let mut sw_points = Vec::new();
            self.query(&sw_boundary, &mut sw_points);
            self.sw = Some(Box::new(Self::new(sw_boundary, sw_points, self.depth + 1)));

            self.divided = true;
            // At this stage, every point has been inserted in one of the subtrees, so we can drop the vector to avoid duplicating data.
            self.points = Vec::new();

            if let Some(ref mut nw) = self.nw {
                nw.divide(threshold, method, mind, maxd, maxerrors);
            }
            if let Some(ref mut ne) = self.ne {
                ne.divide(threshold, method, mind, maxd, maxerrors);
            }
            if let Some(ref mut se) = self.se {
                se.divide(threshold, method, mind, maxd, maxerrors);
            }
            if let Some(ref mut sw) = self.sw {
                sw.divide(threshold, method, mind, maxd, maxerrors);
            }
        } else {
            self.divided = false;
            self.maxerror = Some(maxerror);
            if !self.data.is_empty() {
                self.positions = self.points.iter().map(DatalessPoint::from_point).collect();
                self.data = self.block_data_repr(method);
                self.points.clear();
            }
        }
        maxerror
    }

    #[allow(dead_code)]
    fn len(&self) -> usize {
        let mut npoints = self.points.len();
        if self.divided {
            if let Some(ref nw) = self.nw {
                npoints += nw.len();
            }
            if let Some(ref ne) = self.ne {
                npoints += ne.len();
            }
            if let Some(ref se) = self.se {
                npoints += se.len();
            }
            if let Some(ref sw) = self.sw {
                npoints += sw.len();
            }
        }
        npoints
    }

    fn blocks(&self) -> usize {
        if !self.divided {
            1
        } else {
            let mut npoints = 0;
            if let Some(ref nw) = self.nw {
                npoints += nw.blocks();
            }
            if let Some(ref ne) = self.ne {
                npoints += ne.blocks();
            }
            if let Some(ref se) = self.se {
                npoints += se.blocks();
            }
            if let Some(ref sw) = self.sw {
                npoints += sw.blocks();
            }
            npoints
        }
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

    fn block_data_to_sarray(&self, sparse: bool) -> (SparseArray, BitFieldVec) {
        let block_mean_j = match sparse {
            true => {
                let mut sarray = SparseArray::new(self.points[0].data.len());
                for j in 0..self.points[0].data.len() {
                    let mut values: Vec<f32> = self.points.iter().map(|p| p.data[j]).collect();
                    // Collect non-zero values for this column
                    let mut column_values = Vec::new();
                    for v in values {
                        if v != 0.0 {
                            column_values.push(v as u16);
                        }
                    }
                    //values.sort_by(|a, b| a.partial_cmp(b).unwrap()); //for float type
                    column_values.sort_unstable();
                    // Find median with no ties (using lower middle value if even)
                    let median = if column_values.len() % 2 == 1 {
                        column_values[column_values.len() / 2]
                    } else {
                        column_values[column_values.len() / 2 - 1]
                    };
                    if median != 0 {
                        sarray.append(median, j as u16);
                    // Calculate differences from median for each value
                        let mut differences = Vec::new();
                        for &value in &column_values {
                            let diff = value - median;
                            differences.push(diff);
                        }
                        // Find the minimum value to determine shift needed to make all values non-negative
                        let min_value = differences.iter().min().unwrap_or(&0);
                        
                        // Calculate shift - if min value is negative, we'll add its absolute value to all values
                        let shift_value = if min_value < &0 { min_value.unsigned_abs() } else { 0 };
        
                        // Apply the shift and find the maximum value to determine bit width
                        let shifted_values: Vec<u16> = differences.iter()
                            .map(|&x| (x + shift_value) as u16)
                            .collect();

                    // Find max difference to determine bit width needed
                        let max_diff = shifted_values.iter().max().copied().unwrap_or(0);
                        let bit_width = if max_diff == 0 { 1 } else { (max_diff as u32).ilog2() as usize + 1 };

                    // Create bit field vector
                        let mut bit_field = BitFieldVec::new(bit_width, differences.len());
                        for (i, &diff) in shifted_values.iter().enumerate() {
                            bit_field.set(i, diff as u16);
                        }
                    }
                }
            }
            false => {
                eprintln!("Dense case not implemented yet");
            }
        };
        (sarray, bit_field)
    }

    /// Traverses the quadtree and computes median bit representations at each leaf node
    /// Returns a tree structure of bit fields
    pub fn compute_quadtree_bit_fields(&self) -> BitFieldQuadTree {
        let (sarray, bit_field) = self.block_data_to_sarray(true);///!!!! CHANGE to sparse
        // Create node data
        let mut node = BitFieldQuadTree {
            boundary: self.boundary,
            sarray,
            bit_field,
            divided: self.divided,
            nw: None,
            ne: None,
            se: None,
            sw: None,
            positions: Vec::new(),
        };
        
        // If the node is divided, recursively process children
        if self.divided {
            // Process children (if they exist)
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
    idx_cell: usize,
    //threshold: f32,
    step: f32,
    //loop_flag: bool,
    method: ErrorMetric,
    endpt: Option<usize>,
    allgenes: bool,
) -> (Vec<f32>, QuadTree) {
    let mut coords = Vec::new();
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut mind = Vec::new();
    let mut maxd = Vec::new();

    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path(file_path)
        .expect("Failed to open file");

    for result in rdr.records() {
        let record = result.expect("Failed to read record");
        let x: f32 = record[idx_x].parse().unwrap();
        let y: f32 = record[idx_y].parse().unwrap();
        xs.push(x);
        ys.push(y);

        let mut cells = Vec::new();
        let end = endpt.unwrap_or(if allgenes { record.len() } else { idx_cell + 1 });
        for i in idx_cell..end {
            let value: u16 = record[i].parse().unwrap();
            cells.push(value as f32);
            if mind.len() <= i - idx_cell {
                mind.push(value);
                maxd.push(value);
            } else {
                mind[i - idx_cell] = mind[i - idx_cell].min(value);
                maxd[i - idx_cell] = maxd[i - idx_cell].max(value);
            }
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
    let mut qtree = QuadTree::new(domain, coords, 1);
    /*
    if loop_flag {
        let sequence: Vec<f32> = (0..).map(|x| x as f32 * step).take_while(|&x| x < threshold).collect();
        let mut y_points = Vec::new();
        let mut maxerrorsl = Vec::new();

        for x in sequence {
            let maxerrorl = qtree.divide(x, method, &mind, &maxd, &mut maxerrorsl);
            y_points.push(qtree.non0blocks());
        }

        (maxerrorsl, qtree)
    } else {
     */
    let mut maxerrors = Vec::new();
    let _maxerrorl = qtree.divide(step, method, &mind, &maxd, &mut maxerrors);
    let bit_field_tree = qtree.compute_quadtree_bit_fields();
    (maxerrors, qtree)
    //}
}

/// Build a quadtree representation of spatial transcriptomics data
#[derive(Debug, Parser)]
#[command(version, about, long_about = None)]
struct CmdArgs {
    /// Input csv
    input: PathBuf,
    /// Output file (default "output.bin.gz")
    output: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
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
    let (_maxerrorl, qtree) = tree_from_csv(file_path, 5, 6, 9, 0.5, ErrorMetric::Mean, None, true);
    
    // You can decide whether to serialize the original tree or the bit fields or both
    let config = bincode::config::standard()
        .with_little_endian()
        .with_fixed_int_encoding();
    let ofname = args.output.unwrap_or(PathBuf::from("output.bin.gz"));
    let file = File::create(ofname).unwrap();
    //let mut encoder = GzEncoder::new(file, Compression::default());
    bincode::encode_into_std_write(&qtree, &mut file, config).unwrap();
    //info!("Max Errors: {:?}", maxerrorl);
    info!(
        "QuadTree Blocks: {} (non-zero blocks: {})",
        qtree.blocks(),
        qtree.non_zero_blocks()
    );
    Ok(())
}
