use csv::ReaderBuilder;
use std::fs::{self, File};
use std::io::{self, BufRead};
use std::path::Path;
use ndarray::{Array1, Array2};
use serde::{Serialize, Deserialize};
use bincode;
use flate2::{Compression, write::GzEncoder};
use serde_json;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Point {
    x: f64,
    y: f64,
    data: Vec<f64>,
}

impl Point {
    fn new(x: f64, y: f64, data: Vec<f64>) -> Self {
        Point { x, y, data }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Rect {
    cx: f64,
    cy: f64,
    w: f64,
    h: f64,
    west_edge: f64,
    east_edge: f64,
    north_edge: f64,
    south_edge: f64,
}

impl Rect {
    fn new(cx: f64, cy: f64, w: f64, h: f64) -> Self {
        Rect {
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

    fn contains(&self, point: &Point) -> bool {
        point.x >= self.west_edge && point.x < self.east_edge && point.y >= self.north_edge && point.y < self.south_edge
    }

    fn intersects(&self, other: &Rect) -> bool {
        !(other.west_edge > self.east_edge
            || other.east_edge < self.west_edge
            || other.north_edge > self.south_edge
            || other.south_edge < self.north_edge)
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct QuadTree {
    boundary: Rect,
    points: Vec<Point>,
    depth: usize,
    divided: bool,
    maxerror: Option<f64>,
    nw: Option<Box<QuadTree>>,
    ne: Option<Box<QuadTree>>,
    se: Option<Box<QuadTree>>,
    sw: Option<Box<QuadTree>>,
}

impl QuadTree {
    fn new(boundary: Rect, points: Vec<Point>, depth: usize) -> Self {
        QuadTree {
            boundary,
            points,
            depth,
            divided: false,
            maxerror: None,
            nw: None,
            ne: None,
            se: None,
            sw: None,
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

    fn calculate_error(&self, method: &str, mind: &[f64], maxd: &[f64], prob: f64) -> f64 {
        let mut found_points = Vec::new();
        self.query(&self.boundary, &mut found_points);

        if found_points.is_empty() {
            return 0.0;
        }

        let mut maxerrors = Vec::new();
        for j in 0..found_points[0].data.len() {
            let block_mean = match method {
                "median" => {
                    let mut values: Vec<f64> = found_points.iter().map(|p| p.data[j]).collect();
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    values[values.len() / 2]
                }
                "mean" => {
                    found_points.iter().map(|p| p.data[j]).sum::<f64>() / found_points.len() as f64
                }
                _ => 0.0,
            };

            let maxerror = found_points
                .iter()
                .map(|p| (p.data[j] - block_mean).abs())
                .collect::<Vec<f64>>();
            
            let maxerror = maxerror.iter().fold(0.0, |a, &b| f64::max(a, b)) / (maxd[j] - mind[j] + 0.01);
            maxerrors.push(maxerror);
        }

        maxerrors.iter().fold(0.0, |a, &b| f64::max(a, b))
    }

    fn divide(&mut self, threshold: f64, method: &str, mind: &[f64], maxd: &[f64], maxerrors: &mut Vec<f64>) -> f64 {
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
            self.nw = Some(Box::new(QuadTree::new(nw_boundary, nw_points, self.depth + 1)));

            let ne_boundary = Rect::new(cx + w / 2.0, cy - h / 2.0, w, h);
            let mut ne_points = Vec::new();
            self.query(&ne_boundary, &mut ne_points);
            self.ne = Some(Box::new(QuadTree::new(ne_boundary, ne_points, self.depth + 1)));

            let se_boundary = Rect::new(cx + w / 2.0, cy + h / 2.0, w, h);
            let mut se_points = Vec::new();
            self.query(&se_boundary, &mut se_points);
            self.se = Some(Box::new(QuadTree::new(se_boundary, se_points, self.depth + 1)));

            let sw_boundary = Rect::new(cx - w / 2.0, cy + h / 2.0, w, h);
            let mut sw_points = Vec::new();
            self.query(&sw_boundary, &mut sw_points);
            self.sw = Some(Box::new(QuadTree::new(sw_boundary, sw_points, self.depth + 1)));

            self.divided = true;

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
            
            maxerror
        } else {
            self.divided = false;
            self.maxerror = Some(maxerror);
            maxerror
        }
    }

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

    fn non0blocks(&self) -> usize {
        let mut npoints = 0;
        if !self.divided {
            if !self.points.is_empty() {
                npoints = 1;
            }
        } else {
            if let Some(ref nw) = self.nw {
                npoints += nw.non0blocks();
            }
            if let Some(ref ne) = self.ne {
                npoints += ne.non0blocks();
            }
            if let Some(ref se) = self.se {
                npoints += se.non0blocks();
            }
            if let Some(ref sw) = self.sw {
                npoints += sw.non0blocks();
            }
        }
        npoints
    }
}

fn tree_from_csv(
    file_path: &str,
    idx_x: usize,
    idx_y: usize,
    idx_cell: usize,
    //threshold: f64,
    step: f64,
    //loop_flag: bool,
    method: &str,
    endpt: Option<usize>,
    allgenes: bool,
) -> (Vec<f64>, QuadTree) {
    let mut coords = Vec::new();
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut mind = Vec::new();
    let mut maxd = Vec::new();

    let file = File::open(file_path).expect("Failed to open file");
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);

    for result in rdr.records() {
        let record = result.expect("Failed to read record");
        let x: f64 = record[idx_x].parse().unwrap();
        let y: f64 = record[idx_y].parse().unwrap();
        xs.push(x);
        ys.push(y);

        let mut cells = Vec::new();
        let end = endpt.unwrap_or(if allgenes { record.len() } else { idx_cell + 1 });
        for i in idx_cell..end {
            let value: f64 = record[i].parse().unwrap();
            cells.push(value);
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

    let minx = xs.iter().cloned().fold(f64::INFINITY, f64::min) - 1.0;
    let miny = ys.iter().cloned().fold(f64::INFINITY, f64::min) - 1.0;
    let maxx = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 1.0;
    let maxy = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 1.0;
    let w = maxx - minx;
    let h = maxy - miny;

    let domain = Rect::new(minx + w / 2.0, miny + h / 2.0, w, h);
    let mut qtree = QuadTree::new(domain, coords, 1);
/* 
    if loop_flag {
        let sequence: Vec<f64> = (0..).map(|x| x as f64 * step).take_while(|&x| x < threshold).collect();
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
        let maxerrorl = qtree.divide(step, method, &mind, &maxd, &mut maxerrors);
        (maxerrors, qtree)
    //}

}

fn main() {
    let file_path = "/Users/zhezhenwang/Documents/patro/Moffitt_and_Bambah-Mukku_et_al_merfish_all_cells.csv";
    let (maxerrorl, qtree) = tree_from_csv(file_path, 5, 6, 9, 0.5, "mean", None, true);
    
    let file = File::create("output.bin.gz").unwrap();
    let encoder = GzEncoder::new(file, Compression::default());
    bincode::serialize_into(encoder, &qtree).unwrap();
    
    //println!("Max Errors: {:?}", maxerrorl);
    //println!("QuadTree Blocks: {}", qtree.blocks());
}

