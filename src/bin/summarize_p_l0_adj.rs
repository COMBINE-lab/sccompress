use anyhow::{Context, Result};
use clap::Parser;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Parser)]
#[command(
    version,
    about = "Summarize BH-adjusted p_l0 significance counts across CSV files"
)]
struct Args {
    /// Directory containing CSV files to scan.
    #[arg(long, default_value = ".")]
    input_dir: PathBuf,
    /// Output summary CSV path.
    #[arg(long, default_value = "p_l0_adj_summary.csv")]
    output: PathBuf,
}

fn benjamini_hochberg(p_values: &[f64]) -> Vec<f64> {
    let m = p_values.len();
    if m == 0 {
        return vec![];
    }

    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| {
        let pa = if p_values[a].is_finite() {
            p_values[a]
        } else {
            f64::INFINITY
        };
        let pb = if p_values[b].is_finite() {
            p_values[b]
        } else {
            f64::INFINITY
        };
        pa.partial_cmp(&pb).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut sorted_q = vec![f64::NAN; m];
    let mut next = 1.0_f64;
    for k in (0..m).rev() {
        let p = p_values[order[k]];
        let rank = (k + 1) as f64;
        let q = if p.is_finite() && p >= 0.0 {
            ((m as f64) * p / rank).min(next)
        } else {
            f64::NAN
        };
        sorted_q[k] = q;
        if q.is_finite() {
            next = q;
        }
    }

    let mut adjusted = vec![f64::NAN; m];
    for k in 0..m {
        adjusted[order[k]] = sorted_q[k];
    }
    adjusted
}

fn is_target_csv(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
        return false;
    };
    let lower = name.to_ascii_lowercase();
    lower.ends_with(".csv") && lower.contains("_l0")
}

#[derive(Clone, Copy)]
enum L0Kind {
    Spatial,
    Exprs,
}

fn parse_dataset_and_kind(path: &Path) -> Option<(String, L0Kind)> {
    let stem = path.file_stem()?.to_str()?;
    let lower = stem.to_ascii_lowercase();
    if let Some(idx) = lower.find("_spatial") {
        return Some((stem[..idx].to_string(), L0Kind::Spatial));
    }
    if let Some(idx) = lower.find("_exprs") {
        return Some((stem[..idx].to_string(), L0Kind::Exprs));
    }
    None
}

fn process_csv(path: &Path) -> Result<Option<(usize, usize)>> {
    let mut rdr = csv::Reader::from_path(path)
        .with_context(|| format!("failed to open CSV: {}", path.display()))?;
    let headers = rdr
        .headers()
        .with_context(|| format!("failed to read header: {}", path.display()))?
        .clone();

    let p_col = match headers.iter().position(|h| h == "p_l0") {
        Some(i) => i,
        None => return Ok(None),
    };

    let mut pvals = Vec::new();
    for rec in rdr.records() {
        let rec = rec.with_context(|| format!("failed reading row in {}", path.display()))?;
        let p = rec
            .get(p_col)
            .map(str::trim)
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(f64::NAN);
        pvals.push(p);
    }

    let adj = benjamini_hochberg(&pvals);
    let count = adj
        .iter()
        .filter(|&&p| p.is_finite() && p >= 0.0 && p < 0.05)
        .count();
    Ok(Some((count, pvals.len())))
}

fn main() -> Result<()> {
    let args = Args::parse();

    let mut grouped: BTreeMap<String, (Option<(usize, usize)>, Option<(usize, usize)>)> =
        BTreeMap::new();

    for entry in fs::read_dir(&args.input_dir)
        .with_context(|| format!("failed to read dir: {}", args.input_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() || !is_target_csv(&path) {
            continue;
        }
        let Some((dataset, kind)) = parse_dataset_and_kind(&path) else {
            continue;
        };
        if let Some((count, total)) = process_csv(&path)? {
            let slot = grouped.entry(dataset).or_insert((None, None));
            match kind {
                L0Kind::Spatial => slot.0 = Some((count, total)),
                L0Kind::Exprs => slot.1 = Some((count, total)),
            }
        }
    }

    let mut wtr = csv::Writer::from_path(&args.output)
        .with_context(|| format!("failed to create output: {}", args.output.display()))?;
    wtr.write_record(["filename", "spatial_l0", "spatial_l0 rate", "exprs_l0"])?;
    for (name, (spatial, exprs)) in grouped {
        let spatial_txt = spatial.map(|(c, t)| format!("{c}/{t}")).unwrap_or_default();
        let exprs_txt = exprs.map(|(c, t)| format!("{c}/{t}")).unwrap_or_default();
        let spatial_rate_txt = spatial
            .map(|(c, t)| format!("{:.2}", c as f64 / t as f64))
            .unwrap_or_default();
        wtr.write_record([name, spatial_txt, spatial_rate_txt, exprs_txt])?;
    }
    wtr.flush()?;

    println!("Wrote {}", args.output.display());
    Ok(())
}
