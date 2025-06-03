use std::path::{Path, PathBuf};
use std::error::Error;
use std::fs::File;
use std::io::Read;
use csv::Writer;
use bincode::config;
use flate2::read::GzDecoder;
use quadtree::BitFieldQuadTree;
use clap::Parser;
use sux::traits::bit_field_slice::{BitFieldSlice, BitFieldSliceCore};

/// Build a quadtree representation of spatial transcriptomics data
#[derive(Parser)]
#[command(version, about, long_about = None)]
struct CmdArgs {
    /// Input file (bin.gz)
    #[arg(short = 'i', long)]
    input: PathBuf,
    /// Output file (default "output.bin.gz")
    #[arg(short = 'o', long)]
    output: Option<PathBuf>,
}

fn bitfield_to_csv(tree: &BitFieldQuadTree, output_path: &Path) -> Result<(), Box<dyn Error>> {
    let mut writer = Writer::from_path(output_path)?;

    // Write header row
    let mut header = vec!["x".to_string(), "y".to_string()];
    // Add gene columns - number them based on medians length
    for i in 0..tree.medians.len() {
        header.push(format!("gene_{}", i+1));
    }
    writer.write_record(&header)?;

    // Recursively traverse tree and write data
    write_node_data(tree, &mut writer)?;

    writer.flush()?;
    Ok(())
}

fn write_node_data(node: &BitFieldQuadTree, writer: &mut Writer<File>) -> Result<(), Box<dyn Error>> {
    // If this is a leaf node with data
    if !node.divided {
        if !node.medians.is_empty() {
            // Get center coordinates of boundary
            let x = node.boundary.cx;
            let y = node.boundary.cy;

            // Initialize gene values array with zeros
            let mut gene_values = vec![0u16; node.medians.len()];

            // For each gene that has data
            for (gene_idx, &median) in node.medians.iter().enumerate() {
                if let Some(bit_field) = node.data.get(gene_idx) {
                    // Get the differences from the bit field
                    for i in 0..bit_field.bit_field.len() {
                        let diff = bit_field.bit_field.get(i) as u16;
                        // Add back median to get original value
                        gene_values[gene_idx] = median.wrapping_add(diff);
                    }
                }
            }

            // Write row with coordinates and gene values
            let mut row = vec![x.to_string(), y.to_string()];
            row.extend(gene_values.iter().map(|v| v.to_string()));
            writer.write_record(&row)?;
        }
    } else {
        // Recursively process child nodes
        if let Some(ref nw) = node.nw {
            write_node_data(nw, writer)?;
        }
        if let Some(ref ne) = node.ne {
            write_node_data(ne, writer)?;
        }
        if let Some(ref se) = node.se {
            write_node_data(se, writer)?;
        }
        if let Some(ref sw) = node.sw {
            write_node_data(sw, writer)?;
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = CmdArgs::parse();
    
    // Open the compressed file
    let file = File::open(&args.input)?;
    //let mut decoder = GzDecoder::new(file);
    let mut decoder = file;
    
    // Read the compressed data
    let mut buffer = Vec::new();
    decoder.read_to_end(&mut buffer)?;
    
    // Decode the binary data
    let config = config::standard();
    let bit_field_tree: BitFieldQuadTree = bincode::decode_from_slice(&buffer, config)?.0;
    
    // Print some basic information about the decompressed data
    println!("Number of medians: {}", bit_field_tree.medians.len());
    println!("Number of bit fields: {}", bit_field_tree.data.len());
    
    // If output file is specified, write to CSV
    if let Some(output_path) = args.output {
        bitfield_to_csv(&bit_field_tree, &output_path)?;
        println!("Data written to {}", output_path.display());
    }
    
    Ok(())
}
