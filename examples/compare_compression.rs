// Simple example to compare MST vs Cluster compression

use quadtree::quad_tree::tree::{Point};
use sprs::CsMat;
use tracing::info;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    // For now, create synthetic data for demonstration
    use sprs::TriMatBase;
    
    let ncells = 100;
    let ngenes = 1000;
    
    let mut tri_mat: TriMatBase<Vec<usize>, Vec<u16>> = TriMatBase::new((ncells, ngenes));
    
    // Create heterogeneous clusters
    for cluster in 0..5 {
        let start_cell = cluster * 20;
        let end_cell = start_cell + 20;
        let start_gene = cluster * 200;
        let end_gene = start_gene + 100;
        
        for cell in start_cell..end_cell {
            for gene in start_gene..end_gene {
                let value = 50 + ((cell + gene) % 100) as u16;
                tri_mat.add_triplet(cell, gene, value);
            }
        }
    }
    
    let csr: CsMat<u16> = tri_mat.to_csr();
    let points: Vec<Point> = (0..ncells)
        .map(|i| Point::new(i as f64, 0.0, i))
        .collect();
    
    info!("Data: {} cells × {} genes", ncells, ngenes);
    info!("Created synthetic heterogeneous data with 5 distinct clusters");
    
    // Note: encode functions are not pub, so this example demonstrates the approach
    // In practice, would need to expose these functions or use through main API
    
    Ok(())
}
