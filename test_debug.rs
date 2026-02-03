fn main() {
    let flat_genes: Vec<u32> = vec![0, 100, 1000, 18084];
    let flat_genes_u64: Vec<u64> = flat_genes.iter().map(|&g| g as u64).collect();
    let max_gene_idx = flat_genes_u64.iter().max().copied().unwrap_or(0);
    let num_genes = 18085u32;
    let gene_universe = ((max_gene_idx + 1) as usize).max(num_genes as usize);
    
    println!("max_gene_idx: {}", max_gene_idx);
    println!("num_genes: {}", num_genes);
    println!("gene_universe: {}", gene_universe);
    println!("Should be > max_gene_idx: {}", gene_universe > max_gene_idx as usize);
}
