# Phase 2: Index Encoding Optimization Plan

## Current Status
- **Completed Phase 1**: Adaptive k-parameter selection
  - Savings: 40,420 bytes (2.2% reduction)
  - delta_vals improved from 44.95% to 44.53%
  - root_vals improved from 3.04% to 2.25%

## Phase 2 Target
- **indices**: Currently 915,808 bytes (51.25% of total)
- **Target**: Reduce to ~450,000 bytes (25% of total)
- **Expected savings**: ~460,000 bytes (25% additional reduction)

## Problem Analysis
Current encoding: `index = (dfs_pos-1) * num_genes + gene`
- Creates very large index values (up to ncells × 18,000)
- Example: cell 500, gene 10,000 → index = 499 × 18,000 + 10,000 = 8,992,000
- Even Elias-Fano encoding struggles with such large numbers

## Proposed Solution: Separate Cell/Gene Encoding

### Approach 1: Cell Boundaries + Gene Indices (RECOMMENDED)
Store two separate arrays:
1. **cell_boundaries**: Cumulative count of deltas per cell [n_deltas_cell0, n_deltas_cell0+cell1, ...]
2. **gene_indices**: Just the gene IDs (0 to num_genes-1)

Benefits:
- Gene indices are small (0-18,000 vs 0-9,000,000)
- Cell boundaries are small cumulative sums
- Expected compression: 50-60% reduction in indices size

### Implementation Steps:

#### 1. Modify EncodedDiffsMST structure
```rust
pub(crate) struct EncodedDiffsMST {
    pub(crate) num_genes: u32,
    pub(crate) parent_offset: DacsOpt,
    pub(crate) root_indices: HybridSparseVec,
    pub(crate) root_vals: DacsOpt,
    // NEW: Separate encoding
    pub(crate) cell_boundaries: DacsOpt,    // Cumulative delta counts per cell
    pub(crate) gene_indices: HybridSparseVec, // Just gene IDs (0-num_genes)
    pub(crate) delta_vals: DacsOpt,
    // OLD: Keep for backward compat during transition
    pub(crate) indices: HybridSparseVec,    // Combined indices (deprecated)
    pub(crate) use_v2_format: bool,         // Format version flag
}
```

#### 2. Modify encoding logic (encode_subarray_mst)
```rust
// Instead of combined_indices
let mut gene_indices_per_cell: Vec<Vec<u32>> = vec![Vec::new(); ncells-1];
let mut delta_vals_per_cell: Vec<Vec<u32>> = vec![Vec::new(); ncells-1];

for (dfs_pos, &orig_cell) in dfs_order_vec.iter().enumerate().skip(1) {
    let diff_list = sparse_subtract(child_expr, parent_expr);
    for (g, d) in &diff_list {
        gene_indices_per_cell[dfs_pos-1].push(*g);
        delta_vals_per_cell[dfs_pos-1].push(*d as u32);
    }
}

// Build cumulative boundaries
let mut boundaries = Vec::new();
let mut cumsum = 0u32;
for genes in &gene_indices_per_cell {
    cumsum += genes.len() as u32;
    boundaries.push(cumsum);
}

// Flatten gene indices
let flat_genes: Vec<u32> = gene_indices_per_cell.into_iter().flatten().collect();
let flat_deltas: Vec<u32> = delta_vals_per_cell.into_iter().flatten().collect();

// Encode
let cell_boundaries = DacsOpt::from_slice(&boundaries, Some(3))?;
let gene_indices = HybridSparseVec::from_indices(
    &flat_genes.iter().map(|&g| g as u64).collect::<Vec<_>>(),
    flat_genes.len() as f64 / (num_genes as f64),
    num_genes as usize
);
```

#### 3. Modify decoding logic (decode_cell_at_dfs_pos)
```rust
fn get_cell_deltas(&self, dfs_pos: usize) -> Vec<(u32, i32)> {
    if dfs_pos == 0 || !self.use_v2_format {
        return self.get_cell_deltas_old(dfs_pos); // Backward compat
    }
    
    // Get boundaries for this cell
    let start_idx = if dfs_pos == 1 {
        0
    } else {
        self.cell_boundaries.access(dfs_pos - 2).unwrap_or(0) as usize
    };
    let end_idx = self.cell_boundaries.access(dfs_pos - 1).unwrap_or(0) as usize;
    
    // Extract genes and deltas for this cell
    let all_genes = self.gene_indices.indices_vec();
    let mut deltas = Vec::new();
    for i in start_idx..end_idx {
        let gene = all_genes[i] as u32;
        let delta = self.delta_vals.access(i).unwrap_or(0) as i32;
        deltas.push((gene, delta));
    }
    deltas
}
```

#### 4. Update serialization (Encode/Decode impls)
Add cell_boundaries and gene_indices to bincode serialization

#### 5. Update bytes_breakdown() for logging
Show separate sizes for cell_boundaries and gene_indices

### Expected Results
Current: indices = 915,808 bytes
After optimization:
- cell_boundaries ≈ 5,000 bytes (small cumulative sums)
- gene_indices ≈ 400,000 bytes (small gene IDs instead of huge combined indices)
- **Total indices ≈ 405,000 bytes (56% reduction)**

Combined with Phase 1:
- **Total size reduction**: ~500,000 bytes (27% overall compression improvement)

## Alternative Approach: Delta Encoding on Indices

Simpler but less effective:
- Store first index, then deltas between consecutive indices
- Benefits from sorted indices having small gaps
- Expected savings: 20-30% on indices (vs 50-60% for cell/gene separation)

## Testing Strategy
1. Implement with backward compatibility flag
2. Test with HD16 dataset
3. Validate lossless round-trip
4. Compare sizes before/after
5. Merge if successful

## Timeline Estimate
- Structure modification: 1-2 hours
- Encoding logic: 2-3 hours
- Decoding logic: 1-2 hours
- Serialization updates: 1 hour
- Testing and validation: 2-3 hours
- **Total: 7-11 hours of focused development**
