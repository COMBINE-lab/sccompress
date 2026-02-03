# Cluster-Based Compression for Spatial Transcriptomics

## Overview

This document describes the cluster-based compression strategy implemented as an alternative to MST-based compression.

## Motivation

The MST-based approach performs well for spatially coherent data where nearby cells have similar expression. However, when the MST has many "heavy edges" (large differences between parent-child cells), cluster-based compression can be more effective.

## Algorithm

### 1. Clustering

Cells are grouped by expression similarity using a k-means-like algorithm:
- **Distance metric**: L0 distance (number of genes with different expression patterns)
- **Number of clusters**: Adaptive, roughly sqrt(n) cells
- **Initialization**: Evenly spaced cells as initial representatives
- **Refinement**: Iterative (max 10 iterations) until convergence
- **Representative selection**: Cell with minimum total distance to other cluster members

### 2. Encoding

Each cluster forms a "star" topology:
- **Representative**: Stored directly (gene indices + values)
- **Member cells**: Stored as deltas from representative
- **Delta encoding**: Same as MST (sparse_subtract + zigzag)
- **Adaptive k-parameter**: Uses optimal DacsOpt k for each component

### 3. Decoding

Reconstruction is straightforward:
1. Look up cluster assignment for cell
2. Start with cluster representative
3. Apply cell's delta

No tree traversal needed!

## Data Structure

```rust
struct EncodedDiffsCluster {
    num_genes: u32,
    num_clusters: u32,
    cluster_assignments: DacsOpt,           // Which cluster per cell
    cluster_rep_indices: Vec<HybridSparseVec>, // Representative gene indices
    cluster_rep_vals: Vec<DacsOpt>,         // Representative values
    indices: HybridSparseVec,               // Delta indices
    delta_vals: DacsOpt,                    // Delta values
}
```

## Advantages

1. **Simpler structure**: Stars vs tree traversal
2. **Better for heterogeneous data**: Multiple independent clusters
3. **Direct access**: No need to walk tree path
4. **More robust**: No dependency on tree structure quality

## Disadvantages

1. **Clustering overhead**: Need to run clustering algorithm
2. **Multiple representatives**: One per cluster (vs one root in MST)
3. **May be worse for homogeneous data**: MST can be more efficient

## When to Use

Cluster-based compression is better when:
- Data has distinct subpopulations
- MST has many heavy edges
- Cells cluster naturally by expression

MST is better when:
- Data is spatially smooth
- Strong spatial coherence
- Cells form a gradient

## Adaptive Selection

The `encode_subarray_adaptive()` function tries both methods and automatically selects the one with smaller compressed size.

## Performance

On heterogeneous data (5 distinct cell types):
- Cluster: ~X% better compression
- MST: Longer parent paths, more deltas

On homogeneous spatial data:
- MST: ~X% better compression  
- Cluster: Redundant representatives

## Testing

Comprehensive tests verify:
- Lossless round-trip reconstruction
- Clustering algorithm correctness
- Comparison with MST encoding

All tests pass successfully.

## Future Improvements

1. **Hybrid approach**: Use MST within clusters, clusters across tissue
2. **Adaptive cluster count**: Better heuristic than sqrt(n)
3. **Hierarchical clustering**: Tree of clusters
4. **Parallel encoding**: Encode clusters independently
