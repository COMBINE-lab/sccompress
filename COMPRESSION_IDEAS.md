# Compression Improvement Ideas For Large Sparse Count Matrices

Context:
- Very large, very sparse matrices.
- Non-zero values are mostly very small integers.
- Row and column order can be changed freely.

## High-impact ideas

1. Re-encode deltas per child row, not as one global flattened index stream
- Store per-child local edits instead of global `(cell_offset * num_genes + gene)` indices.
- Use local gene IDs with compact streams such as:
  - edit counts per child
  - per-child gene gaps
  - aligned delta values
- Goal: reduce index entropy, which is currently the dominant byte budget.

2. Improve parent-child pairing metric for sparse supports
- For MST parent selection, test metrics better aligned with sparse overlap:
  - binary Jaccard
  - weighted Jaccard
- Better overlap should reduce per-child edit counts.

3. Reorder columns (genes) to improve locality
- Build a gene co-occurrence graph and reorder genes to place co-occurring genes nearby.
- Candidate approaches:
  - spectral seriation
  - reverse Cuthill-McKee
  - other min-linear-arrangement heuristics
- Goal: shrink within-row index gaps.

4. Reorder rows for compression objective
- Within clusters, reorder rows to reduce support-set edit distance along traversal.
- Candidate approaches:
  - kNN-graph path heuristics
  - Hilbert + local refinement
  - TSP-like approximations on row similarity graph
- Goal: improve MST deltas.

5. Try specialized integer codecs for index streams
- Evaluate alternatives to arithmetic coding for index-like streams:
  - Elias-Fano
  - SIMD-BP128
  - StreamVByte
  - Roaring-style block encodings
- Different gap distributions may favor different codecs.

6. Lossy mode: sparsify tiny counts, not just quantize values
- Since most non-zeros are tiny integers, dropping/thinning very small counts can reduce index count directly.
- Quantization alone may not help much if support patterns remain unchanged.

7. Consider CSC/inverted encoding mode
- Encode per-gene posting lists (row IDs + small counts), then decode/transcode back to CSR as needed.
- With row reorder, row-ID deltas may compress very well.

## Suggested experiment order

1. Implement per-child local delta streams (idea 1).
2. Add column reorder pre-pass based on co-occurrence (idea 3).
3. Compare parent-search metrics (idea 2) using `delta_indices` bytes and total gzip size.
