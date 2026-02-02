# MST Compression Optimization Opportunities

## Current Performance (merfish6k.csv - 6509 cells × 155 genes)

### Compression Breakdown
- **parent_offset**: 5,228 bytes (1.61%) ✓ Excellent
- **root_indices**: 18 bytes (0.01%) ✓ Excellent  
- **root_vals**: 170 bytes (0.05%) ✓ Excellent
- **indices**: 126,096 bytes (38.91%) ⚠️ Optimization opportunity
- **delta_vals**: 192,564 bytes (59.42%) ⚠️ Largest component
- **Total MST encoding**: 324,080 bytes
- **Final compressed size**: 372 KB (after gzip)

### Key Metrics
- Pattern changes: 141,902 (14.067% of possible edges)
- Average parent offset: 58.6 positions
- Compression ratio: ~20x vs original CSV

## Optimization Opportunities

### 1. Delta Value Encoding (59% of data - HIGH PRIORITY)

**Current approach**: DacsOpt with zigzag encoding

**Issue**: Most deltas are small (±1, ±2, etc.) but DacsOpt may not be optimally tuned for this distribution.

**Potential improvements**:
- **Run-length encoding** for consecutive zeros in deltas
- **Entropy coding** (e.g., Huffman or Arithmetic coding) based on actual delta distribution
- **Dictionary-based encoding** for frequently occurring delta patterns
- **Bit-packing with variable-width fields** tuned to delta magnitude distribution

**Expected gain**: 10-30% reduction in delta_vals size

### 2. Index Encoding (39% of data - HIGH PRIORITY)

**Current approach**: HybridSparseVec (Elias-Fano or bitvector based on 75% threshold)

**Issue**: Combined index format `(dfs_pos-1) * num_genes + gene` creates large numbers

**Potential improvements**:
- **Separate encoding** for cell offsets vs gene indices
  - Cell offset changes rarely (only at cell boundaries)
  - Gene indices are typically small and repeated
- **Delta encoding for indices**: encode difference from previous index rather than absolute position
- **Block-based encoding**: group deltas by cell and use more compact local indexing
- **Adjust Elias-Fano threshold**: current 75% may not be optimal for this data

**Expected gain**: 20-40% reduction in indices size

### 3. Adaptive kNN Parameter (MEDIUM PRIORITY)

**Current**: Fixed k=8 neighbors

**Potential improvements**:
- **Adaptive k** based on local cell density
- **Spatial distance weighting** in kNN graph construction
- **Better root selection** using centrality measures (currently arbitrary)

**Expected gain**: 5-10% reduction by improving MST quality (fewer/smaller deltas)

### 4. Parallel Encoding (LOW PRIORITY - Performance, not size)

**Current**: Sequential encoding

**Potential improvements**:
- Parallelize sparse expression computation
- Parallelize delta computation for independent subtrees

**Expected gain**: Faster compression time, no size change

### 5. DacsOpt Tuning (MEDIUM PRIORITY)

**Current**: DacsOpt with k=3 (Some(3))

**Issue**: Fixed k may not be optimal for this data

**Potential improvements**:
- **Adaptive k selection** based on value distribution analysis
- **Separate k values** for root_vals vs delta_vals (different distributions)
- **Profile-guided optimization**: analyze actual value distributions in test data

**Expected gain**: 5-15% reduction in value encoding size

## Recommended Implementation Order

1. **Index encoding optimization** (HIGH impact, MEDIUM complexity)
   - Implement separate cell/gene encoding
   - Test delta encoding for indices
   
2. **Delta value encoding optimization** (HIGH impact, HIGH complexity)
   - Analyze delta value distribution
   - Implement entropy-based encoding or run-length encoding
   
3. **DacsOpt parameter tuning** (MEDIUM impact, LOW complexity)
   - Quick experiment with different k values
   - Measure impact on real data

4. **Adaptive kNN** (MEDIUM impact, MEDIUM complexity)
   - Implement density-based k selection
   - Better root selection algorithm

## Testing Recommendations

For each optimization:
1. Test on provided H5/Parquet data (when available)
2. Measure both compression ratio AND decompression speed
3. Ensure lossless round-trip still works
4. Profile memory usage for large datasets

## Notes

- Current implementation achieves ~20x compression on merfish6k.csv
- MST approach is working well (14% pattern changes is good)
- Main bottleneck is encoding the actual delta values and their positions
- Any optimization must maintain lossless property
