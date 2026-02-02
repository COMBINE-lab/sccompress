# Testing MST Compression with H5/Parquet Data

## Test Data Source

The test data is available via Google Drive:
- **Link**: https://drive.google.com/file/d/1cxrYhnYPqtgUbu-TbaoMB12t7ttNI0Rj/view?usp=sharing
- **Format**: tar.gz archive containing H5 and Parquet files

## Download Instructions

### Option 1: Direct Download (Browser)
1. Click the Google Drive link above
2. Click "Download" button
3. Save as `test_data.tar.gz`

### Option 2: Command Line (requires gdown)
```bash
pip install gdown
gdown 1cxrYhnYPqtgUbu-TbaoMB12t7ttNI0Rj
```

### Option 3: wget/curl
```bash
# Using wget
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1cxrYhnYPqtgUbu-TbaoMB12t7ttNI0Rj' -O test_data.tar.gz

# Or using curl
curl -L 'https://drive.google.com/uc?export=download&id=1cxrYhnYPqtgUbu-TbaoMB12t7ttNI0Rj' -o test_data.tar.gz
```

## Setup and Extraction

```bash
# Extract the archive
tar -xzf test_data.tar.gz

# Verify contents
ls -lh test_data/
# Should show:
# - H5_spatial_subset2.h5
# - H5_spatial_subset2.parquet
```

## Building the Application

```bash
# Ensure HDF5 is installed (Ubuntu/Debian)
sudo apt-get install libhdf5-dev

# Build in release mode for performance
cargo build --release

# Verify binary exists
./target/release/quadtree --version
```

## Running the Compression Test

```bash
./target/release/quadtree build \
  -i test_data/H5_spatial_subset2.h5 \
  -p test_data/H5_spatial_subset2.parquet \
  -f csr \
  -P visium \
  -o H5_spatial_subset2.bin.gz
```

### Command Breakdown

- `-i test_data/H5_spatial_subset2.h5`: Input HDF5 file with expression data
- `-p test_data/H5_spatial_subset2.parquet`: Position data (cell coordinates)
- `-f csr`: Input format (Compressed Sparse Row)
- `-P visium`: Platform type (Visium spatial transcriptomics)
- `-o H5_spatial_subset2.bin.gz`: Output compressed file

## Expected Output

The compression process should log MST encoding statistics:

```
[Level 0] MST encoding: XXXX cells, pattern_changes=XXXXX, change_pct=XX.XXX%, parent_offset avg: XX.X
  Size breakdown: parent_offset=XXXX bytes, root_indices=XX bytes, root_vals=XXX bytes, indices=XXXXX bytes, delta_vals=XXXXX bytes
```

### Key Metrics to Check

1. **Pattern Changes Percentage**: Should be <20% (lower is better)
   - Indicates how different child cells are from parents
   - Low percentage means good spatial coherence

2. **Size Breakdown**:
   - `parent_offset`: Should be ~1-3% (tree structure)
   - `root_indices`: Should be minimal (<1%)
   - `root_vals`: Should be minimal (<1%)
   - `indices`: 30-40% (position encoding)
   - `delta_vals`: 50-65% (value encoding)

3. **Average Parent Offset**: Should be relatively small (10-100)
   - Indicates MST structure quality
   - Lower values mean better spatial locality

4. **Final Compression Ratio**:
   - Compare output file size to original data size
   - Target: >10× compression

## Validation Steps

### 1. Check Compression Succeeded
```bash
ls -lh H5_spatial_subset2.bin.gz
# Should show compressed output file
```

### 2. Verify File Integrity
```bash
gunzip -t H5_spatial_subset2.bin.gz
# Should report: "OK"
```

### 3. Compare Sizes
```bash
# Original data size
du -h test_data/H5_spatial_subset2.h5

# Compressed size
du -h H5_spatial_subset2.bin.gz

# Calculate ratio
# ratio = original_size / compressed_size
```

### 4. Test Decompression (if dump command exists)
```bash
./target/release/quadtree dump \
  -i H5_spatial_subset2.bin.gz \
  -o decompressed_check.csv

# Compare with original (random sampling)
# Ensure lossless reconstruction
```

## Troubleshooting

### Issue: "Unable to locate HDF5"
```bash
# Install HDF5 development files
sudo apt-get install libhdf5-dev

# Or on macOS
brew install hdf5

# Clean and rebuild
cargo clean
cargo build --release
```

### Issue: "File format not recognized"
- Verify the download completed successfully
- Check file size matches expected
- Try re-downloading from Google Drive

### Issue: High pattern_changes percentage (>25%)
- This is expected for less spatially coherent data
- May indicate heterogeneous tissue regions
- Not necessarily a problem, just less compressible

## Optimization Analysis

After successful compression, analyze the output to identify optimization opportunities:

1. **If delta_vals > 60%**: Consider entropy coding improvements
2. **If indices > 40%**: Consider separate cell/gene encoding
3. **If pattern_changes > 20%**: Consider adaptive kNN parameters

See `OPTIMIZATION_NOTES.md` for detailed improvement strategies.

## Reporting Results

Please report:
1. ✅ Compression succeeded/failed
2. 📊 Pattern changes percentage
3. 📦 Compression ratio (original size / compressed size)
4. 🔍 Size breakdown percentages
5. ⏱️ Compression time
6. 💾 Peak memory usage (if available)

Example report:
```
✅ Compression succeeded
📊 Pattern changes: 15.2%
📦 Compression ratio: 18.5×
🔍 Breakdown: parent_offset=1.8%, indices=37.2%, delta_vals=60.4%
⏱️ Time: 12.3 seconds
💾 Memory: 2.1 GB
```

## Next Steps

Based on test results:
1. If compression ratio < 10×: Investigate data characteristics
2. If pattern_changes > 20%: Try different kNN parameters
3. For optimal performance: Implement optimizations from OPTIMIZATION_NOTES.md
