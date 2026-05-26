# csCompress quick tutorial

This guide walks through the two halves of using the `compress` tool:

1. **Compress** — turn a 10x-style count matrix into a single binary
   payload, optionally with spatial coordinates.
2. **Decompress** — recover the count matrix (and verify it against
   the original input).

Everything below assumes you've already built the release binary
(`target/release/compress`) — see [Build](#build) at the bottom if you
haven't.

---

## Compress

The encoder is invoked through the `build` subcommand. The same
subcommand handles both single-cell and spatial inputs; the only
difference is whether you also pass a positions file and a spatial
`--platform`.

### Input file

The encoder reads a **10x Genomics-style HDF5 matrix** (`matrix/data`,
`matrix/indices`, `matrix/indptr`, `matrix/barcodes`, `matrix/shape`),
the same `filtered_feature_bc_matrix.h5` format produced by Cell
Ranger (and `cell_feature_matrix.h5` produced by Xenium Ranger).

### General usage

The minimal compress command, suitable for any single-cell count
matrix and the starting point for the spatial variants below:

```bash
./target/release/compress build \
    --input            /path/to/filtered_feature_bc_matrix.h5 \
    --output           out/payload.bin.zst
```

What each flag does:

| Flag | What it controls | Recommended |
|---|---|---|
| `--input` | The 10x H5 file | required |
| `--output PATH` | Where to write the payload | required if you want the file on disk |

The command above relies on the encoder's defaults for everything
else. The following defaults are applied implicitly and are
equivalent to passing them on the command line:

| Default flag | Value | What it does |
|---|---|---|
| `--platform` | `single-cell` | Treats the input as a count matrix only, with no positions. Use `--platform visium` or `--platform xenium` together with `--input-pos` for spatial data |
| `--matrix-orientation` | `gene-cell` | Treats the on-disk layout as `(genes × cells)`, which matches raw 10x H5 files. Override with `--matrix-orientation cell-gene` only if your file is already transposed to `(cells × genes)` |
| `--cell-blocks` | `6` | Number of cell clusters; robust across single-cell datasets |
| `--gene-blocks` | `5` | Split genes into 5 contiguous blocks after reordering; one (cluster × block) tile per piece. Bigger blocks reduce overhead but limit parallelism |
| `--output-compression` | `zstd` | Outer compressor wrapping the bincode payload. Other options are `gzip` and `seven-zip` |
| `--index-codec` | `arithmetic` | Codec for integer index streams (rANS via `constriction`). `stream-vbyte` is faster at decode |
| `--sorted-index-codec` | `delta` | Codec for sorted index lists. `elias-fano` is the alternative |
| `--knn-metric` | `l0` | Distance metric for MST candidate edges |
| `--mst-weight` | `metric` | Edge-weight scheme used by Kruskal |
| `--row-mst-window` | `8` | Window of nearby nodes considered for candidate edges |
| `--gene-reorder-method` | `projection` | Hash-projection seriation for the gene order |

If you want to override any of these, pass the corresponding flag
explicitly — for example `--cell-blocks 8` or
`--output-compression gzip`. See [Tuning knobs at a glance](#tuning-knobs-at-a-glance)
for the knobs that move the size/decode-speed tradeoff most.

### Optional: spatial mode (Visium / Xenium)

Use spatial mode when you have a positions file and want the
coordinates captured inside the payload. The encoder will use the
coordinates as a secondary signal during cell clustering, which
usually improves compression for visually structured tissue and
always preserves the coordinates for downstream consumers.

In addition to the H5 matrix, you also need a per-barcode positions
file in **Parquet** or **CSV** format. The first column must be the
cell barcode (matching `matrix/barcodes`); the x and y columns are
picked automatically from `--platform`: `(4, 5)` for `visium`
(full-resolution pixel coordinates) and `(1, 2)` for `xenium`
(`x_centroid` / `y_centroid`).

Two example positions schemas:

```
# 10x Visium tissue_positions.csv (no header by default)
# col 0       col 1            col 2    col 3      col 4         col 5
# barcode  in_tissue        array_row array_col pxl_row_in_fullres pxl_col_in_fullres
AAACAACGAATAGTTC-1,1,0,16,8580,8580
AAACAAGTATCTCCCA-1,1,50,102,3873,11148
...

# Xenium cells.parquet (Arrow schema)
# cell_id (utf8), x_centroid (f64), y_centroid (f64), area (f64), ...
```

#### Visium

```bash
./target/release/compress build \
    --input            /path/to/Visium/filtered_feature_bc_matrix.h5 \
    --input-pos        /path/to/Visium/spatial/tissue_positions.parquet \
    --platform         visium \
    --output           out/visium.bin.zst
```

All the defaults from [General usage](#general-usage) still apply.
Visium uses columns `4` (x) and `5` (y) of `tissue_positions.parquet`
— the standard full-resolution pixel coordinates produced by Space
Ranger — and does not need them to be specified on the command line.

If your positions file is the older CSV form, replace
`tissue_positions.parquet` with the CSV path and add
`--pos-format csv`. `--pos-format` itself defaults to `parquet`.

#### Xenium

```bash
./target/release/compress build \
    --input            /path/to/Xenium/cell_feature_matrix.h5 \
    --input-pos        /path/to/Xenium/cells.parquet \
    --platform         xenium \
    --output           out/xenium.bin.zst
```

Same defaults as [General usage](#general-usage) apply. Xenium uses
columns `1` (`x_centroid`) and `2` (`y_centroid`) of `cells.parquet`
for the position values, matching the standard Xenium Ranger output
schema.

---

## Decompress

The compressor ships a single decode entry point, the `dump`
subcommand, which decodes the encoded payload back into a CSR count
matrix and then compares it against the original H5 to confirm the
reconstruction is bit-exact. There is no separate `decompress`
subcommand — the decompression step itself is the first half of
`dump`, and the verify step that follows reads the ground-truth H5
(and the positions file, for spatial payloads) and checks the two
matrices agree under any row/column permutation chosen by the
encoder. Pass `--output PATH` to additionally write the decoded CSR
matrix to disk as a 10x-style HDF5 file (see [Dumping the decoded
matrix to disk](#dumping-the-decoded-matrix-to-disk) below).

Decompression is **platform-agnostic** — it produces the same kind
of CSR matrix regardless of whether the payload was encoded under
`single-cell`, `visium`, or `xenium`. The platform-specific bit is
only on the verify side: spatial payloads need `--input-pos` resupplied
so the verifier can re-resolve barcode-to-position. The position
columns themselves are picked automatically from `--platform`, just
as on the compress side.

### General usage

For a single-cell payload, point the decoder at the encoded file and
the original H5:

```bash
./target/release/compress dump \
    --encoded          out/payload.bin.zst \
    --input            /path/to/filtered_feature_bc_matrix.h5
```

`--platform` defaults to `single-cell`, so no platform flag is needed
for single-cell payloads. `--input-pos` is optional for single-cell
and is only required when decoding payloads encoded under
`--platform visium` or `--platform xenium` (see below).

### Optional: spatial mode (Visium / Xenium)

For spatial payloads, pass `--platform` explicitly and re-supply the
positions file so the verifier can re-resolve barcode-to-position:

```bash
./target/release/compress dump \
    --encoded          out/visium.bin.zst \
    --input            /path/to/Visium/filtered_feature_bc_matrix.h5 \
    --input-pos        /path/to/Visium/spatial/tissue_positions.parquet \
    --platform         visium
```

Swap `visium` for `xenium` (and the corresponding inputs) to decode a
Xenium payload. `--platform` must match whatever the payload was
encoded under; the same `dump` decodes payloads from all three
platforms.

### Output

On success the command prints something like:

```
Decompression PASSED: matches input under row/column permutation.
verify=890ms, end-to-end=2426ms (decode=1348ms + truth-load=187ms + verify=890ms)
```

If you only want to **time decompression** itself (without the
ground-truth H5 load and the verify pass), look for the
`total decode time = …ms` line in the same output — that is the wall
time of the decompression step in isolation.

### Dumping the decoded matrix to disk

By default `dump` only verifies in memory — nothing is written to
disk. Pass `--output PATH` to additionally write the decoded CSR
matrix to a 10x-style HDF5 file:

```bash
./target/release/compress dump \
    --encoded          out/payload.bin.zst \
    --input            /path/to/filtered_feature_bc_matrix.h5 \
    --output           out/decoded.h5
```

The output file is laid out exactly like a Cell Ranger
`filtered_feature_bc_matrix.h5` — `matrix/data`, `matrix/indices`,
`matrix/indptr`, `matrix/shape`, `matrix/barcodes`,
`matrix/features/{name,id,feature_type,genome}` — so any tool that
reads 10x H5 (scanpy, sprs, scran, …) can load it without changes.

**Important caveat about row/column order.** The decoded matrix is in
the encoder's internal *decoder order* (cells grouped by cluster,
genes ordered by the encoder's SVD-based seriation), not the original
input H5's row/column order. Because the payload does not store the
back-permutation to the input, the dumped H5 uses **placeholder
labels**: barcodes are `cell_0..cell_{n_cells-1}`, gene names and ids
are `gene_0..gene_{n_genes-1}`, feature types are `Gene Expression`
and genome is `decoded`. The counts themselves are bit-exact, so
analyses that don't depend on barcode-specific lookups (count totals,
PCA, clustering on the decoded matrix, marker scoring against gene
names you supply separately) work as expected; analyses that look up
specific barcodes or gene ids need a separate mapping step.

---

## Tuning knobs at a glance

You almost never need to touch these, but they're useful to know:

- `--cell-blocks N` — number of cell clusters. Default 6. Larger N
  increases parallelism and shrinks per-tile MST cost, but the
  per-cluster cost increases and very small tiles compress worse.
- `--gene-blocks K` — split the genes into K contiguous blocks after
  the gene reordering step. Default 5. Larger K means more, smaller
  tiles → more parallelism but slightly worse compression. Set to 1
  to disable splitting.
- `--max-cluster-size M` — recursively split any cluster larger than M
  cells. Useful on very large spatial datasets (Visium HD, 320k PBMCs,
  …) where one cluster can otherwise blow up memory.
- `--output-compression {zstd,gzip,seven-zip}` — the outer wrapping
  compressor. `zstd` is the fastest at decode time; `seven-zip` gives
  the smallest payload but is slow to compress; `gzip` is the most
  portable.
- `--joint-svd-fast` — use a randomized SVD for cell clustering. Much
  faster on large matrices with negligible impact on compression.
- `--index-codec {arithmetic,stream-vbyte}` — codec for integer index
  streams. `arithmetic` (default, actually rANS via `constriction`)
  gives smaller payloads; `stream-vbyte` is faster to decode.
- `--set-seed N` — deterministic RNG for clustering / randomized SVD.
  Pass any integer for reproducible runs.
- `RAYON_NUM_THREADS` (environment variable) — number of worker
  threads used for the parallel decode / encode passes. Defaults to
  `4` when not set; override with e.g. `export RAYON_NUM_THREADS=8`
  on bigger machines.

---

## Build

If you haven't built the binary yet:

```bash
cd compress
cargo build --release
```

The binary lands at `target/release/compress`. macOS users on Apple
Silicon: no special flags needed; the default release profile already
enables LTO.

A few system-level dependencies that may be required:

- **HDF5** — usually via `brew install hdf5` on macOS or `apt-get install
  libhdf5-dev` on Debian/Ubuntu.
- **7-Zip** binary, *only* if you plan to use `--output-compression seven-zip`.
  Install via `brew install p7zip` (macOS) or `apt-get install p7zip-full`
  (Debian/Ubuntu). Plain `gzip` and `zstd` output do not require any
  external binaries.
