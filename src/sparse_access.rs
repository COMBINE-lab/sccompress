use sprs::CsMat;
use std::time::Instant;

/// Thin abstraction for sparse row reads so row/column-oriented backends can be
/// introduced incrementally without rewriting call sites.
pub trait SparseRowAccess {
    fn cols(&self) -> usize;
    fn for_each_nonzero_in_row(&self, row_idx: usize, f: &mut dyn FnMut(usize, u16));
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SparseRowBackend {
    Csr,
    CscRowsCache,
}

pub enum SparseRowAccessView<'a> {
    Csr(&'a CsMat<u16>),
    CscRowsCache {
        cols: usize,
        rows: Vec<Vec<(usize, u16)>>,
    },
}

impl<'a> SparseRowAccessView<'a> {
    pub fn from_csr_with_timing(csr: &'a CsMat<u16>, backend: SparseRowBackend) -> (Self, u64) {
        let t0 = Instant::now();
        let view = match backend {
            SparseRowBackend::Csr => Self::Csr(csr),
            SparseRowBackend::CscRowsCache => {
                let csc = csr.to_csc();
                let mut rows = vec![Vec::<(usize, u16)>::new(); csr.rows()];
                for (col_idx, col) in csc.outer_iterator().enumerate() {
                    for (row_idx, &value) in col.iter() {
                        if value != 0 {
                            rows[row_idx].push((col_idx, value));
                        }
                    }
                }
                Self::CscRowsCache {
                    cols: csr.cols(),
                    rows,
                }
            }
        };
        (view, t0.elapsed().as_millis() as u64)
    }
}

impl SparseRowAccess for CsMat<u16> {
    fn cols(&self) -> usize {
        self.cols()
    }

    fn for_each_nonzero_in_row(&self, row_idx: usize, f: &mut dyn FnMut(usize, u16)) {
        if let Some(row) = self.outer_view(row_idx) {
            for (gene_idx, &value) in row.iter() {
                if value != 0 {
                    f(gene_idx, value);
                }
            }
        }
    }
}

impl SparseRowAccess for SparseRowAccessView<'_> {
    fn cols(&self) -> usize {
        match self {
            Self::Csr(csr) => csr.cols(),
            Self::CscRowsCache { cols, .. } => *cols,
        }
    }

    fn for_each_nonzero_in_row(&self, row_idx: usize, f: &mut dyn FnMut(usize, u16)) {
        match self {
            Self::Csr(csr) => csr.for_each_nonzero_in_row(row_idx, f),
            Self::CscRowsCache { rows, .. } => {
                if let Some(row) = rows.get(row_idx) {
                    for &(gene_idx, value) in row {
                        f(gene_idx, value);
                    }
                }
            }
        }
    }
}
