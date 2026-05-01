use crate::arith_encode::ArithmeticEncoded;
use crate::index_stream::{EncodedU32Stream, IndexStreamCodec};
use crate::sorted_indices::{EncodedSortedIndices, SortedIndexCodec};
use bincode::{Decode, Encode};
use petgraph::algo::{connected_components, min_spanning_tree};
use petgraph::data::FromElements;
use petgraph::graph::UnGraph;
use petgraph::visit::EdgeRef;
use rayon::prelude::*;
use sprs::CsMat;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug)]
pub struct Point {
    pub x: f64,
    pub y: f64,
    pub row_index: usize,
}

impl Point {
    pub const fn new(x: f64, y: f64, row_index: usize) -> Self {
        Self { x, y, row_index }
    }
}

#[derive(Clone, Debug, Encode, Decode)]
pub struct DatalessPoint {
    x: f64,
    y: f64,
}

impl DatalessPoint {
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

#[derive(Clone, Encode, Decode)]
pub struct EncodedDiffsMST {
    pub num_genes: u32,
    pub parent_offset: ArithmeticEncoded,
    pub local_to_global: EncodedSortedIndices,
    pub root_indices: EncodedSortedIndices,
    pub root_vals: ArithmeticEncoded,
    pub child_modes: ArithmeticEncoded,
    pub child_full_counts: ArithmeticEncoded,
    pub child_remove_counts: ArithmeticEncoded,
    pub child_add_counts: ArithmeticEncoded,
    pub child_update_counts: ArithmeticEncoded,
    pub row_template_counts: ArithmeticEncoded,
    pub row_template_first_genes: EncodedU32Stream,
    pub row_template_gene_gaps: EncodedU32Stream,
    pub row_template_vals: ArithmeticEncoded,
    pub child_template_ids: EncodedU32Stream,
    pub child_full_first_genes: EncodedU32Stream,
    pub child_full_gene_gaps: EncodedU32Stream,
    pub child_remove_first_genes: EncodedU32Stream,
    pub child_remove_gene_gaps: EncodedU32Stream,
    pub child_add_first_genes: EncodedU32Stream,
    pub child_add_gene_gaps: EncodedU32Stream,
    pub child_update_first_genes: EncodedU32Stream,
    pub child_update_gene_gaps: EncodedU32Stream,
    pub child_full_vals: ArithmeticEncoded,
    pub child_add_vals: ArithmeticEncoded,
    pub child_update_vals: ArithmeticEncoded,
}

impl EncodedDiffsMST {
    pub fn num_cells(&self) -> usize {
        self.parent_offset.len()
    }

    pub fn bytes_breakdown(&self) -> (usize, usize, usize, usize, usize, usize) {
        (
            self.parent_offset.size_in_bytes(),
            self.local_to_global.size_in_bytes() + self.root_indices.size_in_bytes(),
            self.root_vals.size_in_bytes(),
            self.child_modes.size_in_bytes()
                + self.child_full_counts.size_in_bytes()
                + self.child_remove_counts.size_in_bytes()
                + self.child_add_counts.size_in_bytes()
                + self.child_update_counts.size_in_bytes()
                + self.row_template_counts.size_in_bytes()
                + self.child_full_first_genes.size_in_bytes()
                + self.child_full_gene_gaps.size_in_bytes()
                + self.row_template_first_genes.size_in_bytes()
                + self.row_template_gene_gaps.size_in_bytes()
                + self.child_template_ids.size_in_bytes()
                + self.child_remove_first_genes.size_in_bytes()
                + self.child_remove_gene_gaps.size_in_bytes()
                + self.child_add_first_genes.size_in_bytes()
                + self.child_add_gene_gaps.size_in_bytes()
                + self.child_update_first_genes.size_in_bytes()
                + self.child_update_gene_gaps.size_in_bytes(),
            self.child_full_vals.size_in_bytes()
                + self.row_template_vals.size_in_bytes()
                + self.child_add_vals.size_in_bytes()
                + self.child_update_vals.size_in_bytes(),
            4,
        )
    }

    pub fn total_bytes(&self) -> usize {
        let (p, ri, rv, i, dv, ng) = self.bytes_breakdown();
        p + ri + rv + i + dv + ng
    }

    fn parent_dfs_pos(&self, dfs_pos: usize) -> usize {
        if dfs_pos == 0 {
            return 0;
        }
        let offset = self.parent_offset.access(dfs_pos).unwrap_or(0) as usize;
        dfs_pos.saturating_sub(offset)
    }

    fn apply_edit_ops(
        parent: &[(u32, i32)],
        removes: &[u32],
        adds: &[(u32, u32)],
        updates: &[(u32, u32)],
    ) -> Vec<(u32, i32)> {
        let mut result = Vec::with_capacity(parent.len().saturating_add(adds.len()));
        let mut i = 0usize;
        let mut r = 0usize;
        let mut a = 0usize;
        let mut u = 0usize;

        while i < parent.len() || a < adds.len() {
            let next_parent_gene = parent.get(i).map(|(g, _)| *g);
            let next_add_gene = adds.get(a).map(|(g, _)| *g);

            match (next_parent_gene, next_add_gene) {
                (Some(pg), Some(ag)) if ag < pg => {
                    let value = adds[a].1 as i32;
                    if value > 0 {
                        result.push((ag, value));
                    }
                    a += 1;
                }
                (Some(pg), _) => {
                    if removes.get(r).copied() == Some(pg) {
                        i += 1;
                        r += 1;
                        continue;
                    }
                    if updates.get(u).map(|(g, _)| *g) == Some(pg) {
                        let value = updates[u].1 as i32;
                        if value > 0 {
                            result.push((pg, value));
                        }
                        i += 1;
                        u += 1;
                        continue;
                    }

                    let value = parent[i].1;
                    if value > 0 {
                        result.push((pg, value));
                    }
                    i += 1;
                }
                (None, Some(ag)) => {
                    let value = adds[a].1 as i32;
                    if value > 0 {
                        result.push((ag, value));
                    }
                    a += 1;
                }
                (None, None) => break,
            }
        }

        result
    }

    pub fn sparse_expression_iter(&self) -> SparseExpressionIterMST<'_> {
        SparseExpressionIterMST::new(self)
    }

    #[allow(dead_code)]
    pub fn expression_vec_iter(&self) -> ExpressionVecIterMST<'_> {
        ExpressionVecIterMST::new(self)
    }
}

#[derive(Clone, Encode, Decode)]
pub enum ColumnValuesEncoded {
    Global(ArithmeticEncoded),
    PerGene(Vec<ArithmeticEncoded>),
}

impl ColumnValuesEncoded {
    fn from_flat(values: &[u32]) -> Self {
        let encoded = if values.is_empty() {
            ArithmeticEncoded::default()
        } else {
            ArithmeticEncoded::from_slice(values).expect("valid posting values")
        };
        ColumnValuesEncoded::Global(encoded)
    }

    fn from_per_gene(values_by_gene: &[Vec<u32>]) -> Self {
        let mut streams = Vec::new();
        for values in values_by_gene {
            if values.is_empty() {
                continue;
            }
            streams.push(ArithmeticEncoded::from_slice(values).expect("valid posting values"));
        }
        ColumnValuesEncoded::PerGene(streams)
    }

    fn size_in_bytes(&self) -> usize {
        match self {
            ColumnValuesEncoded::Global(stream) => stream.size_in_bytes(),
            ColumnValuesEncoded::PerGene(streams) => {
                streams.iter().map(ArithmeticEncoded::size_in_bytes).sum()
            }
        }
    }

    fn decode_flattened(&self, posting_counts: &[u32]) -> Vec<u32> {
        match self {
            ColumnValuesEncoded::Global(stream) => stream.decode_all().unwrap_or_default(),
            ColumnValuesEncoded::PerGene(streams) => {
                let total: usize = posting_counts.iter().map(|&x| x as usize).sum();
                let mut out = Vec::with_capacity(total);
                let mut stream_idx = 0usize;
                for &count in posting_counts {
                    if count == 0 {
                        continue;
                    }
                    let decoded = streams
                        .get(stream_idx)
                        .map(|s| s.decode_all().unwrap_or_default())
                        .unwrap_or_default();
                    stream_idx += 1;
                    out.extend(decoded.into_iter().take(count as usize));
                }
                out
            }
        }
    }
}

#[derive(Clone, Encode, Decode)]
pub struct EncodedColumnBlock {
    pub num_cells: u32,
    pub num_genes: u32,
    pub local_to_global: EncodedSortedIndices,
    pub posting_modes: ArithmeticEncoded,
    pub posting_counts: ArithmeticEncoded,
    pub raw_row_firsts: EncodedU32Stream,
    pub raw_row_gaps: EncodedU32Stream,
    pub template_counts: ArithmeticEncoded,
    pub template_row_firsts: EncodedU32Stream,
    pub template_row_gaps: EncodedU32Stream,
    pub ref_parent_deltas: EncodedU32Stream,
    pub ref_remove_counts: ArithmeticEncoded,
    pub ref_add_counts: ArithmeticEncoded,
    pub ref_remove_firsts: EncodedU32Stream,
    pub ref_remove_gaps: EncodedU32Stream,
    pub ref_add_firsts: EncodedU32Stream,
    pub ref_add_gaps: EncodedU32Stream,
    pub template_ids: EncodedU32Stream,
    pub template_remove_counts: ArithmeticEncoded,
    pub template_add_counts: ArithmeticEncoded,
    pub template_remove_firsts: EncodedU32Stream,
    pub template_remove_gaps: EncodedU32Stream,
    pub template_add_firsts: EncodedU32Stream,
    pub template_add_gaps: EncodedU32Stream,
    pub vals: ColumnValuesEncoded,
}

impl EncodedColumnBlock {
    pub(crate) const MODE_RAW: u32 = 0;
    pub(crate) const MODE_REF: u32 = 1;
    pub(crate) const MODE_TEMPLATE: u32 = 2;

    pub fn total_bytes(&self) -> usize {
        self.local_to_global.size_in_bytes()
            + self.posting_modes.size_in_bytes()
            + self.posting_counts.size_in_bytes()
            + self.raw_row_firsts.size_in_bytes()
            + self.raw_row_gaps.size_in_bytes()
            + self.template_counts.size_in_bytes()
            + self.template_row_firsts.size_in_bytes()
            + self.template_row_gaps.size_in_bytes()
            + self.ref_parent_deltas.size_in_bytes()
            + self.ref_remove_counts.size_in_bytes()
            + self.ref_add_counts.size_in_bytes()
            + self.ref_remove_firsts.size_in_bytes()
            + self.ref_remove_gaps.size_in_bytes()
            + self.ref_add_firsts.size_in_bytes()
            + self.ref_add_gaps.size_in_bytes()
            + self.template_ids.size_in_bytes()
            + self.template_remove_counts.size_in_bytes()
            + self.template_add_counts.size_in_bytes()
            + self.template_remove_firsts.size_in_bytes()
            + self.template_remove_gaps.size_in_bytes()
            + self.template_add_firsts.size_in_bytes()
            + self.template_add_gaps.size_in_bytes()
            + self.vals.size_in_bytes()
            + 8
    }

    pub fn bytes_breakdown(&self) -> (usize, usize, usize, usize, usize, usize) {
        (
            0,
            self.local_to_global.size_in_bytes(),
            self.posting_modes.size_in_bytes()
                + self.posting_counts.size_in_bytes()
                + self.template_counts.size_in_bytes()
                + self.ref_remove_counts.size_in_bytes()
                + self.ref_add_counts.size_in_bytes()
                + self.template_remove_counts.size_in_bytes()
                + self.template_add_counts.size_in_bytes(),
            self.raw_row_firsts.size_in_bytes()
                + self.raw_row_gaps.size_in_bytes()
                + self.template_row_firsts.size_in_bytes()
                + self.template_row_gaps.size_in_bytes()
                + self.ref_parent_deltas.size_in_bytes()
                + self.ref_remove_firsts.size_in_bytes()
                + self.ref_remove_gaps.size_in_bytes()
                + self.ref_add_firsts.size_in_bytes()
                + self.ref_add_gaps.size_in_bytes()
                + self.template_ids.size_in_bytes()
                + self.template_remove_firsts.size_in_bytes()
                + self.template_remove_gaps.size_in_bytes()
                + self.template_add_firsts.size_in_bytes()
                + self.template_add_gaps.size_in_bytes(),
            self.vals.size_in_bytes(),
            8,
        )
    }

    pub fn uses_per_gene_values(&self) -> bool {
        matches!(self.vals, ColumnValuesEncoded::PerGene(_))
    }

    fn decode_sorted_rows(
        count: usize,
        firsts: &[u32],
        gaps: &[u32],
        first_cursor: &mut usize,
        gap_cursor: &mut usize,
    ) -> Vec<u32> {
        if count == 0 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(count);
        let mut row = firsts.get(*first_cursor).copied().unwrap_or(0);
        *first_cursor += 1;
        out.push(row);
        for _ in 1..count {
            let gap = gaps.get(*gap_cursor).copied().unwrap_or(0);
            *gap_cursor += 1;
            row = row.saturating_add(gap);
            out.push(row);
        }
        out
    }

    fn apply_row_edits(parent: &[u32], removes: &[u32], adds: &[u32]) -> Vec<u32> {
        let mut out = Vec::with_capacity(parent.len().saturating_add(adds.len()));
        let mut p = 0usize;
        let mut r = 0usize;
        let mut a = 0usize;

        while p < parent.len() || a < adds.len() {
            let parent_row = parent.get(p).copied();
            let add_row = adds.get(a).copied();
            match (parent_row, add_row) {
                (Some(pr), Some(ar)) if ar < pr => {
                    out.push(ar);
                    a += 1;
                }
                (Some(pr), _) => {
                    if removes.get(r).copied() == Some(pr) {
                        p += 1;
                        r += 1;
                        continue;
                    }
                    out.push(pr);
                    p += 1;
                }
                (None, Some(ar)) => {
                    out.push(ar);
                    a += 1;
                }
                (None, None) => break,
            }
        }
        out
    }

    pub fn decode_rows(&self) -> Vec<Vec<(u32, u16)>> {
        #[derive(Clone)]
        enum SupportPlan {
            Empty,
            Raw(Vec<u32>),
            Ref {
                parent_idx: usize,
                removes: Vec<u32>,
                adds: Vec<u32>,
            },
            Template {
                template_idx: usize,
                removes: Vec<u32>,
                adds: Vec<u32>,
            },
        }

        fn resolve_support_rows(
            gene_idx: usize,
            plans: &[SupportPlan],
            templates: &[Vec<u32>],
            memo: &mut [Option<Vec<u32>>],
            visiting: &mut [bool],
        ) -> Vec<u32> {
            if let Some(rows) = memo.get(gene_idx).and_then(Option::as_ref) {
                return rows.clone();
            }
            if gene_idx >= plans.len() || visiting.get(gene_idx).copied().unwrap_or(false) {
                return Vec::new();
            }

            visiting[gene_idx] = true;
            let rows = match &plans[gene_idx] {
                SupportPlan::Empty => Vec::new(),
                SupportPlan::Raw(rows) => rows.clone(),
                SupportPlan::Ref {
                    parent_idx,
                    removes,
                    adds,
                } => {
                    let parent_rows = if *parent_idx == gene_idx {
                        Vec::new()
                    } else {
                        resolve_support_rows(*parent_idx, plans, templates, memo, visiting)
                    };
                    EncodedColumnBlock::apply_row_edits(&parent_rows, removes, adds)
                }
                SupportPlan::Template {
                    template_idx,
                    removes,
                    adds,
                } => {
                    let template_rows = templates.get(*template_idx).cloned().unwrap_or_default();
                    EncodedColumnBlock::apply_row_edits(&template_rows, removes, adds)
                }
            };
            visiting[gene_idx] = false;
            memo[gene_idx] = Some(rows.clone());
            rows
        }

        let nrows = self.num_cells as usize;
        let mut rows: Vec<Vec<(u32, u16)>> = vec![Vec::new(); nrows];

        let local_to_global = self.local_to_global.decode_all_u32();
        let modes = self.posting_modes.decode_all().unwrap_or_default();
        let counts = self.posting_counts.decode_all().unwrap_or_default();
        let raw_firsts = self.raw_row_firsts.decode_all();
        let raw_gaps = self.raw_row_gaps.decode_all();
        let template_counts = self.template_counts.decode_all().unwrap_or_default();
        let template_row_firsts = self.template_row_firsts.decode_all();
        let template_row_gaps = self.template_row_gaps.decode_all();
        let ref_parent_deltas = self.ref_parent_deltas.decode_all();
        let ref_remove_counts = self.ref_remove_counts.decode_all().unwrap_or_default();
        let ref_add_counts = self.ref_add_counts.decode_all().unwrap_or_default();
        let ref_remove_firsts = self.ref_remove_firsts.decode_all();
        let ref_remove_gaps = self.ref_remove_gaps.decode_all();
        let ref_add_firsts = self.ref_add_firsts.decode_all();
        let ref_add_gaps = self.ref_add_gaps.decode_all();
        let template_ids = self.template_ids.decode_all();
        let template_remove_counts = self.template_remove_counts.decode_all().unwrap_or_default();
        let template_add_counts = self.template_add_counts.decode_all().unwrap_or_default();
        let template_remove_firsts = self.template_remove_firsts.decode_all();
        let template_remove_gaps = self.template_remove_gaps.decode_all();
        let template_add_firsts = self.template_add_firsts.decode_all();
        let template_add_gaps = self.template_add_gaps.decode_all();
        let vals = self.vals.decode_flattened(&counts);

        let mut raw_first_cursor = 0usize;
        let mut raw_gap_cursor = 0usize;
        let mut template_row_first_cursor = 0usize;
        let mut template_row_gap_cursor = 0usize;
        let mut ref_parent_cursor = 0usize;
        let mut ref_count_cursor = 0usize;
        let mut ref_remove_first_cursor = 0usize;
        let mut ref_remove_gap_cursor = 0usize;
        let mut ref_add_first_cursor = 0usize;
        let mut ref_add_gap_cursor = 0usize;
        let mut template_ref_cursor = 0usize;
        let mut template_ref_count_cursor = 0usize;
        let mut template_remove_first_cursor = 0usize;
        let mut template_remove_gap_cursor = 0usize;
        let mut template_add_first_cursor = 0usize;
        let mut template_add_gap_cursor = 0usize;
        let mut val_cursor = 0usize;
        let mut plans: Vec<SupportPlan> = Vec::with_capacity(local_to_global.len());
        let mut counts_per_gene = Vec::with_capacity(local_to_global.len());
        let mut templates = Vec::with_capacity(template_counts.len());
        for &count_u32 in &template_counts {
            let count = count_u32 as usize;
            let rows = Self::decode_sorted_rows(
                count,
                &template_row_firsts,
                &template_row_gaps,
                &mut template_row_first_cursor,
                &mut template_row_gap_cursor,
            );
            templates.push(rows);
        }

        for gene_pos in 0..local_to_global.len() {
            let count = counts.get(gene_pos).copied().unwrap_or(0) as usize;
            let mode = modes.get(gene_pos).copied().unwrap_or(Self::MODE_RAW);
            counts_per_gene.push(count);

            let plan = if count == 0 {
                SupportPlan::Empty
            } else if mode == Self::MODE_REF {
                let parent_delta = ref_parent_deltas
                    .get(ref_parent_cursor)
                    .copied()
                    .unwrap_or(0);
                ref_parent_cursor += 1;
                let signed_delta = zigzag_decode_i64(parent_delta);
                let parent_idx = (gene_pos as i64)
                    .checked_add(signed_delta)
                    .filter(|idx| *idx >= 0 && (*idx as usize) < local_to_global.len())
                    .map(|idx| idx as usize)
                    .unwrap_or(gene_pos);
                let remove_count = ref_remove_counts
                    .get(ref_count_cursor)
                    .copied()
                    .unwrap_or(0) as usize;
                let add_count = ref_add_counts.get(ref_count_cursor).copied().unwrap_or(0) as usize;
                ref_count_cursor += 1;

                let removes = Self::decode_sorted_rows(
                    remove_count,
                    &ref_remove_firsts,
                    &ref_remove_gaps,
                    &mut ref_remove_first_cursor,
                    &mut ref_remove_gap_cursor,
                );
                let adds = Self::decode_sorted_rows(
                    add_count,
                    &ref_add_firsts,
                    &ref_add_gaps,
                    &mut ref_add_first_cursor,
                    &mut ref_add_gap_cursor,
                );
                SupportPlan::Ref {
                    parent_idx,
                    removes,
                    adds,
                }
            } else if mode == Self::MODE_TEMPLATE {
                let template_idx =
                    template_ids.get(template_ref_cursor).copied().unwrap_or(0) as usize;
                template_ref_cursor += 1;
                let remove_count = template_remove_counts
                    .get(template_ref_count_cursor)
                    .copied()
                    .unwrap_or(0) as usize;
                let add_count = template_add_counts
                    .get(template_ref_count_cursor)
                    .copied()
                    .unwrap_or(0) as usize;
                template_ref_count_cursor += 1;

                let removes = Self::decode_sorted_rows(
                    remove_count,
                    &template_remove_firsts,
                    &template_remove_gaps,
                    &mut template_remove_first_cursor,
                    &mut template_remove_gap_cursor,
                );
                let adds = Self::decode_sorted_rows(
                    add_count,
                    &template_add_firsts,
                    &template_add_gaps,
                    &mut template_add_first_cursor,
                    &mut template_add_gap_cursor,
                );
                SupportPlan::Template {
                    template_idx,
                    removes,
                    adds,
                }
            } else {
                let raw_rows = Self::decode_sorted_rows(
                    count,
                    &raw_firsts,
                    &raw_gaps,
                    &mut raw_first_cursor,
                    &mut raw_gap_cursor,
                );
                SupportPlan::Raw(raw_rows)
            };
            plans.push(plan);
        }

        let mut memo: Vec<Option<Vec<u32>>> = vec![None; local_to_global.len()];
        let mut visiting = vec![false; local_to_global.len()];
        for (gene_pos, &gene) in local_to_global.iter().enumerate() {
            let count = counts_per_gene.get(gene_pos).copied().unwrap_or(0);
            let posting_rows =
                resolve_support_rows(gene_pos, &plans, &templates, &mut memo, &mut visiting);
            for value_idx in 0..count {
                let value = vals.get(val_cursor).copied().unwrap_or(0) as u16;
                val_cursor += 1;
                let row = posting_rows.get(value_idx).copied().unwrap_or(0) as usize;
                if row < nrows && value > 0 {
                    rows[row].push((gene, value));
                }
            }
        }

        rows
    }
}

#[derive(Clone, Encode, Decode)]
pub enum EncodedClusterBlock {
    RowMst(EncodedDiffsMST),
    Column(EncodedColumnBlock),
}

impl EncodedClusterBlock {
    pub fn bytes_breakdown(&self) -> (usize, usize, usize, usize, usize, usize) {
        match self {
            EncodedClusterBlock::RowMst(block) => block.bytes_breakdown(),
            EncodedClusterBlock::Column(block) => block.bytes_breakdown(),
        }
    }

    pub fn decode_rows(&self) -> Vec<Vec<(u32, u16)>> {
        match self {
            EncodedClusterBlock::RowMst(block) => block.sparse_expression_iter().collect(),
            EncodedClusterBlock::Column(block) => block.decode_rows(),
        }
    }
}

type SparseExpression = Vec<(u32, u16)>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KnnDistanceMetric {
    L0,
    L2,
    Hamming,
    Jaccard,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MstWeightMode {
    Metric,
    EncodingCost,
}

fn compute_sparse_expression(
    point: &Point,
    data: &CsMat<u16>,
    gene_old_to_new: Option<&[u32]>,
) -> SparseExpression {
    let mut expression = Vec::new();
    if let Some(row) = data.outer_view(point.row_index) {
        for (gene_idx, &value) in row.iter() {
            if value == 0 {
                continue;
            }
            let mapped_gene = gene_old_to_new
                .and_then(|mapping| mapping.get(gene_idx).copied())
                .unwrap_or(gene_idx as u32);
            // In gene-block mode, u32::MAX marks genes outside the active block.
            // Skip them so each tile only encodes its own columns.
            if mapped_gene == u32::MAX {
                continue;
            }
            expression.push((mapped_gene, value));
        }
    }
    if gene_old_to_new.is_some() {
        expression.sort_unstable_by_key(|(gene, _)| *gene);
    }
    expression
}

fn build_local_gene_remap(expressions: &[SparseExpression]) -> (Vec<SparseExpression>, Vec<u32>) {
    let mut local_to_global = Vec::<u32>::new();
    for expr in expressions {
        for &(gene, _) in expr {
            local_to_global.push(gene);
        }
    }
    local_to_global.sort_unstable();
    local_to_global.dedup();

    if local_to_global.is_empty() {
        return (expressions.to_vec(), local_to_global);
    }

    let mut global_to_local = HashMap::<u32, u32>::with_capacity(local_to_global.len());
    for (local, &global) in local_to_global.iter().enumerate() {
        global_to_local.insert(global, local as u32);
    }

    let remapped = expressions
        .iter()
        .map(|expr| {
            expr.iter()
                .map(|&(gene, value)| {
                    let local_gene = *global_to_local
                        .get(&gene)
                        .expect("global gene must exist in cluster dictionary");
                    (local_gene, value)
                })
                .collect::<SparseExpression>()
        })
        .collect::<Vec<_>>();

    (remapped, local_to_global)
}

fn support_projection_weight(idx: u32, seed: u64) -> f64 {
    let mut z = (idx as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(seed);
    z ^= z >> 30;
    z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z ^= z >> 27;
    z = z.wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    let unit = (z as f64) / (u64::MAX as f64);
    (unit * 2.0) - 1.0
}

fn zigzag_encode_i64(delta: i64) -> u32 {
    ((delta << 1) ^ (delta >> 63)) as u32
}

fn zigzag_decode_i64(v: u32) -> i64 {
    ((v >> 1) as i64) ^ (-((v & 1) as i64))
}

fn optimize_row_order_for_column(expressions: &[SparseExpression]) -> Vec<u32> {
    let mut scored: Vec<(u32, f64, f64, usize)> = Vec::with_capacity(expressions.len());
    for (row_idx, expr) in expressions.iter().enumerate() {
        let mut s1 = 0.0f64;
        let mut s2 = 0.0f64;
        for &(gene, value) in expr {
            let w = (value as f64).ln_1p();
            s1 += w * support_projection_weight(gene, 0xA5A5_A5A5_A5A5_A5A5);
            s2 += w * support_projection_weight(gene, 0x3C3C_3C3C_3C3C_3C3C);
        }
        scored.push((row_idx as u32, s1, s2, expr.len()));
    }

    scored.sort_unstable_by(|a, b| {
        a.1.total_cmp(&b.1)
            .then_with(|| a.2.total_cmp(&b.2))
            .then_with(|| a.3.cmp(&b.3))
            .then_with(|| a.0.cmp(&b.0))
    });
    scored.into_iter().map(|(idx, _, _, _)| idx).collect()
}

fn compute_row_set_edits(parent: &[u32], child: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let mut removes = Vec::new();
    let mut adds = Vec::new();
    let mut p = 0usize;
    let mut c = 0usize;

    while p < parent.len() && c < child.len() {
        match parent[p].cmp(&child[c]) {
            std::cmp::Ordering::Less => {
                removes.push(parent[p]);
                p += 1;
            }
            std::cmp::Ordering::Greater => {
                adds.push(child[c]);
                c += 1;
            }
            std::cmp::Ordering::Equal => {
                p += 1;
                c += 1;
            }
        }
    }

    while p < parent.len() {
        removes.push(parent[p]);
        p += 1;
    }
    while c < child.len() {
        adds.push(child[c]);
        c += 1;
    }

    (removes, adds)
}

fn support_ref_cost_bytes(
    parent_idx: usize,
    child_idx: usize,
    parent_rows: &[u32],
    child_rows: &[u32],
    index_codec: IndexStreamCodec,
) -> u64 {
    let (removes, adds) = compute_row_set_edits(parent_rows, child_rows);
    let parent_delta = zigzag_encode_i64((parent_idx as i64) - (child_idx as i64));
    index_symbol_bytes(parent_delta, index_codec)
        + encoded_gene_only_cost_bytes(&removes, index_codec)
        + encoded_gene_only_cost_bytes(&adds, index_codec)
        + 2
}

fn build_gene_reference_parents(
    posting_rows: &[Vec<u32>],
    raw_costs: &[u64],
    index_codec: IndexStreamCodec,
) -> Vec<Option<usize>> {
    let mut parents = vec![None; posting_rows.len()];
    let non_empty: Vec<usize> = posting_rows
        .iter()
        .enumerate()
        .filter_map(|(gene_idx, rows)| (!rows.is_empty()).then_some(gene_idx))
        .collect();
    if non_empty.len() <= 1 {
        return parents;
    }

    let mut graph = UnGraph::<usize, u64>::new_undirected();
    let mut node_for_gene = vec![None; posting_rows.len()];
    for &gene_idx in &non_empty {
        let node = graph.add_node(gene_idx);
        node_for_gene[gene_idx] = Some(node);
    }

    let mut projected: Vec<(usize, f64, f64, usize)> = non_empty
        .iter()
        .map(|&gene_idx| {
            let mut s1 = 0.0f64;
            let mut s2 = 0.0f64;
            for &row in &posting_rows[gene_idx] {
                s1 += support_projection_weight(row, 0xD1B5_4A32_9C73_8E19);
                s2 += support_projection_weight(row, 0x5F19_77CB_8A31_2D43);
            }
            (gene_idx, s1, s2, posting_rows[gene_idx].len())
        })
        .collect();
    projected.sort_unstable_by(|a, b| {
        a.1.total_cmp(&b.1)
            .then_with(|| a.2.total_cmp(&b.2))
            .then_with(|| a.3.cmp(&b.3))
            .then_with(|| a.0.cmp(&b.0))
    });

    let mut added = HashSet::<(usize, usize)>::new();
    let mut add_edge = |a_gene: usize, b_gene: usize, graph: &mut UnGraph<usize, u64>| {
        if a_gene == b_gene {
            return;
        }
        let (u_gene, v_gene) = if a_gene < b_gene {
            (a_gene, b_gene)
        } else {
            (b_gene, a_gene)
        };
        if !added.insert((u_gene, v_gene)) {
            return;
        }
        let ab = support_ref_cost_bytes(
            u_gene,
            v_gene,
            &posting_rows[u_gene],
            &posting_rows[v_gene],
            index_codec,
        );
        let ba = support_ref_cost_bytes(
            v_gene,
            u_gene,
            &posting_rows[v_gene],
            &posting_rows[u_gene],
            index_codec,
        );
        let weight = ab.min(ba);
        let Some(u_node) = node_for_gene[u_gene] else {
            return;
        };
        let Some(v_node) = node_for_gene[v_gene] else {
            return;
        };
        graph.update_edge(u_node, v_node, weight);
    };

    const GENE_MST_WINDOW: usize = 12;
    for i in 0..projected.len() {
        let upper = (i + GENE_MST_WINDOW + 1).min(projected.len());
        for j in (i + 1)..upper {
            add_edge(projected[i].0, projected[j].0, &mut graph);
        }
    }
    for pair in non_empty.windows(2) {
        add_edge(pair[0], pair[1], &mut graph);
    }

    if connected_components(&graph) > 1 {
        let anchor = non_empty[0];
        for &gene_idx in non_empty.iter().skip(1) {
            add_edge(anchor, gene_idx, &mut graph);
        }
    }

    let mst_graph = UnGraph::<usize, u64>::from_elements(min_spanning_tree(&graph));
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); posting_rows.len()];
    for edge in mst_graph.edge_references() {
        let u = mst_graph[edge.source()];
        let v = mst_graph[edge.target()];
        adjacency[u].push(v);
        adjacency[v].push(u);
    }

    let mut visited = vec![false; posting_rows.len()];
    for &start in &non_empty {
        if visited[start] {
            continue;
        }
        let mut stack = vec![start];
        let mut component = Vec::new();
        visited[start] = true;
        while let Some(node) = stack.pop() {
            component.push(node);
            for &nbr in &adjacency[node] {
                if !visited[nbr] {
                    visited[nbr] = true;
                    stack.push(nbr);
                }
            }
        }

        let root = component
            .iter()
            .copied()
            .min_by_key(|&gene_idx| raw_costs[gene_idx])
            .unwrap_or(start);
        let mut orient_stack = vec![root];
        let mut oriented_seen = HashSet::<usize>::new();
        oriented_seen.insert(root);
        parents[root] = None;
        while let Some(node) = orient_stack.pop() {
            for &nbr in &adjacency[node] {
                if oriented_seen.insert(nbr) {
                    parents[nbr] = Some(node);
                    orient_stack.push(nbr);
                }
            }
        }
    }

    parents
}

fn build_gene_reference_parents_local_window(
    posting_rows: &[Vec<u32>],
    raw_costs: &[u64],
    index_codec: IndexStreamCodec,
    window: usize,
) -> Vec<Option<usize>> {
    let mut parents = vec![None; posting_rows.len()];
    let non_empty: Vec<usize> = posting_rows
        .iter()
        .enumerate()
        .filter_map(|(gene_idx, rows)| (!rows.is_empty()).then_some(gene_idx))
        .collect();
    if non_empty.len() <= 1 {
        return parents;
    }

    let mut graph = UnGraph::<usize, u64>::new_undirected();
    let mut node_for_gene = vec![None; posting_rows.len()];
    for &gene_idx in &non_empty {
        let node = graph.add_node(gene_idx);
        node_for_gene[gene_idx] = Some(node);
    }

    let mut added = HashSet::<(usize, usize)>::new();
    let mut add_edge = |a_gene: usize, b_gene: usize, graph: &mut UnGraph<usize, u64>| {
        if a_gene == b_gene {
            return;
        }
        let (u_gene, v_gene) = if a_gene < b_gene {
            (a_gene, b_gene)
        } else {
            (b_gene, a_gene)
        };
        if !added.insert((u_gene, v_gene)) {
            return;
        }
        let ab = support_ref_cost_bytes(
            u_gene,
            v_gene,
            &posting_rows[u_gene],
            &posting_rows[v_gene],
            index_codec,
        );
        let ba = support_ref_cost_bytes(
            v_gene,
            u_gene,
            &posting_rows[v_gene],
            &posting_rows[u_gene],
            index_codec,
        );
        let weight = ab.min(ba);
        let Some(u_node) = node_for_gene[u_gene] else {
            return;
        };
        let Some(v_node) = node_for_gene[v_gene] else {
            return;
        };
        graph.update_edge(u_node, v_node, weight);
    };

    let w = window.max(1).min(non_empty.len().saturating_sub(1));
    for i in 0..non_empty.len() {
        let upper = (i + w + 1).min(non_empty.len());
        for j in (i + 1)..upper {
            add_edge(non_empty[i], non_empty[j], &mut graph);
        }
    }

    if connected_components(&graph) > 1 {
        let anchor = non_empty[0];
        for &gene_idx in non_empty.iter().skip(1) {
            add_edge(anchor, gene_idx, &mut graph);
        }
    }

    let mst_graph = UnGraph::<usize, u64>::from_elements(min_spanning_tree(&graph));
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); posting_rows.len()];
    for edge in mst_graph.edge_references() {
        let u = mst_graph[edge.source()];
        let v = mst_graph[edge.target()];
        adjacency[u].push(v);
        adjacency[v].push(u);
    }

    let mut visited = vec![false; posting_rows.len()];
    for &start in &non_empty {
        if visited[start] {
            continue;
        }
        let mut stack = vec![start];
        let mut component = Vec::new();
        visited[start] = true;
        while let Some(node) = stack.pop() {
            component.push(node);
            for &nbr in &adjacency[node] {
                if !visited[nbr] {
                    visited[nbr] = true;
                    stack.push(nbr);
                }
            }
        }

        let root = component
            .iter()
            .copied()
            .min_by_key(|&gene_idx| raw_costs[gene_idx])
            .unwrap_or(start);
        let mut orient_stack = vec![root];
        let mut oriented_seen = HashSet::<usize>::new();
        oriented_seen.insert(root);
        parents[root] = None;
        while let Some(node) = orient_stack.pop() {
            for &nbr in &adjacency[node] {
                if oriented_seen.insert(nbr) {
                    parents[nbr] = Some(node);
                    orient_stack.push(nbr);
                }
            }
        }
    }

    parents
}

fn quantile_pick_positions(n: usize, limit: usize) -> Vec<usize> {
    if n == 0 || limit == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }

    let mut out = Vec::with_capacity(limit.min(n));
    let mut seen = HashSet::new();
    let mut level = 0usize;
    while out.len() < limit.min(n) {
        let denom = 1usize << (level + 1);
        let entries = 1usize << level;
        for i in 0..entries {
            let numer = (2 * i) + 1;
            let pos = (((numer as f64) / (denom as f64)) * ((n - 1) as f64)).round() as usize;
            let pos = pos.min(n - 1);
            if seen.insert(pos) {
                out.push(pos);
                if out.len() >= limit.min(n) {
                    break;
                }
            }
        }
        level += 1;
        if level > 20 {
            break;
        }
    }

    if out.len() < limit.min(n) {
        for pos in 0..n {
            if seen.insert(pos) {
                out.push(pos);
                if out.len() >= limit.min(n) {
                    break;
                }
            }
        }
    }
    out
}

fn build_support_template_candidates(
    posting_rows: &[Vec<u32>],
    max_templates: usize,
) -> Vec<Vec<u32>> {
    if max_templates == 0 {
        return Vec::new();
    }

    let non_empty: Vec<usize> = posting_rows
        .iter()
        .enumerate()
        .filter_map(|(idx, rows)| (!rows.is_empty()).then_some(idx))
        .collect();
    if non_empty.is_empty() {
        return Vec::new();
    }

    let mut projected: Vec<(usize, f64, f64, usize)> = non_empty
        .iter()
        .map(|&gene_idx| {
            let mut s1 = 0.0f64;
            let mut s2 = 0.0f64;
            for &row in &posting_rows[gene_idx] {
                s1 += support_projection_weight(row, 0x916A_D5EF_C3A9_7B21);
                s2 += support_projection_weight(row, 0x2F7B_9A10_84DE_63C5);
            }
            (gene_idx, s1, s2, posting_rows[gene_idx].len())
        })
        .collect();
    projected.sort_unstable_by(|a, b| {
        a.1.total_cmp(&b.1)
            .then_with(|| a.2.total_cmp(&b.2))
            .then_with(|| a.3.cmp(&b.3))
            .then_with(|| a.0.cmp(&b.0))
    });

    let positions = quantile_pick_positions(projected.len(), max_templates.min(projected.len()));
    positions
        .into_iter()
        .map(|pos| posting_rows[projected[pos].0].clone())
        .collect()
}

fn build_row_template_candidates(
    expressions: &[SparseExpression],
    max_templates: usize,
) -> Vec<SparseExpression> {
    if max_templates == 0 || expressions.is_empty() {
        return Vec::new();
    }

    let mut scored: Vec<(usize, f64, f64, usize)> = Vec::with_capacity(expressions.len());
    for (row_idx, expr) in expressions.iter().enumerate() {
        if expr.is_empty() {
            continue;
        }
        let mut s1 = 0.0f64;
        let mut s2 = 0.0f64;
        for &(gene, value) in expr {
            let w = (value as f64).ln_1p();
            s1 += w * support_projection_weight(gene, 0x41A8_36D5_C20F_9871);
            s2 += w * support_projection_weight(gene, 0xD37E_6CB4_9A51_204F);
        }
        scored.push((row_idx, s1, s2, expr.len()));
    }
    if scored.is_empty() {
        return Vec::new();
    }
    scored.sort_unstable_by(|a, b| {
        a.1.total_cmp(&b.1)
            .then_with(|| a.2.total_cmp(&b.2))
            .then_with(|| a.3.cmp(&b.3))
            .then_with(|| a.0.cmp(&b.0))
    });

    let positions = quantile_pick_positions(scored.len(), max_templates.min(scored.len()));
    positions
        .into_iter()
        .map(|pos| expressions[scored[pos].0].clone())
        .collect()
}

#[derive(Default)]
struct EditOps {
    removes: Vec<u32>,
    adds: Vec<(u32, u16)>,
    updates: Vec<(u32, u16)>,
}

fn compute_edit_ops(parent: &SparseExpression, child: &SparseExpression) -> EditOps {
    let mut ops = EditOps::default();
    let mut i = 0usize;
    let mut j = 0usize;

    while i < parent.len() && j < child.len() {
        match parent[i].0.cmp(&child[j].0) {
            std::cmp::Ordering::Less => {
                ops.removes.push(parent[i].0);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                ops.adds.push(child[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                if parent[i].1 != child[j].1 {
                    ops.updates.push(child[j]);
                }
                i += 1;
                j += 1;
            }
        }
    }

    while i < parent.len() {
        ops.removes.push(parent[i].0);
        i += 1;
    }

    while j < child.len() {
        ops.adds.push(child[j]);
        j += 1;
    }

    ops
}

fn l0_binary_diff(a: &[(u32, u16)], b: &[(u32, u16)]) -> u32 {
    let mut count = 0u32;
    let mut i = 0;
    let mut j = 0;

    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Less => {
                count += 1;
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                count += 1;
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }

    count + (a.len() - i) as u32 + (b.len() - j) as u32
}

fn l2_squared_diff(a: &[(u32, u16)], b: &[(u32, u16)]) -> u64 {
    let mut sum = 0u64;
    let mut i = 0;
    let mut j = 0;

    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Less => {
                let v = a[i].1 as u64;
                sum += v * v;
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                let v = b[j].1 as u64;
                sum += v * v;
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                let lhs = a[i].1 as i64;
                let rhs = b[j].1 as i64;
                let diff = lhs - rhs;
                sum += (diff * diff) as u64;
                i += 1;
                j += 1;
            }
        }
    }

    while i < a.len() {
        let v = a[i].1 as u64;
        sum += v * v;
        i += 1;
    }
    while j < b.len() {
        let v = b[j].1 as u64;
        sum += v * v;
        j += 1;
    }

    sum
}

fn hamming_diff(a: &[(u32, u16)], b: &[(u32, u16)]) -> u32 {
    let mut count = 0u32;
    let mut i = 0;
    let mut j = 0;

    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Less => {
                count += 1;
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                count += 1;
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                if a[i].1 != b[j].1 {
                    count += 1;
                }
                i += 1;
                j += 1;
            }
        }
    }

    count + (a.len() - i) as u32 + (b.len() - j) as u32
}

fn binary_jaccard_distance(a: &[(u32, u16)], b: &[(u32, u16)]) -> f32 {
    let mut intersection = 0u32;
    let mut union = 0u32;
    let mut i = 0usize;
    let mut j = 0usize;

    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Less => {
                union += 1;
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                union += 1;
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                intersection += 1;
                union += 1;
                i += 1;
                j += 1;
            }
        }
    }

    union += (a.len() - i) as u32 + (b.len() - j) as u32;
    if union == 0 {
        return 0.0;
    }
    1.0 - (intersection as f32 / union as f32)
}

const JACCARD_WEIGHT_SCALE: f32 = 1_000_000.0;

fn pair_distance(metric: KnnDistanceMetric, a: &[(u32, u16)], b: &[(u32, u16)]) -> u64 {
    match metric {
        KnnDistanceMetric::L0 => l0_binary_diff(a, b) as u64,
        KnnDistanceMetric::L2 => l2_squared_diff(a, b),
        KnnDistanceMetric::Hamming => hamming_diff(a, b) as u64,
        KnnDistanceMetric::Jaccard => {
            (binary_jaccard_distance(a, b) * JACCARD_WEIGHT_SCALE).round() as u64
        }
    }
}

fn stream_vbyte_symbol_bytes(v: u32) -> u64 {
    if v <= 0xFF {
        1
    } else if v <= 0xFFFF {
        2
    } else if v <= 0x00FF_FFFF {
        3
    } else {
        4
    }
}

fn arithmetic_symbol_bytes(v: u32) -> u64 {
    // Cheap monotonic proxy for arithmetic-coded symbol cost.
    if v == 0 {
        1
    } else {
        let bits = (u32::BITS - v.leading_zeros()) as u64;
        bits.div_ceil(6).max(1)
    }
}

fn index_symbol_bytes(v: u32, index_codec: IndexStreamCodec) -> u64 {
    match index_codec {
        IndexStreamCodec::Arithmetic => arithmetic_symbol_bytes(v),
        IndexStreamCodec::StreamVByte => stream_vbyte_symbol_bytes(v),
    }
}

fn encoded_gene_only_cost_bytes(genes: &[u32], index_codec: IndexStreamCodec) -> u64 {
    if genes.is_empty() {
        return 0;
    }

    let mut cost = index_symbol_bytes(genes[0], index_codec);
    let mut prev_gene = genes[0];
    for &gene in genes.iter().skip(1) {
        cost += index_symbol_bytes(gene.saturating_sub(prev_gene), index_codec);
        prev_gene = gene;
    }
    cost
}

fn encoded_entries_cost_bytes(entries: &[(u32, u32)], index_codec: IndexStreamCodec) -> u64 {
    if entries.is_empty() {
        return 0;
    }

    let mut cost = index_symbol_bytes(entries[0].0, index_codec);
    let mut prev_gene = entries[0].0;
    cost += arithmetic_symbol_bytes(entries[0].1);

    for &(gene, value) in entries.iter().skip(1) {
        cost += index_symbol_bytes(gene.saturating_sub(prev_gene), index_codec);
        cost += arithmetic_symbol_bytes(value);
        prev_gene = gene;
    }
    cost
}

fn child_payload_cost_bytes(
    parent: &SparseExpression,
    child: &SparseExpression,
    index_codec: IndexStreamCodec,
    full_row_fallback_ratio: Option<f32>,
) -> u64 {
    let (delta_cost, ops) = row_delta_cost_and_ops(parent, child, index_codec);
    let total_edits = ops.removes.len() + ops.adds.len() + ops.updates.len();

    let use_full_row = match full_row_fallback_ratio {
        Some(ratio) => {
            if child.is_empty() {
                total_edits > 0
            } else {
                (total_edits as f32) > ratio * (child.len() as f32)
            }
        }
        None => false,
    };

    if use_full_row {
        full_row_cost_bytes(child, index_codec)
    } else {
        delta_cost
    }
}

fn full_row_cost_bytes(child: &SparseExpression, index_codec: IndexStreamCodec) -> u64 {
    let entries: Vec<(u32, u32)> = child
        .iter()
        .map(|&(gene, value)| (gene, value as u32))
        .collect();
    encoded_entries_cost_bytes(&entries, index_codec) + 2
}

fn row_delta_cost_and_ops(
    base: &SparseExpression,
    child: &SparseExpression,
    index_codec: IndexStreamCodec,
) -> (u64, EditOps) {
    let ops = compute_edit_ops(base, child);
    let remove_cost = encoded_gene_only_cost_bytes(&ops.removes, index_codec);
    let add_entries: Vec<(u32, u32)> = ops
        .adds
        .iter()
        .map(|&(gene, value)| (gene, value as u32))
        .collect();
    let update_entries: Vec<(u32, u32)> = ops
        .updates
        .iter()
        .map(|&(gene, value)| (gene, value as u32))
        .collect();
    (
        remove_cost
            + encoded_entries_cost_bytes(&add_entries, index_codec)
            + encoded_entries_cost_bytes(&update_entries, index_codec)
            + 4,
        ops,
    )
}

fn symmetric_edge_encoding_cost_bytes(
    a: &SparseExpression,
    b: &SparseExpression,
    index_codec: IndexStreamCodec,
    full_row_fallback_ratio: Option<f32>,
) -> u64 {
    let ab = child_payload_cost_bytes(a, b, index_codec, full_row_fallback_ratio);
    let ba = child_payload_cost_bytes(b, a, index_codec, full_row_fallback_ratio);
    ab.min(ba)
}

fn edge_weight(
    metric: KnnDistanceMetric,
    a: &SparseExpression,
    b: &SparseExpression,
    mst_weight_mode: MstWeightMode,
    index_codec: IndexStreamCodec,
    full_row_fallback_ratio: Option<f32>,
) -> u64 {
    match mst_weight_mode {
        MstWeightMode::Metric => pair_distance(metric, a, b),
        MstWeightMode::EncodingCost => {
            symmetric_edge_encoding_cost_bytes(a, b, index_codec, full_row_fallback_ratio)
        }
    }
}

fn build_mst_prim(
    expressions: &[SparseExpression],
    _k: usize,
    metric: KnnDistanceMetric,
    mst_weight_mode: MstWeightMode,
    index_codec: IndexStreamCodec,
    full_row_fallback_ratio: Option<f32>,
    forest_cut_factor: Option<f32>,
    row_mst_window: usize,
) -> (usize, Vec<u32>) {
    let n = expressions.len();
    if n == 0 {
        return (0, Vec::new());
    }
    if n == 1 {
        return (0, vec![0]);
    }

    let mut graph = UnGraph::<usize, u64>::new_undirected();
    let nodes: Vec<_> = (0..n).map(|i| graph.add_node(i)).collect();

    let window = row_mst_window.max(1).min(n.saturating_sub(1));
    for i in 0..n {
        let upper = (i + window + 1).min(n);
        for j in (i + 1)..upper {
            let weight = edge_weight(
                metric,
                &expressions[i],
                &expressions[j],
                mst_weight_mode,
                index_codec,
                full_row_fallback_ratio,
            );
            graph.update_edge(nodes[i], nodes[j], weight);
        }
    }

    if connected_components(&graph) > 1 {
        for i in 1..n {
            let dist = edge_weight(
                metric,
                &expressions[0],
                &expressions[i],
                mst_weight_mode,
                index_codec,
                full_row_fallback_ratio,
            );
            graph.update_edge(nodes[0], nodes[i], dist);
        }
    }

    let mst_graph = UnGraph::<usize, u64>::from_elements(min_spanning_tree(&graph));
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];

    let cut_threshold = if let Some(factor) = forest_cut_factor.filter(|f| *f > 0.0) {
        let mut weights: Vec<u64> = mst_graph.edge_references().map(|e| *e.weight()).collect();
        if weights.is_empty() {
            None
        } else {
            weights.sort_unstable();
            let median = weights[weights.len() / 2];
            Some(((median as f64) * (factor as f64)).ceil() as u64)
        }
    } else {
        None
    };

    for edge in mst_graph.edge_references() {
        let keep = cut_threshold
            .map(|threshold| *edge.weight() <= threshold)
            .unwrap_or(true);
        if !keep {
            continue;
        }
        let u = mst_graph[edge.source()];
        let v = mst_graph[edge.target()];
        adjacency[u].push(v);
        adjacency[v].push(u);
    }

    let mut parent = vec![u32::MAX; n];
    let mut visited = vec![false; n];
    let mut roots = Vec::new();

    for start in 0..n {
        if visited[start] {
            continue;
        }
        roots.push(start);
        parent[start] = start as u32;
        visited[start] = true;
        let mut stack = vec![start];
        while let Some(u) = stack.pop() {
            for &v in &adjacency[u] {
                if !visited[v] {
                    visited[v] = true;
                    parent[v] = u as u32;
                    stack.push(v);
                }
            }
        }
    }

    roots.sort_unstable();
    let primary_root = roots.first().copied().unwrap_or(0);
    (primary_root, parent)
}

fn compute_dfs_order(_root: usize, parent: &[u32], n: usize) -> (Vec<u32>, Vec<u32>) {
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut roots = Vec::new();
    for i in 0..n {
        if parent[i] as usize == i {
            roots.push(i);
        } else {
            children[parent[i] as usize].push(i);
        }
    }
    roots.sort_unstable();
    if roots.is_empty() {
        roots.push(0);
    }

    let mut dfs_order = Vec::with_capacity(n);
    let mut pos_in_dfs = vec![0u32; n];

    for &root in &roots {
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            pos_in_dfs[node] = dfs_order.len() as u32;
            dfs_order.push(node as u32);
            for &child in children[node].iter().rev() {
                stack.push(child);
            }
        }
    }

    let mut parent_offset = Vec::with_capacity(n);
    for (dfs_pos, &orig_cell) in dfs_order.iter().enumerate() {
        if parent[orig_cell as usize] as usize == orig_cell as usize {
            parent_offset.push(0);
        } else {
            let parent_orig = parent[orig_cell as usize] as usize;
            parent_offset.push((dfs_pos as u32) - pos_in_dfs[parent_orig]);
        }
    }

    (dfs_order, parent_offset)
}

fn append_gene_stream(genes: &[u32], first: &mut Vec<u32>, gaps: &mut Vec<u32>) {
    if let Some((&first_gene, rest)) = genes.split_first() {
        first.push(first_gene);
        let mut prev = first_gene;
        for &gene in rest {
            gaps.push(gene.saturating_sub(prev));
            prev = gene;
        }
    }
}

fn append_entry_stream(
    entries: &[(u32, u16)],
    first: &mut Vec<u32>,
    gaps: &mut Vec<u32>,
    vals: &mut Vec<u32>,
) {
    if let Some((&(first_gene, first_val), rest)) = entries.split_first() {
        first.push(first_gene);
        vals.push(first_val as u32);
        let mut prev = first_gene;
        for &(gene, value) in rest {
            gaps.push(gene.saturating_sub(prev));
            vals.push(value as u32);
            prev = gene;
        }
    }
}

pub fn encode_subarray_column(
    points: &[Point],
    data: &CsMat<u16>,
    gene_old_to_new: Option<&[u32]>,
    index_codec: IndexStreamCodec,
    sorted_index_codec: SortedIndexCodec,
    template_count: usize,
    template_adaptive: bool,
    template_max: usize,
    gene_ref_local_window: bool,
) -> Option<(EncodedColumnBlock, Vec<u32>)> {
    if points.is_empty() {
        return None;
    }

    let expressions_global: Vec<SparseExpression> = points
        .par_iter()
        .map(|p| compute_sparse_expression(p, data, gene_old_to_new))
        .collect();
    let (expressions, local_to_global_raw) = build_local_gene_remap(&expressions_global);
    let local_order = optimize_row_order_for_column(&expressions);
    let mut reordered = Vec::with_capacity(expressions.len());
    for &old_local in &local_order {
        reordered.push(expressions[old_local as usize].clone());
    }

    let mut postings: Vec<Vec<(u32, u16)>> = vec![Vec::new(); local_to_global_raw.len()];
    for (local_row, expr) in reordered.iter().enumerate() {
        let row_id = local_row as u32;
        for &(gene, value) in expr {
            postings[gene as usize].push((row_id, value));
        }
    }

    let local_to_global =
        EncodedSortedIndices::from_sorted_u32(&local_to_global_raw, sorted_index_codec);
    let mut posting_modes_raw = Vec::<u32>::with_capacity(postings.len());
    let mut posting_counts = Vec::<u32>::with_capacity(postings.len());
    let mut raw_row_firsts = Vec::<u32>::new();
    let mut raw_row_gaps = Vec::<u32>::new();
    let mut template_counts_raw = Vec::<u32>::new();
    let mut template_row_firsts = Vec::<u32>::new();
    let mut template_row_gaps = Vec::<u32>::new();
    let mut ref_parent_deltas = Vec::<u32>::new();
    let mut ref_remove_counts = Vec::<u32>::new();
    let mut ref_add_counts = Vec::<u32>::new();
    let mut ref_remove_firsts = Vec::<u32>::new();
    let mut ref_remove_gaps = Vec::<u32>::new();
    let mut ref_add_firsts = Vec::<u32>::new();
    let mut ref_add_gaps = Vec::<u32>::new();
    let mut template_ids = Vec::<u32>::new();
    let mut template_remove_counts = Vec::<u32>::new();
    let mut template_add_counts = Vec::<u32>::new();
    let mut template_remove_firsts = Vec::<u32>::new();
    let mut template_remove_gaps = Vec::<u32>::new();
    let mut template_add_firsts = Vec::<u32>::new();
    let mut template_add_gaps = Vec::<u32>::new();
    let mut vals_flat = Vec::<u32>::new();
    let mut vals_by_gene = Vec::<Vec<u32>>::with_capacity(postings.len());

    let posting_rows: Vec<Vec<u32>> = postings
        .iter()
        .map(|entries| entries.iter().map(|&(row, _)| row).collect())
        .collect();
    let raw_support_costs: Vec<u64> = posting_rows
        .iter()
        .map(|rows| encoded_gene_only_cost_bytes(rows, index_codec))
        .collect();
    let ref_parents = if gene_ref_local_window {
        build_gene_reference_parents_local_window(
            &posting_rows,
            &raw_support_costs,
            index_codec,
            12,
        )
    } else {
        build_gene_reference_parents(&posting_rows, &raw_support_costs, index_codec)
    };

    let requested_template_count = if template_adaptive {
        template_max.max(template_count)
    } else {
        template_count
    };
    let template_candidates =
        build_support_template_candidates(&posting_rows, requested_template_count);
    let selected_templates = if template_adaptive && !template_candidates.is_empty() {
        let mut base_costs = vec![0u64; posting_rows.len()];
        for gene_idx in 0..posting_rows.len() {
            let rows = &posting_rows[gene_idx];
            if rows.is_empty() {
                base_costs[gene_idx] = 0;
                continue;
            }
            let raw_cost = raw_support_costs[gene_idx];
            let ref_cost = ref_parents
                .get(gene_idx)
                .and_then(|p| *p)
                .map(|parent_idx| {
                    support_ref_cost_bytes(
                        parent_idx,
                        gene_idx,
                        &posting_rows[parent_idx],
                        rows,
                        index_codec,
                    )
                })
                .unwrap_or(u64::MAX);
            base_costs[gene_idx] = raw_cost.min(ref_cost);
        }

        let mut template_costs = vec![vec![0u64; posting_rows.len()]; template_candidates.len()];
        for (template_idx, template_rows) in template_candidates.iter().enumerate() {
            for gene_idx in 0..posting_rows.len() {
                let rows = &posting_rows[gene_idx];
                if rows.is_empty() {
                    template_costs[template_idx][gene_idx] = 0;
                    continue;
                }
                let (removes, adds) = compute_row_set_edits(template_rows, rows);
                template_costs[template_idx][gene_idx] =
                    index_symbol_bytes(template_idx as u32, index_codec)
                        + encoded_gene_only_cost_bytes(&removes, index_codec)
                        + encoded_gene_only_cost_bytes(&adds, index_codec)
                        + 2;
            }
        }

        let mut running_best = base_costs.clone();
        let mut running_sum: u64 = running_best.iter().copied().sum();
        let mut dict_sum = 0u64;
        let mut best_total = running_sum;
        let mut best_k = 0usize;

        for k in 1..=template_candidates.len() {
            dict_sum += encoded_gene_only_cost_bytes(&template_candidates[k - 1], index_codec);
            for gene_idx in 0..posting_rows.len() {
                let c = template_costs[k - 1][gene_idx];
                if c < running_best[gene_idx] {
                    running_sum = running_sum
                        .saturating_sub(running_best[gene_idx])
                        .saturating_add(c);
                    running_best[gene_idx] = c;
                }
            }
            let total = running_sum.saturating_add(dict_sum);
            if total < best_total {
                best_total = total;
                best_k = k;
            }
        }

        template_candidates
            .into_iter()
            .take(best_k)
            .collect::<Vec<_>>()
    } else {
        template_candidates
    };

    enum PostingDecision {
        Raw,
        Ref {
            parent_delta: u32,
            removes: Vec<u32>,
            adds: Vec<u32>,
        },
        Template {
            template_idx: usize,
            removes: Vec<u32>,
            adds: Vec<u32>,
        },
    }
    let mut decisions = Vec::with_capacity(postings.len());

    for (gene_idx, _) in postings.iter().enumerate() {
        let rows = &posting_rows[gene_idx];
        let raw_cost = raw_support_costs[gene_idx];

        if rows.is_empty() {
            decisions.push(PostingDecision::Raw);
            continue;
        }

        let mut best_cost = raw_cost as f64;
        let mut best_decision = PostingDecision::Raw;

        if let Some(parent_idx) = ref_parents.get(gene_idx).and_then(|p| *p) {
            let (removes, adds) = compute_row_set_edits(&posting_rows[parent_idx], rows);
            let parent_delta = zigzag_encode_i64((parent_idx as i64) - (gene_idx as i64));
            let ref_cost = index_symbol_bytes(parent_delta, index_codec)
                + encoded_gene_only_cost_bytes(&removes, index_codec)
                + encoded_gene_only_cost_bytes(&adds, index_codec)
                + 2;
            if (ref_cost as f64) < best_cost {
                best_cost = ref_cost as f64;
                best_decision = PostingDecision::Ref {
                    parent_delta,
                    removes,
                    adds,
                };
            }
        }

        if !selected_templates.is_empty() {
            for (template_idx, template_rows) in selected_templates.iter().enumerate() {
                let (removes, adds) = compute_row_set_edits(template_rows, rows);
                let template_cost = index_symbol_bytes(template_idx as u32, index_codec)
                    + encoded_gene_only_cost_bytes(&removes, index_codec)
                    + encoded_gene_only_cost_bytes(&adds, index_codec)
                    + 2;
                if (template_cost as f64) < best_cost {
                    best_cost = template_cost as f64;
                    best_decision = PostingDecision::Template {
                        template_idx,
                        removes,
                        adds,
                    };
                }
            }
        }

        decisions.push(best_decision);
    }

    for template_rows in &selected_templates {
        template_counts_raw.push(template_rows.len() as u32);
        append_gene_stream(
            template_rows,
            &mut template_row_firsts,
            &mut template_row_gaps,
        );
    }

    for (gene_idx, entries) in postings.iter().enumerate() {
        posting_counts.push(entries.len() as u32);
        let rows = &posting_rows[gene_idx];
        match decisions.get(gene_idx) {
            Some(PostingDecision::Ref {
                parent_delta,
                removes,
                adds,
            }) => {
                posting_modes_raw.push(EncodedColumnBlock::MODE_REF);
                ref_parent_deltas.push(*parent_delta);
                ref_remove_counts.push(removes.len() as u32);
                ref_add_counts.push(adds.len() as u32);
                append_gene_stream(removes, &mut ref_remove_firsts, &mut ref_remove_gaps);
                append_gene_stream(adds, &mut ref_add_firsts, &mut ref_add_gaps);
            }
            Some(PostingDecision::Template {
                template_idx,
                removes,
                adds,
            }) => {
                posting_modes_raw.push(EncodedColumnBlock::MODE_TEMPLATE);
                template_ids.push(*template_idx as u32);
                template_remove_counts.push(removes.len() as u32);
                template_add_counts.push(adds.len() as u32);
                append_gene_stream(
                    removes,
                    &mut template_remove_firsts,
                    &mut template_remove_gaps,
                );
                append_gene_stream(adds, &mut template_add_firsts, &mut template_add_gaps);
            }
            _ => {
                posting_modes_raw.push(EncodedColumnBlock::MODE_RAW);
                append_gene_stream(rows, &mut raw_row_firsts, &mut raw_row_gaps);
            }
        };

        let mut gene_vals = Vec::with_capacity(entries.len());
        for &(_, value) in entries {
            let v = value as u32;
            vals_flat.push(v);
            gene_vals.push(v);
        }
        vals_by_gene.push(gene_vals);
    }

    let posting_modes = if posting_modes_raw.is_empty() {
        ArithmeticEncoded::default()
    } else {
        ArithmeticEncoded::from_slice(&posting_modes_raw).expect("valid posting modes")
    };
    let posting_counts = if posting_counts.is_empty() {
        ArithmeticEncoded::default()
    } else {
        ArithmeticEncoded::from_slice(&posting_counts).expect("valid posting counts")
    };
    let raw_row_firsts = EncodedU32Stream::from_slice(&raw_row_firsts, index_codec);
    let raw_row_gaps = EncodedU32Stream::from_slice(&raw_row_gaps, index_codec);
    let template_counts = if template_counts_raw.is_empty() {
        ArithmeticEncoded::default()
    } else {
        ArithmeticEncoded::from_slice(&template_counts_raw).expect("valid template counts")
    };
    let template_row_firsts = EncodedU32Stream::from_slice(&template_row_firsts, index_codec);
    let template_row_gaps = EncodedU32Stream::from_slice(&template_row_gaps, index_codec);
    let ref_parent_deltas = EncodedU32Stream::from_slice(&ref_parent_deltas, index_codec);
    let ref_remove_counts = if ref_remove_counts.is_empty() {
        ArithmeticEncoded::default()
    } else {
        ArithmeticEncoded::from_slice(&ref_remove_counts).expect("valid ref remove counts")
    };
    let ref_add_counts = if ref_add_counts.is_empty() {
        ArithmeticEncoded::default()
    } else {
        ArithmeticEncoded::from_slice(&ref_add_counts).expect("valid ref add counts")
    };
    let ref_remove_firsts = EncodedU32Stream::from_slice(&ref_remove_firsts, index_codec);
    let ref_remove_gaps = EncodedU32Stream::from_slice(&ref_remove_gaps, index_codec);
    let ref_add_firsts = EncodedU32Stream::from_slice(&ref_add_firsts, index_codec);
    let ref_add_gaps = EncodedU32Stream::from_slice(&ref_add_gaps, index_codec);
    let template_ids = EncodedU32Stream::from_slice(&template_ids, index_codec);
    let template_remove_counts = if template_remove_counts.is_empty() {
        ArithmeticEncoded::default()
    } else {
        ArithmeticEncoded::from_slice(&template_remove_counts)
            .expect("valid template remove counts")
    };
    let template_add_counts = if template_add_counts.is_empty() {
        ArithmeticEncoded::default()
    } else {
        ArithmeticEncoded::from_slice(&template_add_counts).expect("valid template add counts")
    };
    let template_remove_firsts = EncodedU32Stream::from_slice(&template_remove_firsts, index_codec);
    let template_remove_gaps = EncodedU32Stream::from_slice(&template_remove_gaps, index_codec);
    let template_add_firsts = EncodedU32Stream::from_slice(&template_add_firsts, index_codec);
    let template_add_gaps = EncodedU32Stream::from_slice(&template_add_gaps, index_codec);
    let vals_global = ColumnValuesEncoded::from_flat(&vals_flat);
    let vals_per_gene = ColumnValuesEncoded::from_per_gene(&vals_by_gene);
    let vals = if vals_per_gene.size_in_bytes() < vals_global.size_in_bytes() {
        vals_per_gene
    } else {
        vals_global
    };

    Some((
        EncodedColumnBlock {
            num_cells: points.len() as u32,
            num_genes: data.cols() as u32,
            local_to_global,
            posting_modes,
            posting_counts,
            raw_row_firsts,
            raw_row_gaps,
            template_counts,
            template_row_firsts,
            template_row_gaps,
            ref_parent_deltas,
            ref_remove_counts,
            ref_add_counts,
            ref_remove_firsts,
            ref_remove_gaps,
            ref_add_firsts,
            ref_add_gaps,
            template_ids,
            template_remove_counts,
            template_add_counts,
            template_remove_firsts,
            template_remove_gaps,
            template_add_firsts,
            template_add_gaps,
            vals,
        },
        local_order,
    ))
}

pub fn encode_subarray_mst_with_metric(
    points: &[Point],
    data: &CsMat<u16>,
    knn_metric: KnnDistanceMetric,
    mst_weight_mode: MstWeightMode,
    gene_old_to_new: Option<&[u32]>,
    index_codec: IndexStreamCodec,
    sorted_index_codec: SortedIndexCodec,
    full_row_fallback_ratio: Option<f32>,
    forest_cut_factor: Option<f32>,
    row_mst_window: usize,
    row_template_adaptive: bool,
    row_template_max: usize,
) -> Option<(EncodedDiffsMST, Vec<u32>)> {
    if points.is_empty() {
        return None;
    }
    let expressions_global: Vec<SparseExpression> = points
        .par_iter()
        .map(|p| compute_sparse_expression(p, data, gene_old_to_new))
        .collect();
    let (expressions, local_to_global_raw) = build_local_gene_remap(&expressions_global);
    let k = 8usize.min(points.len().saturating_sub(1)).max(1);
    let (root, parent) = build_mst_prim(
        &expressions,
        k,
        knn_metric,
        mst_weight_mode,
        index_codec,
        full_row_fallback_ratio,
        forest_cut_factor,
        row_mst_window,
    );
    encode_subarray_mst_from_parts(
        data.cols() as u32,
        &expressions,
        &local_to_global_raw,
        root,
        &parent,
        sorted_index_codec,
        index_codec,
        full_row_fallback_ratio,
        row_template_adaptive,
        row_template_max,
    )
}

fn encode_subarray_mst_from_parts(
    num_genes: u32,
    expressions: &[SparseExpression],
    local_to_global_raw: &[u32],
    root: usize,
    parent: &[u32],
    sorted_index_codec: SortedIndexCodec,
    index_codec: IndexStreamCodec,
    full_row_fallback_ratio: Option<f32>,
    row_template_adaptive: bool,
    row_template_max: usize,
) -> Option<(EncodedDiffsMST, Vec<u32>)> {
    let local_to_global =
        EncodedSortedIndices::from_sorted_u32(local_to_global_raw, sorted_index_codec);
    let (dfs_order, parent_offset_raw) = compute_dfs_order(root, parent, expressions.len());
    let root_expr = &expressions[root];
    let root_genes: Vec<u32> = root_expr.iter().map(|(g, _)| *g).collect();
    let root_vals_raw: Vec<u32> = root_expr.iter().map(|(_, v)| *v as u32).collect();

    const ROW_MODE_PARENT: u32 = 0;
    const ROW_MODE_FULL: u32 = 1;
    const ROW_MODE_TEMPLATE: u32 = 2;

    let mut child_modes_raw = Vec::<u32>::with_capacity(expressions.len().saturating_sub(1));
    let mut child_full_counts_raw = Vec::<u32>::with_capacity(expressions.len().saturating_sub(1));
    let mut child_remove_counts_raw =
        Vec::<u32>::with_capacity(expressions.len().saturating_sub(1));
    let mut child_add_counts_raw = Vec::<u32>::with_capacity(expressions.len().saturating_sub(1));
    let mut child_update_counts_raw =
        Vec::<u32>::with_capacity(expressions.len().saturating_sub(1));
    let mut row_template_counts_raw = Vec::<u32>::new();

    let mut row_template_first_genes_raw = Vec::<u32>::new();
    let mut row_template_gene_gaps_raw = Vec::<u32>::new();
    let mut child_full_first_genes_raw = Vec::<u32>::new();
    let mut child_full_gene_gaps_raw = Vec::<u32>::new();
    let mut child_template_ids_raw = Vec::<u32>::new();
    let mut child_remove_first_genes_raw = Vec::<u32>::new();
    let mut child_remove_gene_gaps_raw = Vec::<u32>::new();
    let mut child_add_first_genes_raw = Vec::<u32>::new();
    let mut child_add_gene_gaps_raw = Vec::<u32>::new();
    let mut child_update_first_genes_raw = Vec::<u32>::new();
    let mut child_update_gene_gaps_raw = Vec::<u32>::new();

    let mut row_template_vals_raw = Vec::<u32>::new();
    let mut child_full_vals_raw = Vec::<u32>::new();
    let mut child_add_vals_raw = Vec::<u32>::new();
    let mut child_update_vals_raw = Vec::<u32>::new();

    let child_origs: Vec<usize> = dfs_order.iter().skip(1).map(|&v| v as usize).collect();
    let selected_row_templates = if row_template_adaptive && row_template_max > 0 {
        let template_candidates = build_row_template_candidates(&expressions, row_template_max);
        if template_candidates.is_empty() {
            Vec::new()
        } else {
            let mut baseline_costs = Vec::with_capacity(child_origs.len());
            for &orig_cell in &child_origs {
                let parent_orig = parent[orig_cell] as usize;
                let child_expr = &expressions[orig_cell];
                let parent_expr = &expressions[parent_orig];
                let (parent_delta_cost, parent_ops) =
                    row_delta_cost_and_ops(parent_expr, child_expr, index_codec);
                let edit_count =
                    parent_ops.removes.len() + parent_ops.adds.len() + parent_ops.updates.len();
                let use_full_row = match full_row_fallback_ratio {
                    Some(ratio) => {
                        if child_expr.is_empty() {
                            edit_count > 0
                        } else {
                            (edit_count as f32) > ratio * (child_expr.len() as f32)
                        }
                    }
                    None => false,
                };
                let baseline = if use_full_row {
                    full_row_cost_bytes(child_expr, index_codec)
                } else {
                    parent_delta_cost
                };
                baseline_costs.push(baseline);
            }

            let mut template_costs = vec![vec![0u64; child_origs.len()]; template_candidates.len()];
            for (template_idx, template_expr) in template_candidates.iter().enumerate() {
                for (child_idx, &orig_cell) in child_origs.iter().enumerate() {
                    let child_expr = &expressions[orig_cell];
                    let (delta_cost, _) =
                        row_delta_cost_and_ops(template_expr, child_expr, index_codec);
                    template_costs[template_idx][child_idx] =
                        delta_cost + index_symbol_bytes(template_idx as u32, index_codec);
                }
            }

            let mut running_best = baseline_costs.clone();
            let mut running_sum: u64 = running_best.iter().copied().sum();
            let mut dict_sum = 0u64;
            let mut best_total = running_sum;
            let mut best_k = 0usize;

            for k in 1..=template_candidates.len() {
                let template_entries: Vec<(u32, u32)> = template_candidates[k - 1]
                    .iter()
                    .map(|&(gene, value)| (gene, value as u32))
                    .collect();
                dict_sum = dict_sum
                    .saturating_add(encoded_entries_cost_bytes(&template_entries, index_codec) + 1);
                for child_idx in 0..child_origs.len() {
                    let c = template_costs[k - 1][child_idx];
                    if c < running_best[child_idx] {
                        running_sum = running_sum
                            .saturating_sub(running_best[child_idx])
                            .saturating_add(c);
                        running_best[child_idx] = c;
                    }
                }
                let total = running_sum.saturating_add(dict_sum);
                if total < best_total {
                    best_total = total;
                    best_k = k;
                }
            }

            template_candidates
                .into_iter()
                .take(best_k)
                .collect::<Vec<_>>()
        }
    } else {
        Vec::new()
    };

    for template_expr in &selected_row_templates {
        row_template_counts_raw.push(template_expr.len() as u32);
        append_entry_stream(
            template_expr,
            &mut row_template_first_genes_raw,
            &mut row_template_gene_gaps_raw,
            &mut row_template_vals_raw,
        );
    }

    for &orig_cell in &child_origs {
        let parent_orig = parent[orig_cell] as usize;
        let child_expr = &expressions[orig_cell];
        let parent_expr = &expressions[parent_orig];
        let (parent_delta_cost, parent_ops) =
            row_delta_cost_and_ops(parent_expr, child_expr, index_codec);
        let edit_count =
            parent_ops.removes.len() + parent_ops.adds.len() + parent_ops.updates.len();

        let use_full_row = match full_row_fallback_ratio {
            Some(ratio) => {
                if child_expr.is_empty() {
                    edit_count > 0
                } else {
                    (edit_count as f32) > ratio * (child_expr.len() as f32)
                }
            }
            None => false,
        };

        let mut best_mode = if use_full_row {
            ROW_MODE_FULL
        } else {
            ROW_MODE_PARENT
        };
        let mut best_template_id: Option<usize> = None;
        let mut best_ops = parent_ops;
        let mut best_cost = if use_full_row {
            full_row_cost_bytes(child_expr, index_codec)
        } else {
            parent_delta_cost
        };

        for (template_idx, template_expr) in selected_row_templates.iter().enumerate() {
            let (template_cost, template_ops) =
                row_delta_cost_and_ops(template_expr, child_expr, index_codec);
            let total_template_cost =
                template_cost + index_symbol_bytes(template_idx as u32, index_codec);
            if total_template_cost < best_cost {
                best_cost = total_template_cost;
                best_mode = ROW_MODE_TEMPLATE;
                best_template_id = Some(template_idx);
                best_ops = template_ops;
            }
        }

        if best_mode == ROW_MODE_FULL {
            child_modes_raw.push(ROW_MODE_FULL);
            child_full_counts_raw.push(child_expr.len() as u32);
            child_remove_counts_raw.push(0);
            child_add_counts_raw.push(0);
            child_update_counts_raw.push(0);
            append_entry_stream(
                child_expr,
                &mut child_full_first_genes_raw,
                &mut child_full_gene_gaps_raw,
                &mut child_full_vals_raw,
            );
        } else {
            child_modes_raw.push(best_mode);
            child_full_counts_raw.push(0);
            child_remove_counts_raw.push(best_ops.removes.len() as u32);
            child_add_counts_raw.push(best_ops.adds.len() as u32);
            child_update_counts_raw.push(best_ops.updates.len() as u32);

            if let Some(template_id) = best_template_id {
                child_template_ids_raw.push(template_id as u32);
            }

            append_gene_stream(
                &best_ops.removes,
                &mut child_remove_first_genes_raw,
                &mut child_remove_gene_gaps_raw,
            );
            append_entry_stream(
                &best_ops.adds,
                &mut child_add_first_genes_raw,
                &mut child_add_gene_gaps_raw,
                &mut child_add_vals_raw,
            );
            append_entry_stream(
                &best_ops.updates,
                &mut child_update_first_genes_raw,
                &mut child_update_gene_gaps_raw,
                &mut child_update_vals_raw,
            );
        }
    }

    let parent_offset =
        ArithmeticEncoded::from_slice(&parent_offset_raw).expect("valid parent offset");
    let root_indices = EncodedSortedIndices::from_sorted_u32(&root_genes, sorted_index_codec);
    let root_vals = if root_vals_raw.is_empty() {
        ArithmeticEncoded::default()
    } else {
        ArithmeticEncoded::from_slice(&root_vals_raw).expect("valid root vals")
    };
    let child_modes = ArithmeticEncoded::from_slice(&child_modes_raw).expect("valid child modes");
    let child_full_counts =
        ArithmeticEncoded::from_slice(&child_full_counts_raw).expect("valid full counts");
    let child_remove_counts =
        ArithmeticEncoded::from_slice(&child_remove_counts_raw).expect("valid remove counts");
    let child_add_counts =
        ArithmeticEncoded::from_slice(&child_add_counts_raw).expect("valid add counts");
    let child_update_counts =
        ArithmeticEncoded::from_slice(&child_update_counts_raw).expect("valid update counts");
    let row_template_counts = if row_template_counts_raw.is_empty() {
        ArithmeticEncoded::default()
    } else {
        ArithmeticEncoded::from_slice(&row_template_counts_raw).expect("valid template counts")
    };

    let row_template_first_genes =
        EncodedU32Stream::from_slice(&row_template_first_genes_raw, index_codec);
    let row_template_gene_gaps =
        EncodedU32Stream::from_slice(&row_template_gene_gaps_raw, index_codec);
    let child_full_first_genes =
        EncodedU32Stream::from_slice(&child_full_first_genes_raw, index_codec);
    let child_full_gene_gaps = EncodedU32Stream::from_slice(&child_full_gene_gaps_raw, index_codec);
    let child_template_ids = EncodedU32Stream::from_slice(&child_template_ids_raw, index_codec);
    let child_remove_first_genes =
        EncodedU32Stream::from_slice(&child_remove_first_genes_raw, index_codec);
    let child_remove_gene_gaps =
        EncodedU32Stream::from_slice(&child_remove_gene_gaps_raw, index_codec);
    let child_add_first_genes =
        EncodedU32Stream::from_slice(&child_add_first_genes_raw, index_codec);
    let child_add_gene_gaps = EncodedU32Stream::from_slice(&child_add_gene_gaps_raw, index_codec);
    let child_update_first_genes =
        EncodedU32Stream::from_slice(&child_update_first_genes_raw, index_codec);
    let child_update_gene_gaps =
        EncodedU32Stream::from_slice(&child_update_gene_gaps_raw, index_codec);

    let row_template_vals = if row_template_vals_raw.is_empty() {
        ArithmeticEncoded::default()
    } else {
        ArithmeticEncoded::from_slice(&row_template_vals_raw).expect("valid template values")
    };
    let child_full_vals = if child_full_vals_raw.is_empty() {
        ArithmeticEncoded::default()
    } else {
        ArithmeticEncoded::from_slice(&child_full_vals_raw).expect("valid full values")
    };
    let child_add_vals = if child_add_vals_raw.is_empty() {
        ArithmeticEncoded::default()
    } else {
        ArithmeticEncoded::from_slice(&child_add_vals_raw).expect("valid add values")
    };
    let child_update_vals = if child_update_vals_raw.is_empty() {
        ArithmeticEncoded::default()
    } else {
        ArithmeticEncoded::from_slice(&child_update_vals_raw).expect("valid update values")
    };

    Some((
        EncodedDiffsMST {
            num_genes,
            parent_offset,
            local_to_global,
            root_indices,
            root_vals,
            child_modes,
            child_full_counts,
            child_remove_counts,
            child_add_counts,
            child_update_counts,
            row_template_counts,
            row_template_first_genes,
            row_template_gene_gaps,
            row_template_vals,
            child_template_ids,
            child_full_first_genes,
            child_full_gene_gaps,
            child_remove_first_genes,
            child_remove_gene_gaps,
            child_add_first_genes,
            child_add_gene_gaps,
            child_update_first_genes,
            child_update_gene_gaps,
            child_full_vals,
            child_add_vals,
            child_update_vals,
        },
        dfs_order,
    ))
}

#[allow(dead_code)]
pub fn encode_subarray_mst(
    points: &[Point],
    data: &CsMat<u16>,
) -> Option<(EncodedDiffsMST, Vec<u32>)> {
    encode_subarray_mst_with_metric(
        points,
        data,
        KnnDistanceMetric::L0,
        MstWeightMode::Metric,
        None,
        IndexStreamCodec::Arithmetic,
        SortedIndexCodec::EliasFano,
        Some(1.0),
        None,
        8,
        false,
        0,
    )
}

pub struct ExpressionVecIterMST<'a> {
    sparse_iter: SparseExpressionIterMST<'a>,
    num_genes: usize,
}

impl<'a> ExpressionVecIterMST<'a> {
    fn new(ediff: &'a EncodedDiffsMST) -> Self {
        Self {
            sparse_iter: SparseExpressionIterMST::new(ediff),
            num_genes: ediff.num_genes as usize,
        }
    }
}

impl Iterator for ExpressionVecIterMST<'_> {
    type Item = Vec<u16>;

    fn next(&mut self) -> Option<Self::Item> {
        let sparse = self.sparse_iter.next()?;
        let mut dense = vec![0u16; self.num_genes];
        for (gene_idx, value) in sparse {
            dense[gene_idx as usize] = value;
        }
        Some(dense)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.sparse_iter.size_hint()
    }
}

impl ExactSizeIterator for ExpressionVecIterMST<'_> {}

pub struct SparseExpressionIterMST<'a> {
    encoded: &'a EncodedDiffsMST,
    next_dfs: usize,
    local_to_global: Vec<u32>,
    states: Vec<Vec<(u32, i32)>>,
    child_modes: Vec<u32>,
    row_templates: Vec<Vec<(u32, i32)>>,
    child_template_base: Vec<Option<usize>>,
    child_full_entries: Vec<Vec<(u32, u32)>>,
    child_removes: Vec<Vec<u32>>,
    child_adds: Vec<Vec<(u32, u32)>>,
    child_updates: Vec<Vec<(u32, u32)>>,
}

impl<'a> SparseExpressionIterMST<'a> {
    fn new(encoded: &'a EncodedDiffsMST) -> Self {
        let ncells = encoded.num_cells();
        let local_to_global = encoded.local_to_global.decode_all_u32();
        let mut states = Vec::with_capacity(ncells);
        let mut child_modes = vec![0u32; ncells];
        let mut child_template_base = vec![None; ncells];
        let mut child_full_entries = vec![Vec::new(); ncells];
        let mut child_removes = vec![Vec::new(); ncells];
        let mut child_adds = vec![Vec::new(); ncells];
        let mut child_updates = vec![Vec::new(); ncells];
        let mut row_templates: Vec<Vec<(u32, i32)>> = Vec::new();

        const ROW_MODE_PARENT: u32 = 0;
        const ROW_MODE_FULL: u32 = 1;
        const ROW_MODE_TEMPLATE: u32 = 2;

        if ncells > 0 {
            let mut root_state = Vec::new();
            let root_indices = encoded.root_indices.decode_all_u32();
            let root_vals = encoded.root_vals.decode_all().unwrap_or_default();
            for (i, &gene) in root_indices.iter().enumerate() {
                let value = root_vals.get(i).copied().unwrap_or(0) as i32;
                root_state.push((gene, value));
            }
            states.push(root_state);
        }

        if ncells > 1 {
            let modes = encoded.child_modes.decode_all().unwrap_or_default();
            let full_counts = encoded.child_full_counts.decode_all().unwrap_or_default();
            let remove_counts = encoded.child_remove_counts.decode_all().unwrap_or_default();
            let add_counts = encoded.child_add_counts.decode_all().unwrap_or_default();
            let update_counts = encoded.child_update_counts.decode_all().unwrap_or_default();
            let template_counts = encoded.row_template_counts.decode_all().unwrap_or_default();

            let template_first_genes = encoded.row_template_first_genes.decode_all();
            let template_gaps = encoded.row_template_gene_gaps.decode_all();
            let full_first_genes = encoded.child_full_first_genes.decode_all();
            let full_gaps = encoded.child_full_gene_gaps.decode_all();
            let template_ids = encoded.child_template_ids.decode_all();
            let remove_first_genes = encoded.child_remove_first_genes.decode_all();
            let remove_gaps = encoded.child_remove_gene_gaps.decode_all();
            let add_first_genes = encoded.child_add_first_genes.decode_all();
            let add_gaps = encoded.child_add_gene_gaps.decode_all();
            let update_first_genes = encoded.child_update_first_genes.decode_all();
            let update_gaps = encoded.child_update_gene_gaps.decode_all();

            let template_vals = encoded.row_template_vals.decode_all().unwrap_or_default();
            let full_vals = encoded.child_full_vals.decode_all().unwrap_or_default();
            let add_vals = encoded.child_add_vals.decode_all().unwrap_or_default();
            let update_vals = encoded.child_update_vals.decode_all().unwrap_or_default();

            let mut template_first_cursor = 0usize;
            let mut template_gap_cursor = 0usize;
            let mut template_val_cursor = 0usize;
            let mut full_first_cursor = 0usize;
            let mut full_gap_cursor = 0usize;
            let mut full_val_cursor = 0usize;
            let mut template_id_cursor = 0usize;
            let mut remove_first_cursor = 0usize;
            let mut remove_gap_cursor = 0usize;
            let mut add_first_cursor = 0usize;
            let mut add_gap_cursor = 0usize;
            let mut add_val_cursor = 0usize;
            let mut update_first_cursor = 0usize;
            let mut update_gap_cursor = 0usize;
            let mut update_val_cursor = 0usize;

            for &count_raw in &template_counts {
                let count = count_raw as usize;
                let mut entries = Vec::with_capacity(count);
                if count > 0 {
                    let mut gene = template_first_genes
                        .get(template_first_cursor)
                        .copied()
                        .unwrap_or(0);
                    template_first_cursor += 1;
                    let first_val = template_vals.get(template_val_cursor).copied().unwrap_or(0);
                    template_val_cursor += 1;
                    entries.push((gene, first_val as i32));
                    for _ in 1..count {
                        let gap = template_gaps.get(template_gap_cursor).copied().unwrap_or(0);
                        template_gap_cursor += 1;
                        gene = gene.saturating_add(gap);
                        let value = template_vals.get(template_val_cursor).copied().unwrap_or(0);
                        template_val_cursor += 1;
                        entries.push((gene, value as i32));
                    }
                }
                row_templates.push(entries);
            }

            for dfs_pos in 1..ncells {
                let mode = modes.get(dfs_pos - 1).copied().unwrap_or(ROW_MODE_PARENT);
                child_modes[dfs_pos] = mode;

                if mode == ROW_MODE_FULL {
                    let count = full_counts.get(dfs_pos - 1).copied().unwrap_or(0) as usize;
                    if count > 0 {
                        let mut entries = Vec::with_capacity(count);
                        let mut gene = full_first_genes
                            .get(full_first_cursor)
                            .copied()
                            .unwrap_or(0);
                        full_first_cursor += 1;
                        let first_val = full_vals.get(full_val_cursor).copied().unwrap_or(0);
                        full_val_cursor += 1;
                        entries.push((gene, first_val));
                        for _ in 1..count {
                            let gap = full_gaps.get(full_gap_cursor).copied().unwrap_or(0);
                            full_gap_cursor += 1;
                            gene = gene.saturating_add(gap);
                            let value = full_vals.get(full_val_cursor).copied().unwrap_or(0);
                            full_val_cursor += 1;
                            entries.push((gene, value));
                        }
                        child_full_entries[dfs_pos] = entries;
                    }
                    continue;
                }

                if mode == ROW_MODE_TEMPLATE {
                    let template_idx =
                        template_ids.get(template_id_cursor).copied().unwrap_or(0) as usize;
                    template_id_cursor += 1;
                    if template_idx < row_templates.len() {
                        child_template_base[dfs_pos] = Some(template_idx);
                    }
                }

                let remove_count = remove_counts.get(dfs_pos - 1).copied().unwrap_or(0) as usize;
                if remove_count > 0 {
                    let mut genes = Vec::with_capacity(remove_count);
                    let mut gene = remove_first_genes
                        .get(remove_first_cursor)
                        .copied()
                        .unwrap_or(0);
                    remove_first_cursor += 1;
                    genes.push(gene);
                    for _ in 1..remove_count {
                        let gap = remove_gaps.get(remove_gap_cursor).copied().unwrap_or(0);
                        remove_gap_cursor += 1;
                        gene = gene.saturating_add(gap);
                        genes.push(gene);
                    }
                    child_removes[dfs_pos] = genes;
                }

                let add_count = add_counts.get(dfs_pos - 1).copied().unwrap_or(0) as usize;
                if add_count > 0 {
                    let mut entries = Vec::with_capacity(add_count);
                    let mut gene = add_first_genes.get(add_first_cursor).copied().unwrap_or(0);
                    add_first_cursor += 1;
                    let first_val = add_vals.get(add_val_cursor).copied().unwrap_or(0);
                    add_val_cursor += 1;
                    entries.push((gene, first_val));
                    for _ in 1..add_count {
                        let gap = add_gaps.get(add_gap_cursor).copied().unwrap_or(0);
                        add_gap_cursor += 1;
                        gene = gene.saturating_add(gap);
                        let value = add_vals.get(add_val_cursor).copied().unwrap_or(0);
                        add_val_cursor += 1;
                        entries.push((gene, value));
                    }
                    child_adds[dfs_pos] = entries;
                }

                let update_count = update_counts.get(dfs_pos - 1).copied().unwrap_or(0) as usize;
                if update_count > 0 {
                    let mut entries = Vec::with_capacity(update_count);
                    let mut gene = update_first_genes
                        .get(update_first_cursor)
                        .copied()
                        .unwrap_or(0);
                    update_first_cursor += 1;
                    let first_val = update_vals.get(update_val_cursor).copied().unwrap_or(0);
                    update_val_cursor += 1;
                    entries.push((gene, first_val));
                    for _ in 1..update_count {
                        let gap = update_gaps.get(update_gap_cursor).copied().unwrap_or(0);
                        update_gap_cursor += 1;
                        gene = gene.saturating_add(gap);
                        let value = update_vals.get(update_val_cursor).copied().unwrap_or(0);
                        update_val_cursor += 1;
                        entries.push((gene, value));
                    }
                    child_updates[dfs_pos] = entries;
                }
            }
        }

        Self {
            encoded,
            next_dfs: 0,
            local_to_global,
            states,
            child_modes,
            row_templates,
            child_template_base,
            child_full_entries,
            child_removes,
            child_adds,
            child_updates,
        }
    }
}

impl Iterator for SparseExpressionIterMST<'_> {
    type Item = Vec<(u32, u16)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_dfs >= self.encoded.num_cells() {
            return None;
        }

        let current_dfs = self.next_dfs;
        self.next_dfs += 1;

        if current_dfs > 0 {
            let parent_dfs = self.encoded.parent_dfs_pos(current_dfs);
            let mode = self.child_modes[current_dfs];
            let current_state = if mode == 1 {
                self.child_full_entries[current_dfs]
                    .iter()
                    .filter_map(|(gene, value)| {
                        if *value > 0 {
                            Some((*gene, *value as i32))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            } else {
                let base_state = if mode == 2 {
                    self.child_template_base[current_dfs]
                        .and_then(|idx| self.row_templates.get(idx))
                        .unwrap_or(&self.states[parent_dfs])
                } else {
                    &self.states[parent_dfs]
                };
                EncodedDiffsMST::apply_edit_ops(
                    base_state,
                    &self.child_removes[current_dfs],
                    &self.child_adds[current_dfs],
                    &self.child_updates[current_dfs],
                )
            };
            self.states.push(current_state);
        }

        let sparse = self.states[current_dfs]
            .iter()
            .filter_map(|(gene, value)| {
                if *value > 0 {
                    let mapped_gene = self
                        .local_to_global
                        .get(*gene as usize)
                        .copied()
                        .unwrap_or(*gene);
                    Some((mapped_gene, *value as u16))
                } else {
                    None
                }
            })
            .collect();

        Some(sparse)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.encoded.num_cells().saturating_sub(self.next_dfs);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for SparseExpressionIterMST<'_> {}

#[cfg(test)]
mod tests {
    use super::*;
    use sprs::TriMatI;

    #[test]
    fn test_mst_round_trip_sparse_iter() {
        let mut tri = TriMatI::<u16, usize>::new((3, 6));
        tri.add_triplet(0, 0, 4);
        tri.add_triplet(0, 2, 1);
        tri.add_triplet(1, 0, 5);
        tri.add_triplet(1, 3, 2);
        tri.add_triplet(2, 2, 1);
        tri.add_triplet(2, 4, 7);
        let csr = tri.to_csr::<usize>();

        let points = vec![
            Point::new(0.0, 0.0, 0),
            Point::new(1.0, 0.0, 1),
            Point::new(2.0, 0.0, 2),
        ];

        let (encoded, dfs_order) = encode_subarray_mst(&points, &csr).expect("must encode");
        let mut decoded_rows = Vec::new();
        for sparse in encoded.sparse_expression_iter() {
            decoded_rows.push(sparse);
        }

        for (dfs_pos, row_sparse) in decoded_rows.into_iter().enumerate() {
            let row_idx = dfs_order[dfs_pos] as usize;
            let mut dense = vec![0u16; csr.cols()];
            for (g, v) in row_sparse {
                dense[g as usize] = v;
            }

            let mut expected = vec![0u16; csr.cols()];
            for (g, &v) in csr.outer_view(row_idx).expect("row").iter() {
                expected[g] = v;
            }
            assert_eq!(dense, expected);
        }
    }

    #[test]
    fn test_column_template_round_trip_sparse() {
        let mut tri = TriMatI::<u16, usize>::new((6, 8));
        tri.add_triplet(0, 0, 2);
        tri.add_triplet(0, 2, 1);
        tri.add_triplet(1, 0, 3);
        tri.add_triplet(1, 2, 1);
        tri.add_triplet(1, 4, 1);
        tri.add_triplet(2, 1, 2);
        tri.add_triplet(2, 3, 1);
        tri.add_triplet(3, 1, 2);
        tri.add_triplet(3, 3, 1);
        tri.add_triplet(3, 5, 1);
        tri.add_triplet(4, 0, 1);
        tri.add_triplet(4, 2, 1);
        tri.add_triplet(4, 4, 1);
        tri.add_triplet(5, 1, 1);
        tri.add_triplet(5, 3, 1);
        tri.add_triplet(5, 5, 1);
        let csr = tri.to_csr::<usize>();

        let points = vec![
            Point::new(0.0, 0.0, 0),
            Point::new(1.0, 0.0, 1),
            Point::new(2.0, 0.0, 2),
            Point::new(3.0, 0.0, 3),
            Point::new(4.0, 0.0, 4),
            Point::new(5.0, 0.0, 5),
        ];

        let (encoded, local_order) = encode_subarray_column(
            &points,
            &csr,
            None,
            IndexStreamCodec::Arithmetic,
            SortedIndexCodec::Delta,
            3,
            false,
            32,
        )
        .expect("column encoding should succeed");
        let decoded_rows = encoded.decode_rows();
        assert_eq!(decoded_rows.len(), points.len());

        for (local_row, sparse_row) in decoded_rows.iter().enumerate() {
            let orig_row = local_order[local_row] as usize;
            let mut expected = vec![0u16; csr.cols()];
            for (g, &v) in csr.outer_view(orig_row).expect("csr row").iter() {
                expected[g] = v;
            }
            let mut actual = vec![0u16; csr.cols()];
            for &(g, v) in sparse_row {
                actual[g as usize] = v;
            }
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_compute_sparse_expression_skips_genes_outside_block() {
        let mut tri = TriMatI::<u16, usize>::new((1, 4));
        tri.add_triplet(0, 0, 3);
        tri.add_triplet(0, 1, 5);
        tri.add_triplet(0, 2, 7);
        tri.add_triplet(0, 3, 11);
        let csr = tri.to_csr::<usize>();
        let point = Point::new(0.0, 0.0, 0);

        // Keep genes 1 and 3 in this block; genes 0 and 2 are outside.
        let block_map = vec![u32::MAX, 0, u32::MAX, 1];
        let sparse = compute_sparse_expression(&point, &csr, Some(&block_map));

        assert_eq!(sparse, vec![(0, 5), (1, 11)]);
    }
}
