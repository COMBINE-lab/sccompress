use crate::arith_encode::ArithmeticEncoded;
use crate::delta_indices::DeltaEncodedIndices;
use crate::index_stream::{EncodedU32Stream, IndexStreamCodec};
use bincode::{Decode, Encode};
use hnsw_rs::prelude::*;
use petgraph::algo::{connected_components, min_spanning_tree};
use petgraph::data::FromElements;
use petgraph::graph::UnGraph;
use petgraph::visit::EdgeRef;
use rayon::prelude::*;
use sprs::CsMat;
use std::collections::BTreeMap;

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
    pub root_indices: DeltaEncodedIndices,
    pub root_vals: ArithmeticEncoded,
    pub child_full_row_flags: ArithmeticEncoded,
    pub child_full_counts: ArithmeticEncoded,
    pub child_remove_counts: ArithmeticEncoded,
    pub child_add_counts: ArithmeticEncoded,
    pub child_update_counts: ArithmeticEncoded,
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
            self.root_indices.size_in_bytes(),
            self.root_vals.size_in_bytes(),
            self.child_full_row_flags.size_in_bytes()
                + self.child_full_counts.size_in_bytes()
                + self.child_remove_counts.size_in_bytes()
                + self.child_add_counts.size_in_bytes()
                + self.child_update_counts.size_in_bytes()
                + self.child_full_first_genes.size_in_bytes()
                + self.child_full_gene_gaps.size_in_bytes()
                + self.child_remove_first_genes.size_in_bytes()
                + self.child_remove_gene_gaps.size_in_bytes()
                + self.child_add_first_genes.size_in_bytes()
                + self.child_add_gene_gaps.size_in_bytes()
                + self.child_update_first_genes.size_in_bytes()
                + self.child_update_gene_gaps.size_in_bytes(),
            self.child_full_vals.size_in_bytes()
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
pub struct EncodedColumnBlock {
    pub num_cells: u32,
    pub num_genes: u32,
    pub gene_ids: DeltaEncodedIndices,
    pub posting_counts: ArithmeticEncoded,
    pub row_firsts: EncodedU32Stream,
    pub row_gaps: EncodedU32Stream,
    pub vals: ArithmeticEncoded,
}

impl EncodedColumnBlock {
    pub fn total_bytes(&self) -> usize {
        self.gene_ids.size_in_bytes()
            + self.posting_counts.size_in_bytes()
            + self.row_firsts.size_in_bytes()
            + self.row_gaps.size_in_bytes()
            + self.vals.size_in_bytes()
            + 8
    }

    pub fn bytes_breakdown(&self) -> (usize, usize, usize, usize, usize, usize) {
        (
            0,
            self.gene_ids.size_in_bytes(),
            self.posting_counts.size_in_bytes(),
            self.row_firsts.size_in_bytes() + self.row_gaps.size_in_bytes(),
            self.vals.size_in_bytes(),
            8,
        )
    }

    pub fn decode_rows(&self) -> Vec<Vec<(u32, u16)>> {
        let nrows = self.num_cells as usize;
        let mut rows: Vec<Vec<(u32, u16)>> = vec![Vec::new(); nrows];

        let genes = self.gene_ids.decode_all();
        let counts = self.posting_counts.decode_all().unwrap_or_default();
        let firsts = self.row_firsts.decode_all();
        let gaps = self.row_gaps.decode_all();
        let vals = self.vals.decode_all().unwrap_or_default();

        let mut first_cursor = 0usize;
        let mut gap_cursor = 0usize;
        let mut val_cursor = 0usize;

        for (gene_pos, &gene_u64) in genes.iter().enumerate() {
            let gene = gene_u64 as u32;
            let count = counts.get(gene_pos).copied().unwrap_or(0) as usize;
            if count == 0 {
                continue;
            }

            let mut row = firsts.get(first_cursor).copied().unwrap_or(0) as usize;
            first_cursor += 1;

            let value0 = vals.get(val_cursor).copied().unwrap_or(0) as u16;
            val_cursor += 1;
            if row < nrows && value0 > 0 {
                rows[row].push((gene, value0));
            }

            for _ in 1..count {
                let gap = gaps.get(gap_cursor).copied().unwrap_or(0) as usize;
                gap_cursor += 1;
                row = row.saturating_add(gap);
                let value = vals.get(val_cursor).copied().unwrap_or(0) as u16;
                val_cursor += 1;
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
            if value != 0 {
                let mapped_gene = gene_old_to_new
                    .and_then(|mapping| mapping.get(gene_idx).copied())
                    .unwrap_or(gene_idx as u32);
                expression.push((mapped_gene, value));
            }
        }
    }
    if gene_old_to_new.is_some() {
        expression.sort_unstable_by_key(|(gene, _)| *gene);
    }
    expression
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

#[derive(Clone, Copy)]
struct L0Distance;

impl Distance<(u32, u16)> for L0Distance {
    fn eval(&self, va: &[(u32, u16)], vb: &[(u32, u16)]) -> f32 {
        l0_binary_diff(va, vb) as f32
    }
}

#[derive(Clone, Copy)]
struct L2SquaredDistance;

impl Distance<(u32, u16)> for L2SquaredDistance {
    fn eval(&self, va: &[(u32, u16)], vb: &[(u32, u16)]) -> f32 {
        l2_squared_diff(va, vb) as f32
    }
}

#[derive(Clone, Copy)]
struct HammingDistance;

impl Distance<(u32, u16)> for HammingDistance {
    fn eval(&self, va: &[(u32, u16)], vb: &[(u32, u16)]) -> f32 {
        hamming_diff(va, vb) as f32
    }
}

#[derive(Clone, Copy)]
struct JaccardDistance;

impl Distance<(u32, u16)> for JaccardDistance {
    fn eval(&self, va: &[(u32, u16)], vb: &[(u32, u16)]) -> f32 {
        binary_jaccard_distance(va, vb)
    }
}

const JACCARD_WEIGHT_SCALE: f32 = 1_000_000.0;

fn hnsw_distance_to_weight(metric: KnnDistanceMetric, distance: f32) -> u64 {
    match metric {
        KnnDistanceMetric::Jaccard => (distance * JACCARD_WEIGHT_SCALE).round() as u64,
        _ => distance as u64,
    }
}

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
    let ops = compute_edit_ops(parent, child);
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
        let entries: Vec<(u32, u32)> = child
            .iter()
            .map(|&(gene, value)| (gene, value as u32))
            .collect();
        // full-row count + flag
        encoded_entries_cost_bytes(&entries, index_codec) + 2
    } else {
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
        // full-row flag + 3 op-count fields
        remove_cost
            + encoded_entries_cost_bytes(&add_entries, index_codec)
            + encoded_entries_cost_bytes(&update_entries, index_codec)
            + 4
    }
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
    metric_distance_hint: Option<f32>,
) -> u64 {
    match mst_weight_mode {
        MstWeightMode::Metric => metric_distance_hint
            .map(|d| hnsw_distance_to_weight(metric, d))
            .unwrap_or_else(|| pair_distance(metric, a, b)),
        MstWeightMode::EncodingCost => {
            symmetric_edge_encoding_cost_bytes(a, b, index_codec, full_row_fallback_ratio)
        }
    }
}

fn build_mst_prim(
    expressions: &[SparseExpression],
    k: usize,
    metric: KnnDistanceMetric,
    mst_weight_mode: MstWeightMode,
    index_codec: IndexStreamCodec,
    full_row_fallback_ratio: Option<f32>,
    forest_cut_factor: Option<f32>,
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

    if n >= 64 {
        let max_nb_connection = 16;
        let ef_construction = 100;
        let nb_layers = 16;
        let ef_search = 50;
        match metric {
            KnnDistanceMetric::L0 => {
                let hnsw = Hnsw::<(u32, u16), L0Distance>::new(
                    max_nb_connection,
                    n,
                    nb_layers,
                    ef_construction,
                    L0Distance,
                );

                for (i, expr) in expressions.iter().enumerate() {
                    hnsw.insert((expr, i));
                }

                let knn_results: Vec<Vec<Neighbour>> = expressions
                    .par_iter()
                    .map(|expr| hnsw.search(expr, k, ef_search))
                    .collect();

                for (i, neighbors) in knn_results.into_iter().enumerate() {
                    for neighbor in neighbors {
                        if neighbor.d_id != i {
                            let weight = edge_weight(
                                metric,
                                &expressions[i],
                                &expressions[neighbor.d_id],
                                mst_weight_mode,
                                index_codec,
                                full_row_fallback_ratio,
                                Some(neighbor.distance),
                            );
                            graph.update_edge(nodes[i], nodes[neighbor.d_id], weight);
                        }
                    }
                }
            }
            KnnDistanceMetric::L2 => {
                let hnsw = Hnsw::<(u32, u16), L2SquaredDistance>::new(
                    max_nb_connection,
                    n,
                    nb_layers,
                    ef_construction,
                    L2SquaredDistance,
                );

                for (i, expr) in expressions.iter().enumerate() {
                    hnsw.insert((expr, i));
                }

                let knn_results: Vec<Vec<Neighbour>> = expressions
                    .par_iter()
                    .map(|expr| hnsw.search(expr, k, ef_search))
                    .collect();

                for (i, neighbors) in knn_results.into_iter().enumerate() {
                    for neighbor in neighbors {
                        if neighbor.d_id != i {
                            let weight = edge_weight(
                                metric,
                                &expressions[i],
                                &expressions[neighbor.d_id],
                                mst_weight_mode,
                                index_codec,
                                full_row_fallback_ratio,
                                Some(neighbor.distance),
                            );
                            graph.update_edge(nodes[i], nodes[neighbor.d_id], weight);
                        }
                    }
                }
            }
            KnnDistanceMetric::Hamming => {
                let hnsw = Hnsw::<(u32, u16), HammingDistance>::new(
                    max_nb_connection,
                    n,
                    nb_layers,
                    ef_construction,
                    HammingDistance,
                );

                for (i, expr) in expressions.iter().enumerate() {
                    hnsw.insert((expr, i));
                }

                let knn_results: Vec<Vec<Neighbour>> = expressions
                    .par_iter()
                    .map(|expr| hnsw.search(expr, k, ef_search))
                    .collect();

                for (i, neighbors) in knn_results.into_iter().enumerate() {
                    for neighbor in neighbors {
                        if neighbor.d_id != i {
                            let weight = edge_weight(
                                metric,
                                &expressions[i],
                                &expressions[neighbor.d_id],
                                mst_weight_mode,
                                index_codec,
                                full_row_fallback_ratio,
                                Some(neighbor.distance),
                            );
                            graph.update_edge(nodes[i], nodes[neighbor.d_id], weight);
                        }
                    }
                }
            }
            KnnDistanceMetric::Jaccard => {
                let hnsw = Hnsw::<(u32, u16), JaccardDistance>::new(
                    max_nb_connection,
                    n,
                    nb_layers,
                    ef_construction,
                    JaccardDistance,
                );

                for (i, expr) in expressions.iter().enumerate() {
                    hnsw.insert((expr, i));
                }

                let knn_results: Vec<Vec<Neighbour>> = expressions
                    .par_iter()
                    .map(|expr| hnsw.search(expr, k, ef_search))
                    .collect();

                for (i, neighbors) in knn_results.into_iter().enumerate() {
                    for neighbor in neighbors {
                        if neighbor.d_id != i {
                            let weight = edge_weight(
                                metric,
                                &expressions[i],
                                &expressions[neighbor.d_id],
                                mst_weight_mode,
                                index_codec,
                                full_row_fallback_ratio,
                                Some(neighbor.distance),
                            );
                            graph.update_edge(nodes[i], nodes[neighbor.d_id], weight);
                        }
                    }
                }
            }
        }
    } else {
        for i in 0..n {
            let mut candidates: Vec<(usize, u64)> = Vec::with_capacity(n - 1);
            for j in 0..n {
                if i == j {
                    continue;
                }
                candidates.push((
                    j,
                    edge_weight(
                        metric,
                        &expressions[i],
                        &expressions[j],
                        mst_weight_mode,
                        index_codec,
                        full_row_fallback_ratio,
                        None,
                    ),
                ));
            }

            let k_actual = k.min(candidates.len());
            if k_actual > 0 {
                candidates.select_nth_unstable_by(k_actual - 1, |a, b| a.1.cmp(&b.1));
                for &(j, dist) in &candidates[..k_actual] {
                    graph.update_edge(nodes[i], nodes[j], dist);
                }
            }
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
                None,
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
) -> Option<EncodedColumnBlock> {
    if points.is_empty() {
        return None;
    }

    let expressions: Vec<SparseExpression> = points
        .par_iter()
        .map(|p| compute_sparse_expression(p, data, gene_old_to_new))
        .collect();

    let mut postings: BTreeMap<u32, Vec<(u32, u16)>> = BTreeMap::new();
    for (local_row, expr) in expressions.iter().enumerate() {
        let row_id = local_row as u32;
        for &(gene, value) in expr {
            postings.entry(gene).or_default().push((row_id, value));
        }
    }

    let gene_ids_u64: Vec<u64> = postings.keys().map(|&g| g as u64).collect();
    let mut posting_counts = Vec::<u32>::with_capacity(gene_ids_u64.len());
    let mut row_firsts = Vec::<u32>::new();
    let mut row_gaps = Vec::<u32>::new();
    let mut vals = Vec::<u32>::new();

    for entries in postings.values() {
        posting_counts.push(entries.len() as u32);
        if let Some(&(first_row, first_val)) = entries.first() {
            row_firsts.push(first_row);
            vals.push(first_val as u32);
            let mut prev = first_row;
            for &(row, value) in entries.iter().skip(1) {
                row_gaps.push(row.saturating_sub(prev));
                vals.push(value as u32);
                prev = row;
            }
        }
    }

    let gene_ids = DeltaEncodedIndices::from_indices(&gene_ids_u64);
    let posting_counts = if posting_counts.is_empty() {
        ArithmeticEncoded::default()
    } else {
        ArithmeticEncoded::from_slice(&posting_counts).expect("valid posting counts")
    };
    let row_firsts = EncodedU32Stream::from_slice(&row_firsts, index_codec);
    let row_gaps = EncodedU32Stream::from_slice(&row_gaps, index_codec);
    let vals = if vals.is_empty() {
        ArithmeticEncoded::default()
    } else {
        ArithmeticEncoded::from_slice(&vals).expect("valid posting values")
    };

    Some(EncodedColumnBlock {
        num_cells: points.len() as u32,
        num_genes: data.cols() as u32,
        gene_ids,
        posting_counts,
        row_firsts,
        row_gaps,
        vals,
    })
}

pub fn encode_subarray_mst_with_metric(
    points: &[Point],
    data: &CsMat<u16>,
    knn_metric: KnnDistanceMetric,
    mst_weight_mode: MstWeightMode,
    gene_old_to_new: Option<&[u32]>,
    index_codec: IndexStreamCodec,
    full_row_fallback_ratio: Option<f32>,
    forest_cut_factor: Option<f32>,
) -> Option<(EncodedDiffsMST, Vec<u32>)> {
    if points.is_empty() {
        return None;
    }

    let num_genes = data.cols() as u32;
    let expressions: Vec<SparseExpression> = points
        .par_iter()
        .map(|p| compute_sparse_expression(p, data, gene_old_to_new))
        .collect();

    let k = 8usize.min(points.len().saturating_sub(1)).max(1);
    let (root, parent) = build_mst_prim(
        &expressions,
        k,
        knn_metric,
        mst_weight_mode,
        index_codec,
        full_row_fallback_ratio,
        forest_cut_factor,
    );
    let (dfs_order, parent_offset_raw) = compute_dfs_order(root, &parent, points.len());

    let root_expr = &expressions[root];
    let root_genes_u64: Vec<u64> = root_expr.iter().map(|(g, _)| *g as u64).collect();
    let root_vals_raw: Vec<u32> = root_expr.iter().map(|(_, v)| *v as u32).collect();

    let mut child_full_row_flags_raw = Vec::<u32>::with_capacity(points.len().saturating_sub(1));
    let mut child_full_counts_raw = Vec::<u32>::with_capacity(points.len().saturating_sub(1));
    let mut child_remove_counts_raw = Vec::<u32>::with_capacity(points.len().saturating_sub(1));
    let mut child_add_counts_raw = Vec::<u32>::with_capacity(points.len().saturating_sub(1));
    let mut child_update_counts_raw = Vec::<u32>::with_capacity(points.len().saturating_sub(1));

    let mut child_full_first_genes_raw = Vec::<u32>::new();
    let mut child_full_gene_gaps_raw = Vec::<u32>::new();
    let mut child_remove_first_genes_raw = Vec::<u32>::new();
    let mut child_remove_gene_gaps_raw = Vec::<u32>::new();
    let mut child_add_first_genes_raw = Vec::<u32>::new();
    let mut child_add_gene_gaps_raw = Vec::<u32>::new();
    let mut child_update_first_genes_raw = Vec::<u32>::new();
    let mut child_update_gene_gaps_raw = Vec::<u32>::new();

    let mut child_full_vals_raw = Vec::<u32>::new();
    let mut child_add_vals_raw = Vec::<u32>::new();
    let mut child_update_vals_raw = Vec::<u32>::new();

    for &orig_cell in dfs_order.iter().skip(1) {
        let parent_orig = parent[orig_cell as usize] as usize;
        let child_expr = &expressions[orig_cell as usize];
        let parent_expr = &expressions[parent_orig];
        let ops = compute_edit_ops(parent_expr, child_expr);
        let edit_count = ops.removes.len() + ops.adds.len() + ops.updates.len();

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

        if use_full_row {
            child_full_row_flags_raw.push(1);
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
            child_full_row_flags_raw.push(0);
            child_full_counts_raw.push(0);
            child_remove_counts_raw.push(ops.removes.len() as u32);
            child_add_counts_raw.push(ops.adds.len() as u32);
            child_update_counts_raw.push(ops.updates.len() as u32);

            append_gene_stream(
                &ops.removes,
                &mut child_remove_first_genes_raw,
                &mut child_remove_gene_gaps_raw,
            );
            append_entry_stream(
                &ops.adds,
                &mut child_add_first_genes_raw,
                &mut child_add_gene_gaps_raw,
                &mut child_add_vals_raw,
            );
            append_entry_stream(
                &ops.updates,
                &mut child_update_first_genes_raw,
                &mut child_update_gene_gaps_raw,
                &mut child_update_vals_raw,
            );
        }
    }

    let parent_offset =
        ArithmeticEncoded::from_slice(&parent_offset_raw).expect("valid parent offset");
    let root_indices = DeltaEncodedIndices::from_indices(&root_genes_u64);
    let root_vals = if root_vals_raw.is_empty() {
        ArithmeticEncoded::default()
    } else {
        ArithmeticEncoded::from_slice(&root_vals_raw).expect("valid root vals")
    };
    let child_full_row_flags =
        ArithmeticEncoded::from_slice(&child_full_row_flags_raw).expect("valid full-row flags");
    let child_full_counts =
        ArithmeticEncoded::from_slice(&child_full_counts_raw).expect("valid full counts");
    let child_remove_counts =
        ArithmeticEncoded::from_slice(&child_remove_counts_raw).expect("valid remove counts");
    let child_add_counts =
        ArithmeticEncoded::from_slice(&child_add_counts_raw).expect("valid add counts");
    let child_update_counts =
        ArithmeticEncoded::from_slice(&child_update_counts_raw).expect("valid update counts");

    let child_full_first_genes =
        EncodedU32Stream::from_slice(&child_full_first_genes_raw, index_codec);
    let child_full_gene_gaps = EncodedU32Stream::from_slice(&child_full_gene_gaps_raw, index_codec);
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
            root_indices,
            root_vals,
            child_full_row_flags,
            child_full_counts,
            child_remove_counts,
            child_add_counts,
            child_update_counts,
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
        Some(1.0),
        None,
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
    states: Vec<Vec<(u32, i32)>>,
    child_is_full_row: Vec<bool>,
    child_full_entries: Vec<Vec<(u32, u32)>>,
    child_removes: Vec<Vec<u32>>,
    child_adds: Vec<Vec<(u32, u32)>>,
    child_updates: Vec<Vec<(u32, u32)>>,
}

impl<'a> SparseExpressionIterMST<'a> {
    fn new(encoded: &'a EncodedDiffsMST) -> Self {
        let ncells = encoded.num_cells();
        let mut states = Vec::with_capacity(ncells);
        let mut child_is_full_row = vec![false; ncells];
        let mut child_full_entries = vec![Vec::new(); ncells];
        let mut child_removes = vec![Vec::new(); ncells];
        let mut child_adds = vec![Vec::new(); ncells];
        let mut child_updates = vec![Vec::new(); ncells];

        if ncells > 0 {
            let mut root_state = Vec::new();
            let root_indices = encoded.root_indices.decode_all();
            let root_vals = encoded.root_vals.decode_all().unwrap_or_default();
            for (i, &gene) in root_indices.iter().enumerate() {
                let value = root_vals.get(i).copied().unwrap_or(0) as i32;
                root_state.push((gene as u32, value));
            }
            states.push(root_state);
        }

        if ncells > 1 {
            let full_row_flags = encoded
                .child_full_row_flags
                .decode_all()
                .unwrap_or_default();
            let full_counts = encoded.child_full_counts.decode_all().unwrap_or_default();
            let remove_counts = encoded.child_remove_counts.decode_all().unwrap_or_default();
            let add_counts = encoded.child_add_counts.decode_all().unwrap_or_default();
            let update_counts = encoded.child_update_counts.decode_all().unwrap_or_default();

            let full_first_genes = encoded.child_full_first_genes.decode_all();
            let full_gaps = encoded.child_full_gene_gaps.decode_all();
            let remove_first_genes = encoded.child_remove_first_genes.decode_all();
            let remove_gaps = encoded.child_remove_gene_gaps.decode_all();
            let add_first_genes = encoded.child_add_first_genes.decode_all();
            let add_gaps = encoded.child_add_gene_gaps.decode_all();
            let update_first_genes = encoded.child_update_first_genes.decode_all();
            let update_gaps = encoded.child_update_gene_gaps.decode_all();

            let full_vals = encoded.child_full_vals.decode_all().unwrap_or_default();
            let add_vals = encoded.child_add_vals.decode_all().unwrap_or_default();
            let update_vals = encoded.child_update_vals.decode_all().unwrap_or_default();

            let mut full_first_cursor = 0usize;
            let mut full_gap_cursor = 0usize;
            let mut full_val_cursor = 0usize;
            let mut remove_first_cursor = 0usize;
            let mut remove_gap_cursor = 0usize;
            let mut add_first_cursor = 0usize;
            let mut add_gap_cursor = 0usize;
            let mut add_val_cursor = 0usize;
            let mut update_first_cursor = 0usize;
            let mut update_gap_cursor = 0usize;
            let mut update_val_cursor = 0usize;

            for dfs_pos in 1..ncells {
                let is_full_row = full_row_flags.get(dfs_pos - 1).copied().unwrap_or(0) != 0;
                child_is_full_row[dfs_pos] = is_full_row;
                if is_full_row {
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
            states,
            child_is_full_row,
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
            let current_state = if self.child_is_full_row[current_dfs] {
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
                EncodedDiffsMST::apply_edit_ops(
                    &self.states[parent_dfs],
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
                    Some((*gene, *value as u16))
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
}
