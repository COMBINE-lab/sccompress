use crate::arith_encode::ArithmeticEncoded;
use crate::delta_indices::DeltaEncodedIndices;
use crate::index_stream::{EncodedU32Stream, IndexStreamCodec};
use bincode::{Decode, Encode};
use hnsw_rs::prelude::*;
use petgraph::algo::{connected_components, min_spanning_tree};
use petgraph::data::FromElements;
use petgraph::graph::UnGraph;
use rayon::prelude::*;
use sprs::CsMat;

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
    pub child_delta_counts: ArithmeticEncoded,
    pub child_full_row_flags: ArithmeticEncoded,
    pub child_first_genes: EncodedU32Stream,
    pub child_gene_gaps: EncodedU32Stream,
    pub child_vals: ArithmeticEncoded,
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
            self.child_delta_counts.size_in_bytes()
                + self.child_full_row_flags.size_in_bytes()
                + self.child_first_genes.size_in_bytes()
                + self.child_gene_gaps.size_in_bytes(),
            self.child_vals.size_in_bytes(),
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

    fn apply_deltas(acc: &[(u32, i32)], deltas: &[(u32, i32)]) -> Vec<(u32, i32)> {
        let mut result = Vec::with_capacity(acc.len().saturating_add(deltas.len()));
        let mut i = 0;
        let mut j = 0;

        while i < acc.len() && j < deltas.len() {
            match acc[i].0.cmp(&deltas[j].0) {
                std::cmp::Ordering::Less => {
                    result.push(acc[i]);
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    let decoded_delta = zigzag_decode(deltas[j].1);
                    if decoded_delta != 0 {
                        result.push((deltas[j].0, decoded_delta));
                    }
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    let decoded_delta = zigzag_decode(deltas[j].1);
                    let sum = acc[i].1 + decoded_delta;
                    if sum != 0 {
                        result.push((acc[i].0, sum));
                    }
                    i += 1;
                    j += 1;
                }
            }
        }

        while i < acc.len() {
            result.push(acc[i]);
            i += 1;
        }

        while j < deltas.len() {
            let decoded_delta = zigzag_decode(deltas[j].1);
            if decoded_delta != 0 {
                result.push((deltas[j].0, decoded_delta));
            }
            j += 1;
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

fn encoded_entries_cost_bytes(entries: &[(u32, u32)], index_codec: IndexStreamCodec) -> u64 {
    if entries.is_empty() {
        return 0;
    }

    let mut cost = 0u64;
    cost += index_symbol_bytes(entries[0].0, index_codec);
    let mut prev_gene = entries[0].0;
    cost += arithmetic_symbol_bytes(entries[0].1);

    for &(gene, value) in entries.iter().skip(1) {
        let gap = gene.saturating_sub(prev_gene);
        cost += index_symbol_bytes(gap, index_codec);
        cost += arithmetic_symbol_bytes(value);
        prev_gene = gene;
    }

    // Count + full-row-flag marginal overhead.
    cost + 2
}

fn child_payload_cost_bytes(
    parent: &SparseExpression,
    child: &SparseExpression,
    index_codec: IndexStreamCodec,
    full_row_fallback_ratio: Option<f32>,
) -> u64 {
    let deltas = sparse_subtract(child, parent);

    let use_full_row = match full_row_fallback_ratio {
        Some(ratio) => {
            if child.is_empty() {
                !deltas.is_empty()
            } else {
                (deltas.len() as f32) > ratio * (child.len() as f32)
            }
        }
        None => false,
    };

    if use_full_row {
        let entries: Vec<(u32, u32)> = child
            .iter()
            .map(|&(gene, value)| (gene, value as u32))
            .collect();
        encoded_entries_cost_bytes(&entries, index_codec)
    } else {
        let entries: Vec<(u32, u32)> = deltas
            .into_iter()
            .map(|(gene, delta)| (gene, delta as u32))
            .collect();
        encoded_entries_cost_bytes(&entries, index_codec)
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

fn zigzag_encode(v: i32) -> i32 {
    if v < 0 {
        (-2 * v) - 1
    } else {
        2 * v
    }
}

fn zigzag_decode(v: i32) -> i32 {
    if v % 2 == 0 {
        v / 2
    } else {
        -(v + 1) / 2
    }
}

fn sparse_subtract(child: &SparseExpression, parent: &SparseExpression) -> Vec<(u32, i32)> {
    let mut result = Vec::with_capacity(child.len().saturating_add(parent.len()));
    let mut i = 0;
    let mut j = 0;

    while i < child.len() && j < parent.len() {
        match child[i].0.cmp(&parent[j].0) {
            std::cmp::Ordering::Less => {
                result.push((child[i].0, zigzag_encode(child[i].1 as i32)));
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                result.push((parent[j].0, zigzag_encode(-(parent[j].1 as i32))));
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                let delta = child[i].1 as i32 - parent[j].1 as i32;
                if delta != 0 {
                    result.push((child[i].0, zigzag_encode(delta)));
                }
                i += 1;
                j += 1;
            }
        }
    }

    while i < child.len() {
        result.push((child[i].0, zigzag_encode(child[i].1 as i32)));
        i += 1;
    }

    while j < parent.len() {
        result.push((parent[j].0, zigzag_encode(-(parent[j].1 as i32))));
        j += 1;
    }

    result
}

fn build_mst_prim(
    expressions: &[SparseExpression],
    k: usize,
    metric: KnnDistanceMetric,
    mst_weight_mode: MstWeightMode,
    index_codec: IndexStreamCodec,
    full_row_fallback_ratio: Option<f32>,
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
    let mut parent = vec![u32::MAX; n];
    let mut visited = vec![false; n];
    let mut stack = vec![nodes[0]];
    visited[0] = true;
    parent[0] = 0;

    let mut cell_to_mst_node = vec![None; n];
    for node_idx in mst_graph.node_indices() {
        cell_to_mst_node[mst_graph[node_idx]] = Some(node_idx);
    }

    while let Some(u_idx) = stack.pop() {
        let u = graph[u_idx];
        if let Some(mst_u_idx) = cell_to_mst_node[u] {
            for v_idx in mst_graph.neighbors(mst_u_idx) {
                let v = mst_graph[v_idx];
                if !visited[v] {
                    visited[v] = true;
                    parent[v] = u as u32;
                    stack.push(nodes[v]);
                }
            }
        }
    }

    for p in &mut parent {
        if *p == u32::MAX {
            *p = 0;
        }
    }

    (0, parent)
}

fn compute_dfs_order(root: usize, parent: &[u32], n: usize) -> (Vec<u32>, Vec<u32>) {
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        if i != root {
            children[parent[i] as usize].push(i);
        }
    }

    let mut dfs_order = Vec::with_capacity(n);
    let mut pos_in_dfs = vec![0u32; n];
    let mut stack = vec![root];

    while let Some(node) = stack.pop() {
        pos_in_dfs[node] = dfs_order.len() as u32;
        dfs_order.push(node as u32);
        for &child in children[node].iter().rev() {
            stack.push(child);
        }
    }

    let mut parent_offset = Vec::with_capacity(n);
    for (dfs_pos, &orig_cell) in dfs_order.iter().enumerate() {
        if orig_cell as usize == root {
            parent_offset.push(0);
        } else {
            let parent_orig = parent[orig_cell as usize] as usize;
            parent_offset.push((dfs_pos as u32) - pos_in_dfs[parent_orig]);
        }
    }

    (dfs_order, parent_offset)
}

pub fn encode_subarray_mst_with_metric(
    points: &[Point],
    data: &CsMat<u16>,
    knn_metric: KnnDistanceMetric,
    mst_weight_mode: MstWeightMode,
    gene_old_to_new: Option<&[u32]>,
    index_codec: IndexStreamCodec,
    full_row_fallback_ratio: Option<f32>,
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
    );
    let (dfs_order, parent_offset_raw) = compute_dfs_order(root, &parent, points.len());

    let root_expr = &expressions[root];
    let root_genes_u64: Vec<u64> = root_expr.iter().map(|(g, _)| *g as u64).collect();
    let root_vals_raw: Vec<u32> = root_expr.iter().map(|(_, v)| *v as u32).collect();

    let mut child_delta_counts_raw = Vec::<u32>::with_capacity(points.len().saturating_sub(1));
    let mut child_full_row_flags_raw = Vec::<u32>::with_capacity(points.len().saturating_sub(1));
    let mut child_first_genes_raw = Vec::<u32>::new();
    let mut child_gene_gaps_raw = Vec::<u32>::new();
    let mut child_vals_raw = Vec::<u32>::new();

    for &orig_cell in dfs_order.iter().skip(1) {
        let parent_orig = parent[orig_cell as usize] as usize;
        let child_expr = &expressions[orig_cell as usize];
        let parent_expr = &expressions[parent_orig];
        let deltas = sparse_subtract(child_expr, parent_expr);

        let use_full_row = match full_row_fallback_ratio {
            Some(ratio) => {
                if child_expr.is_empty() {
                    !deltas.is_empty()
                } else {
                    (deltas.len() as f32) > ratio * (child_expr.len() as f32)
                }
            }
            None => false,
        };

        if use_full_row {
            child_full_row_flags_raw.push(1);
            child_delta_counts_raw.push(child_expr.len() as u32);

            if let Some((first_gene, first_val)) = child_expr.first() {
                child_first_genes_raw.push(*first_gene);
                child_vals_raw.push(*first_val as u32);
                let mut prev_gene = *first_gene;
                for &(gene, value) in child_expr.iter().skip(1) {
                    child_gene_gaps_raw.push(gene - prev_gene);
                    child_vals_raw.push(value as u32);
                    prev_gene = gene;
                }
            }
        } else {
            child_full_row_flags_raw.push(0);
            child_delta_counts_raw.push(deltas.len() as u32);

            if let Some((first_gene, first_delta)) = deltas.first() {
                child_first_genes_raw.push(*first_gene);
                child_vals_raw.push(*first_delta as u32);
                let mut prev_gene = *first_gene;
                for &(gene, delta) in deltas.iter().skip(1) {
                    child_gene_gaps_raw.push(gene - prev_gene);
                    child_vals_raw.push(delta as u32);
                    prev_gene = gene;
                }
            }
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
    let child_delta_counts =
        ArithmeticEncoded::from_slice(&child_delta_counts_raw).expect("valid child counts");
    let child_full_row_flags =
        ArithmeticEncoded::from_slice(&child_full_row_flags_raw).expect("valid full-row flags");
    let child_first_genes = EncodedU32Stream::from_slice(&child_first_genes_raw, index_codec);
    let child_gene_gaps = EncodedU32Stream::from_slice(&child_gene_gaps_raw, index_codec);
    let child_vals = if child_vals_raw.is_empty() {
        ArithmeticEncoded::default()
    } else {
        ArithmeticEncoded::from_slice(&child_vals_raw).expect("valid child values")
    };

    Some((
        EncodedDiffsMST {
            num_genes,
            parent_offset,
            root_indices,
            root_vals,
            child_delta_counts,
            child_full_row_flags,
            child_first_genes,
            child_gene_gaps,
            child_vals,
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
    child_entries: Vec<Vec<(u32, u32)>>,
    child_is_full_row: Vec<bool>,
}

impl<'a> SparseExpressionIterMST<'a> {
    fn new(encoded: &'a EncodedDiffsMST) -> Self {
        let ncells = encoded.num_cells();
        let mut states = Vec::with_capacity(ncells);
        let mut child_entries = vec![Vec::new(); ncells];
        let mut child_is_full_row = vec![false; ncells];

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
            let counts = encoded.child_delta_counts.decode_all().unwrap_or_default();
            let full_row_flags = encoded
                .child_full_row_flags
                .decode_all()
                .unwrap_or_default();
            let first_genes = encoded.child_first_genes.decode_all();
            let gaps = encoded.child_gene_gaps.decode_all();
            let vals = encoded.child_vals.decode_all().unwrap_or_default();

            let mut first_cursor = 0usize;
            let mut gap_cursor = 0usize;
            let mut val_cursor = 0usize;

            for dfs_pos in 1..ncells {
                let entry_count = counts.get(dfs_pos - 1).copied().unwrap_or(0) as usize;
                let is_full_row = full_row_flags.get(dfs_pos - 1).copied().unwrap_or(0) != 0;
                child_is_full_row[dfs_pos] = is_full_row;

                if entry_count == 0 {
                    continue;
                }

                let mut entries = Vec::with_capacity(entry_count);
                let mut gene = first_genes.get(first_cursor).copied().unwrap_or(0);
                first_cursor += 1;

                let first_val = vals.get(val_cursor).copied().unwrap_or(0);
                val_cursor += 1;
                entries.push((gene, first_val));

                for _ in 1..entry_count {
                    let gap = gaps.get(gap_cursor).copied().unwrap_or(0);
                    gap_cursor += 1;
                    gene = gene.saturating_add(gap);
                    let value = vals.get(val_cursor).copied().unwrap_or(0);
                    val_cursor += 1;
                    entries.push((gene, value));
                }

                child_entries[dfs_pos] = entries;
            }
        }

        Self {
            encoded,
            next_dfs: 0,
            states,
            child_entries,
            child_is_full_row,
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
                self.child_entries[current_dfs]
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
                let deltas: Vec<(u32, i32)> = self.child_entries[current_dfs]
                    .iter()
                    .map(|&(gene, val)| (gene, val as i32))
                    .collect();
                EncodedDiffsMST::apply_deltas(&self.states[parent_dfs], &deltas)
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
