# Lossy Compression Approach

## Overview

The goal here is to adopt a lossy compression that quantizes values prior to additional compression steps, and to test how well this allows us to shrink the size of the final file
compared to the error introduced.

## Main approach

The main approach will be to learn different quantization models for different clusters of cells, and to re-arrange the order of cells in the matrix so that all cells belonging to the same cluster are adjacent and we
simply store the prefix sum of cluster sizes to determine which quantization model to use.

## Algorithm Phases

- First, we should cluster the input data. Here we can first compute a nearest neighbor matrix efficiently and then use an approach like k-means clustering to cluster the input cells. We are looking to cluster together cells that have a similar expression profile.

- Second, we want to reorder the cells in the matrix so that all cells belonging to cluster 1 come first, then all cells from cluster 2, etc.  Make sure to re-arrange the metadata (2D cell positions) as well as the cells expression vectors themselves.

- Third, we will learn a quantizer (16 bins) for each cluster.  This quantizer will be used to quantize the gene expression values for the cells in the current cluster.

- Fourth, **optional** recursion (see `recursion` below).

- Fifth, once we have the quantized expression values, we should use our MST-based encoding to compress the expression values for all cells in this cluster.

We should perform this compression for all clusters. Our final representation will consist of (a) the MST-compressed blocks for each cluster, (b) the quantization models used for each cluster (so that we can map quantization bins to actual expression values) (c) a sparsely-encoded (using Elias-Fano) vector of the prefix sum of cluster sizes, this will allow us to map a re-ordered cell's index back to the cluster to which it belongs, so that we can use the right quantization model to recover its expression values.

- `recursion` : Let's do the above and validate correctness before considering this, but after the basic scheme works, we likely want to enable the ability to recursively decompose specific clusters into sub-clusters to achieve better compression or better MSE.  Here the idea is, *within* a cluster, we will cluster just the cells in this cluster, and apply the re-ordering, quantization, and compression to those.  This leads to a hierarchical represenation.  For the final encoding, we needn't care specifically about this hierarhical structure (i.e. we can flatten the representation to just the final set of "leaf" clusters and their corresponding quantization models).



