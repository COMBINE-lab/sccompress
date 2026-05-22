/// Delta-based encoding for sorted indices using arithmetic coding
///
/// This module provides efficient compression of sorted indices by encoding
/// the deltas (differences) between consecutive indices using arithmetic encoding.
///
/// # Approach
///
/// Instead of storing absolute indices, we store:
/// 1. The first index (directly)
/// 2. Deltas between consecutive indices (using arithmetic encoding)
///
/// For decoding, we reconstruct by accumulating deltas starting from the first index.
///
/// # Benefits
///
/// - Deltas are typically small (1-100 for consecutive genes or nearby cells)
/// - Arithmetic encoding excels at compressing small integer distributions
/// - Expected 40-60% compression improvement over absolute index storage
///
/// # Trade-offs
///
/// - No random access to individual indices (must decode sequentially)
/// - This is acceptable since we always decode in deterministic order (DFS for MST, cluster order for clusters)
use crate::arith_encode::ArithmeticEncoded;
use bincode::{Decode, Encode};

/// Delta-encoded indices structure
///
/// Stores sorted indices as (first_index, deltas) where deltas are compressed
/// using arithmetic encoding.
#[derive(Clone, Encode, Decode)]
pub struct DeltaEncodedIndices {
    /// First index in the sequence (stored directly, no delta)
    first_index: u64,

    /// Deltas between consecutive indices, encoded with arithmetic coding
    /// delta[i] = index[i+1] - index[i]
    deltas: ArithmeticEncoded,

    /// Total number of indices (including first)
    count: usize,
}

impl DeltaEncodedIndices {
    /// Create an empty DeltaEncodedIndices
    #[allow(dead_code)]
    pub fn empty() -> Self {
        Self::default()
    }

    /// Create delta-encoded indices from a sorted slice of indices
    ///
    /// # Arguments
    ///
    /// * `indices` - Sorted slice of u64 indices
    ///
    /// # Returns
    ///
    /// Delta-encoded representation
    ///
    /// # Panics
    ///
    /// Panics if indices are not sorted in ascending order
    pub fn from_indices(indices: &[u64]) -> Self {
        if indices.is_empty() {
            return Self {
                first_index: 0,
                deltas: ArithmeticEncoded::default(),
                count: 0,
            };
        }

        // Verify sorted order
        for i in 1..indices.len() {
            assert!(indices[i] >= indices[i - 1], "Indices must be sorted");
        }

        let first_index = indices[0];
        let count = indices.len();

        if count == 1 {
            // Special case: single index, no deltas
            return Self {
                first_index,
                deltas: ArithmeticEncoded::default(),
                count: 1,
            };
        }

        // Compute deltas
        let deltas_vec: Vec<u32> = indices
            .windows(2)
            .map(|w| {
                let delta = w[1] - w[0];
                assert!(delta <= u32::MAX as u64, "Delta too large");
                delta as u32
            })
            .collect();

        let deltas = ArithmeticEncoded::from_slice(&deltas_vec).expect("Failed to encode deltas");

        Self {
            first_index,
            deltas,
            count,
        }
    }

    /// Decode all indices from the delta encoding
    ///
    /// # Returns
    ///
    /// Vector of reconstructed indices in sorted order
    pub fn decode_all(&self) -> Vec<u64> {
        if self.count == 0 {
            return Vec::new();
        }

        if self.count == 1 {
            return vec![self.first_index];
        }

        let mut result = Vec::with_capacity(self.count);
        result.push(self.first_index);

        // Decode deltas and accumulate
        let deltas = self.deltas.decode_all().expect("Failed to decode deltas");
        let mut current = self.first_index;

        for delta in deltas {
            current += delta as u64;
            result.push(current);
        }

        assert_eq!(result.len(), self.count, "Decoded count mismatch");
        result
    }

    /// Get the number of indices
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if empty
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Get the size in bytes of the compressed representation
    pub fn size_in_bytes(&self) -> usize {
        8 + // first_index (u64)
        self.deltas.size_in_bytes() +
        8 // count (usize)
    }
}

impl Default for DeltaEncodedIndices {
    fn default() -> Self {
        Self {
            first_index: 0,
            deltas: ArithmeticEncoded::default(),
            count: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delta_encoding_roundtrip() {
        let indices = vec![10, 12, 13, 15, 20, 21, 22, 100];
        let encoded = DeltaEncodedIndices::from_indices(&indices);
        let decoded = encoded.decode_all();

        assert_eq!(indices, decoded, "Round-trip failed");
        assert_eq!(encoded.len(), indices.len());
    }

    #[test]
    fn test_delta_encoding_empty() {
        let indices: Vec<u64> = vec![];
        let encoded = DeltaEncodedIndices::from_indices(&indices);
        let decoded = encoded.decode_all();

        assert_eq!(indices, decoded);
        assert!(encoded.is_empty());
        assert_eq!(encoded.len(), 0);
    }

    #[test]
    fn test_delta_encoding_single() {
        let indices = vec![42];
        let encoded = DeltaEncodedIndices::from_indices(&indices);
        let decoded = encoded.decode_all();

        assert_eq!(indices, decoded);
        assert_eq!(encoded.len(), 1);
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_delta_encoding_consecutive() {
        // Consecutive indices should compress very well (all deltas = 1)
        let indices: Vec<u64> = (0..100).collect();
        let encoded = DeltaEncodedIndices::from_indices(&indices);
        let decoded = encoded.decode_all();

        assert_eq!(indices, decoded);
    }

    #[test]
    fn test_delta_encoding_large_gaps() {
        // Test with large gaps between indices
        let indices = vec![0, 100, 200, 10000, 10001, 20000];
        let encoded = DeltaEncodedIndices::from_indices(&indices);
        let decoded = encoded.decode_all();

        assert_eq!(indices, decoded);
    }

    #[test]
    fn test_delta_encoding_serialization() {
        use bincode::config::standard;

        let indices = vec![5, 10, 15, 20, 100, 101, 102];
        let encoded = DeltaEncodedIndices::from_indices(&indices);

        // Serialize
        let bytes = bincode::encode_to_vec(&encoded, standard()).unwrap();

        // Deserialize
        let (decoded_struct, _): (DeltaEncodedIndices, usize) =
            bincode::decode_from_slice(&bytes, standard()).unwrap();

        let decoded_indices = decoded_struct.decode_all();
        assert_eq!(indices, decoded_indices);
    }
}
