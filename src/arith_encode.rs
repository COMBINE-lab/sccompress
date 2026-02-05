//! Arithmetic encoding for integer sequences
//!
//! This module provides an efficient arithmetic encoding implementation
//! for compressing sequences of unsigned integers, specifically designed
//! for gene expression values and deltas.

use bincode::{Decode, Encode};
use constriction::stream::{
    model::DefaultContiguousCategoricalEntropyModel, 
    stack::DefaultAnsCoder, 
    Decode as AnsDecode,
    Encode as AnsEncode,
};
use std::convert::TryInto;

/// Arithmetic-encoded integer sequence
///
/// Uses ANS (Asymmetric Numeral Systems) coding for efficient compression
/// of integer sequences. The encoder builds a histogram of values and uses
/// it to assign shorter codes to more frequent values.
#[derive(Clone)]
pub struct ArithmeticEncoded {
    /// Compressed data
    compressed: Vec<u32>,
    /// Number of original values
    length: usize,
    /// Symbol probabilities (stored for decoding)
    probabilities: Vec<u32>,
    /// Maximum symbol value + 1
    alphabet_size: u32,
}

impl ArithmeticEncoded {
    /// Create an empty ArithmeticEncoded
    pub fn default() -> Self {
        ArithmeticEncoded {
            compressed: Vec::new(),
            length: 0,
            probabilities: Vec::new(),
            alphabet_size: 0,
        }
    }

    /// Encode a sequence of values using arithmetic coding
    ///
    /// # Arguments
    ///
    /// * `values` - Slice of u32 values to encode
    ///
    /// # Returns
    ///
    /// Encoded representation or error
    pub fn from_slice(values: &[u32]) -> Result<Self, String> {
        if values.is_empty() {
            return Ok(Self::default());
        }

        // Find alphabet size (max value + 1)
        let max_val = *values.iter().max().unwrap();
        let alphabet_size = max_val + 1;

        // Special case: if all values are the same, use a simple encoding
        if alphabet_size == 1 {
            // All values are 0, just store the count
            return Ok(ArithmeticEncoded {
                compressed: vec![],
                length: values.len(),
                probabilities: vec![values.len() as u32],
                alphabet_size: 1,
            });
        }

        // Build histogram
        let mut histogram = vec![0u32; alphabet_size as usize];
        for &val in values {
            histogram[val as usize] += 1;
        }

        // Ensure no zero probabilities (add Laplace smoothing)
        // This also handles the case where we have very few symbols
        for count in histogram.iter_mut() {
            *count += 1;  // Add 1 to all (Laplace smoothing)
        }

        // Create entropy model with normalized probabilities
        let probabilities = histogram.clone();
        
        // Convert to f64 and normalize
        let total: f64 = probabilities.iter().map(|&x| x as f64).sum();
        let normalized_probs: Vec<f64> = probabilities.iter()
            .map(|&x| (x as f64) / total)
            .collect();
        
        let model = DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities_fast(
            &normalized_probs,
            None,
        )
        .map_err(|e| format!("Failed to create entropy model for {} symbols: {:?}", alphabet_size, e))?;

        // Encode values (ANS doesn't require reversing on encode in this implementation)
        let mut coder = DefaultAnsCoder::new();
        for &val in values.iter() {
            coder
                .encode_symbol(val as usize, &model)
                .map_err(|e| format!("Encoding failed: {:?}", e))?;
        }

        let compressed = coder.into_compressed().map_err(|e| format!("Failed to get compressed: {:?}", e))?;

        Ok(ArithmeticEncoded {
            compressed: compressed.to_vec(),
            length: values.len(),
            probabilities,
            alphabet_size,
        })
    }

    /// Get the size in bytes of the compressed representation
    pub fn size_in_bytes(&self) -> usize {
        // Compressed data + metadata
        self.compressed.len() * 4 + // compressed data (u32s)
        self.probabilities.len() * 4 + // probabilities (u32s)
        8 // length + alphabet_size
    }

    /// Get the number of encoded values
    pub fn len(&self) -> usize {
        self.length
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Access a value at a specific index
    ///
    /// Note: This requires decoding all values up to the index,
    /// so it's O(n) unlike DacsOpt which supports random access.
    /// For the compression use case, this is acceptable as we
    /// typically decode entire sequences.
    pub fn access(&self, index: usize) -> Option<u32> {
        if index >= self.length {
            return None;
        }

        // Decode all values (cached in future optimization)
        let values = self.decode_all().ok()?;
        values.get(index).copied()
    }

    /// Decode all values from the compressed representation
    ///
    /// # Returns
    ///
    /// Vector of decoded values
    pub fn decode_all(&self) -> Result<Vec<u32>, String> {
        if self.length == 0 {
            return Ok(Vec::new());
        }

        // Special case: if alphabet size is 1, all values are 0
        if self.alphabet_size == 1 {
            return Ok(vec![0; self.length]);
        }

        // Create entropy model from stored probabilities with normalization
        let total: f64 = self.probabilities.iter().map(|&x| x as f64).sum();
        let normalized_probs: Vec<f64> = self.probabilities.iter()
            .map(|&x| (x as f64) / total)
            .collect();
        
        let model = DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities_fast(
            &normalized_probs,
            None,
        )
        .map_err(|e| format!("Failed to create entropy model: {:?}", e))?;

        // Decode values - need to convert to owned Vec for the decoder
        let compressed_vec = self.compressed.clone();
        let mut coder = DefaultAnsCoder::from_compressed(compressed_vec)
            .map_err(|e| format!("Failed to create decoder: {:?}", e))?;

        let mut values = Vec::with_capacity(self.length);
        for _ in 0..self.length {
            let symbol = coder
                .decode_symbol(&model)
                .map_err(|e| format!("Decoding failed: {:?}", e))?;
            values.push(symbol as u32);
        }

        // ANS stack decoder returns values in reverse order of encoding
        // Since we encoded in forward order, we need to reverse the decoded values
        values.reverse();
        Ok(values)
    }

    /// Serialize to bytes
    pub fn serialize_into(&self, writer: &mut Vec<u8>) -> Result<(), String> {
        // Write length
        writer.extend_from_slice(&(self.length as u64).to_le_bytes());
        // Write alphabet_size
        writer.extend_from_slice(&self.alphabet_size.to_le_bytes());
        // Write probabilities length
        writer.extend_from_slice(&(self.probabilities.len() as u32).to_le_bytes());
        // Write probabilities
        for &p in &self.probabilities {
            writer.extend_from_slice(&p.to_le_bytes());
        }
        // Write compressed length
        writer.extend_from_slice(&(self.compressed.len() as u32).to_le_bytes());
        // Write compressed data
        for &c in &self.compressed {
            writer.extend_from_slice(&c.to_le_bytes());
        }
        Ok(())
    }

    /// Deserialize from bytes
    pub fn deserialize_from(reader: &[u8]) -> Result<Self, String> {
        let mut pos = 0;

        // Read length
        let length = u64::from_le_bytes(
            reader[pos..pos + 8]
                .try_into()
                .map_err(|_| "Failed to read length")?,
        ) as usize;
        pos += 8;

        // Read alphabet_size
        let alphabet_size = u32::from_le_bytes(
            reader[pos..pos + 4]
                .try_into()
                .map_err(|_| "Failed to read alphabet_size")?,
        );
        pos += 4;

        // Read probabilities length
        let prob_len = u32::from_le_bytes(
            reader[pos..pos + 4]
                .try_into()
                .map_err(|_| "Failed to read probabilities length")?,
        ) as usize;
        pos += 4;

        // Read probabilities
        let mut probabilities = Vec::with_capacity(prob_len);
        for _ in 0..prob_len {
            let p = u32::from_le_bytes(
                reader[pos..pos + 4]
                    .try_into()
                    .map_err(|_| "Failed to read probability")?,
            );
            probabilities.push(p);
            pos += 4;
        }

        // Read compressed length
        let compressed_len = u32::from_le_bytes(
            reader[pos..pos + 4]
                .try_into()
                .map_err(|_| "Failed to read compressed length")?,
        ) as usize;
        pos += 4;

        // Read compressed data
        let mut compressed = Vec::with_capacity(compressed_len);
        for _ in 0..compressed_len {
            let c = u32::from_le_bytes(
                reader[pos..pos + 4]
                    .try_into()
                    .map_err(|_| "Failed to read compressed data")?,
            );
            compressed.push(c);
            pos += 4;
        }

        Ok(ArithmeticEncoded {
            compressed,
            length,
            probabilities,
            alphabet_size,
        })
    }
}

// Implement bincode traits for serialization
impl Encode for ArithmeticEncoded {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> core::result::Result<(), bincode::error::EncodeError> {
        let mut bytes = Vec::new();
        self.serialize_into(&mut bytes)
            .map_err(|e| bincode::error::EncodeError::OtherString(e))?;
        Encode::encode(&bytes, encoder)?;
        Ok(())
    }
}

impl<Context> Decode<Context> for ArithmeticEncoded {
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> core::result::Result<Self, bincode::error::DecodeError> {
        let bytes: Vec<u8> = Decode::decode(decoder)?;
        Self::deserialize_from(&bytes)
            .map_err(|e| bincode::error::DecodeError::OtherString(e.into()))
    }
}

impl<'de, Context> bincode::BorrowDecode<'de, Context> for ArithmeticEncoded {
    fn borrow_decode<D: bincode::de::BorrowDecoder<'de, Context = Context>>(
        decoder: &mut D,
    ) -> core::result::Result<Self, bincode::error::DecodeError> {
        let bytes: Vec<u8> = bincode::BorrowDecode::borrow_decode(decoder)?;
        Self::deserialize_from(&bytes)
            .map_err(|e| bincode::error::DecodeError::OtherString(e.into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arithmetic_encoding_roundtrip() {
        let values = vec![1, 2, 3, 2, 1, 5, 2, 3, 1];
        let encoded = ArithmeticEncoded::from_slice(&values).unwrap();

        // Test decoding
        let decoded = encoded.decode_all().unwrap();
        assert_eq!(values, decoded);

        // Test access
        for (i, &val) in values.iter().enumerate() {
            assert_eq!(encoded.access(i), Some(val));
        }
    }

    #[test]
    fn test_arithmetic_encoding_empty() {
        let values: Vec<u32> = vec![];
        let encoded = ArithmeticEncoded::from_slice(&values).unwrap();
        assert_eq!(encoded.len(), 0);
        assert!(encoded.is_empty());

        let decoded = encoded.decode_all().unwrap();
        assert_eq!(decoded.len(), 0);
    }

    #[test]
    fn test_arithmetic_encoding_serialization() {
        let values = vec![10, 20, 30, 20, 10, 50, 20, 30, 10];
        let encoded = ArithmeticEncoded::from_slice(&values).unwrap();

        // Serialize
        let mut bytes = Vec::new();
        encoded.serialize_into(&mut bytes).unwrap();

        // Deserialize
        let deserialized = ArithmeticEncoded::deserialize_from(&bytes).unwrap();

        // Verify
        let decoded = deserialized.decode_all().unwrap();
        assert_eq!(values, decoded);
    }

    #[test]
    fn test_arithmetic_encoding_size() {
        let values = vec![1, 2, 3, 2, 1, 5, 2, 3, 1, 2, 3, 4, 5];
        let encoded = ArithmeticEncoded::from_slice(&values).unwrap();

        let size = encoded.size_in_bytes();
        assert!(size > 0);
        println!("Encoded {} values into {} bytes", values.len(), size);
    }
}
