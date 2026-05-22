use crate::delta_indices::DeltaEncodedIndices;
use bincode::{Decode, Encode};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SortedIndexCodec {
    Delta,
    EliasFano,
}

#[derive(Clone, Encode, Decode)]
pub enum EncodedSortedIndices {
    Delta(DeltaEncodedIndices),
    EliasFano(EliasFanoEncoded),
    Raw(Vec<u32>),
}

impl EncodedSortedIndices {
    pub fn from_u32(indices: &[u32], codec: SortedIndexCodec) -> Self {
        if indices.windows(2).all(|w| w[0] <= w[1]) {
            Self::from_sorted_u32(indices, codec)
        } else {
            EncodedSortedIndices::Raw(indices.to_vec())
        }
    }

    pub fn from_sorted_u32(indices: &[u32], codec: SortedIndexCodec) -> Self {
        match codec {
            SortedIndexCodec::Delta => {
                let indices_u64: Vec<u64> = indices.iter().map(|&x| x as u64).collect();
                EncodedSortedIndices::Delta(DeltaEncodedIndices::from_indices(&indices_u64))
            }
            SortedIndexCodec::EliasFano => {
                EncodedSortedIndices::EliasFano(EliasFanoEncoded::from_sorted_u32(indices))
            }
        }
    }

    pub fn decode_all_u32(&self) -> Vec<u32> {
        match self {
            EncodedSortedIndices::Delta(d) => {
                d.decode_all().into_iter().map(|x| x as u32).collect()
            }
            EncodedSortedIndices::EliasFano(e) => e.decode_all_u32(),
            EncodedSortedIndices::Raw(v) => v.clone(),
        }
    }

    pub fn size_in_bytes(&self) -> usize {
        match self {
            EncodedSortedIndices::Delta(d) => d.size_in_bytes(),
            EncodedSortedIndices::EliasFano(e) => e.size_in_bytes(),
            EncodedSortedIndices::Raw(v) => v.len() * 4 + 8,
        }
    }
}

#[derive(Clone, Encode, Decode)]
pub struct EliasFanoEncoded {
    count: usize,
    low_bits: u8,
    max_high: u32,
    low_parts: Vec<u32>,
    high_bits: Vec<u64>,
}

impl EliasFanoEncoded {
    pub fn from_sorted_u32(indices: &[u32]) -> Self {
        if indices.is_empty() {
            return Self {
                count: 0,
                low_bits: 0,
                max_high: 0,
                low_parts: Vec::new(),
                high_bits: Vec::new(),
            };
        }

        for i in 1..indices.len() {
            assert!(indices[i] >= indices[i - 1], "Indices must be sorted");
        }

        let count = indices.len();
        let max_value = *indices.last().unwrap() as u64;
        let universe = max_value + 1;
        let ratio = (universe / (count as u64)).max(1);
        let low_bits = (u64::BITS - 1 - ratio.leading_zeros()).min(31) as u8;

        let low_mask = if low_bits == 0 {
            0
        } else {
            (1u64 << low_bits) - 1
        };

        let mut low_parts = Vec::with_capacity(count);
        let mut max_high = 0u32;
        for &value in indices {
            let value_u64 = value as u64;
            let high = (value_u64 >> low_bits) as u32;
            max_high = max_high.max(high);
            low_parts.push((value_u64 & low_mask) as u32);
        }

        let bit_len = (max_high as usize).saturating_add(count).saturating_add(1);
        let mut high_bits = vec![0u64; bit_len.div_ceil(64)];

        for (i, &value) in indices.iter().enumerate() {
            let high = (value as u64) >> low_bits;
            let pos = high as usize + i;
            let word = pos / 64;
            let bit = pos % 64;
            high_bits[word] |= 1u64 << bit;
        }

        Self {
            count,
            low_bits,
            max_high,
            low_parts,
            high_bits,
        }
    }

    pub fn decode_all_u32(&self) -> Vec<u32> {
        if self.count == 0 {
            return Vec::new();
        }

        let mut select_positions = Vec::with_capacity(self.count);
        'outer: for (word_idx, &word_raw) in self.high_bits.iter().enumerate() {
            let mut word = word_raw;
            while word != 0 {
                let tz = word.trailing_zeros() as usize;
                let pos = word_idx * 64 + tz;
                select_positions.push(pos);
                if select_positions.len() == self.count {
                    break 'outer;
                }
                word &= word - 1;
            }
        }

        let mut decoded = Vec::with_capacity(self.count);
        for i in 0..self.count {
            let high = (select_positions[i] - i) as u64;
            let low = self.low_parts.get(i).copied().unwrap_or(0) as u64;
            let value = (high << self.low_bits) | low;
            decoded.push(value as u32);
        }
        decoded
    }

    pub fn size_in_bytes(&self) -> usize {
        self.low_parts.len() * 4 + self.high_bits.len() * 8 + 17
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elias_fano_roundtrip_sparse() {
        let input = vec![1u32, 2, 3, 10, 50, 51, 52, 10_000];
        let ef = EliasFanoEncoded::from_sorted_u32(&input);
        assert_eq!(ef.decode_all_u32(), input);
    }

    #[test]
    fn elias_fano_roundtrip_dense() {
        let input: Vec<u32> = (0..10_000u32).collect();
        let ef = EliasFanoEncoded::from_sorted_u32(&input);
        assert_eq!(ef.decode_all_u32(), input);
    }

    #[test]
    fn encoded_sorted_indices_roundtrip() {
        let input = vec![5u32, 7, 9, 10, 100, 500];
        let delta = EncodedSortedIndices::from_sorted_u32(&input, SortedIndexCodec::Delta);
        let ef = EncodedSortedIndices::from_sorted_u32(&input, SortedIndexCodec::EliasFano);
        assert_eq!(delta.decode_all_u32(), input);
        assert_eq!(ef.decode_all_u32(), input);
    }
}
