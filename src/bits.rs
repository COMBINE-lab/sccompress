use std::io::Cursor;

use bincode::{BorrowDecode, Decode, Encode};
use bitm::{self, BitAccess};
use tracing::error;

pub(crate) type InnerEFVector =
    cseq::elias_fano::Sequence<bitm::BinaryRankSearch, bitm::BinaryRankSearch>;
pub(crate) struct EFVector(pub InnerEFVector);

#[derive(Clone, Encode, Decode)]
pub(crate) enum HybridSparseVec {
    EF(EFVector),
    Bit(Vec<u64>),
}

#[allow(dead_code)]
struct HybridSparseIterator<'a> {
    bit_it: Option<bitm::BitBIterator<'a, true>>,
    ef_it: Option<
        cseq::elias_fano::Cursor<'a, bitm::BinaryRankSearch, bitm::BinaryRankSearch, Box<[u64]>>,
    >,
}

impl Iterator for HybridSparseIterator<'_> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(bit_it) = &mut self.bit_it {
            bit_it.next().map(|x| x as u64)
        } else if let Some(ef_it) = &mut self.ef_it {
            let r = ef_it.value();
            ef_it.advance();
            r
        } else {
            None
        }
    }
}

impl HybridSparseVec {
    pub fn tyname(&self) -> &'static str {
        match self {
            Self::EF(_) => "EFVector",
            Self::Bit(_) => "Bitvector",
        }
    }

    pub fn empty() -> Self {
        Self::Bit(Vec::new())
    }

    pub fn from_indices(inds: &[u64], _s: f64, tot: usize) -> Self {
        let ef = InnerEFVector::with_items_from_slice_s(inds);
        let nw = bitm::ceiling_div(tot, 64);
        if ef.write_bytes() < nw * 8 {
            for v in ef.iter().zip(inds.iter()) {
                assert_eq!(v.0, *v.1);
            }
            Self::EF(EFVector(ef))
        } else {
            let mut v = vec![0; nw];
            for i in inds {
                v.set_bit(*i as usize);
            }
            assert_eq!(v.count_bit_ones(), inds.len());
            Self::Bit(v)
        }
    }

    pub fn num_bits(&self) -> usize {
        match self {
            Self::Bit(b) => b.len() * 8,
            Self::EF(b) => b.0.write_bytes() * 8,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::EF(e) => e.0.len(),
            Self::Bit(e) => e.count_bit_ones(),
        }
    }

    //    pub fn
}

impl EFVector {
    #[allow(dead_code)]
    pub fn empty() -> Self {
        let e = Vec::<u64>::new();
        Self(InnerEFVector::with_items_from_slice_s(&e))
    }
}

impl Clone for EFVector {
    // Required method
    fn clone(&self) -> Self {
        let mut c = Cursor::new(Vec::new());
        if let Err(_e) = self.0.write(&mut c) {
            error!("failed to clone!");
        }
        c.set_position(0);
        let seq = InnerEFVector::read_s(&mut c).expect("works");
        Self(seq)
    }
}

impl Encode for EFVector {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> core::result::Result<(), bincode::error::EncodeError> {
        let b = self.0.write_bytes();
        let mut d = Cursor::new(Vec::<u8>::with_capacity(b));
        self.0.write(&mut d).expect("can write to cursor");
        d.set_position(0);
        Encode::encode(&d.into_inner(), encoder)
    }
}

impl<Context> Decode<Context> for EFVector {
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> core::result::Result<Self, bincode::error::DecodeError> {
        let d: Vec<u8> = Decode::decode(decoder)?;
        let mut c = Cursor::new(d);
        c.set_position(0);
        let seq = InnerEFVector::read_s(&mut c).expect("valid EF sequence");
        Ok(Self(seq))
    }
}

impl<'de, Context> BorrowDecode<'de, Context> for EFVector {
    fn borrow_decode<D: bincode::de::BorrowDecoder<'de, Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let d: Vec<u8> = BorrowDecode::borrow_decode(decoder)?;
        let mut c = Cursor::new(d);
        c.set_position(0);
        let indices = InnerEFVector::read_s(&mut c).expect("valid EF sequence");
        Ok(Self(indices))
    }
}
