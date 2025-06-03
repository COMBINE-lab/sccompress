use bincode::{BorrowDecode, Decode, Encode};
use sux::prelude::BitFieldVec;
use sux::traits::bit_field_slice::{BitFieldSliceMut};

#[derive(Clone)]
pub struct BitField {
    pub bit_field: BitFieldVec,
}

impl BitField {
    pub fn new(bit_field: BitFieldVec) -> Self {
        Self { bit_field }
    }
}

impl Encode for BitField {
    fn encode<E: bincode::enc::Encoder>(&self, encoder: &mut E) -> Result<(), bincode::error::EncodeError> {
        let (data, width, len) = self.bit_field.clone().into_raw_parts();
        Encode::encode(&data, encoder)?;
        Encode::encode(&width, encoder)?;
        Encode::encode(&len, encoder)?;
        Ok(())
    }
}

impl<Context> Decode<Context> for BitField {
    fn decode<D: bincode::de::Decoder<Context = Context>>(decoder: &mut D) -> Result<Self, bincode::error::DecodeError> {
        let data: Vec<u64> = Decode::decode(decoder)?;
        let width = Decode::decode(decoder)?;
        let len = Decode::decode(decoder)?;
        let mut bit_field = BitFieldVec::new(width, len);
        for i in 0..len {
            bit_field.set(i, data[i] as usize);
        }
        Ok(BitField::new(bit_field))
    }
}

#[derive(Clone)]
pub struct BitFieldQuadTree {
    pub boundary: Rect,
    pub medians: Vec<u16>,
    pub indexes: Vec<usize>,
    pub data: Vec<BitField>,
    pub divided: bool,
    pub nw: Option<Box<BitFieldQuadTree>>,
    pub ne: Option<Box<BitFieldQuadTree>>,
    pub se: Option<Box<BitFieldQuadTree>>,
    pub sw: Option<Box<BitFieldQuadTree>>,
    pub positions: Vec<DatalessPoint>,
}

impl Encode for BitFieldQuadTree {
    fn encode<E: bincode::enc::Encoder>(&self, encoder: &mut E) -> Result<(), bincode::error::EncodeError> {
        Encode::encode(&self.boundary, encoder)?;
        Encode::encode(&self.medians, encoder)?;
        Encode::encode(&self.indexes, encoder)?;
        Encode::encode(&self.data, encoder)?;
        Encode::encode(&self.divided, encoder)?;
        Encode::encode(&self.nw, encoder)?;
        Encode::encode(&self.ne, encoder)?;
        Encode::encode(&self.se, encoder)?;
        Encode::encode(&self.sw, encoder)?;
        Encode::encode(&self.positions, encoder)?;
        Ok(())
    }
}

impl<Context> Decode<Context> for BitFieldQuadTree {
    fn decode<D: bincode::de::Decoder<Context = Context>>(decoder: &mut D) -> Result<Self, bincode::error::DecodeError> {
        Ok(Self {
            boundary: Decode::decode(decoder)?,
            medians: Decode::decode(decoder)?,
            indexes: Decode::decode(decoder)?,
            data: Decode::decode(decoder)?,
            divided: Decode::decode(decoder)?,
            nw: Decode::decode(decoder)?,
            ne: Decode::decode(decoder)?,
            se: Decode::decode(decoder)?,
            sw: Decode::decode(decoder)?,
            positions: Decode::decode(decoder)?,
        })
    }
}

#[derive(Debug, Clone, Encode, Decode)]
pub struct Rect {
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
    pub west_edge: f32,
    pub east_edge: f32,
    pub north_edge: f32,
    pub south_edge: f32,
}

#[derive(Debug, Clone, Encode, Decode)]
pub struct DatalessPoint {
    pub x: f32,
    pub y: f32,
} 