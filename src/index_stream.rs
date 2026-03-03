use crate::arith_encode::ArithmeticEncoded;
use bincode::{Decode, Encode};
use stream_vbyte::{decode::decode, encode::encode, scalar::Scalar};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexStreamCodec {
    Arithmetic,
    StreamVByte,
}

#[derive(Clone, Encode, Decode)]
pub struct StreamVByteEncoded {
    compressed: Vec<u8>,
    len: usize,
}

impl StreamVByteEncoded {
    pub fn from_slice(values: &[u32]) -> Self {
        if values.is_empty() {
            return Self {
                compressed: Vec::new(),
                len: 0,
            };
        }

        let mut encoded = vec![0u8; values.len() * 5];
        let used = encode::<Scalar>(values, &mut encoded);
        encoded.truncate(used);

        Self {
            compressed: encoded,
            len: values.len(),
        }
    }

    pub fn decode_all(&self) -> Vec<u32> {
        if self.len == 0 {
            return Vec::new();
        }

        let mut out = vec![0u32; self.len];
        let consumed = decode::<Scalar>(&self.compressed, self.len, &mut out);
        debug_assert_eq!(consumed, self.compressed.len());
        out
    }

    pub fn size_in_bytes(&self) -> usize {
        self.compressed.len() + 8
    }
}

#[derive(Clone, Encode, Decode)]
pub enum EncodedU32Stream {
    Arithmetic(ArithmeticEncoded),
    StreamVByte(StreamVByteEncoded),
}

impl EncodedU32Stream {
    pub fn from_slice(values: &[u32], codec: IndexStreamCodec) -> Self {
        match codec {
            IndexStreamCodec::Arithmetic => EncodedU32Stream::Arithmetic(
                ArithmeticEncoded::from_slice(values).expect("valid arithmetic u32 stream"),
            ),
            IndexStreamCodec::StreamVByte => {
                EncodedU32Stream::StreamVByte(StreamVByteEncoded::from_slice(values))
            }
        }
    }

    pub fn decode_all(&self) -> Vec<u32> {
        match self {
            EncodedU32Stream::Arithmetic(stream) => stream.decode_all().unwrap_or_default(),
            EncodedU32Stream::StreamVByte(stream) => stream.decode_all(),
        }
    }

    pub fn size_in_bytes(&self) -> usize {
        match self {
            EncodedU32Stream::Arithmetic(stream) => stream.size_in_bytes(),
            EncodedU32Stream::StreamVByte(stream) => stream.size_in_bytes(),
        }
    }
}
