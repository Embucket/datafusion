// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! File Compression type abstraction

use std::io::Read;
use std::str::FromStr;

use datafusion_common::error::{DataFusionError, Result};

use datafusion_common::GetExt;
use datafusion_common::parsers::CompressionTypeVariant::{self, *};

#[cfg(feature = "compression")]
use async_compression::tokio::bufread::{
    BrotliDecoder as AsyncBrotliDecoder, BrotliEncoder as AsyncBrotliEncoder,
    BzDecoder as AsyncBzDecoder, BzEncoder as AsyncBzEncoder,
    DeflateDecoder as AsyncDeflateDecoder, DeflateEncoder as AsyncDeflateEncoder,
    GzipDecoder as AsyncGzDecoder, GzipEncoder as AsyncGzEncoder,
    XzDecoder as AsyncXzDecoder, XzEncoder as AsyncXzEncoder,
    ZlibDecoder as AsyncZlibDecoder, ZlibEncoder as AsyncZlibEncoder,
    ZstdDecoder as AsyncZstdDecoer, ZstdEncoder as AsyncZstdEncoder,
};

#[cfg(feature = "compression")]
use async_compression::tokio::write::{
    BrotliEncoder, BzEncoder, DeflateEncoder, GzipEncoder, XzEncoder, ZlibEncoder,
    ZstdEncoder,
};
#[cfg(feature = "compression")]
use brotli::Decompressor as BrotliReader;
use bytes::Bytes;
#[cfg(feature = "compression")]
use bzip2::read::MultiBzDecoder;
#[cfg(feature = "compression")]
use flate2::read::{DeflateDecoder, MultiGzDecoder, ZlibDecoder};
use futures::StreamExt;
#[cfg(feature = "compression")]
use futures::TryStreamExt;
use futures::stream::BoxStream;
#[cfg(feature = "compression")]
use liblzma::read::XzDecoder;
use object_store::buffered::BufWriter;
use tokio::io::AsyncWrite;
#[cfg(feature = "compression")]
use tokio_util::io::{ReaderStream, StreamReader};
#[cfg(feature = "compression")]
use zstd::Decoder as ZstdDecoder;

/// Readable file compression type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileCompressionType {
    variant: CompressionTypeVariant,
}

impl GetExt for FileCompressionType {
    fn get_ext(&self) -> String {
        match self.variant {
            GZIP => ".gz".to_owned(),
            BZIP2 => ".bz2".to_owned(),
            XZ => ".xz".to_owned(),
            ZSTD => ".zst".to_owned(),
            UNCOMPRESSED => "".to_owned(),
            AUTO => "".to_owned(),
            BROTLI => ".br".to_owned(),
            DEFLATE => ".zlib".to_owned(),
            RAW_DEFLATE => ".raw_deflate".to_owned(),
        }
    }
}

impl From<CompressionTypeVariant> for FileCompressionType {
    fn from(t: CompressionTypeVariant) -> Self {
        Self { variant: t }
    }
}

impl From<FileCompressionType> for CompressionTypeVariant {
    fn from(t: FileCompressionType) -> Self {
        t.variant
    }
}

impl FromStr for FileCompressionType {
    type Err = DataFusionError;

    fn from_str(s: &str) -> Result<Self> {
        let variant = CompressionTypeVariant::from_str(s).map_err(|_| {
            DataFusionError::NotImplemented(format!("Unknown FileCompressionType: {s}"))
        })?;
        Ok(Self { variant })
    }
}

/// `FileCompressionType` implementation
impl FileCompressionType {
    /// Whether this build can decode compressed input or use AUTO detection.
    pub const fn compression_enabled() -> bool {
        cfg!(feature = "compression")
    }

    /// Gzip-ed file
    pub const GZIP: Self = Self { variant: GZIP };

    /// Bzip2-ed file
    pub const BZIP2: Self = Self { variant: BZIP2 };

    /// Xz-ed file (liblzma)
    pub const XZ: Self = Self { variant: XZ };

    /// Zstd-ed file
    pub const ZSTD: Self = Self { variant: ZSTD };

    /// Uncompressed file
    pub const UNCOMPRESSED: Self = Self {
        variant: UNCOMPRESSED,
    };

    /// Detect the compression type from the file header while reading
    pub const AUTO: Self = Self { variant: AUTO };

    /// Brotli-compressed file
    pub const BROTLI: Self = Self { variant: BROTLI };

    /// Deflate-compressed file with a zlib header
    pub const DEFLATE: Self = Self { variant: DEFLATE };

    /// Deflate-compressed file without a zlib header
    pub const RAW_DEFLATE: Self = Self {
        variant: RAW_DEFLATE,
    };

    /// Read only access to self.variant
    pub fn get_variant(&self) -> &CompressionTypeVariant {
        &self.variant
    }

    /// The file is compressed or not
    pub const fn is_compressed(&self) -> bool {
        self.variant.is_compressed()
    }

    /// Given a `Stream`, create a `Stream` which data are compressed with `FileCompressionType`.
    pub fn convert_to_compress_stream<'a>(
        &self,
        s: BoxStream<'a, Result<Bytes>>,
    ) -> Result<BoxStream<'a, Result<Bytes>>> {
        Ok(match self.variant {
            #[cfg(feature = "compression")]
            GZIP => ReaderStream::new(AsyncGzEncoder::new(StreamReader::new(s)))
                .map_err(DataFusionError::from)
                .boxed(),
            #[cfg(feature = "compression")]
            BZIP2 => ReaderStream::new(AsyncBzEncoder::new(StreamReader::new(s)))
                .map_err(DataFusionError::from)
                .boxed(),
            #[cfg(feature = "compression")]
            XZ => ReaderStream::new(AsyncXzEncoder::new(StreamReader::new(s)))
                .map_err(DataFusionError::from)
                .boxed(),
            #[cfg(feature = "compression")]
            ZSTD => ReaderStream::new(AsyncZstdEncoder::new(StreamReader::new(s)))
                .map_err(DataFusionError::from)
                .boxed(),
            #[cfg(feature = "compression")]
            BROTLI => ReaderStream::new(AsyncBrotliEncoder::new(StreamReader::new(s)))
                .map_err(DataFusionError::from)
                .boxed(),
            #[cfg(feature = "compression")]
            DEFLATE => ReaderStream::new(AsyncZlibEncoder::new(StreamReader::new(s)))
                .map_err(DataFusionError::from)
                .boxed(),
            #[cfg(feature = "compression")]
            RAW_DEFLATE => {
                ReaderStream::new(AsyncDeflateEncoder::new(StreamReader::new(s)))
                    .map_err(DataFusionError::from)
                    .boxed()
            }
            #[cfg(not(feature = "compression"))]
            GZIP | BZIP2 | XZ | ZSTD | BROTLI | DEFLATE | RAW_DEFLATE => {
                return Err(DataFusionError::NotImplemented(
                    "Compression feature is not enabled".to_owned(),
                ));
            }
            UNCOMPRESSED => s.boxed(),
            AUTO => {
                return Err(DataFusionError::NotImplemented(
                    "AUTO compression is only supported for reading".to_owned(),
                ));
            }
        })
    }

    /// Wrap the given `BufWriter` so that it performs compressed writes
    /// according to this `FileCompressionType` using the default compression level.
    pub fn convert_async_writer(
        &self,
        w: BufWriter,
    ) -> Result<Box<dyn AsyncWrite + Send + Unpin>> {
        self.convert_async_writer_with_level(w, None)
    }

    /// Wrap the given `BufWriter` so that it performs compressed writes
    /// according to this `FileCompressionType`.
    ///
    /// If `compression_level` is `Some`, the encoder will use the specified
    /// compression level. If `None`, the default level for each algorithm is used.
    pub fn convert_async_writer_with_level(
        &self,
        w: BufWriter,
        compression_level: Option<u32>,
    ) -> Result<Box<dyn AsyncWrite + Send + Unpin>> {
        #[cfg(feature = "compression")]
        use async_compression::Level;

        Ok(match self.variant {
            #[cfg(feature = "compression")]
            GZIP => match compression_level {
                Some(level) => {
                    Box::new(GzipEncoder::with_quality(w, Level::Precise(level as i32)))
                }
                None => Box::new(GzipEncoder::new(w)),
            },
            #[cfg(feature = "compression")]
            BZIP2 => match compression_level {
                Some(level) => {
                    Box::new(BzEncoder::with_quality(w, Level::Precise(level as i32)))
                }
                None => Box::new(BzEncoder::new(w)),
            },
            #[cfg(feature = "compression")]
            XZ => match compression_level {
                Some(level) => {
                    Box::new(XzEncoder::with_quality(w, Level::Precise(level as i32)))
                }
                None => Box::new(XzEncoder::new(w)),
            },
            #[cfg(feature = "compression")]
            ZSTD => match compression_level {
                Some(level) => {
                    Box::new(ZstdEncoder::with_quality(w, Level::Precise(level as i32)))
                }
                None => Box::new(ZstdEncoder::new(w)),
            },
            #[cfg(feature = "compression")]
            BROTLI => match compression_level {
                Some(level) => {
                    Box::new(BrotliEncoder::with_quality(w, Level::Precise(level as i32)))
                }
                None => Box::new(BrotliEncoder::new(w)),
            },
            #[cfg(feature = "compression")]
            DEFLATE => match compression_level {
                Some(level) => {
                    Box::new(ZlibEncoder::with_quality(w, Level::Precise(level as i32)))
                }
                None => Box::new(ZlibEncoder::new(w)),
            },
            #[cfg(feature = "compression")]
            RAW_DEFLATE => match compression_level {
                Some(level) => Box::new(DeflateEncoder::with_quality(
                    w,
                    Level::Precise(level as i32),
                )),
                None => Box::new(DeflateEncoder::new(w)),
            },
            #[cfg(not(feature = "compression"))]
            GZIP | BZIP2 | XZ | ZSTD | BROTLI | DEFLATE | RAW_DEFLATE => {
                // compression_level is not used when compression feature is disabled
                let _ = compression_level;
                return Err(DataFusionError::NotImplemented(
                    "Compression feature is not enabled".to_owned(),
                ));
            }
            UNCOMPRESSED => Box::new(w),
            AUTO => {
                return Err(DataFusionError::NotImplemented(
                    "AUTO compression is only supported for reading".to_owned(),
                ));
            }
        })
    }

    /// Given a `Stream`, create a `Stream` which data are decompressed with `FileCompressionType`.
    pub fn convert_stream<'a>(
        &self,
        s: BoxStream<'a, Result<Bytes>>,
    ) -> Result<BoxStream<'a, Result<Bytes>>> {
        Ok(match self.variant {
            #[cfg(feature = "compression")]
            GZIP => {
                let mut decoder = AsyncGzDecoder::new(StreamReader::new(s));
                decoder.multiple_members(true);

                ReaderStream::new(decoder)
                    .map_err(DataFusionError::from)
                    .boxed()
            }
            #[cfg(feature = "compression")]
            BZIP2 => ReaderStream::new(AsyncBzDecoder::new(StreamReader::new(s)))
                .map_err(DataFusionError::from)
                .boxed(),
            #[cfg(feature = "compression")]
            XZ => ReaderStream::new(AsyncXzDecoder::new(StreamReader::new(s)))
                .map_err(DataFusionError::from)
                .boxed(),
            #[cfg(feature = "compression")]
            ZSTD => ReaderStream::new(AsyncZstdDecoer::new(StreamReader::new(s)))
                .map_err(DataFusionError::from)
                .boxed(),
            #[cfg(feature = "compression")]
            BROTLI => ReaderStream::new(AsyncBrotliDecoder::new(StreamReader::new(s)))
                .map_err(DataFusionError::from)
                .boxed(),
            #[cfg(feature = "compression")]
            DEFLATE => ReaderStream::new(AsyncZlibDecoder::new(StreamReader::new(s)))
                .map_err(DataFusionError::from)
                .boxed(),
            #[cfg(feature = "compression")]
            RAW_DEFLATE => {
                ReaderStream::new(AsyncDeflateDecoder::new(StreamReader::new(s)))
                    .map_err(DataFusionError::from)
                    .boxed()
            }
            #[cfg(not(feature = "compression"))]
            GZIP | BZIP2 | XZ | ZSTD | BROTLI | DEFLATE | RAW_DEFLATE => {
                return Err(DataFusionError::NotImplemented(
                    "Compression feature is not enabled".to_owned(),
                ));
            }
            UNCOMPRESSED => s.boxed(),
            #[cfg(feature = "compression")]
            AUTO => futures::stream::once(async move {
                let mut source = s;
                let mut header = [0_u8; 6];
                let mut header_len = 0;
                let mut initial_chunks = Vec::new();
                while header_len < header.len() {
                    match source.next().await {
                        Some(Ok(bytes)) if bytes.is_empty() => continue,
                        Some(Ok(bytes)) => {
                            let copied = (header.len() - header_len).min(bytes.len());
                            header[header_len..header_len + copied]
                                .copy_from_slice(&bytes[..copied]);
                            header_len += copied;
                            initial_chunks.push(Ok(bytes));
                        }
                        Some(Err(error)) => return Err(error),
                        None => break,
                    }
                }
                let replay = futures::stream::iter(initial_chunks).chain(source).boxed();
                Self::detect_from_header(&header[..header_len]).convert_stream(replay)
            })
            .try_flatten()
            .boxed(),
            #[cfg(not(feature = "compression"))]
            AUTO => {
                return Err(DataFusionError::NotImplemented(
                    "Compression feature is not enabled".to_owned(),
                ));
            }
        })
    }

    /// Given a `Read`, create a `Read` which data are decompressed with `FileCompressionType`.
    pub fn convert_read<T: Read + Send + 'static>(
        &self,
        r: T,
    ) -> Result<Box<dyn Read + Send>> {
        Ok(match self.variant {
            #[cfg(feature = "compression")]
            GZIP => Box::new(MultiGzDecoder::new(r)),
            #[cfg(feature = "compression")]
            BZIP2 => Box::new(MultiBzDecoder::new(r)),
            #[cfg(feature = "compression")]
            XZ => Box::new(XzDecoder::new_multi_decoder(r)),
            #[cfg(feature = "compression")]
            ZSTD => match ZstdDecoder::new(r) {
                Ok(decoder) => Box::new(decoder),
                Err(e) => return Err(DataFusionError::External(Box::new(e))),
            },
            #[cfg(feature = "compression")]
            BROTLI => Box::new(BrotliReader::new(r, 4096)),
            #[cfg(feature = "compression")]
            DEFLATE => Box::new(ZlibDecoder::new(r)),
            #[cfg(feature = "compression")]
            RAW_DEFLATE => Box::new(DeflateDecoder::new(r)),
            #[cfg(not(feature = "compression"))]
            GZIP | BZIP2 | XZ | ZSTD | BROTLI | DEFLATE | RAW_DEFLATE => {
                return Err(DataFusionError::NotImplemented(
                    "Compression feature is not enabled".to_owned(),
                ));
            }
            UNCOMPRESSED => Box::new(r),
            #[cfg(feature = "compression")]
            AUTO => {
                let mut source = r;
                let mut header = [0_u8; 6];
                let mut header_len = 0;
                while header_len < header.len() {
                    let read = source.read(&mut header[header_len..])?;
                    if read == 0 {
                        break;
                    }
                    header_len += read;
                }
                let replay: Box<dyn Read + Send> = Box::new(
                    std::io::Cursor::new(header[..header_len].to_vec()).chain(source),
                );
                Self::detect_from_header(&header[..header_len]).convert_read(replay)?
            }
            #[cfg(not(feature = "compression"))]
            AUTO => {
                return Err(DataFusionError::NotImplemented(
                    "Compression feature is not enabled".to_owned(),
                ));
            }
        })
    }
}

impl FileCompressionType {
    /// Identify codecs supported by AUTO from up to the first six file bytes.
    /// Headerless codecs such as Brotli and raw Deflate require explicit selection.
    pub fn detect_from_header(header: &[u8]) -> Self {
        if header.starts_with(&[0x1f, 0x8b]) {
            Self::GZIP
        } else if header.starts_with(b"BZh") {
            Self::BZIP2
        } else if header.starts_with(&[0x28, 0xb5, 0x2f, 0xfd]) {
            Self::ZSTD
        } else if header.starts_with(&[0xfd, b'7', b'z', b'X', b'Z', 0x00]) {
            Self::XZ
        } else if header.len() >= 2
            && header[0] & 0x0f == 8
            && header[0] >> 4 <= 7
            && u16::from_be_bytes([header[0], header[1]]).is_multiple_of(31)
        {
            Self::DEFLATE
        } else {
            Self::UNCOMPRESSED
        }
    }
}

/// Trait for extending the functionality of the `FileType` enum.
pub trait FileTypeExt {
    /// Given a `FileCompressionType`, return the `FileType`'s extension with compression suffix
    fn get_ext_with_compression(&self, c: FileCompressionType) -> Result<String>;
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use super::FileCompressionType;
    use datafusion_common::error::DataFusionError;

    use bytes::Bytes;
    use futures::StreamExt;

    #[test]
    fn from_str() {
        for (ext, compression_type) in [
            ("gz", FileCompressionType::GZIP),
            ("GZ", FileCompressionType::GZIP),
            ("gzip", FileCompressionType::GZIP),
            ("GZIP", FileCompressionType::GZIP),
            ("xz", FileCompressionType::XZ),
            ("XZ", FileCompressionType::XZ),
            ("bz2", FileCompressionType::BZIP2),
            ("BZ2", FileCompressionType::BZIP2),
            ("bzip2", FileCompressionType::BZIP2),
            ("BZIP2", FileCompressionType::BZIP2),
            ("zst", FileCompressionType::ZSTD),
            ("ZST", FileCompressionType::ZSTD),
            ("zstd", FileCompressionType::ZSTD),
            ("ZSTD", FileCompressionType::ZSTD),
            ("AUTO", FileCompressionType::AUTO),
            ("BROTLI", FileCompressionType::BROTLI),
            ("BR", FileCompressionType::BROTLI),
            ("DEFLATE", FileCompressionType::DEFLATE),
            ("ZLIB", FileCompressionType::DEFLATE),
            ("RAW_DEFLATE", FileCompressionType::RAW_DEFLATE),
            ("", FileCompressionType::UNCOMPRESSED),
        ] {
            assert_eq!(
                FileCompressionType::from_str(ext).unwrap(),
                compression_type
            );
        }

        assert!(matches!(
            FileCompressionType::from_str("Unknown"),
            Err(DataFusionError::NotImplemented(_))
        ));
    }

    #[cfg(feature = "compression")]
    #[tokio::test]
    async fn auto_detects_compression_across_stream_chunks() -> Result<(), DataFusionError>
    {
        use futures::TryStreamExt;
        use std::io::Read;

        let plain = b"1,alice\n2,bob\n";
        for codec in [
            FileCompressionType::UNCOMPRESSED,
            FileCompressionType::GZIP,
            FileCompressionType::BZIP2,
            FileCompressionType::ZSTD,
            FileCompressionType::DEFLATE,
            FileCompressionType::XZ,
        ] {
            let input = futures::stream::once(async {
                Ok::<Bytes, DataFusionError>(Bytes::from_static(plain))
            });
            let encoded = codec
                .convert_to_compress_stream(input.boxed())?
                .try_collect::<Vec<_>>()
                .await?
                .concat();
            let chunks = std::iter::repeat_with(|| Ok(Bytes::new()))
                .take(100)
                .chain(
                    encoded
                        .chunks(1)
                        .map(|chunk| Ok(Bytes::copy_from_slice(chunk))),
                )
                .collect::<Vec<Result<Bytes, DataFusionError>>>();
            let decoded = FileCompressionType::AUTO
                .convert_stream(futures::stream::iter(chunks).boxed())?
                .try_collect::<Vec<_>>()
                .await?
                .concat();
            assert_eq!(decoded, plain, "stream decoder for {codec:?}");

            let mut reader =
                FileCompressionType::AUTO.convert_read(std::io::Cursor::new(encoded))?;
            let mut decoded = Vec::new();
            reader.read_to_end(&mut decoded)?;
            assert_eq!(decoded, plain, "sync decoder for {codec:?}");
        }
        Ok(())
    }

    #[cfg(feature = "compression")]
    #[tokio::test]
    async fn explicit_brotli_and_deflate_roundtrip() -> Result<(), DataFusionError> {
        use futures::TryStreamExt;
        use std::io::Read;

        let plain = b"1,alice\n2,bob\n";
        for codec in [
            FileCompressionType::BROTLI,
            FileCompressionType::DEFLATE,
            FileCompressionType::RAW_DEFLATE,
        ] {
            let input = futures::stream::once(async {
                Ok::<Bytes, DataFusionError>(Bytes::from_static(plain))
            });
            let encoded = codec
                .convert_to_compress_stream(input.boxed())?
                .try_collect::<Vec<_>>()
                .await?
                .concat();
            let chunks = encoded
                .chunks(3)
                .map(|chunk| Ok(Bytes::copy_from_slice(chunk)))
                .collect::<Vec<Result<Bytes, DataFusionError>>>();
            let decoded = codec
                .convert_stream(futures::stream::iter(chunks).boxed())?
                .try_collect::<Vec<_>>()
                .await?
                .concat();
            assert_eq!(decoded, plain, "stream decoder for {codec:?}");

            let mut reader = codec.convert_read(std::io::Cursor::new(encoded))?;
            let mut decoded = Vec::new();
            reader.read_to_end(&mut decoded)?;
            assert_eq!(decoded, plain, "sync decoder for {codec:?}");
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_bgzip_stream_decoding() -> Result<(), DataFusionError> {
        // As described in https://samtools.github.io/hts-specs/SAMv1.pdf ("The BGZF compression format")

        // Ignore rust formatting so the byte array is easier to read
        #[rustfmt::skip]
        let data = [
            // Block header
            0x1f, 0x8b, 0x08, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0x06, 0x00, 0x42, 0x43,
            0x02, 0x00,
            // Block 0, literal: 42
            0x1e, 0x00, 0x33, 0x31, 0xe2, 0x02, 0x00, 0x31, 0x29, 0x86, 0xd1, 0x03, 0x00, 0x00, 0x00,
            // Block header
            0x1f, 0x8b, 0x08, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0x06, 0x00, 0x42, 0x43,
            0x02, 0x00,
            // Block 1, literal: 42
            0x1e, 0x00, 0x33, 0x31, 0xe2, 0x02, 0x00, 0x31, 0x29, 0x86, 0xd1, 0x03, 0x00, 0x00, 0x00,
            // EOF
            0x1f, 0x8b, 0x08, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0x06, 0x00, 0x42, 0x43,
            0x02, 0x00, 0x1b, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];

        // Create a byte stream
        let stream = futures::stream::iter(vec![Ok::<Bytes, DataFusionError>(
            Bytes::from(data.to_vec()),
        )]);
        let converted_stream =
            FileCompressionType::GZIP.convert_stream(stream.boxed())?;

        let vec = converted_stream
            .map(|r| r.unwrap())
            .collect::<Vec<Bytes>>()
            .await;

        let string_value = String::from_utf8_lossy(&vec[0]);

        assert_eq!(string_value, "42\n42\n");

        Ok(())
    }
}
