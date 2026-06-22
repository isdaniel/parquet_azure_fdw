#![forbid(unsafe_code)]
pub mod multifile;
pub mod reader;
pub mod writer;
pub use reader::{open_local_stream, open_stream, ParquetReadOptions};
pub use writer::{Compression, ParquetBatchWriter};
