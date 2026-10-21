//! GPU radix sorting of pairs of `u32` keys on macOS using Metal.
//!
//! Returns sorted indices without modifying the input.

mod gpu;

pub use gpu::GpuSorter;
