# metal-radix

GPU-accelerated, stable radix sorting of pairs of `u32` keys on macOS, built with
[wgpu](https://github.com/gfx-rs/wgpu) and Apple's Metal backend.

`metal-radix` returns a permutation of the original input indices. Your keys and
associated data remain in place, so callers can traverse records in sorted order
without moving their payloads.

## Requirements

- macOS with a hardware Metal-capable GPU.
- A Rust toolchain compatible with the Rust 2024 edition and wgpu 29.

The current implementation uses Metal exclusively. It has no CPU fallback, and
`GpuSorter::new()` returns an error when a supported GPU cannot be initialized.

## Installation

After version `0.1.0` is published to crates.io, add:

```toml
[dependencies]
metal-radix = "0.1.0"
```

For local development before publication, use a path dependency instead:

```toml
[dependencies]
metal-radix = { path = "../metal-radix" }
```

Adjust the path to the directory containing this crate's `Cargo.toml`.

## Usage

```rust
use metal_radix::GpuSorter;

fn main() -> std::io::Result<()> {
    let sorter = GpuSorter::new()?;

    // Each pair is [primary_key, secondary_key].
    let keys = [[2, 10], [1, 20], [1, 5], [1, 5]];
    let indices = sorter.sort_indices(&keys)?;

    // Equal key pairs retain their original order.
    assert_eq!(indices, vec![2, 3, 1, 0]);

    let sorted_keys: Vec<_> = indices
        .iter()
        .map(|&index| keys[index as usize])
        .collect();
    assert_eq!(sorted_keys, vec![[1, 5], [1, 5], [1, 20], [2, 10]]);

    println!("{}", sorter.description());
    Ok(())
}
```

To access associated records, use the returned indices to index your original
record collection in the same way. Reuse the sorter across batches to reuse the
GPU device and compiled pipelines.

## API

| Method | Behavior |
| --- | --- |
| `GpuSorter::new()` | Initializes the GPU and pipelines; returns `std::io::Result<GpuSorter>`. |
| `sorter.sort_indices(&keys)` | Returns `std::io::Result<Vec<u32>>` containing original indices in sorted order. |
| `sorter.description()` | Returns a description of the selected GPU and backend as `&str`. |

## Sorting behavior

- Ordering is ascending: first by the primary key, then by the secondary key.
- Sorting is stable: equal pairs retain their input order.
- The input is not modified; empty input returns an empty vector.
- Both keys support the full `u32` range.
- Calls are synchronous and return after GPU readback and result validation.
- A sorter can be shared across threads; an internal mutex serializes GPU work.

This API sorts numeric key pairs. For strings, assign numeric ranks that preserve
your desired string ordering before calling the sorter.

## Batch limits and performance

The number of keys must fit in `u32`. Each batch must also fit the device's buffer
and dispatch limits; exceeding those limits returns an error. Automatic batching
and merging are not provided.

The implementation uploads packed keys and original indices, performs radix-sort
passes, and reads the result back for validation. GPU buffers are allocated for
each batch. Setup, transfer, allocation, and validation costs all contribute to
execution time, so benchmark representative workloads against CPU sorting before
choosing a backend.

## Development

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

The documentation example is compiled by `cargo test`; it is marked `no_run`
because executing it requires a hardware Metal GPU.

## License

Licensed under the [MIT License](LICENSE).
