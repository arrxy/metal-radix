//! Synchronous GPU radix sorting of unsigned key pairs using Metal.

use std::future::Future;
use std::io;
use std::sync::{Arc, Mutex, mpsc};
use std::task::{Context, Poll, Wake, Waker};
use std::thread;
use std::time::Duration;

use wgpu::util::DeviceExt;

const THREADS: u32 = 256;
type NumericRecord = [u32; 4]; // primary key, secondary key, original index, padding

/// A reusable sorter for pairs of `u32` keys on a hardware Metal GPU.
///
/// Create one sorter and reuse it across batches to reuse the device and
/// pipelines. Concurrent calls are serialized by an internal mutex.
/// GPU buffers are allocated separately for each batch.
pub struct GpuSorter {
    context: Mutex<GpuContext>,
    description: String,
}

impl GpuSorter {
    /// Initializes the Metal device and compiles the radix-sort pipelines.
    ///
    /// Returns an error on unsupported platforms, when no hardware Metal
    /// adapter is available, or when device/pipeline creation fails.
    pub fn new() -> io::Result<Self> {
        let context = GpuContext::new()?;
        let description = format!(
            "gpu ({} via Metal, radix sort)",
            context.device.adapter_info().name
        );
        Ok(Self {
            context: Mutex::new(context),
            description,
        })
    }

    /// Returns original indices in ascending lexicographic key order.
    ///
    /// `keys[i][0]` is the primary key and `keys[i][1]` the secondary key.
    /// Equal pairs retain their input order. The input remains unchanged.
    /// This call blocks until the GPU result has been read back and validated.
    ///
    /// Empty input returns an empty vector. The number of keys must fit in
    /// `u32`, and the packed batch must fit the device's buffer/dispatch limits.
    /// There is no CPU fallback or automatic batching.
    ///
    /// ```no_run
    /// # fn main() -> std::io::Result<()> {
    /// use metal_radix::GpuSorter;
    ///
    /// let sorter = GpuSorter::new()?;
    /// let indices = sorter.sort_indices(&[[2, 10], [1, 20], [1, 5]])?;
    /// assert_eq!(indices, vec![2, 1, 0]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn sort_indices(&self, keys: &[[u32; 2]]) -> io::Result<Vec<u32>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        u32::try_from(keys.len()).map_err(|_| invalid("GPU batch has more than u32::MAX keys"))?;

        let records: Vec<NumericRecord> = keys
            .iter()
            .enumerate()
            .map(|(index, &[primary, secondary])| [primary, secondary, index as u32, 0])
            .collect();
        let context = self
            .context
            .lock()
            .map_err(|_| io::Error::other("GPU sorter mutex poisoned"))?;
        context.sort_indices(&records)
    }

    /// Describes the selected GPU and backend.
    pub fn description(&self) -> &str {
        &self.description
    }
}

struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    local: wgpu::ComputePipeline,
    prefix: wgpu::ComputePipeline,
    scatter: wgpu::ComputePipeline,
}

impl GpuContext {
    fn new() -> io::Result<Self> {
        if !cfg!(target_os = "macos") {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "metal-radix requires macOS and a Metal-capable GPU",
            ));
        }
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::METAL,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .map_err(|e| io::Error::other(format!("No Metal GPU adapter: {e}")))?;
        if adapter.get_info().device_type == wgpu::DeviceType::Cpu {
            return Err(io::Error::other("GPU sorting requires a hardware GPU"));
        }
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Key-pair radix sorter"),
            ..Default::default()
        }))
        .map_err(|e| io::Error::other(format!("Could not initialize GPU: {e}")))?;
        let errors = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let entries: Vec<_> = (0..6)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 5 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding == 0,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Radix buffers"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Radix layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Key-pair radix shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("radix.wgsl").into()),
        });
        let pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let local = pipeline("local_sort");
        let prefix = pipeline("prefix");
        let scatter = pipeline("scatter");
        if let Some(error) = block_on(errors.pop()) {
            return Err(io::Error::other(format!(
                "GPU radix pipeline validation failed: {error}"
            )));
        }
        Ok(Self {
            device,
            queue,
            layout,
            local,
            prefix,
            scatter,
        })
    }

    fn sort_indices(&self, records: &[NumericRecord]) -> io::Result<Vec<u32>> {
        if records.is_empty() {
            return Ok(Vec::new());
        }
        let count = u32::try_from(records.len()).map_err(|_| invalid("GPU batch too large"))?;
        let groups = count.div_ceil(THREADS);
        let bytes = size_of_val(records) as u64;
        let limits = self.device.limits();
        if bytes > u64::from(limits.max_storage_buffer_binding_size)
            || bytes > limits.max_buffer_size
            || groups > limits.max_compute_workgroups_per_dimension
        {
            return Err(invalid(
                "GPU batch exceeds device buffer or dispatch limits; use a smaller batch",
            ));
        }
        let a = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Numeric sort keys"),
                contents: bytemuck::cast_slice(records),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let storage = |label, size| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let b = storage("Radix output", bytes);
        let local = storage("Locally sorted records", bytes);
        let histogram = storage("Radix bucket counts and starts", u64::from(groups) * 16 * 8);
        let offsets = storage("Radix bucket offsets", u64::from(groups) * 16 * 4);
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sorted key readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Key-pair radix sort"),
            });
        let mut input = &a;
        let mut output = &b;
        // LSD order: sort the secondary key first, then the primary key.
        // Leading zero nibbles need no pass. Always execute one pass per key.
        for component in [1, 0] {
            let maximum = records.iter().map(|r| r[component]).max().unwrap();
            let bits = (32 - maximum.leading_zeros()).max(1);
            for shift in (0..bits).step_by(4) {
                let params = self
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Radix pass parameters"),
                        contents: bytemuck::cast_slice(&[count, groups, shift, component as u32]),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
                let buffers = [input, output, &local, &histogram, &offsets, &params];
                let entries: Vec<_> = buffers
                    .iter()
                    .enumerate()
                    .map(|(binding, buffer)| wgpu::BindGroupEntry {
                        binding: binding as u32,
                        resource: buffer.as_entire_binding(),
                    })
                    .collect();
                let bindings = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Radix pass buffers"),
                    layout: &self.layout,
                    entries: &entries,
                });
                for (pipeline, workgroups) in [
                    (&self.local, groups),
                    (&self.prefix, 1),
                    (&self.scatter, groups),
                ] {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Radix stage"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, &bindings, &[]);
                    pass.dispatch_workgroups(workgroups, 1, 1);
                }
                std::mem::swap(&mut input, &mut output);
            }
        }
        encoder.copy_buffer_to_buffer(input, 0, &readback, 0, bytes);
        let submission = self.queue.submit([encoder.finish()]);
        let slice = readback.slice(..);
        let (sender, receiver) = mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(Duration::from_secs(60)),
            })
            .map_err(|e| io::Error::other(format!("GPU sort wait failed: {e}")))?;
        receiver
            .recv_timeout(Duration::from_secs(5))
            .map_err(|e| io::Error::other(format!("GPU readback callback failed: {e}")))?
            .map_err(|e| io::Error::other(format!("GPU readback failed: {e}")))?;
        let view = slice.get_mapped_range();
        let sorted: &[NumericRecord] = bytemuck::cast_slice(&view);
        // Validate ordering and that no record was lost, duplicated or changed
        // before returning the permutation to the caller.
        let mut seen = vec![false; records.len()];
        let mut indices = Vec::with_capacity(records.len());
        let mut previous = None;
        for record in sorted {
            let old = record[2] as usize;
            let key = (record[0], record[1]);
            if old >= records.len()
                || seen[old]
                || *record != records[old]
                || previous.is_some_and(|p| p > key)
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "GPU returned an invalid sort order or permutation",
                ));
            }
            previous = Some(key);
            seen[old] = true;
            indices.push(record[2]);
        }
        drop(view);
        readback.unmap();
        Ok(indices)
    }
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

// Native adapter/device creation returns futures; this small executor needs
// no async runtime and parks until the future's waker signals readiness.
fn block_on<F: Future>(future: F) -> F::Output {
    struct ThreadWake(thread::Thread);
    impl Wake for ThreadWake {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = Waker::from(Arc::new(ThreadWake(thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => thread::park(),
        }
    }
}
