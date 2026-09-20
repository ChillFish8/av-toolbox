#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, doc(auto_cfg))]

//! Accelerator and device management for av-toolbox.
//!
//! An [Accelerator] names a backend, such as CUDA or Vulkan. A [Device] names the hardware
//! inside that backend. The two are chosen separately because most backends expose more than
//! one device.
//!
//! The codegen stack limit and the compilation cache must both be set before the first client is
//! opened. Opening a client fixes CubeCL's global config, and a thread's stack size is fixed
//! when it starts, so either one set afterwards has no effect.

#[cfg(not(any(feature = "cuda", feature = "rocm", feature = "vulkan", feature = "metal")))]
compile_error!("av-toolbox-gpu needs at least one accelerator backend enabled: cuda, rocm, vulkan or metal");

#[cfg(all(feature = "vulkan", feature = "metal"))]
compile_error!("the vulkan and metal backends cannot both be enabled, pick one");

mod accelerator;
mod backend;
mod cache;
mod capabilities;
mod device;
mod enumerate;
mod frame;
mod memory;
mod negotiate;
mod probe;
mod sniff;
mod stack;
mod warmup;

#[cfg(test)]
mod testing;

pub use cubecl::client::ComputeClient;
pub use cubecl::prelude::Runtime;

pub use self::accelerator::{Accelerator, default_accelerators};
pub use self::backend::{Backend, BuildContext, BuildTask};
pub use self::cache::{
    COMPILATION_CACHE_ENV,
    CacheError,
    compilation_cache_dir,
    default_cache_dir,
    install_compilation_cache,
    install_compilation_cache_at,
    install_compilation_cache_once,
};
pub use self::capabilities::{Capabilities, ElementType, capabilities_for};
pub use self::device::{Device, DeviceError, ParseDeviceError};
pub use self::enumerate::{BackendDevices, enumerate_devices};
pub use self::frame::{
    Channel,
    ColourInfo,
    ColourRange,
    Depth,
    DepthError,
    Frame,
    FrameFormat,
    FrameLayout,
    FrameMeta,
    FrameMut,
    HostPlanes,
    Mask,
    MaskFormat,
    Matrix,
    Plane,
    PlaneBuffer,
    Primaries,
    SampleFormat,
    StreamFormat,
    Subsampling,
    Transfer,
};
pub use self::memory::{BufferPool, ClientAllocator, DeviceAllocator, Ring, RingGeometry, RingSpec};
pub use self::negotiate::{FormatSet, OutputFormats, ResolvedFormats, StreamTemplate};
pub use self::sniff::{auto_select_accelerator, sniff_best_accelerator};
pub use self::stack::{CODEGEN_STACK_BYTES, codegen_stack_is_sufficient, raise_codegen_stack_limit};
pub use self::warmup::WarmUp;
