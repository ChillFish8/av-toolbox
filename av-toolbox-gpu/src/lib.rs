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
mod batch;
mod cache;
mod capabilities;
mod context;
mod device;
mod enumerate;
mod frame;
mod memory;
mod negotiate;
mod probe;
mod side;
mod sniff;
mod stack;
mod stream_policy;
mod warmup;

#[cfg(any(test, feature = "test-support"))]
mod testing;
#[cfg(test)]
mod tests;

pub use cubecl::bytes::Bytes;
pub use cubecl::client::ComputeClient;
pub use cubecl::prelude::{ArrayArg, CubeCount, CubeDim, Runtime};

pub use self::accelerator::{Accelerator, default_accelerators};
pub use self::backend::{Backend, BuildContext, BuildTask, DeviceSync};
pub use self::batch::{PendingBatch, TryWait};
pub use self::cache::{
    COMPILATION_CACHE_ENV,
    CacheError,
    compilation_cache_dir,
    default_cache_dir,
    install_compilation_cache,
    install_compilation_cache_at,
    install_compilation_cache_once,
};
pub use self::capabilities::{Capabilities, CapabilitiesBuilder, ElementType, capabilities_for};
pub use self::context::{FrameContext, LoadItem};
pub use self::device::{Device, DeviceError, ParseDeviceError};
pub use self::enumerate::{BackendDevices, enumerate_devices};
pub use self::frame::{
    Attachments,
    Channel,
    ChromaLocation,
    ChunkKeyFn,
    ChunkRoute,
    ColourInfo,
    ColourRange,
    Depth,
    DepthError,
    FieldOrder,
    Frame,
    FrameError,
    FrameFormat,
    FrameLayout,
    FrameMeta,
    FrameMut,
    FrameRate,
    HostFrame,
    HostFrameMut,
    HostItem,
    HostMask,
    HostMaskMut,
    HostPlane,
    HostPlanes,
    HostStreamItem,
    LayerId,
    Layers,
    Mask,
    MaskChroma,
    MaskFormat,
    Matrix,
    MetaValue,
    Plane,
    PlaneBuffer,
    Primaries,
    Provided,
    ProvidedValue,
    SampleFormat,
    SceneScore,
    StreamFormat,
    StreamItem,
    Subsampling,
    Transfer,
};
pub use self::memory::{
    BudgetWaits,
    BufferPool,
    ClientAllocator,
    DeviceAllocator,
    DeviceRead,
    FrameBudget,
    HostBuffer,
    HostPool,
    Ring,
    RingGeometry,
    RingSpec,
    Room,
    RuntimeError,
};
pub use self::negotiate::{FormatSet, Offered, OutputFormats, ResolvedFormats, StreamTemplate};
pub use self::side::{AnyItem, DeviceSide, HostSide, Side};
pub use self::sniff::{auto_select_accelerator, sniff_best_accelerator};
pub use self::stack::{CODEGEN_STACK_BYTES, codegen_stack_is_sufficient, raise_codegen_stack_limit};
pub use self::stream_policy::StreamPolicy;
#[cfg(any(test, feature = "test-support"))]
pub use self::testing::{flush_counting_context, test_device, testing_context};
pub use self::warmup::WarmUp;
