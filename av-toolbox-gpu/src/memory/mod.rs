//! Device memory for frames.
//!
//! A [DeviceAllocator] is the one place memory is reserved, so everything above it is plain code
//! rather than code generic over a runtime.

mod allocator;
mod budget;
mod host_pool;
mod pool;
mod ring;

pub use self::allocator::{ClientAllocator, DeviceAllocator, DeviceRead, RuntimeError};
pub use self::budget::{FrameBudget, Room};
pub use self::host_pool::{HostBuffer, HostPool};
pub use self::pool::BufferPool;
pub use self::ring::{Ring, RingGeometry, RingSpec};
