//! Device memory for frames.
//!
//! A [DeviceAllocator] is the one place memory is reserved, so everything above it is plain code
//! rather than code generic over a runtime.

mod allocator;
mod pool;
mod ring;

pub use self::allocator::{ClientAllocator, DeviceAllocator};
pub use self::pool::BufferPool;
pub use self::ring::{Ring, RingGeometry, RingSpec};
