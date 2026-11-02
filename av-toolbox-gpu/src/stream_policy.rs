use crate::Accelerator;

/// How GPU work queued by different threads is kept in order.
///
/// CubeCL queues each thread's work on that thread's own stream. A buffer handed to another thread
/// could otherwise be used there before the work that wrote it was submitted.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum StreamPolicy {
    /// Every thread queues on one shared stream, so work is submitted in the order it was queued.
    ///
    /// Vulkan and Metal use this. Vulkan's streams all feed one GPU queue anyway, and Metal keeps
    /// its work safely ordered on one command queue.
    SharedStream,
    /// Each thread keeps its own stream and submits it before a buffer it used goes to another
    /// thread.
    ///
    /// CUDA and ROCm use this, since their streams let work overlap on the GPU.
    FlushAtHandOff,
}

impl StreamPolicy {
    /// The policy `accelerator` keeps its work in order with.
    pub const fn for_accelerator(accelerator: Accelerator) -> Self {
        match accelerator {
            #[cfg(any(feature = "cuda", docsrs))]
            Accelerator::Cuda => Self::FlushAtHandOff,
            #[cfg(any(feature = "rocm", docsrs))]
            Accelerator::Rocm => Self::FlushAtHandOff,
            #[cfg(any(feature = "vulkan", docsrs))]
            Accelerator::Vulkan => Self::SharedStream,
            #[cfg(any(feature = "metal", docsrs))]
            Accelerator::Metal => Self::SharedStream,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "vulkan")]
    #[test]
    fn vulkan_shares_one_stream() {
        let policy = StreamPolicy::for_accelerator(Accelerator::Vulkan);

        assert_eq!(policy, StreamPolicy::SharedStream);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_shares_one_stream() {
        let policy = StreamPolicy::for_accelerator(Accelerator::Metal);

        assert_eq!(policy, StreamPolicy::SharedStream);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_flushes_at_hand_off() {
        let policy = StreamPolicy::for_accelerator(Accelerator::Cuda);

        assert_eq!(policy, StreamPolicy::FlushAtHandOff);
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn rocm_flushes_at_hand_off() {
        let policy = StreamPolicy::for_accelerator(Accelerator::Rocm);

        assert_eq!(policy, StreamPolicy::FlushAtHandOff);
    }
}
