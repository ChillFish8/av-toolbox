mod database;
mod lock;
mod private;
mod prune;
#[cfg(test)]
mod tests;

#[cfg(any(test, feature = "test-support"))]
pub(crate) use self::private::atexit;
pub(crate) use self::private::{
    PrivateKernelCache,
    finish_kernel_cache,
    finish_kernel_cache_at_exit,
    merge_kernel_cache,
    refill_kernel_cache,
};
