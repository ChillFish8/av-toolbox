#[cfg_attr(
    not(test),
    expect(dead_code, reason = "used once the resize stages are registered")
)]
mod geometry;
#[expect(dead_code, reason = "used once the resize stages are registered")]
mod kernels;
