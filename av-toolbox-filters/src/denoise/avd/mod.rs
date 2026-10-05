#[expect(dead_code, reason = "registered by the av-denoise stages")]
mod args;
#[cfg_attr(not(test), expect(dead_code, reason = "registered by the av-denoise stages"))]
mod planes;
#[cfg(test)]
mod tests;
