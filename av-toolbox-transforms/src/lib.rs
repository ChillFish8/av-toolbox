//! Transforms that change how a stream is divided into chunks.

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "nothing outside the tests splits scenes yet")
)]
mod split;
