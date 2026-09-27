#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the stage built on these settings has not been registered yet"
    )
)]
mod placebo;
#[cfg(test)]
mod tests;
