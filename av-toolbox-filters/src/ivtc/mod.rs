#[cfg_attr(
    not(test),
    expect(dead_code, reason = "the decision has no caller outside its tests")
)]
mod decision;
#[cfg(test)]
mod tests;
