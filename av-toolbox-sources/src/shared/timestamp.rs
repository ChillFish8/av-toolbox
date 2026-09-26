use std::time::Duration;

/// When frame `index` starts at `numerator / denominator` frames per second, or `None` for a zero rate.
pub(crate) fn timestamp_of(index: u64, (numerator, denominator): (usize, usize)) -> Option<Duration> {
    if numerator == 0 {
        return None;
    }

    let nanos = index as u128 * denominator as u128 * 1_000_000_000 / numerator as u128;
    let nanos = u64::try_from(nanos).ok()?;

    Some(Duration::from_nanos(nanos))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_two_at_25_fps_starts_at_80_ms() {
        assert_eq!(timestamp_of(2, (25, 1)), Some(Duration::from_millis(80)));
    }

    #[test]
    fn a_zero_rate_gives_no_timestamp() {
        assert_eq!(timestamp_of(2, (0, 1)), None);
    }
}
