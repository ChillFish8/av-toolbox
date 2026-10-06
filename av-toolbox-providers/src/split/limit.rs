/// Keeps float error from losing a whole frame when seconds convert to frames.
const FRAME_TOLERANCE: f64 = 1e-9;

/// The whole frames that fit in `seconds` when each lasts `frame_duration` seconds, at least one.
pub(crate) fn frames_in(seconds: f64, frame_duration: f64) -> usize {
    let frames = seconds / frame_duration + FRAME_TOLERANCE;
    let whole = frames.floor() as usize;

    whole.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seconds_convert_by_the_frame_duration() {
        let frames = frames_in(2.0, 0.0625);

        assert_eq!(frames, 32);
    }

    #[test]
    fn ten_seconds_at_twenty_five_frames_a_second_is_two_hundred_and_fifty_frames() {
        let frames = frames_in(10.0, 0.04);

        assert_eq!(frames, 250);
    }

    #[test]
    fn a_tiny_seconds_limit_is_still_one_frame() {
        let frames = frames_in(0.001, 0.0625);

        assert_eq!(frames, 1);
    }
}
