use std::fmt;
use std::num::NonZeroU32;

/// How many frames a stream shows per second, kept in lowest terms.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct FrameRate {
    numerator: u32,
    denominator: NonZeroU32,
}

impl FrameRate {
    /// The rate `numerator / denominator`, or `None` when either part is zero.
    pub fn new(numerator: u32, denominator: u32) -> Option<Self> {
        if numerator == 0 {
            return None;
        }

        let denominator = NonZeroU32::new(denominator)?;
        let divisor = greatest_common_divisor(numerator, denominator.get());
        let reduced_denominator = NonZeroU32::new(denominator.get() / divisor)?;

        Some(Self {
            numerator: numerator / divisor,
            denominator: reduced_denominator,
        })
    }

    pub const fn numerator(self) -> u32 {
        self.numerator
    }

    pub const fn denominator(self) -> u32 {
        self.denominator.get()
    }
}

impl fmt::Display for FrameRate {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}/{}", self.numerator, self.denominator)
    }
}

fn greatest_common_divisor(mut left: u32, mut right: u32) -> u32 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }

    left
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rate_is_kept_in_lowest_terms() {
        let halved = FrameRate::new(50, 2).expect("a valid rate");
        let whole = FrameRate::new(25, 1).expect("a valid rate");

        assert_eq!(halved, whole);
        assert_eq!(halved.numerator(), 25);
        assert_eq!(halved.denominator(), 1);
    }

    #[test]
    fn an_ntsc_rate_is_already_reduced() {
        let rate = FrameRate::new(30000, 1001).expect("a valid rate");

        assert_eq!(rate.numerator(), 30000);
        assert_eq!(rate.denominator(), 1001);
    }

    #[test]
    fn a_zero_part_is_refused() {
        assert_eq!(FrameRate::new(0, 1), None);
        assert_eq!(FrameRate::new(25, 0), None);
    }

    #[test]
    fn a_rate_displays_as_a_fraction() {
        let rate = FrameRate::new(24000, 1001).expect("a valid rate");

        assert_eq!(rate.to_string(), "24000/1001");
    }
}
