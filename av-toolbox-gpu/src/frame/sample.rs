use std::fmt;

use crate::ElementType;

/// How many meaningful bits a `u16` sample carries.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct Depth(u8);

/// A bit depth outside what a `u16` sample can carry.
#[derive(Debug, thiserror::Error)]
#[error("bit depth {bits} is not supported, expected between 9 and 16")]
pub struct DepthError {
    pub bits: u8,
}

impl Depth {
    pub const TEN: Self = Self(10);
    pub const TWELVE: Self = Self(12);
    pub const SIXTEEN: Self = Self(16);

    /// Accepts a depth between 9 and 16.
    ///
    /// Eight-bit content is [SampleFormat::U8] rather than a `u16` at depth eight, so eight is not
    /// a depth this accepts.
    pub const fn new(bits: u8) -> Result<Self, DepthError> {
        if !matches!(bits, 9..=16) {
            return Err(DepthError { bits });
        }

        Ok(Self(bits))
    }

    pub const fn bits(self) -> u8 {
        self.0
    }

    /// The largest sample value this depth can hold.
    pub const fn max_value(self) -> u16 {
        ((1u32 << self.0) - 1) as u16
    }
}

impl fmt::Display for Depth {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}-bit", self.0)
    }
}

/// What one sample of one channel is stored as.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum SampleFormat {
    U8,
    U16 { depth: Depth },
    F32,
}

impl SampleFormat {
    /// How many bytes one sample occupies.
    pub const fn bytes(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U16 { .. } => 2,
            Self::F32 => 4,
        }
    }

    /// The largest value a sample of this format can hold, or `None` for a float format.
    ///
    /// Float samples are normalised rather than counted, so there is no largest value in this
    /// sense.
    pub const fn max_value(self) -> Option<u16> {
        match self {
            Self::U8 => Some(u8::MAX as u16),
            Self::U16 { depth } => Some(depth.max_value()),
            Self::F32 => None,
        }
    }

    /// The element type a plane of these samples is stored in.
    ///
    /// Two depths share an element, so this does not round-trip back to a sample format.
    pub const fn element(self) -> ElementType {
        match self {
            Self::U8 => ElementType::U8,
            Self::U16 { .. } => ElementType::U16,
            Self::F32 => ElementType::F32,
        }
    }

    /// Every sample that holds all of this one's values exactly, narrowest first.
    ///
    /// An integer widens to each deeper integer and then to F32, which holds every integer up to
    /// 16 bits exactly. F32 widens to nothing.
    pub fn lossless_widenings(self) -> Vec<SampleFormat> {
        let bits = match self {
            Self::U8 => 8,
            Self::U16 { depth } => depth.bits(),
            Self::F32 => return Vec::new(),
        };

        let mut widenings: Vec<SampleFormat> = (bits + 1..=16)
            .filter_map(|deeper| Depth::new(deeper).ok())
            .map(|depth| Self::U16 { depth })
            .collect();
        widenings.push(Self::F32);

        widenings
    }
}

impl fmt::Display for SampleFormat {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::U8 => formatter.write_str("u8"),
            Self::U16 { depth } => write!(formatter, "u16 {depth}"),
            Self::F32 => formatter.write_str("f32"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_depth_outside_the_supported_range_is_rejected() {
        for bits in [0, 1, 8, 17, 32] {
            assert!(Depth::new(bits).is_err(), "{bits}");
        }
    }

    #[test]
    fn every_supported_depth_is_accepted() {
        for bits in 9..=16 {
            let depth = Depth::new(bits).expect("should accept");
            assert_eq!(depth.bits(), bits);
        }
    }

    #[test]
    fn a_depth_knows_its_largest_sample() {
        assert_eq!(Depth::TEN.max_value(), 1023);
        assert_eq!(Depth::TWELVE.max_value(), 4095);
        assert_eq!(Depth::SIXTEEN.max_value(), 65535);
    }

    #[test]
    fn a_sample_format_knows_its_width() {
        assert_eq!(SampleFormat::U8.bytes(), 1);
        assert_eq!(SampleFormat::U16 { depth: Depth::TEN }.bytes(), 2);
        assert_eq!(SampleFormat::F32.bytes(), 4);
    }

    #[test]
    fn an_integer_format_reports_the_largest_value_it_holds() {
        assert_eq!(SampleFormat::U8.max_value(), Some(255));
        assert_eq!(SampleFormat::U16 { depth: Depth::TEN }.max_value(), Some(1023));
        assert_eq!(
            SampleFormat::U16 {
                depth: Depth::SIXTEEN
            }
            .max_value(),
            Some(65535)
        );
    }

    /// Float samples are normalised, so there is no largest value for a depth conversion to scale
    /// by.
    #[test]
    fn a_float_format_has_no_largest_value() {
        assert_eq!(SampleFormat::F32.max_value(), None);
    }

    #[test]
    fn a_sample_format_maps_onto_the_element_it_is_stored_in() {
        assert_eq!(SampleFormat::U8.element(), ElementType::U8);
        assert_eq!(
            SampleFormat::U16 { depth: Depth::TEN }.element(),
            ElementType::U16
        );
        assert_eq!(SampleFormat::F32.element(), ElementType::F32);
    }

    /// Ten-bit and twelve-bit content both sit in a `u16`, so the element alone cannot tell them
    /// apart and the depth has to survive alongside it.
    #[test]
    fn two_depths_share_an_element_but_not_a_format() {
        let ten = SampleFormat::U16 { depth: Depth::TEN };
        let twelve = SampleFormat::U16 { depth: Depth::TWELVE };

        assert_eq!(ten.element(), twelve.element());
        assert_ne!(ten, twelve);
    }

    #[test]
    fn a_depth_and_sample_format_render_for_a_reader() {
        assert_eq!(Depth::TEN.to_string(), "10-bit");
        assert_eq!(SampleFormat::U8.to_string(), "u8");
        assert_eq!(SampleFormat::F32.to_string(), "f32");
        assert_eq!(SampleFormat::U16 { depth: Depth::TEN }.to_string(), "u16 10-bit");
    }

    #[test]
    fn an_integer_widens_to_every_deeper_integer_then_f32() {
        let ten = SampleFormat::U16 { depth: Depth::TEN };
        let widenings = ten.lossless_widenings();

        let eleven = Depth::new(11).expect("should accept");
        assert_eq!(widenings.first(), Some(&SampleFormat::U16 { depth: eleven }));
        assert_eq!(widenings.last(), Some(&SampleFormat::F32));
        assert_eq!(widenings.len(), 7);
    }

    #[test]
    fn u8_widens_to_every_u16_depth_then_f32() {
        let widenings = SampleFormat::U8.lossless_widenings();

        let nine = Depth::new(9).expect("should accept");
        assert_eq!(widenings.first(), Some(&SampleFormat::U16 { depth: nine }));
        assert_eq!(widenings.last(), Some(&SampleFormat::F32));
        assert_eq!(widenings.len(), 9);
    }

    #[test]
    fn f32_widens_to_nothing() {
        assert!(SampleFormat::F32.lossless_widenings().is_empty());
    }

    #[test]
    fn a_rejected_depth_says_what_it_was_given() {
        let error = Depth::new(8).expect_err("should reject");

        assert_eq!(error.bits, 8);
        assert!(error.to_string().contains('8'), "{error}");
    }
}
