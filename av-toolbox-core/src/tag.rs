

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
/// A [Tag] is a standardised label used for organisation.
pub enum Tag {
    Denoise,
    Deband,
    SuperResolution,
    Deinterlace,
    Color,
    Conversion,
    Other(&'static str)
}