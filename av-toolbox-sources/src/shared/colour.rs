use av_toolbox_gpu::{
    ChromaLocation,
    ColourInfo,
    ColourRange,
    FieldOrder,
    FrameLayout,
    Matrix,
    Primaries,
    Transfer,
};

/// A frame's colour properties as FFmpeg codes, before mapping.
pub(crate) struct RawColour {
    pub(crate) range: i32,
    pub(crate) matrix: i32,
    pub(crate) primaries: i32,
    pub(crate) transfer: i32,
    pub(crate) chroma_location: i32,
}

/// The field order a decoded frame's interlacing flags give.
pub(crate) fn field_order_of(interlaced: bool, top_field_first: bool) -> FieldOrder {
    if !interlaced {
        return FieldOrder::Progressive;
    }

    if top_field_first {
        FieldOrder::TopFirst
    } else {
        FieldOrder::BottomFirst
    }
}

/// The colour a frame carries, mapped from its FFmpeg codes.
///
/// A forced format that crosses the RGB/YUV family boundary overrides the source's matrix and
/// range, which no longer describe the samples once the decoder has recoloured them. An RGB output made
/// from a YUV source is identity matrix at full range, and a YUV output made from an RGB source
/// carries no matrix or range at all.
pub(crate) fn colour_of(
    raw: RawColour,
    full_range: bool,
    source_layout: Option<FrameLayout>,
    output_layout: FrameLayout,
) -> ColourInfo {
    let mapped = ColourInfo {
        range: range_of(raw.range, full_range),
        matrix: matrix_of(raw.matrix),
        primaries: primaries_of(raw.primaries),
        transfer: transfer_of(raw.transfer),
        chroma_location: chroma_location_of(raw.chroma_location),
    };

    match (source_layout, output_layout) {
        (Some(FrameLayout::Yuv { .. }), FrameLayout::Rgb) => ColourInfo {
            matrix: Matrix::Identity,
            range: ColourRange::Full,
            ..mapped
        },
        (Some(FrameLayout::Rgb), FrameLayout::Yuv { .. }) => ColourInfo {
            matrix: Matrix::Unspecified,
            range: ColourRange::Unspecified,
            ..mapped
        },
        _ => mapped,
    }
}

fn range_of(value: i32, full_range: bool) -> ColourRange {
    if full_range {
        return ColourRange::Full;
    }

    match value {
        1 => ColourRange::Limited,
        2 => ColourRange::Full,
        _ => ColourRange::Unspecified,
    }
}

fn matrix_of(value: i32) -> Matrix {
    match value {
        0 => Matrix::Identity,
        1 => Matrix::Bt709,
        5 | 6 => Matrix::Bt601,
        9 => Matrix::Bt2020Ncl,
        _ => Matrix::Unspecified,
    }
}

fn primaries_of(value: i32) -> Primaries {
    match value {
        1 => Primaries::Bt709,
        5 | 6 => Primaries::Bt601,
        9 => Primaries::Bt2020,
        _ => Primaries::Unspecified,
    }
}

fn transfer_of(value: i32) -> Transfer {
    match value {
        1 | 6 | 14 | 15 => Transfer::Bt709,
        13 => Transfer::Srgb,
        16 => Transfer::Pq,
        18 => Transfer::Hlg,
        _ => Transfer::Unspecified,
    }
}

/// Maps an FFmpeg chroma location, where 0 is unspecified and `1..=6` follow its enum.
fn chroma_location_of(raw: i32) -> ChromaLocation {
    match raw {
        2 => ChromaLocation::Center,
        3 => ChromaLocation::TopLeft,
        4 => ChromaLocation::Top,
        5 => ChromaLocation::BottomLeft,
        6 => ChromaLocation::Bottom,
        _ => ChromaLocation::Left,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffmpeg_chroma_location_values_map_to_their_position() {
        assert_eq!(chroma_location_of(0), ChromaLocation::Left);
        assert_eq!(chroma_location_of(1), ChromaLocation::Left);
        assert_eq!(chroma_location_of(2), ChromaLocation::Center);
        assert_eq!(chroma_location_of(6), ChromaLocation::Bottom);
    }

    #[test]
    fn transfers_on_the_bt709_curve_map_to_bt709() {
        for value in [1, 6, 14, 15] {
            assert_eq!(transfer_of(value), Transfer::Bt709, "{value}");
        }
    }
}
