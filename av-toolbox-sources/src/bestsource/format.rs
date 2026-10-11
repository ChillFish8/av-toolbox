use av_toolbox_gpu::{Channel, Depth, FrameLayout, SampleFormat, Subsampling};
use avtbx_bestsource::{ColorFamily, VideoFormat};

/// The layout and sample format a BestSource format decodes as, when the pipeline carries it.
pub(crate) fn mapped_format(format: &VideoFormat) -> Option<(FrameLayout, SampleFormat)> {
    if format.alpha || format.float {
        return None;
    }

    let layout = layout_of(format)?;
    let sample = sample_of(format.bits)?;

    Some((layout, sample))
}

/// The channel each exported plane fills, in BestSource's plane order.
pub(crate) fn plane_channels(layout: FrameLayout) -> &'static [Channel] {
    match layout {
        FrameLayout::Yuv { .. } => &[Channel::Y, Channel::U, Channel::V],
        FrameLayout::Grey => &[Channel::Y],
        FrameLayout::Rgb => &[Channel::R, Channel::G, Channel::B],
        FrameLayout::Chroma { .. } => unreachable!("bestsource never opens a source as chroma planes"),
    }
}

/// A short readable description of a format, such as `yuv 4:2:0 10 bit`.
pub(crate) fn describe_format(format: &VideoFormat) -> String {
    let family = match format.color_family {
        ColorFamily::Yuv => {
            let chroma = chroma_text(format.sub_sampling_w, format.sub_sampling_h);

            format!("yuv {chroma}")
        },
        ColorFamily::Gray => "gray".to_string(),
        ColorFamily::Rgb => "rgb".to_string(),
        ColorFamily::Bayer => "bayer".to_string(),
        ColorFamily::Unknown => "unknown".to_string(),
    };
    let bits = format.bits;
    let kind = if format.float { " float" } else { "" };
    let alpha = if format.alpha { " with alpha" } else { "" };

    format!("{family} {bits} bit{kind}{alpha}")
}

fn chroma_text(shift_w: i32, shift_h: i32) -> String {
    let horizontal = 4 >> shift_w.clamp(0, 3);
    let vertical = if shift_h == 0 { horizontal } else { 0 };

    format!("4:{horizontal}:{vertical}")
}

fn layout_of(format: &VideoFormat) -> Option<FrameLayout> {
    match format.color_family {
        ColorFamily::Yuv => {
            let subsampling = subsampling_of(format.sub_sampling_w, format.sub_sampling_h)?;

            Some(FrameLayout::Yuv { subsampling })
        },
        ColorFamily::Gray => Some(FrameLayout::Grey),
        ColorFamily::Rgb => Some(FrameLayout::Rgb),
        ColorFamily::Unknown | ColorFamily::Bayer => None,
    }
}

fn subsampling_of(shift_w: i32, shift_h: i32) -> Option<Subsampling> {
    match (shift_w, shift_h) {
        (0, 0) => Some(Subsampling::Yuv444),
        (1, 0) => Some(Subsampling::Yuv422),
        (1, 1) => Some(Subsampling::Yuv420),
        _ => None,
    }
}

fn sample_of(bits: i32) -> Option<SampleFormat> {
    if bits == 8 {
        return Some(SampleFormat::U8);
    }

    let bits = u8::try_from(bits).ok()?;
    let depth = Depth::new(bits).ok()?;

    Some(SampleFormat::U16 { depth })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn yuv(bits: i32, shift_w: i32, shift_h: i32) -> VideoFormat {
        VideoFormat {
            color_family: ColorFamily::Yuv,
            alpha: false,
            float: false,
            bits,
            sub_sampling_w: shift_w,
            sub_sampling_h: shift_h,
        }
    }

    #[test]
    fn formats_are_described_in_words() {
        let float_rgb = VideoFormat {
            color_family: ColorFamily::Rgb,
            alpha: true,
            float: true,
            ..yuv(32, 0, 0)
        };

        assert_eq!(describe_format(&yuv(8, 2, 0)), "yuv 4:1:1 8 bit");
        assert_eq!(describe_format(&yuv(10, 1, 1)), "yuv 4:2:0 10 bit");
        assert_eq!(describe_format(&float_rgb), "rgb 32 bit float with alpha");
    }

    #[test]
    fn each_yuv_subsampling_maps_to_its_layout() {
        let cases = [
            ((0, 0), Subsampling::Yuv444),
            ((1, 0), Subsampling::Yuv422),
            ((1, 1), Subsampling::Yuv420),
        ];

        for ((shift_w, shift_h), subsampling) in cases {
            let format = yuv(8, shift_w, shift_h);
            let mapped = mapped_format(&format);

            assert_eq!(mapped, Some((FrameLayout::Yuv { subsampling }, SampleFormat::U8)));
        }
    }

    #[test]
    fn depths_above_eight_bits_ride_in_sixteen_bit_samples() {
        let format = yuv(10, 1, 1);
        let depth = Depth::new(10).expect("a valid depth");

        let mapped = mapped_format(&format);

        assert_eq!(
            mapped.map(|(_, sample)| sample),
            Some(SampleFormat::U16 { depth })
        );
    }

    #[test]
    fn gray_and_rgb_map_to_their_layouts() {
        let gray = VideoFormat {
            color_family: ColorFamily::Gray,
            ..yuv(8, 0, 0)
        };
        let rgb = VideoFormat {
            color_family: ColorFamily::Rgb,
            ..yuv(8, 0, 0)
        };

        assert_eq!(
            mapped_format(&gray).map(|(layout, _)| layout),
            Some(FrameLayout::Grey)
        );
        assert_eq!(
            mapped_format(&rgb).map(|(layout, _)| layout),
            Some(FrameLayout::Rgb)
        );
    }

    #[test]
    fn alpha_float_bayer_and_other_subsampling_are_refused() {
        let alpha = VideoFormat {
            alpha: true,
            ..yuv(8, 1, 1)
        };
        let float = VideoFormat {
            float: true,
            ..yuv(32, 0, 0)
        };
        let bayer = VideoFormat {
            color_family: ColorFamily::Bayer,
            ..yuv(8, 0, 0)
        };
        let yuv411 = yuv(8, 2, 0);

        for format in [alpha, float, bayer, yuv411] {
            assert_eq!(mapped_format(&format), None, "{format:?}");
        }
    }
}
