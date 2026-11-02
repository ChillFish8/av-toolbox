use av_denoise_core::{ChromaLayout, ChromaSiting};
use av_toolbox_gpu::{
    ChromaLocation,
    ColourInfo,
    Depth,
    FrameFormat,
    FrameLayout,
    SampleFormat,
    Subsampling,
};

use crate::denoise::avd::planes::{
    PlaneGroups,
    engine_geometry,
    engine_sample,
    needs_padding,
    plane_groups,
    stage_geometry,
};
use crate::planes::ChannelName;

const STAGE: &str = "av-denoise/nlmeans";

fn format(layout: FrameLayout, sample: SampleFormat) -> FrameFormat {
    FrameFormat {
        width: 64,
        height: 32,
        layout,
        sample,
        colour: ColourInfo::default(),
        frame_rate: None,
        field_order: None,
    }
}

fn yuv(subsampling: Subsampling) -> FrameFormat {
    let layout = FrameLayout::Yuv { subsampling };

    format(layout, SampleFormat::U8)
}

#[test]
fn every_plane_of_yuv_selects_both_groups() {
    let input = yuv(Subsampling::Yuv420);

    let groups = plane_groups(STAGE, None, &input).expect("should map");

    assert!(groups.luma);
    assert!(groups.chroma);
}

#[test]
fn luma_alone_selects_the_luma_group() {
    let input = yuv(Subsampling::Yuv422);
    let planes = [ChannelName::Y];

    let groups = plane_groups(STAGE, Some(&planes), &input).expect("should map");

    assert!(groups.luma);
    assert!(!groups.chroma);
}

#[test]
fn chroma_alone_selects_the_chroma_group() {
    let input = yuv(Subsampling::Yuv422);
    let planes = [ChannelName::U, ChannelName::V];

    let groups = plane_groups(STAGE, Some(&planes), &input).expect("should map");

    assert!(!groups.luma);
    assert!(groups.chroma);
}

#[test]
fn half_chroma_is_refused() {
    let input = yuv(Subsampling::Yuv420);
    let planes = [ChannelName::Y, ChannelName::U];

    let error = plane_groups(STAGE, Some(&planes), &input).expect_err("should refuse");

    assert!(error.to_string().contains(STAGE), "{error}");
}

#[test]
fn grey_has_no_chroma_group() {
    let input = format(FrameLayout::Grey, SampleFormat::U8);

    let groups = plane_groups(STAGE, None, &input).expect("should map");

    assert!(groups.luma);
    assert!(!groups.chroma);
}

#[test]
fn rgb_is_refused() {
    let input = format(FrameLayout::Rgb, SampleFormat::U8);

    let refused = plane_groups(STAGE, None, &input);

    assert!(refused.is_err());
}

#[test]
fn engine_geometry_maps_each_layout() {
    let layouts = [
        (FrameLayout::Grey, ChromaLayout::Gray),
        (
            FrameLayout::Yuv {
                subsampling: Subsampling::Yuv420,
            },
            ChromaLayout::Yuv420,
        ),
        (
            FrameLayout::Yuv {
                subsampling: Subsampling::Yuv422,
            },
            ChromaLayout::Yuv422,
        ),
        (
            FrameLayout::Yuv {
                subsampling: Subsampling::Yuv444,
            },
            ChromaLayout::Yuv444,
        ),
    ];

    for (layout, chroma) in layouts {
        let input = format(layout, SampleFormat::U8);

        let geometry = engine_geometry(&input);

        assert_eq!(geometry.chroma, chroma, "{layout}");
        assert_eq!((geometry.width, geometry.height), (64, 32), "{layout}");
        assert_eq!(geometry.input, av_denoise_core::SampleFormat::U8, "{layout}");
        assert_eq!(geometry.output, av_denoise_core::SampleFormat::F32, "{layout}");
    }

    let mut odd = yuv(Subsampling::Yuv420);
    odd.width = 33;
    odd.height = 17;

    let geometry = engine_geometry(&odd);

    assert_eq!((geometry.width, geometry.height), (33, 17));
    assert_eq!(geometry.plane_dims(1), (17, 9));
}

#[test]
fn only_a_partial_integer_selection_reads_normalised_floats() {
    let both = PlaneGroups {
        luma: true,
        chroma: true,
    };
    let luma_only = PlaneGroups {
        luma: true,
        chroma: false,
    };
    let chroma_only = PlaneGroups {
        luma: false,
        chroma: true,
    };
    let integer = yuv(Subsampling::Yuv420);
    let float = format(integer.layout, SampleFormat::F32);
    let grey = format(FrameLayout::Grey, SampleFormat::U8);

    let normalised = av_denoise_core::SampleFormat::F32;
    assert_eq!(stage_geometry(&integer, luma_only).input, normalised);
    assert_eq!(stage_geometry(&integer, chroma_only).input, normalised);
    assert_eq!(
        stage_geometry(&integer, both).input,
        av_denoise_core::SampleFormat::U8
    );
    assert_eq!(
        stage_geometry(&float, luma_only).input,
        av_denoise_core::SampleFormat::F32
    );
    assert_eq!(
        stage_geometry(&grey, luma_only).input,
        av_denoise_core::SampleFormat::U8
    );
}

#[test]
fn chroma_siting_follows_the_chroma_location() {
    let locations = [
        (ChromaLocation::Left, ChromaSiting::Left),
        (ChromaLocation::Center, ChromaSiting::Center),
        (ChromaLocation::TopLeft, ChromaSiting::TopLeft),
        (ChromaLocation::Top, ChromaSiting::Unknown),
        (ChromaLocation::BottomLeft, ChromaSiting::Unknown),
        (ChromaLocation::Bottom, ChromaSiting::Unknown),
    ];

    for (location, siting) in locations {
        let mut input = yuv(Subsampling::Yuv420);
        input.colour.chroma_location = location;

        let geometry = engine_geometry(&input);

        assert_eq!(geometry.chroma_siting, siting, "{location:?}");
    }
}

#[test]
fn samples_map_to_av_denoise_formats() {
    let ten_bit = SampleFormat::U16 { depth: Depth::TEN };

    assert_eq!(engine_sample(SampleFormat::U8), av_denoise_core::SampleFormat::U8);
    assert_eq!(
        engine_sample(ten_bit),
        av_denoise_core::SampleFormat::U16 { depth: 10 }
    );
    assert_eq!(
        engine_sample(SampleFormat::F32),
        av_denoise_core::SampleFormat::F32
    );
}

#[test]
fn only_planes_short_of_whole_words_need_padding() {
    let ten_bit = SampleFormat::U16 { depth: Depth::TEN };

    assert!(!needs_padding(SampleFormat::U8, 64, 32));
    assert!(needs_padding(SampleFormat::U8, 33, 17));
    assert!(!needs_padding(ten_bit, 33, 18));
    assert!(needs_padding(ten_bit, 33, 17));
    assert!(!needs_padding(SampleFormat::F32, 33, 17));
}
