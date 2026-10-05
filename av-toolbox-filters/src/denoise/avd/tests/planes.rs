use av_denoise_core::ChannelMode;
use av_toolbox_gpu::{ColourInfo, Depth, FrameFormat, FrameLayout, SampleFormat, Subsampling};

use crate::denoise::avd::planes::{engine_sample, engine_specs, needs_padding};
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
fn every_plane_of_yuv_gets_a_luma_and_a_chroma_engine() {
    let input = yuv(Subsampling::Yuv420);

    let specs = engine_specs(STAGE, None, false, &input).expect("should map");

    assert_eq!(specs.len(), 2);
    assert_eq!(specs[0].channels, ChannelMode::Luma);
    assert_eq!(specs[0].planes, vec![0]);
    assert_eq!((specs[0].width, specs[0].height), (64, 32));
    assert_eq!(specs[1].channels, ChannelMode::Chroma);
    assert_eq!(specs[1].planes, vec![1, 2]);
    assert_eq!((specs[1].width, specs[1].height), (32, 16));
}

#[test]
fn luma_alone_gets_one_engine() {
    let input = yuv(Subsampling::Yuv422);
    let planes = [ChannelName::Y];

    let specs = engine_specs(STAGE, Some(&planes), false, &input).expect("should map");

    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0].channels, ChannelMode::Luma);
}

#[test]
fn chroma_alone_gets_one_engine_at_chroma_size() {
    let input = yuv(Subsampling::Yuv422);
    let planes = [ChannelName::U, ChannelName::V];

    let specs = engine_specs(STAGE, Some(&planes), false, &input).expect("should map");

    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0].channels, ChannelMode::Chroma);
    assert_eq!((specs[0].width, specs[0].height), (32, 32));
}

#[test]
fn u_without_v_is_refused() {
    let input = yuv(Subsampling::Yuv420);
    let planes = [ChannelName::Y, ChannelName::U];

    let error = engine_specs(STAGE, Some(&planes), false, &input).expect_err("should refuse");

    assert!(error.to_string().contains(STAGE), "{error}");
}

#[test]
fn fused_444_gets_one_yuv_engine() {
    let input = yuv(Subsampling::Yuv444);

    let specs = engine_specs(STAGE, None, true, &input).expect("should map");

    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0].channels, ChannelMode::Yuv);
    assert_eq!(specs[0].planes, vec![0, 1, 2]);
}

#[test]
fn fused_420_is_refused() {
    let input = yuv(Subsampling::Yuv420);

    let refused = engine_specs(STAGE, None, true, &input);

    assert!(refused.is_err());
}

#[test]
fn fused_without_every_plane_is_refused() {
    let input = yuv(Subsampling::Yuv444);
    let planes = [ChannelName::Y];

    let refused = engine_specs(STAGE, Some(&planes), true, &input);

    assert!(refused.is_err());
}

#[test]
fn a_grey_clip_gets_one_luma_engine() {
    let input = format(FrameLayout::Grey, SampleFormat::U8);

    let specs = engine_specs(STAGE, None, false, &input).expect("should map");

    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0].channels, ChannelMode::Luma);
}

#[test]
fn fused_on_grey_is_refused() {
    let input = format(FrameLayout::Grey, SampleFormat::U8);

    let refused = engine_specs(STAGE, None, true, &input);

    assert!(refused.is_err());
}

#[test]
fn rgb_is_refused() {
    let input = format(FrameLayout::Rgb, SampleFormat::U8);

    let refused = engine_specs(STAGE, None, false, &input);

    assert!(refused.is_err());
}

#[test]
fn samples_map_to_av_denoise_formats() {
    let ten_bit = SampleFormat::U16 { depth: Depth::TEN };

    assert_eq!(
        engine_sample(SampleFormat::U8),
        Some(av_denoise_core::SampleFormat::U8)
    );
    assert_eq!(
        engine_sample(ten_bit),
        Some(av_denoise_core::SampleFormat::U16 { depth: 10 })
    );
    assert_eq!(
        engine_sample(SampleFormat::F32),
        Some(av_denoise_core::SampleFormat::F32)
    );
    assert_eq!(engine_sample(SampleFormat::F16), None);
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
