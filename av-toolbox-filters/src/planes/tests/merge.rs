use av_toolbox_core::{Filter, FilterDefinition, Inputs};
use av_toolbox_gpu::{
    Capabilities,
    Channel,
    ChromaLocation,
    ColourInfo,
    Depth,
    FormatSet,
    Frame,
    FrameFormat,
    FrameLayout,
    FrameMeta,
    MaskFormat,
    Offered,
    SampleFormat,
    StreamFormat,
    StreamItem,
    Subsampling,
    default_accelerators,
    testing_context,
};
use serde_json::json;

use crate::planes::PlaneName;
use crate::planes::extract::extracted;
use crate::planes::merge::{Merge, MergeArgs, Merger, merged};

/// The one frame a filter emitted for one input.
fn only_frame(mut frames: Vec<Frame>) -> Frame {
    assert_eq!(
        frames.len(),
        1,
        "a filter emitted {} frames for one input",
        frames.len()
    );
    frames.remove(0)
}

fn capabilities() -> Capabilities {
    let accelerator = default_accelerators()[0];

    Capabilities::builder(accelerator)
        .plane_size(32, 32)
        .max_units_per_cube(1024)
        .max_cube_count(65535, 65535, 65535)
        .build()
}

fn yuv(width: u32, height: u32, subsampling: Subsampling) -> FrameFormat {
    FrameFormat {
        width,
        height,
        layout: FrameLayout::Yuv { subsampling },
        sample: SampleFormat::U8,
        colour: ColourInfo::default(),
        frame_rate: None,
        field_order: None,
    }
}

fn args(value: serde_json::Value) -> MergeArgs {
    serde_json::from_value(value).expect("parses")
}

fn split(input: &FrameFormat) -> [FrameFormat; 2] {
    let luma = extracted(input, PlaneName::Y).expect("carries Y");
    let chroma = extracted(input, PlaneName::Chroma).expect("carries chroma");

    [luma, chroma]
}

fn rejoin() -> MergeArgs {
    args(json!({
        "y": { "input": 0 },
        "u": { "input": 1 },
        "v": { "input": 1 },
    }))
}

#[test]
fn every_subsampling_is_worked_out_from_the_plane_sizes() {
    for subsampling in [Subsampling::Yuv444, Subsampling::Yuv422, Subsampling::Yuv420] {
        let original = yuv(64, 32, subsampling);
        let inputs = split(&original);

        let merge = merged(&rejoin(), &inputs).expect("rejoins");

        assert_eq!(merge.format, original, "{subsampling}");
    }
}

#[test]
fn an_odd_sized_420_clip_rejoins_as_420() {
    let original = yuv(63, 31, Subsampling::Yuv420);
    let inputs = split(&original);

    let merge = merged(&rejoin(), &inputs).expect("rejoins");

    assert_eq!(merge.format.layout, original.layout);
    assert_eq!((merge.format.width, merge.format.height), (63, 31));
}

#[test]
fn every_plane_can_come_from_one_clip() {
    let original = yuv(64, 32, Subsampling::Yuv420);
    let value = json!({ "y": { "input": 0 }, "u": { "input": 0 }, "v": { "input": 0 } });

    let merge = merged(&args(value), &[original]).expect("rebuilds");

    assert_eq!(merge.format, original);
}

#[test]
fn a_chroma_size_that_is_not_a_half_or_whole_is_refused() {
    let full_size = yuv(64, 32, Subsampling::Yuv444);
    let luma = extracted(&full_size, PlaneName::Y).expect("carries Y");
    let quarter = FrameFormat {
        width: 16,
        height: 8,
        layout: FrameLayout::Chroma {
            subsampling: Subsampling::Yuv420,
        },
        ..luma
    };

    let refused = merged(&rejoin(), &[luma, quarter]);

    assert!(refused.is_err());
}

#[test]
fn u_and_v_of_different_sizes_are_refused_naming_both() {
    let full_size = yuv(64, 32, Subsampling::Yuv444);
    let half_size = yuv(64, 32, Subsampling::Yuv420);
    let luma = extracted(&full_size, PlaneName::Y).expect("carries Y");
    let full = extracted(&full_size, PlaneName::U).expect("carries U");
    let half = extracted(&half_size, PlaneName::V).expect("carries V");
    let value = json!({ "y": { "input": 0 }, "u": { "input": 1 }, "v": { "input": 2 } });

    let error = merged(&args(value), &[luma, full, half]).expect_err("sizes differ");

    assert!(error.to_string().contains("64x32"), "{error}");
    assert!(error.to_string().contains("32x16"), "{error}");
}

#[test]
fn y_alone_builds_grey() {
    let original = yuv(64, 32, Subsampling::Yuv420);
    let luma = extracted(&original, PlaneName::Y).expect("carries Y");
    let value = json!({ "y": { "input": 0 } });

    let merge = merged(&args(value), &[luma]).expect("builds grey");

    assert_eq!(merge.format.layout, FrameLayout::Grey);
}

#[test]
fn rgb_needs_three_planes_of_one_size() {
    let rgb = FrameFormat {
        layout: FrameLayout::Rgb,
        ..yuv(64, 32, Subsampling::Yuv444)
    };
    let complete = json!({ "r": { "input": 0 }, "g": { "input": 0 }, "b": { "input": 0 } });
    let partial = json!({ "r": { "input": 0 }, "g": { "input": 0 } });

    let merge = merged(&args(complete), &[rgb]).expect("rebuilds");
    let refused = merged(&args(partial), &[rgb]);

    assert_eq!(merge.format, rgb);
    assert!(refused.is_err());
}

#[test]
fn mixing_yuv_and_rgb_is_refused() {
    let original = yuv(64, 32, Subsampling::Yuv444);
    let value = json!({ "y": { "input": 0 }, "u": { "input": 0 }, "v": { "input": 0 }, "r": { "input": 0 } });

    let refused = merged(&args(value), &[original]);

    assert!(refused.is_err());
}

#[test]
fn an_input_past_the_end_is_refused() {
    let original = yuv(64, 32, Subsampling::Yuv420);
    let value = json!({ "y": { "input": 1 }, "u": { "input": 0 }, "v": { "input": 0 } });

    let refused = merged(&args(value), &[original]);

    assert!(refused.is_err());
}

#[test]
fn an_input_nothing_picks_from_is_refused() {
    let original = yuv(64, 32, Subsampling::Yuv420);
    let value = json!({ "y": { "input": 0 }, "u": { "input": 0 }, "v": { "input": 0 } });

    let error = merged(&args(value), &[original, original]).expect_err("input 1 is unused");

    assert!(error.to_string().contains("input 1"), "{error}");
}

#[test]
fn planes_of_different_samples_are_refused() {
    let original = yuv(64, 32, Subsampling::Yuv420);
    let [luma, chroma] = split(&original);
    let wide = FrameFormat {
        sample: SampleFormat::U16 { depth: Depth::TEN },
        ..chroma
    };

    let refused = merged(&rejoin(), &[luma, wide]);

    assert!(refused.is_err());
}

#[test]
fn siting_comes_from_the_clip_supplying_u() {
    let original = yuv(64, 32, Subsampling::Yuv420);
    let [luma, chroma] = split(&original);
    let sited = FrameFormat {
        colour: ColourInfo {
            chroma_location: ChromaLocation::TopLeft,
            ..chroma.colour
        },
        ..chroma
    };

    let merge = merged(&rejoin(), &[luma, sited]).expect("rejoins");

    assert_eq!(merge.format.colour.chroma_location, ChromaLocation::TopLeft);
}

#[test]
fn merging_shares_every_plane() {
    let mut context = testing_context();
    let original = yuv(64, 32, Subsampling::Yuv420);
    let inputs = split(&original);
    let merge = merged(&rejoin(), &inputs).expect("rejoins");
    let luma = context.alloc_frame(&inputs[0], FrameMeta::at(0));
    let chroma = context.alloc_frame(&inputs[1], FrameMeta::at(0));
    let original_v = chroma.plane(Channel::V).expect("carries V").clone();
    let mut merger = Merger::new(merge);

    let slots = vec![Some(StreamItem::Frame(luma)), Some(StreamItem::Frame(chroma))];
    let inputs = Inputs::new(slots);
    let emitted = merger.process(&mut context, inputs).expect("merges");
    let produced = only_frame(emitted);

    let produced_v = produced.plane(Channel::V).expect("carries V");
    assert!(std::ptr::eq(produced_v.buffer(), original_v.buffer()));
}

#[test]
fn a_mask_is_refused() {
    let capabilities = capabilities();
    let arguments = rejoin();
    let inputs = [Offered::Unknown, Offered::Unknown];
    let mask = StreamFormat::Mask(MaskFormat {
        frame_rate: None,
        width: 16,
        height: 8,
        sample: SampleFormat::U8,
        chroma: None,
    });

    let refused = Merge::accepts_input(&capabilities, &arguments, 0, mask, &inputs);

    assert!(refused.is_err());
}

#[test]
fn a_narrower_clip_is_answered_with_the_wider_sample() {
    let capabilities = capabilities();
    let arguments = rejoin();
    let original = yuv(64, 32, Subsampling::Yuv420);
    let float = FrameFormat {
        sample: SampleFormat::F32,
        ..original
    };
    let inputs = [
        Offered::Proposed(StreamFormat::Frames(float)),
        Offered::Unknown,
        Offered::Proposed(StreamFormat::Frames(float)),
    ];
    let offered = StreamFormat::Frames(original);

    let accepted =
        Merge::accepts_input(&capabilities, &arguments, 1, offered, &inputs).expect("frames readable");

    assert_eq!(accepted, FormatSet::of([SampleFormat::F32]));
}

#[test]
fn a_wider_clip_keeps_its_own_sample() {
    let capabilities = capabilities();
    let arguments = rejoin();
    let original = yuv(64, 32, Subsampling::Yuv420);
    let float = FrameFormat {
        sample: SampleFormat::F32,
        ..original
    };
    let inputs = [
        Offered::Proposed(StreamFormat::Frames(original)),
        Offered::Unknown,
        Offered::Proposed(StreamFormat::Frames(original)),
    ];
    let offered = StreamFormat::Frames(float);

    let accepted =
        Merge::accepts_input(&capabilities, &arguments, 1, offered, &inputs).expect("frames readable");

    assert_eq!(accepted, FormatSet::of([SampleFormat::F32]));
}

#[test]
fn produces_refuses_an_unbound_slot() {
    let capabilities = capabilities();
    let arguments = rejoin();
    let original = yuv(64, 32, Subsampling::Yuv420);
    let luma = extracted(&original, PlaneName::Y).expect("carries Y");
    let chosen = [Some(StreamFormat::Frames(luma)), None];

    let refused = Merge::produces(&capabilities, &arguments, &chosen);

    assert!(refused.is_err());
}
