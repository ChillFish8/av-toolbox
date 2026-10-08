use av_toolbox_core::{
    FilterDefinition,
    InputArgument,
    Parallelism,
    PipelineRegistry,
    StageDefinition,
    StageHandle,
    StageKind,
    StreamType,
};
use av_toolbox_cubek::deblock::h264::{Thresholds, thresholds};
use av_toolbox_cubek::matrix::MatrixSamples;
use av_toolbox_gpu::{
    ColourInfo,
    Depth as BitDepth,
    FrameFormat,
    FrameLayout,
    SampleFormat,
    StreamFormat,
    Subsampling,
};
use serde_json::json;

use crate::deblock::Qed;
use crate::deblock::qed::{
    QedArgs,
    QedChroma,
    Treatment,
    accepted_samples,
    checked_qed,
    layout_planes,
    lowpass_sizes,
};
use crate::sample_checks::{assert_integers_and_floats_accepted_and_mask_refused, capabilities_computing};

const STAGE: &str = "deblock/qed";

fn args(value: serde_json::Value) -> QedArgs {
    serde_json::from_value(value).expect("parses")
}

fn format(layout: FrameLayout, sample: SampleFormat) -> FrameFormat {
    FrameFormat {
        width: 64,
        height: 48,
        layout,
        sample,
        colour: ColourInfo::default(),
        frame_rate: None,
        field_order: None,
    }
}

fn yuv420() -> FrameFormat {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };

    format(layout, SampleFormat::U8)
}

fn refusal(value: serde_json::Value, input: FrameFormat) -> String {
    let arguments = args(value);
    let error = checked_qed(&arguments, &input).expect_err("should be refused");

    error.to_string()
}

fn treatments(value: serde_json::Value, input: FrameFormat) -> Vec<Treatment> {
    let arguments = args(value);
    let deblocking = checked_qed(&arguments, &input).expect("accepted");

    deblocking.treatments
}

/// Every message in the chain the pipeline refuses `value` with, one per line.
fn prepared_refusal(value: serde_json::Value) -> String {
    let mut registry = PipelineRegistry::new();
    crate::register(&mut registry).expect("the filters register");

    let handle = StageHandle::new(1);
    let arguments = [InputArgument::positional(handle, StreamType::Frames)];
    let prepared = registry.prepare(StageKind::Filter, STAGE, &value, &arguments);
    let Err(error) = prepared else {
        panic!("{STAGE} should refuse {value}");
    };

    let mut lines = vec![error.to_string()];
    let mut cause = std::error::Error::source(&error);
    while let Some(inner) = cause {
        lines.push(inner.to_string());
        cause = inner.source();
    }

    lines.join("\n")
}

#[test]
fn the_defaults_are_quant_24_26_offsets_1_and_2_and_qed_chroma() {
    let parsed = args(json!({}));

    let deblocking = checked_qed(&parsed, &yuv420()).expect("accepted");

    assert_eq!(parsed.quant, vec![24, 26]);
    assert_eq!(parsed.aoffset, vec![1, 1]);
    assert_eq!(parsed.boffset, vec![2, 2]);
    assert_eq!(parsed.chroma, QedChroma::Qed);
    assert_eq!(parsed.planes, None);
    assert_eq!(deblocking.treatments, vec![Treatment::Qed; 3]);
    assert_eq!(deblocking.normal, thresholds(24, 1, 2, MatrixSamples::U8));
    assert_eq!(deblocking.strong, thresholds(26, 1, 2, MatrixSamples::U8));
}

#[test]
fn an_unknown_argument_is_refused() {
    let parsed = serde_json::from_value::<QedArgs>(json!({ "strength": 3 }));
    let error = parsed.expect_err("strength is not an argument");
    let message = error.to_string();

    assert!(message.contains("strength"), "{message}");
}

#[test]
fn quant_outside_0_to_60_is_refused_naming_its_half() {
    let border = refusal(json!({ "quant": [61, 26] }), yuv420());
    let interior = refusal(json!({ "quant": [24, -1] }), yuv420());

    for (message, half) in [(border, "quant[0] (border)"), (interior, "quant[1] (interior)")] {
        assert!(message.contains(STAGE), "{message}");
        assert!(message.contains(half), "{message}");
        assert!(message.contains("between 0 and 60"), "{message}");
    }

    for quant in [[0, 60], [60, 0]] {
        let arguments = args(json!({ "quant": quant }));
        let checked = checked_qed(&arguments, &yuv420());

        assert!(checked.is_ok(), "quant {quant:?} is in range");
    }
}

#[test]
fn a_pair_of_the_wrong_length_is_refused_naming_the_argument() {
    let short = refusal(json!({ "quant": [24] }), yuv420());
    let long = refusal(json!({ "aoffset": [1, 1, 1] }), yuv420());
    let empty = refusal(json!({ "boffset": [] }), yuv420());

    for (message, name) in [(short, "quant"), (long, "aoffset"), (empty, "boffset")] {
        assert!(message.contains(STAGE), "{message}");
        assert!(message.contains(name), "{message}");
        assert!(message.contains("two values"), "{message}");
    }
}

#[test]
fn an_unknown_chroma_is_refused_naming_chroma() {
    let parsed = serde_json::from_value::<QedArgs>(json!({ "chroma": "soft" }));
    let message = parsed.expect_err("soft is not a chroma").to_string();
    let prepared = prepared_refusal(json!({ "chroma": "soft" }));

    assert!(
        message.contains("soft") && message.contains("normal"),
        "{message}"
    );
    assert!(prepared.contains("/chroma"), "{prepared}");
}

#[test]
fn normal_or_strong_chroma_on_rgb_is_refused_naming_chroma() {
    let rgb = format(FrameLayout::Rgb, SampleFormat::U8);

    for chroma in ["normal", "strong"] {
        let message = refusal(json!({ "chroma": chroma }), rgb);

        assert!(message.contains(STAGE), "{message}");
        assert!(message.contains("chroma"), "{message}");
        assert!(message.contains("RGB"), "{message}");
        assert!(message.contains(chroma), "{message}");
    }
}

#[test]
fn rgb_under_qed_treats_every_plane_like_luma() {
    let rgb = format(FrameLayout::Rgb, SampleFormat::U8);

    let found = treatments(json!({}), rgb);

    assert_eq!(found, vec![Treatment::Qed; 3]);
}

#[test]
fn chroma_decides_how_selected_chroma_planes_are_treated() {
    let normal = treatments(json!({ "chroma": "normal" }), yuv420());
    let strong = treatments(json!({ "chroma": "strong" }), yuv420());
    let only_u = treatments(json!({ "chroma": "strong", "planes": ["u"] }), yuv420());

    assert_eq!(
        normal,
        vec![Treatment::Qed, Treatment::NormalPass, Treatment::NormalPass]
    );
    assert_eq!(
        strong,
        vec![Treatment::Qed, Treatment::StrongPass, Treatment::StrongPass]
    );
    assert_eq!(
        only_u,
        vec![Treatment::Untouched, Treatment::StrongPass, Treatment::Untouched]
    );
}

#[test]
fn a_chroma_layout_takes_chroma_for_both_planes() {
    let layout = FrameLayout::Chroma {
        subsampling: Subsampling::Yuv420,
    };
    let input = format(layout, SampleFormat::U8);

    let qed = treatments(json!({}), input);
    let normal = treatments(json!({ "chroma": "normal" }), input);
    let strong = treatments(json!({ "chroma": "strong" }), input);

    assert_eq!(qed, vec![Treatment::Qed; 2]);
    assert_eq!(normal, vec![Treatment::NormalPass; 2]);
    assert_eq!(strong, vec![Treatment::StrongPass; 2]);
}

#[test]
fn grey_ignores_chroma() {
    let grey = format(FrameLayout::Grey, SampleFormat::U8);

    let found = treatments(json!({ "chroma": "strong" }), grey);

    assert_eq!(found, vec![Treatment::Qed]);
}

#[test]
fn bad_planes_are_refused() {
    let empty = refusal(json!({ "planes": [] }), yuv420());
    let twice = refusal(json!({ "planes": ["u", "u"] }), yuv420());
    let missing = refusal(json!({ "planes": ["r"] }), yuv420());

    for message in [empty, twice, missing] {
        assert!(message.contains(STAGE), "{message}");
        assert!(message.contains("planes"), "{message}");
    }
}

#[test]
fn integers_and_floats_are_accepted_and_a_mask_is_refused() {
    assert_integers_and_floats_accepted_and_mask_refused(STAGE, FrameLayout::Grey, format, accepted_samples);
}

#[test]
fn each_pass_takes_its_own_thresholds_at_the_inputs_depth() {
    let ten_bit = SampleFormat::U16 { depth: BitDepth::TEN };
    let input = format(FrameLayout::Grey, ten_bit);
    let arguments = args(json!({ "quant": [30, 40], "aoffset": [3, -2], "boffset": [-1, 4] }));

    let deblocking = checked_qed(&arguments, &input).expect("accepted");

    let samples = MatrixSamples::U16 { bits: 10 };
    assert_eq!(deblocking.samples, samples);
    assert_eq!(deblocking.normal, thresholds(30, 3, -1, samples));
    assert_eq!(deblocking.strong, thresholds(40, -2, 4, samples));
}

#[test]
fn a_qed_plane_lays_out_both_passes_and_a_chroma_pass_only_its_own() {
    let odd = FrameFormat {
        width: 70,
        height: 46,
        ..yuv420()
    };
    let arguments = args(json!({ "chroma": "normal" }));
    let deblocking = checked_qed(&arguments, &odd).expect("accepted");

    let planes = layout_planes(&odd, &deblocking);

    let sizes: Vec<(u32, u32)> = planes.iter().map(|plane| (plane.width, plane.height)).collect();
    let limits: Vec<Thresholds> = planes.iter().map(|plane| plane.thresholds).collect();
    let normal = deblocking.normal;
    let strong = deblocking.strong;
    assert_eq!(sizes, vec![(72, 48), (72, 48), (36, 24), (36, 24)]);
    assert_eq!(limits, vec![normal, strong, normal, normal]);
}

#[test]
fn the_lowpass_extends_each_plane_to_its_share_of_a_frame_rounded_to_16() {
    let odd_420 = FrameFormat {
        width: 70,
        height: 46,
        ..yuv420()
    };
    let odd_422 = FrameFormat {
        width: 71,
        height: 45,
        layout: FrameLayout::Yuv {
            subsampling: Subsampling::Yuv422,
        },
        ..yuv420()
    };

    let odd_chroma = FrameFormat {
        width: 70,
        height: 46,
        layout: FrameLayout::Chroma {
            subsampling: Subsampling::Yuv420,
        },
        ..yuv420()
    };

    let sizes_420 = lowpass_sizes(&odd_420);
    let sizes_422 = lowpass_sizes(&odd_422);

    assert_eq!(sizes_420, vec![(80, 48), (40, 24), (40, 24)]);
    assert_eq!(sizes_422, vec![(80, 48), (40, 48), (40, 48)]);
    assert_eq!(lowpass_sizes(&odd_chroma), vec![(80, 48), (80, 48)]);
}

#[test]
fn the_output_is_the_input_format_and_the_stage_runs_frame_by_frame() {
    let capabilities = capabilities_computing(&[]);
    let arguments = args(json!({}));
    let input = yuv420();
    let chosen = [Some(StreamFormat::Frames(input))];

    let outputs = Qed::produces(&capabilities, &arguments, &chosen).expect("accepted");

    let output = outputs.template.with_sample(input.sample);
    assert_eq!(output.as_frames(), Some(&input));
    assert_eq!(Qed::parallelism(&arguments), Parallelism::Frame);
}
