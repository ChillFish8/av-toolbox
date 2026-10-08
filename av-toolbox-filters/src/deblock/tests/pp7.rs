use av_toolbox_core::{FilterDefinition, Parallelism, StageDefinition};
use av_toolbox_cubek::deblock::pp7::Pp7Mode;
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

use super::prepared_refusal;
use crate::deblock::Pp7;
use crate::deblock::pp7::{Mode, Pp7Args, accepted_samples, aligned_sizes, checked_pp7};
use crate::sample_checks::{assert_integers_and_floats_accepted_and_mask_refused, capabilities_computing};

const STAGE: &str = "deblock/pp7";

fn args(value: serde_json::Value) -> Pp7Args {
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

fn yuv(subsampling: Subsampling) -> FrameLayout {
    FrameLayout::Yuv { subsampling }
}

fn yuv420() -> FrameFormat {
    let layout = yuv(Subsampling::Yuv420);

    format(layout, SampleFormat::U8)
}

fn refusal(arguments: &Pp7Args, input: FrameFormat) -> String {
    let error = checked_pp7(arguments, &input).expect_err("should be refused");

    error.to_string()
}

#[test]
fn the_defaults_are_qp_2_hard_and_every_plane() {
    let parsed = args(json!({}));
    let input = yuv420();

    let deblocking = checked_pp7(&parsed, &input).expect("accepted");

    assert_eq!(parsed.qp, 2.0);
    assert_eq!(parsed.mode, Mode::Hard);
    assert_eq!(parsed.planes, None);
    assert_eq!(deblocking.selected, vec![true, true, true]);
    assert_eq!(deblocking.samples, MatrixSamples::U8);
    assert_eq!(deblocking.qp, 2.0);
    assert_eq!(deblocking.mode, Pp7Mode::Hard);
}

#[test]
fn an_unknown_argument_is_refused() {
    let parsed = serde_json::from_value::<Pp7Args>(json!({ "quant": 3 }));
    let error = parsed.expect_err("quant is not an argument");
    let message = error.to_string();

    assert!(message.contains("quant"), "{message}");
}

#[test]
fn qp_outside_1_to_63_or_not_finite_is_refused_naming_the_range() {
    let refused = [0.99, 63.01, -2.0, f64::NAN, f64::INFINITY];

    for qp in refused {
        let mut arguments = args(json!({}));
        arguments.qp = qp;
        let input = yuv420();

        let message = refusal(&arguments, input);

        assert!(message.contains(STAGE), "{message}");
        assert!(message.contains("qp between 1 and 63"), "{message}");
    }
}

#[test]
fn qp_at_either_end_and_fractional_qp_are_kept_as_they_are() {
    for qp in [1.0, 2.5, 13.37, 63.0] {
        let arguments = args(json!({ "qp": qp }));
        let input = yuv420();

        let deblocking = checked_pp7(&arguments, &input).expect("in range");

        assert_eq!(deblocking.qp, qp);
    }
}

#[test]
fn each_mode_reaches_the_kernel() {
    let modes = [
        ("hard", Pp7Mode::Hard),
        ("soft", Pp7Mode::Soft),
        ("medium", Pp7Mode::Medium),
    ];

    for (name, expected) in modes {
        let arguments = args(json!({ "mode": name }));
        let input = yuv420();

        let deblocking = checked_pp7(&arguments, &input).expect("accepted");

        assert_eq!(deblocking.mode, expected, "{name}");
    }
}

#[test]
fn an_unknown_mode_is_refused_naming_mode() {
    let parsed = serde_json::from_value::<Pp7Args>(json!({ "mode": "strong" }));
    let error = parsed.expect_err("strong is not a mode");
    let message = error.to_string();
    let prepared = prepared_refusal(STAGE, json!({ "mode": "strong" }));

    assert!(
        message.contains("strong") && message.contains("medium"),
        "{message}"
    );
    assert!(prepared.contains("/mode"), "{prepared}");
}

#[test]
fn bad_planes_are_refused() {
    let no_planes = args(json!({ "planes": [] }));
    let repeated = args(json!({ "planes": ["u", "u"] }));
    let absent = args(json!({ "planes": ["r"] }));
    let input = yuv420();
    let empty = refusal(&no_planes, input);
    let twice = refusal(&repeated, input);
    let missing = refusal(&absent, input);

    for message in [empty, twice, missing] {
        assert!(message.contains(STAGE), "{message}");
        assert!(message.contains("planes"), "{message}");
    }
}

#[test]
fn planes_choose_which_planes_are_deblocked() {
    let arguments = args(json!({ "planes": ["v", "y"] }));
    let input = yuv420();

    let deblocking = checked_pp7(&arguments, &input).expect("accepted");

    assert_eq!(deblocking.selected, vec![true, false, true]);
}

#[test]
fn integers_and_floats_are_accepted_and_a_mask_is_refused() {
    assert_integers_and_floats_accepted_and_mask_refused(STAGE, FrameLayout::Grey, format, accepted_samples);
}

#[test]
fn a_half_float_frame_is_refused() {
    let input = format(FrameLayout::Grey, SampleFormat::F16);
    let arguments = args(json!({}));

    let message = refusal(&arguments, input);

    assert!(message.contains(STAGE), "{message}");
    assert!(message.contains("F16"), "{message}");
}

#[test]
fn deeper_and_float_frames_keep_their_sample_kind() {
    let ten_bit = SampleFormat::U16 { depth: BitDepth::TEN };
    let deep = format(FrameLayout::Grey, ten_bit);
    let float = format(FrameLayout::Rgb, SampleFormat::F32);
    let arguments = args(json!({}));

    let deep_deblocking = checked_pp7(&arguments, &deep).expect("accepted");
    let float_deblocking = checked_pp7(&arguments, &float).expect("accepted");

    assert_eq!(deep_deblocking.samples, MatrixSamples::U16 { bits: 10 });
    assert_eq!(float_deblocking.samples, MatrixSamples::F32);
}

#[test]
fn each_plane_aligns_to_its_share_of_a_frame_rounded_up_to_16() {
    let odd_420 = FrameFormat {
        width: 70,
        height: 46,
        ..yuv420()
    };
    let odd_422 = FrameFormat {
        width: 46,
        height: 22,
        layout: yuv(Subsampling::Yuv422),
        ..yuv420()
    };
    let grey = FrameFormat {
        width: 47,
        height: 30,
        layout: FrameLayout::Grey,
        ..yuv420()
    };
    let rgb = FrameFormat {
        width: 47,
        height: 30,
        layout: FrameLayout::Rgb,
        ..yuv420()
    };
    let chroma_only = FrameFormat {
        width: 70,
        height: 46,
        layout: FrameLayout::Chroma {
            subsampling: Subsampling::Yuv420,
        },
        ..yuv420()
    };

    let odd_420_sizes = aligned_sizes(&odd_420);
    let odd_422_sizes = aligned_sizes(&odd_422);
    let grey_sizes = aligned_sizes(&grey);
    let rgb_sizes = aligned_sizes(&rgb);
    let chroma_only_sizes = aligned_sizes(&chroma_only);

    assert_eq!(odd_420_sizes, vec![(80, 48), (40, 24), (40, 24)]);
    assert_eq!(odd_422_sizes, vec![(48, 32), (24, 32), (24, 32)]);
    assert_eq!(grey_sizes, vec![(48, 32)]);
    assert_eq!(rgb_sizes, vec![(48, 32), (48, 32), (48, 32)]);
    assert_eq!(chroma_only_sizes, vec![(80, 48), (80, 48)]);
}

#[test]
fn the_output_is_the_input_format_and_the_stage_runs_frame_by_frame() {
    let capabilities = capabilities_computing(&[]);
    let arguments = args(json!({}));
    let input = yuv420();
    let chosen = [Some(StreamFormat::Frames(input))];

    let outputs = Pp7::produces(&capabilities, &arguments, &chosen).expect("accepted");

    let output = outputs.template.with_sample(input.sample);
    let output_frames = output.as_frames();
    let parallelism = Pp7::parallelism(&arguments);
    assert_eq!(output_frames, Some(&input));
    assert_eq!(parallelism, Parallelism::Frame);
}
