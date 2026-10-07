use av_toolbox_core::{FilterDefinition, Parallelism, StageDefinition};
use av_toolbox_cubek::cnn::{Candidate, LayerRun};
use av_toolbox_cubek::dpir::Variant;
use av_toolbox_gpu::{
    ColourInfo,
    ElementType,
    FrameFormat,
    FrameLayout,
    SampleFormat,
    StreamFormat,
    Subsampling,
};
use av_toolbox_models::Store;
use serde_json::json;

use crate::cnn::PrecisionName;
use crate::deblock::Dpir;
use crate::deblock::dpir::{
    DpirArgs,
    ModelChoice,
    Runs,
    StrengthArg,
    accepted_samples,
    checked_runs,
    level_summary,
    locate,
};
use crate::sample_checks::{assert_integers_and_floats_accepted_and_mask_refused, capabilities_computing};

const STAGE: &str = "deblock/dpir";

fn args(value: serde_json::Value) -> DpirArgs {
    serde_json::from_value(value).expect("parses")
}

fn format(layout: FrameLayout, sample: SampleFormat) -> FrameFormat {
    FrameFormat {
        width: 48,
        height: 40,
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

fn rgb() -> FrameFormat {
    format(FrameLayout::Rgb, SampleFormat::U8)
}

fn grey() -> FrameFormat {
    format(FrameLayout::Grey, SampleFormat::U8)
}

fn runs(value: serde_json::Value, input: FrameFormat) -> Runs {
    let arguments = args(value);

    checked_runs(&arguments, &input).expect("accepted")
}

fn refusal(value: serde_json::Value, input: FrameFormat) -> String {
    let arguments = args(value);
    let error = checked_runs(&arguments, &input).expect_err("should be refused");

    error.to_string()
}

#[test]
fn the_defaults_are_strength_ten_auto_every_plane_at_f16() {
    let parsed = args(json!({}));

    assert_eq!(parsed.strength, StrengthArg::One(10.0));
    assert_eq!(parsed.model, ModelChoice::Auto);
    assert_eq!(parsed.planes, None);
    assert_eq!(parsed.precision, PrecisionName::F16);
}

#[test]
fn a_strength_parses_as_a_number_or_a_list() {
    let one = args(json!({ "strength": 25 }));
    let list = args(json!({ "strength": [5, 50.5] }));

    assert_eq!(one.strength, StrengthArg::One(25.0));
    assert_eq!(list.strength, StrengthArg::PerPlane(vec![5.0, 50.5]));
}

#[test]
fn every_model_name_parses_and_an_unknown_argument_is_refused() {
    let gray = args(json!({ "model": "gray" }));
    let colour = args(json!({ "model": "colour" }));
    let unknown = serde_json::from_value::<DpirArgs>(json!({ "tiles": 4 }));

    assert_eq!(gray.model, ModelChoice::Gray);
    assert_eq!(colour.model, ModelChoice::Colour);

    let message = unknown.expect_err("tiles is not an argument").to_string();
    assert!(message.contains("tiles"), "{message}");
}

#[test]
fn auto_picks_gray_on_grey_and_yuv_and_colour_on_rgb() {
    let on_grey = runs(json!({}), grey());
    let on_yuv = runs(json!({}), yuv420());
    let on_rgb = runs(json!({}), rgb());

    assert_eq!(on_grey.variant(), Variant::Gray);
    assert_eq!(on_yuv.variant(), Variant::Gray);
    assert_eq!(on_rgb, Runs::Colour { strength: 10.0 });
}

#[test]
fn the_colour_model_on_yuv_is_refused_naming_both_resample_stages() {
    let message = refusal(json!({ "model": "colour" }), yuv420());
    let grey_message = refusal(json!({ "model": "colour" }), grey());

    for text in [message, grey_message] {
        assert!(text.contains(STAGE), "{text}");
        assert!(text.contains("resample/chroma"), "{text}");
        assert!(text.contains("resample/matrix"), "{text}");
        assert!(text.contains("gray"), "{text}");
    }
}

#[test]
fn planes_or_a_strength_list_with_the_colour_model_are_refused() {
    let with_planes = refusal(json!({ "planes": ["r"] }), rgb());
    let with_list = refusal(json!({ "strength": [5, 5, 5] }), rgb());

    assert!(with_planes.contains(STAGE), "{with_planes}");
    assert!(with_planes.contains("planes"), "{with_planes}");
    assert!(with_planes.contains("gray"), "{with_planes}");
    assert!(with_list.contains(STAGE), "{with_list}");
    assert!(with_list.contains("strength"), "{with_list}");
    assert!(with_list.contains("gray"), "{with_list}");
}

#[test]
fn the_gray_model_runs_every_plane_by_default_at_one_strength() {
    let found = runs(json!({ "strength": 20 }), yuv420());

    assert_eq!(
        found,
        Runs::Gray {
            strengths: vec![Some(20.0), Some(20.0), Some(20.0)]
        }
    );
}

#[test]
fn listed_strengths_follow_the_order_planes_lists_them_in() {
    let found = runs(json!({ "planes": ["v", "y"], "strength": [5, 50] }), yuv420());

    assert_eq!(
        found,
        Runs::Gray {
            strengths: vec![Some(50.0), None, Some(5.0)]
        }
    );
}

#[test]
fn the_gray_model_can_run_on_rgb_when_asked() {
    let found = runs(json!({ "model": "gray", "planes": ["g"] }), rgb());

    assert_eq!(
        found,
        Runs::Gray {
            strengths: vec![None, Some(10.0), None]
        }
    );
}

#[test]
fn a_strength_list_of_the_wrong_length_is_refused() {
    let without_planes = refusal(json!({ "strength": [5, 50] }), yuv420());
    let with_planes = refusal(json!({ "planes": ["y"], "strength": [5, 50] }), yuv420());

    for message in [without_planes, with_planes] {
        assert!(message.contains(STAGE), "{message}");
        assert!(message.contains("strength"), "{message}");
    }
}

#[test]
fn a_strength_outside_0_to_255_is_refused_and_the_ends_are_accepted() {
    let below = refusal(json!({ "strength": -1 }), yuv420());
    let above = refusal(json!({ "strength": 256 }), yuv420());
    let lowest = runs(json!({ "strength": 0 }), yuv420());
    let highest = runs(json!({ "strength": 255 }), yuv420());

    for message in [below, above] {
        assert!(message.contains(STAGE), "{message}");
        assert!(message.contains("strength"), "{message}");
    }

    assert_eq!(
        lowest,
        Runs::Gray {
            strengths: vec![Some(0.0); 3]
        }
    );
    assert_eq!(
        highest,
        Runs::Gray {
            strengths: vec![Some(255.0); 3]
        }
    );
}

#[test]
fn a_strength_list_entry_outside_0_to_255_is_refused() {
    let below = refusal(json!({ "strength": [-1, 10, 10] }), yuv420());
    let above = refusal(json!({ "strength": [10, 10, 256] }), yuv420());

    for message in [below, above] {
        assert!(message.contains(STAGE), "{message}");
        assert!(message.contains("strength"), "{message}");
    }
}

#[test]
fn a_strength_that_is_not_a_number_is_refused() {
    let arguments = DpirArgs {
        strength: StrengthArg::One(f32::NAN),
        ..args(json!({}))
    };
    let input = yuv420();
    let error = checked_runs(&arguments, &input).expect_err("NaN is out of range");
    let message = error.to_string();

    assert!(message.contains(STAGE), "{message}");
    assert!(message.contains("strength"), "{message}");
}

#[test]
fn a_plane_the_layout_lacks_is_refused() {
    let message = refusal(json!({ "planes": ["r"] }), yuv420());

    assert!(message.contains(STAGE), "{message}");
}

#[test]
fn integers_and_floats_are_accepted_and_a_mask_is_refused() {
    assert_integers_and_floats_accepted_and_mask_refused(STAGE, FrameLayout::Grey, format, accepted_samples);
}

#[test]
fn a_missing_pack_names_the_pull_command() {
    let root = tempfile::tempdir().expect("a temp dir");
    let store = Store::new(root.path().to_path_buf(), "v0.1.0");

    let error = locate(&store, Variant::Colour).expect_err("nothing is installed");
    let message = error.to_string();

    assert!(message.contains(STAGE), "{message}");
    assert!(message.contains("colour"), "{message}");
    assert!(message.contains("drunet_deblock_color.bin"), "{message}");
    assert!(message.contains("av-toolbox pull dpir"), "{message}");
}

#[test]
fn the_stage_runs_as_a_single_instance() {
    let arguments = args(json!({}));

    let parallelism = Dpir::parallelism(&arguments);

    assert_eq!(parallelism, Parallelism::Stream);
}

#[test]
fn f16_without_f16_arithmetic_is_refused_suggesting_f32() {
    let capabilities = capabilities_computing(&[ElementType::F32]);
    let arguments = args(json!({ "precision": "f16" }));
    let input = yuv420();
    let chosen = [Some(StreamFormat::Frames(input))];

    let error = Dpir::produces(&capabilities, &arguments, &chosen).expect_err("no f16 arithmetic");
    let message = error.to_string();

    assert!(message.contains(STAGE), "{message}");
    assert!(message.contains("\"f32\""), "{message}");
}

#[test]
fn the_level_summary_names_each_level_by_its_features() {
    let layer = LayerRun {
        forced: false,
        fused: true,
        candidate: Some(Candidate::FusedCubek),
        winograd_offered: false,
    };
    let deepest = LayerRun {
        candidate: None,
        ..layer
    };
    let mut layers = vec![layer; 56];
    layers[24..32].fill(deepest);

    let summary = level_summary(&layers);

    assert_eq!(
        summary,
        "64: fused-cubek x16, 128: fused-cubek x16, 256: fused-cubek x16, 512: cubek x8"
    );
}
