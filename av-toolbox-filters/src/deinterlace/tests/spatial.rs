use av_toolbox_core::{FilterDefinition, StageError};
use av_toolbox_gpu::{
    Capabilities,
    ColourInfo,
    FieldOrder,
    FrameFormat,
    FrameLayout,
    FrameRate,
    OutputFormats,
    SampleFormat,
    StreamFormat,
    Subsampling,
    default_accelerators,
};
use serde_json::json;

use crate::aa::eedi2::Eedi2Args;
use crate::aa::eedi3::Eedi3Args;
use crate::aa::sangnom::SangnomArgs;
use crate::deinterlace::Bwdif;
use crate::deinterlace::bwdif::BwdifArgs;
use crate::deinterlace::spatial::Spatial;

fn args(value: serde_json::Value) -> BwdifArgs {
    serde_json::from_value(value).expect("parses")
}

fn interlaced(width: u32, height: u32) -> FrameFormat {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };

    FrameFormat {
        width,
        height,
        layout,
        sample: SampleFormat::U8,
        colour: ColourInfo::default(),
        frame_rate: FrameRate::new(25, 1),
        field_order: Some(FieldOrder::TopFirst),
    }
}

fn produces(value: serde_json::Value, input: FrameFormat) -> Result<OutputFormats, StageError> {
    let accelerator = default_accelerators()[0];
    let capabilities = Capabilities::builder(accelerator)
        .plane_size(32, 32)
        .max_units_per_cube(1024)
        .max_cube_count(65535, 65535, 65535)
        .build();
    let parsed = args(value);
    let chosen = [Some(StreamFormat::Frames(input))];

    Bwdif::produces(&capabilities, &parsed, &chosen)
}

fn refusal(value: serde_json::Value, input: FrameFormat) -> String {
    let error = produces(value, input).expect_err("should be refused");

    error.to_string()
}

#[test]
fn the_default_is_cubic() {
    let parsed = args(json!({}));

    assert!(matches!(parsed.spatial, Spatial::Cubic {}));
}

#[test]
fn cubic_takes_no_other_field() {
    let parsed = serde_json::from_value::<BwdifArgs>(json!({ "spatial": { "kind": "cubic", "alpha": 0.2 } }));
    let message = parsed.expect_err("cubic has no alpha").to_string();

    assert!(message.contains("alpha"), "{message}");
}

#[test]
fn an_unknown_kind_is_refused() {
    let parsed = serde_json::from_value::<BwdifArgs>(json!({ "spatial": { "kind": "nnedi3" } }));

    assert!(parsed.is_err());
}

#[test]
fn eedi2_defaults_are_aa_eedi2_defaults() {
    let parsed = args(json!({ "spatial": { "kind": "eedi2" } }));
    let stage: Eedi2Args = serde_json::from_value(json!({})).expect("parses");
    let Spatial::Eedi2(tuning) = parsed.spatial else {
        panic!("expected eedi2");
    };

    assert_eq!(tuning, stage.tuning());
}

#[test]
fn eedi3_defaults_are_aa_eedi3_defaults() {
    let parsed = args(json!({ "spatial": { "kind": "eedi3" } }));
    let stage: Eedi3Args = serde_json::from_value(json!({})).expect("parses");
    let Spatial::Eedi3(spatial) = parsed.spatial else {
        panic!("expected eedi3");
    };

    assert_eq!(spatial.tuning(), stage.tuning());
}

#[test]
fn sangnom_defaults_are_aa_sangnom_defaults() {
    let parsed = args(json!({ "spatial": { "kind": "sangnom" } }));
    let stage: SangnomArgs = serde_json::from_value(json!({})).expect("parses");
    let Spatial::Sangnom(tuning) = parsed.spatial else {
        panic!("expected sangnom");
    };

    assert_eq!(tuning.aa, stage.aa);
}

#[test]
fn planes_and_direction_are_not_spatial_settings() {
    let planes =
        serde_json::from_value::<BwdifArgs>(json!({ "spatial": { "kind": "eedi3", "planes": ["y"] } }));
    let direction =
        serde_json::from_value::<BwdifArgs>(json!({ "spatial": { "kind": "sangnom", "direction": "both" } }));

    assert!(planes.is_err());
    assert!(direction.is_err());
}

#[test]
fn a_bad_eedi2_setting_is_refused_naming_bwdif() {
    let input = interlaced(32, 32);
    let message = refusal(json!({ "spatial": { "kind": "eedi2", "maxd": 30 } }), input);

    assert!(message.contains("deinterlace/bwdif"), "{message}");
    assert!(message.contains("maxd"), "{message}");
}

#[test]
fn a_bad_eedi3_setting_is_refused_naming_bwdif() {
    let input = interlaced(32, 32);
    let message = refusal(
        json!({ "spatial": { "kind": "eedi3", "alpha": 0.7, "beta": 0.7 } }),
        input,
    );

    assert!(message.contains("deinterlace/bwdif"), "{message}");
}

#[test]
fn a_bad_eedi3_threshold_count_names_bwdif() {
    let parsed =
        serde_json::from_value::<BwdifArgs>(json!({ "spatial": { "kind": "eedi3", "vthresh": [1, 2] } }));
    let message = parsed.expect_err("two thresholds").to_string();

    assert!(message.contains("deinterlace/bwdif"), "{message}");
}

#[test]
fn a_bad_sangnom_setting_is_refused_naming_bwdif() {
    let input = interlaced(32, 32);
    let message = refusal(json!({ "spatial": { "kind": "sangnom", "aa": 200 } }), input);

    assert!(message.contains("deinterlace/bwdif"), "{message}");
    assert!(message.contains("aa"), "{message}");
}

#[test]
fn a_plane_too_short_for_eedi2_is_refused_naming_bwdif() {
    // 4:2:0 chroma of a 12 row frame is 6 rows, short of EEDI2's 14.
    let input = interlaced(32, 12);
    let message = refusal(json!({ "spatial": { "kind": "eedi2" } }), input);

    assert!(message.contains("deinterlace/bwdif"), "{message}");

    let luma_only = produces(
        json!({ "planes": ["y"], "spatial": { "kind": "eedi2" } }),
        interlaced(32, 14),
    );
    assert!(luma_only.is_ok());
}
