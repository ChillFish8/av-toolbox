use av_toolbox_gpu::{ColourInfo, FrameFormat, FrameLayout, FrameRate, SampleFormat, Subsampling};
use avtbx_vship::DisplayModel;
use serde_json::json;

use crate::vship::cvvdp::{CvvdpArgs, display_model, frame_rate, map_size};
use crate::vship::output::Output;

fn args_from(value: serde_json::Value) -> CvvdpArgs {
    serde_json::from_value(value).expect("parses")
}

fn clip(width: u32, height: u32, frame_rate: Option<FrameRate>) -> FrameFormat {
    FrameFormat {
        width,
        height,
        layout: FrameLayout::Yuv {
            subsampling: Subsampling::Yuv420,
        },
        sample: SampleFormat::U8,
        colour: ColourInfo::default(),
        frame_rate,
        field_order: None,
    }
}

#[test]
fn args_default_to_the_spec() {
    let args = args_from(json!({}));

    assert_eq!(args.display, "standard_fhd");
    assert_eq!(args.display_config, None);
    assert!(!args.resize_to_display);
    assert!(args.temporal);
    assert_eq!(args.output, Output::Clip);
    assert_eq!(args.gpu, None);
}

#[test]
fn unknown_args_are_refused() {
    let parsed = serde_json::from_value::<CvvdpArgs>(json!({ "model": "standard_4k" }));
    let error = parsed.expect_err("should be refused");

    assert!(error.to_string().contains("unknown field `model`"), "{error}");
}

#[test]
fn every_built_in_display_name_is_known() {
    let expected = [
        ("standard_4k", DisplayModel::Standard4k),
        ("standard_fhd", DisplayModel::StandardFhd),
        ("standard_hdr_pq", DisplayModel::StandardHdrPq),
        ("standard_hdr_hlg", DisplayModel::StandardHdrHlg),
        ("standard_hdr_dark", DisplayModel::StandardHdrDark),
        ("standard_hdr_linear_zoom", DisplayModel::StandardHdrLinearZoom),
    ];

    for (name, model) in expected {
        let args = args_from(json!({ "display": name }));
        assert_eq!(display_model(&args).expect("a built-in name"), model);
    }
}

#[test]
fn an_unknown_display_is_an_error() {
    let args = args_from(json!({ "display": "my_tv" }));

    let error = display_model(&args).expect_err("an unknown name");

    let message = error.to_string();
    assert!(message.contains("my_tv"), "{message}");
    assert!(message.contains("standard_hdr_linear_zoom"), "{message}");
}

#[test]
fn a_display_config_makes_a_custom_display() {
    let args = args_from(json!({ "display": "my_tv", "display_config": "{}" }));

    let model = display_model(&args).expect("a custom display");

    let expected = DisplayModel::Custom {
        name: "my_tv".to_string(),
        json: "{}".to_string(),
    };
    assert_eq!(model, expected);
}

#[test]
fn the_clip_rate_is_used() {
    let format = clip(64, 64, FrameRate::new(24000, 1001));

    let rate = frame_rate(&format).expect("a rate");

    assert!((rate - 23.976).abs() < 1e-3, "{rate}");
}

#[test]
fn a_clip_with_no_frame_rate_is_an_error() {
    let format = clip(64, 64, None);

    let error = frame_rate(&format).expect_err("the clip has no rate");

    let message = error.to_string();
    assert!(
        message.contains("cvvdp needs the clip's frame rate, and this clip has none"),
        "{message}"
    );
}

#[test]
fn a_clip_rate_above_1000_is_an_error() {
    let format = clip(64, 64, FrameRate::new(2001, 2));

    let error = frame_rate(&format).expect_err("out of range");

    let message = error.to_string();
    assert!(message.contains("above 0 and at most 1000"), "{message}");
}

#[test]
fn a_clip_rate_of_1000_is_kept() {
    let format = clip(64, 64, FrameRate::new(1000, 1));

    assert_eq!(frame_rate(&format).expect("a rate"), 1000.0);
}

#[test]
fn an_fps_arg_is_refused() {
    let parsed = serde_json::from_value::<CvvdpArgs>(json!({ "fps": 24.0 }));
    let error = parsed.expect_err("fps should be refused");

    assert!(error.to_string().contains("unknown field `fps`"), "{error}");
}

#[test]
fn the_map_is_the_clip_size_without_resizing() {
    let args = args_from(json!({ "display": "standard_4k" }));
    let format = clip(640, 360, None);
    let model = display_model(&args).expect("a built-in name");

    assert_eq!(map_size(&args, &model, &format).expect("a size"), (640, 360));
}

#[test]
fn resizing_matches_the_display_width_for_a_wide_clip() {
    let args = args_from(json!({ "resize_to_display": true }));
    let format = clip(1000, 400, None);
    let model = display_model(&args).expect("a built-in name");

    assert_eq!(map_size(&args, &model, &format).expect("a size"), (1920, 768));
}

#[test]
fn resizing_matches_the_display_height_for_a_tall_clip() {
    let args = args_from(json!({ "display": "standard_4k", "resize_to_display": true }));
    let format = clip(300, 400, None);
    let model = display_model(&args).expect("a built-in name");

    assert_eq!(map_size(&args, &model, &format).expect("a size"), (1620, 2160));
}

#[test]
fn resizing_reads_a_custom_display_resolution() {
    let config = r#"{ "my_tv": { "resolution": [1280, 720] } }"#;
    let args = args_from(json!({
        "display": "my_tv",
        "display_config": config,
        "resize_to_display": true,
    }));
    let format = clip(640, 360, None);
    let model = display_model(&args).expect("a custom display");

    assert_eq!(map_size(&args, &model, &format).expect("a size"), (1280, 720));
}

#[test]
fn resizing_keeps_a_built_in_resolution_a_config_does_not_override() {
    let config = r#"{ "standard_4k": { "max_luminance": 400 } }"#;
    let args = args_from(json!({
        "display": "standard_4k",
        "display_config": config,
        "resize_to_display": true,
    }));
    let format = clip(1920, 1080, None);
    let model = display_model(&args).expect("a custom display");

    assert_eq!(map_size(&args, &model, &format).expect("a size"), (3840, 2160));
}

#[test]
fn resizing_a_custom_display_with_no_resolution_is_an_error() {
    let config = r#"{ "my_tv": { "max_luminance": 400 } }"#;
    let args = args_from(json!({
        "display": "my_tv",
        "display_config": config,
        "resize_to_display": true,
    }));
    let format = clip(640, 360, None);
    let model = display_model(&args).expect("a custom display");

    let error = map_size(&args, &model, &format).expect_err("no resolution");

    assert!(error.to_string().contains("no resolution for my_tv"), "{error}");
}

#[cfg(feature = "gpu-tests")]
mod device {
    use av_toolbox_gpu::{FrameFormat, FrameLayout, FrameRate, HostFrame, MetaValue, SampleFormat};
    use serde_json::json;

    use crate::vship::tests::device::{
        Run,
        SIZE,
        blurred_luma,
        build_error,
        built,
        detailed_luma,
        f32_plane,
        frame,
        frame_float,
        run,
        summary_values,
        yuv420,
    };

    const STAGE: &str = "vship/cvvdp";

    const FRAMES: u64 = 4;

    fn clip() -> FrameFormat {
        FrameFormat {
            frame_rate: FrameRate::new(24, 1),
            ..yuv420(SampleFormat::U8)
        }
    }

    fn pairs(format: &FrameFormat, distorted_luma: fn(u32, u32) -> u32) -> Vec<(HostFrame, HostFrame)> {
        (0..FRAMES)
            .map(|index| {
                let reference = frame(format, index, detailed_luma);
                let distorted = frame(format, index, distorted_luma);
                (reference, distorted)
            })
            .collect()
    }

    fn scored(args: serde_json::Value, distorted_luma: fn(u32, u32) -> u32) -> Run {
        let format = clip();
        let mut filter = built(STAGE, args, format, format).expect("should build");
        let pairs = pairs(&format, distorted_luma);

        run(filter.as_mut(), pairs)
    }

    fn scores(run: &Run) -> Vec<f64> {
        run.frames
            .iter()
            .map(|frame| frame_float(frame, STAGE, "score"))
            .collect()
    }

    /// The summary's `score`, once its names are checked.
    fn summary_score(run: &Run) -> f64 {
        let values = summary_values(run, STAGE);
        let names: Vec<&str> = values.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names, ["frames", "score"]);
        assert_eq!(values[0].1, MetaValue::Int(FRAMES as i64));

        match values[1].1 {
            MetaValue::Float(score) => score,
            ref other => panic!("expected a float score, found {other:?}"),
        }
    }

    #[test]
    fn identical_clips_score_near_ten() {
        let run = scored(json!({}), detailed_luma);
        let per_frame = scores(&run);
        let summary = summary_score(&run);

        eprintln!("identical scores: {per_frame:?}, summary: {summary}");
        assert_eq!(per_frame.len(), FRAMES as usize);
        for score in &per_frame {
            assert!(*score > 9.5, "{score}");
        }
        assert_eq!(summary, per_frame[per_frame.len() - 1]);
    }

    #[test]
    fn a_distorted_clip_scores_lower() {
        let identical = scored(json!({}), detailed_luma);
        let distorted = scored(json!({}), blurred_luma);
        let identical_score = summary_score(&identical);
        let distorted_score = summary_score(&distorted);

        eprintln!(
            "distorted scores: {:?}, summary: {distorted_score}",
            scores(&distorted)
        );
        assert!(distorted_score < identical_score - 0.1, "{distorted_score}");
    }

    #[test]
    fn temporal_off_scores_every_frame_near_ten() {
        let run = scored(json!({ "temporal": false }), detailed_luma);
        let per_frame = scores(&run);
        let summary = summary_score(&run);

        eprintln!("temporal off identical scores: {per_frame:?}, summary: {summary}");
        for score in &per_frame {
            assert!(*score > 9.5, "{score}");
        }
    }

    #[test]
    fn temporal_off_summarises_the_mean_frame_score() {
        let run = scored(json!({ "temporal": false }), blurred_luma);
        let per_frame = scores(&run);
        let summary = summary_score(&run);

        let mean = per_frame.iter().sum::<f64>() / per_frame.len() as f64;
        eprintln!("temporal off distorted scores: {per_frame:?}, summary: {summary}");
        assert!((summary - mean).abs() < 1e-9, "{summary} against {mean}");
    }

    #[test]
    fn map_output_is_an_f32_grey_frame_at_the_clip_size() {
        let run = scored(json!({ "output": "map" }), blurred_luma);

        let format = run.frames[0].format();
        assert_eq!((format.width, format.height), (SIZE, SIZE));
        assert_eq!(format.layout, FrameLayout::Grey);
        assert_eq!(format.sample, SampleFormat::F32);
        assert_eq!(f32_plane(&run.frames[0]).len(), (SIZE * SIZE) as usize);
    }

    #[test]
    fn map_values_are_jod_lost() {
        let run = scored(json!({ "output": "map" }), blurred_luma);

        let values = f32_plane(&run.frames[0]);
        let rounding = 1e-6;
        let range = -rounding..=10.0;
        assert!(values.iter().all(|value| range.contains(value)));

        let largest = values.iter().copied().fold(f32::MIN, f32::max);
        assert!(largest > 0.0, "{largest}");
    }

    #[test]
    fn a_resized_map_is_at_the_display_size() {
        let args = json!({ "output": "map", "resize_to_display": true });

        let run = scored(args, blurred_luma);

        let format = run.frames[0].format();
        assert_eq!((format.width, format.height), (1080, 1080));
    }

    #[test]
    fn a_missing_frame_rate_is_an_error() {
        let format = yuv420(SampleFormat::U8);

        let refused = build_error(STAGE, json!({}), format, format);

        assert!(
            refused.contains("cvvdp needs the clip's frame rate, and this clip has none"),
            "{refused}"
        );
    }

    #[test]
    fn an_unknown_display_is_an_error() {
        let format = clip();

        let refused = build_error(STAGE, json!({ "display": "my_tv" }), format, format);

        assert!(refused.contains("my_tv"), "{refused}");
    }
}
