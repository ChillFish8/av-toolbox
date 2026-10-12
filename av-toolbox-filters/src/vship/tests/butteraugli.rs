use serde_json::json;

use crate::vship::butteraugli::{ButteraugliArgs, checked_intensity, checked_qnorm};
use crate::vship::output::Output;

#[test]
fn args_default_to_the_spec() {
    let args: ButteraugliArgs = serde_json::from_value(json!({})).expect("parses");

    assert_eq!(args.qnorm, 2);
    assert_eq!(args.intensity, 203.0);
    assert_eq!(args.output, Output::Clip);
    assert_eq!(args.gpu, None);
}

#[test]
fn args_parse_every_field() {
    let value = json!({ "qnorm": 3, "intensity": 80.0, "output": "map", "gpu": 1 });

    let args: ButteraugliArgs = serde_json::from_value(value).expect("parses");

    assert_eq!(args.qnorm, 3);
    assert_eq!(args.intensity, 80.0);
    assert_eq!(args.output, Output::Map);
    assert_eq!(args.gpu, Some(1));
}

#[test]
fn unknown_args_are_refused() {
    let parsed = serde_json::from_value::<ButteraugliArgs>(json!({ "norm": 2 }));
    let error = parsed.expect_err("should be refused");

    assert!(error.to_string().contains("unknown field `norm`"), "{error}");
}

#[test]
fn a_zero_qnorm_is_refused() {
    let error = checked_qnorm(0).expect_err("a zero qnorm passed");

    assert!(error.to_string().contains("qnorm between 1"), "{error}");
}

#[test]
fn a_qnorm_above_i32_is_refused() {
    let error = checked_qnorm(u32::MAX).expect_err("a huge qnorm passed");

    assert!(error.to_string().contains("qnorm between 1"), "{error}");
}

#[test]
fn a_positive_qnorm_is_kept() {
    assert_eq!(checked_qnorm(2).expect("a usable qnorm"), 2);
}

#[test]
fn a_non_positive_or_non_finite_intensity_is_refused() {
    for intensity in [0.0, -80.0, f32::NAN, f32::INFINITY] {
        let error = checked_intensity(intensity).expect_err("an unusable intensity passed");
        assert!(error.to_string().contains("positive intensity"), "{error}");
    }
}

#[cfg(feature = "gpu-tests")]
mod device {
    use av_toolbox_gpu::{FrameFormat, FrameLayout, HostFrame, MetaValue, SampleFormat};
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

    const STAGE: &str = "vship/butteraugli";

    const FRAMES: u64 = 3;

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
        let format = yuv420(SampleFormat::U8);
        let mut filter = built(STAGE, args, format, format).expect("should build");
        let pairs = pairs(&format, distorted_luma);

        run(filter.as_mut(), pairs)
    }

    /// The value named `name` on each output frame, in order.
    fn per_frame(run: &Run, name: &str) -> Vec<f64> {
        run.frames
            .iter()
            .map(|frame| frame_float(frame, STAGE, name))
            .collect()
    }

    #[test]
    fn identical_clips_score_zero() {
        let run = scored(json!({}), detailed_luma);
        let norm_inf = per_frame(&run, "norm_inf");

        eprintln!("identical norm_inf: {norm_inf:?}");
        assert_eq!(norm_inf.len(), FRAMES as usize);
        for score in norm_inf {
            assert!(score < 1e-6, "{score}");
        }
    }

    #[test]
    fn a_distorted_clip_scores_higher() {
        let run = scored(json!({}), blurred_luma);
        let norm_q = per_frame(&run, "norm_q");
        let norm3 = per_frame(&run, "norm3");
        let norm_inf = per_frame(&run, "norm_inf");

        eprintln!("distorted norm_q: {norm_q:?}, norm3: {norm3:?}, norm_inf: {norm_inf:?}");
        for score in norm_q.iter().chain(&norm3).chain(&norm_inf) {
            assert!(*score > 0.1, "{score}");
        }
    }

    #[test]
    fn the_distorted_frame_passes_through() {
        let run = scored(json!({}), blurred_luma);

        let indices: Vec<u64> = run.frames.iter().map(|frame| frame.meta().index).collect();
        assert_eq!(indices, vec![0, 1, 2]);
        assert_eq!(run.frames[0].format(), &yuv420(SampleFormat::U8));
    }

    #[test]
    fn the_summary_writes_norm_q_statistics_then_max_norm_inf() {
        let run = scored(json!({}), blurred_luma);
        let norm_inf = per_frame(&run, "norm_inf");

        let values = summary_values(&run, STAGE);

        let names: Vec<&str> = values.iter().map(|(name, _)| name.as_str()).collect();
        let expected = ["frames", "mean", "min", "max", "p5", "p95", "max_norm_inf"];
        assert_eq!(names, expected);
        assert_eq!(values[0].1, MetaValue::Int(3));

        let largest = norm_inf.into_iter().reduce(f64::max).expect("three frames");
        assert_eq!(values[6].1, MetaValue::Float(largest));
    }

    #[test]
    fn map_output_is_an_f32_grey_frame_at_the_clip_size() {
        let run = scored(json!({ "output": "map" }), blurred_luma);

        assert_eq!(run.frames.len(), FRAMES as usize);
        let map = &run.frames[1];
        let format = map.format();
        assert_eq!((format.width, format.height), (SIZE, SIZE));
        assert_eq!(format.layout, FrameLayout::Grey);
        assert_eq!(format.sample, SampleFormat::F32);
        assert_eq!(map.meta().index, 1);

        let values = f32_plane(map);
        let largest = values.iter().copied().fold(0.0, f32::max);
        assert_eq!(values.len(), (SIZE * SIZE) as usize);
        assert!(largest > 0.0, "{largest}");

        let norm_inf = frame_float(map, STAGE, "norm_inf");
        assert!(norm_inf > 0.1, "{norm_inf}");
    }

    #[test]
    fn the_map_is_all_zero_for_identical_clips() {
        let run = scored(json!({ "output": "map" }), detailed_luma);

        for map in &run.frames {
            let values = f32_plane(map);
            let nonzero = values.iter().filter(|value| **value != 0.0).count();
            assert_eq!(nonzero, 0);
        }
    }

    #[test]
    fn a_qnorm_too_large_for_vship_fails_to_build() {
        let format = yuv420(SampleFormat::U8);
        let args = json!({ "qnorm": u32::MAX });

        let refused = build_error(STAGE, args, format, format);

        assert!(refused.contains("qnorm"), "{refused}");
    }
}
