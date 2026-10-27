use serde_json::json;

use crate::vship::ssimulacra2::Ssimulacra2Args;

#[test]
fn gpu_defaults_to_unset() {
    let args: Ssimulacra2Args = serde_json::from_value(json!({})).expect("parses");

    assert_eq!(args.gpu, None);
}

#[test]
fn gpu_parses_an_index() {
    let args: Ssimulacra2Args = serde_json::from_value(json!({ "gpu": 1 })).expect("parses");

    assert_eq!(args.gpu, Some(1));
}

#[test]
fn an_unknown_field_is_refused() {
    let parsed = serde_json::from_value::<Ssimulacra2Args>(json!({ "device": 1 }));
    let error = parsed.expect_err("should be refused");

    assert!(error.to_string().contains("unknown field `device`"), "{error}");
}

#[cfg(feature = "gpu-tests")]
mod device {
    use av_toolbox_gpu::{Depth, FrameFormat, FrameLayout, HostFrame, MetaValue, SampleFormat};
    use serde_json::json;

    use crate::vship::tests::device::{
        Run,
        blurred_luma,
        build_error,
        built,
        detailed_luma,
        float_copy,
        float_named,
        frame,
        run,
        yuv420,
    };

    const STAGE: &str = "vship/ssimulacra2";

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

    fn scored(format: FrameFormat, distorted_luma: fn(u32, u32) -> u32) -> Run {
        let mut filter = built(STAGE, json!({}), format, format).expect("should build");
        let pairs = pairs(&format, distorted_luma);

        run(filter.as_mut(), pairs)
    }

    /// Each output frame's `score`, in order.
    fn scores(run: &Run) -> Vec<f64> {
        run.frames
            .iter()
            .map(|frame| {
                let mut attachments = frame.meta().attachments.clone();
                attachments.stamp(STAGE);
                float_named(&attachments, "score")
            })
            .collect()
    }

    #[test]
    fn identical_clips_score_near_100() {
        let format = yuv420(SampleFormat::U8);

        let run = scored(format, detailed_luma);
        let per_frame = scores(&run);

        assert_eq!(per_frame.len(), FRAMES as usize);
        for score in per_frame {
            assert!(score > 99.5, "{score}");
        }
    }

    #[test]
    fn a_distorted_clip_scores_lower() {
        let format = yuv420(SampleFormat::U8);

        let run = scored(format, blurred_luma);
        let per_frame = scores(&run);

        assert_eq!(per_frame.len(), FRAMES as usize);
        for score in per_frame {
            assert!(score < 99.0, "{score}");
        }
    }

    #[test]
    fn ten_bit_clips_score_like_eight_bit() {
        let sample = SampleFormat::U16 { depth: Depth::TEN };
        let format = yuv420(sample);

        let run = scored(format, detailed_luma);
        let per_frame = scores(&run);

        assert_eq!(per_frame.len(), FRAMES as usize);
        for score in per_frame {
            assert!(score > 99.5, "{score}");
        }
    }

    #[test]
    fn an_f32_clip_scores_like_its_integer_source() {
        let format = yuv420(SampleFormat::U8);
        let integer_run = scored(format, blurred_luma);
        let integer_scores = scores(&integer_run);

        let mut float_format = format;
        let mut float_pairs = Vec::new();
        for (index, (reference, distorted)) in pairs(&format, blurred_luma).into_iter().enumerate() {
            let (converted_format, converted) = float_copy(&format, &distorted, index as u64);
            float_format = converted_format;
            float_pairs.push((reference, converted));
        }
        let mut filter = built(STAGE, json!({}), format, float_format).expect("should build");
        let float_run = run(filter.as_mut(), float_pairs);
        let float_scores = scores(&float_run);

        assert_eq!(float_scores.len(), integer_scores.len());
        for (float_score, integer_score) in float_scores.iter().zip(&integer_scores) {
            assert!(
                (float_score - integer_score).abs() < 0.5,
                "{float_score} against {integer_score}"
            );
        }
    }

    #[test]
    fn the_distorted_frame_passes_through() {
        let format = yuv420(SampleFormat::U8);

        let run = scored(format, blurred_luma);

        let indices: Vec<u64> = run.frames.iter().map(|frame| frame.meta().index).collect();
        assert_eq!(indices, vec![0, 1, 2]);
        assert_eq!(run.frames[0].format(), &format);
    }

    #[test]
    fn the_summary_record_counts_every_frame() {
        let format = yuv420(SampleFormat::U8);

        let run = scored(format, detailed_luma);

        assert_eq!(run.records.len(), 1);
        let record = &run.records[0];
        assert_eq!(record.kind(), "summary");

        let mut attachments = record.attachments().clone();
        attachments.stamp(STAGE);
        let frames = attachments
            .stages()
            .flat_map(|(_, values)| values.iter())
            .find(|(key, _)| key == "frames");
        assert_eq!(frames, Some(&("frames".to_string(), MetaValue::Int(3))));

        let mean = float_named(&attachments, "mean");
        assert!(mean > 99.5, "{mean}");
    }

    #[test]
    fn different_sizes_fail_to_build() {
        let reference = yuv420(SampleFormat::U8);
        let distorted = FrameFormat {
            width: reference.width / 2,
            ..reference
        };

        let refused = build_error(STAGE, json!({}), reference, distorted);

        assert!(refused.contains("both clips at one size"), "{refused}");
    }

    #[test]
    fn grey_input_fails_to_build() {
        let reference = yuv420(SampleFormat::U8);
        let grey = FrameFormat {
            layout: FrameLayout::Grey,
            ..reference
        };

        let refused = build_error(STAGE, json!({}), grey, grey);

        assert!(refused.contains("got a grey clip"), "{refused}");
    }

    #[test]
    fn an_out_of_range_gpu_fails_to_build() {
        let format = yuv420(SampleFormat::U8);

        let refused = build_error(STAGE, json!({ "gpu": 1000 }), format, format);

        assert!(refused.contains("gpu 1000 is out of range"), "{refused}");
    }
}
