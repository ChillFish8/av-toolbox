use av_toolbox_cubek::resize::{Kernel, centred_taps};
use av_toolbox_cubek::testing::reference_resize_float;
use av_toolbox_gpu::{ChromaLocation, FrameLayout, SampleFormat, Subsampling};
use serde_json::json;

use crate::support::{
    ROUND_TRIP,
    assert_codes_within,
    frame,
    merge_frames,
    patterned_planes,
    planes_as_codes,
    run_filter_on_frame,
    run_filter_on_output,
};

#[test]
fn extracting_then_merging_gives_back_the_original() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let input = frame(32, 16, layout, SampleFormat::U8, ChromaLocation::Left);
    let planes = patterned_planes(&input);

    let luma = run_filter_on_frame("planes/extract", json!({ "plane": "y" }), &input, &planes);
    let chroma = run_filter_on_frame("planes/extract", json!({ "plane": "chroma" }), &input, &planes);
    let merged = merge_frames(
        json!({ "y": { "input": 0 }, "u": { "input": 1 }, "v": { "input": 1 } }),
        &[luma, chroma],
    );

    assert_eq!(merged.format, input);
    assert_eq!(merged.planes, planes);
}

#[test]
fn upscaled_chroma_merges_as_444() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let input = frame(32, 16, layout, SampleFormat::U8, ChromaLocation::Left);
    let planes = patterned_planes(&input);

    let luma = run_filter_on_frame("planes/extract", json!({ "plane": "y" }), &input, &planes);
    let chroma = run_filter_on_frame("planes/extract", json!({ "plane": "chroma" }), &input, &planes);
    let float_luma = run_filter_on_output("resample/depth", json!({ "depth": "f32" }), &luma);
    let upscaled = run_filter_on_output("resize/spline36", json!({ "width": 32, "height": 16 }), &chroma);
    let merged = merge_frames(
        json!({ "y": { "input": 0 }, "u": { "input": 1 }, "v": { "input": 1 } }),
        &[float_luma, upscaled],
    );

    let full_chroma = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv444,
    };
    assert_eq!(merged.format.layout, full_chroma);
    assert_eq!(merged.format.sample, SampleFormat::F32);

    let found = planes_as_codes(&merged.format, &merged.planes, 8);
    let luma_codes = codes_of(&planes[0]);
    assert_codes_within(&found[0], &luma_codes, ROUND_TRIP, "Y");

    let kernel = Kernel::Spline36;
    let (chroma_width, chroma_height) = (chroma.format.width, chroma.format.height);
    let across = centred_taps(chroma_width, 32, &kernel);
    let down = centred_taps(chroma_height, 16, &kernel);
    for (plane, label) in [(1, "U"), (2, "V")] {
        let codes = codes_of(&chroma.planes[plane - 1]);
        let expected = reference_resize_float(&codes, chroma_width, chroma_height, &across, &down);

        assert_codes_within(&found[plane], &expected, 0.01, label);
    }
}

fn codes_of(samples: &[u8]) -> Vec<f32> {
    samples.iter().map(|sample| f32::from(*sample)).collect()
}
