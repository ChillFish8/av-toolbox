use av_toolbox_gpu::{ChromaLocation, FrameLayout, SampleFormat, Subsampling};
use serde_json::json;

use crate::support::{frame, merge_frames, patterned_planes, run_filter_on_frame, run_filter_on_output};

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
    let upscaled = run_filter_on_output("resize/spline36", json!({ "width": 32, "height": 16 }), &chroma);
    let merged = merge_frames(
        json!({ "y": { "input": 0 }, "u": { "input": 1 }, "v": { "input": 1 } }),
        &[luma, upscaled],
    );

    assert_eq!(
        merged.format.layout,
        FrameLayout::Yuv {
            subsampling: Subsampling::Yuv444
        }
    );
}
