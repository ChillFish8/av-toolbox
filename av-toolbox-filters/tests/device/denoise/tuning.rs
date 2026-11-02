use std::path::Path;

use av_toolbox_gpu::{SampleFormat, Tuning, install_compilation_cache_at};
use serde_json::json;

use super::{float_of, noisy_clip, yuv420};
use crate::stream::{Input, Output, Stream};
use crate::support::{backend, built_filter};

fn run_tuned(stage: &str, args: serde_json::Value, chunks: Vec<Vec<Input>>) -> Vec<Output> {
    let format = yuv420(64, 32, SampleFormat::U8);
    let output = float_of(&format);
    let backend = backend().with_tuning(Tuning::Auto);
    let filter = built_filter(&backend, stage, format, output, args);
    let stream = Stream {
        filter,
        input: format,
        output,
        chunks,
    };

    backend.build(stream)
}

fn holds_a_result(store: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(store) else {
        return false;
    };

    entries.flatten().any(|entry| {
        entry
            .path()
            .extension()
            .is_some_and(|extension| extension == "json")
    })
}

#[test]
fn auto_tuning_stores_a_result_beside_the_kernel_cache() {
    let cache = tempfile::tempdir().expect("a temp dir");
    // CubeCL's config is process wide, so this relies on nextest running each test in its own process.
    install_compilation_cache_at(cache.path()).expect("no client is open yet");

    let format = yuv420(64, 32, SampleFormat::U8);
    let clip = noisy_clip(&format, 4);

    let outputs = run_tuned(
        "av-denoise/nlmeans",
        json!({ "preset": "veryfast" }),
        vec![clip.clone()],
    );

    assert_eq!(outputs.len(), clip.len());

    let store = cache.path().join("av-denoise-tune");
    assert!(
        holds_a_result(&store),
        "nothing was stored in {}",
        store.display()
    );
}

#[test]
fn frames_held_for_tuning_come_out_in_order() {
    let format = yuv420(64, 32, SampleFormat::U8);
    let clip = noisy_clip(&format, 6);

    let outputs = run_tuned("av-denoise/nlmeans", json!({}), vec![clip.clone()]);

    assert_eq!(outputs.len(), clip.len());
    for (output, input) in outputs.iter().zip(&clip) {
        assert_eq!(output.meta, input.meta);
    }
}

#[test]
fn a_stream_shorter_than_the_tuning_prefix_is_tuned_at_its_end() {
    let format = yuv420(64, 32, SampleFormat::U8);
    let short = noisy_clip(&format, 1);
    let long = noisy_clip(&format, 5);
    let chunks = vec![short.clone(), long.clone()];

    let outputs = run_tuned("av-denoise/nlmeans", json!({}), chunks);

    let inputs: Vec<&Input> = short.iter().chain(&long).collect();
    assert_eq!(outputs.len(), inputs.len());
    for (output, input) in outputs.iter().zip(inputs) {
        assert_eq!(output.meta, input.meta);
    }
}
