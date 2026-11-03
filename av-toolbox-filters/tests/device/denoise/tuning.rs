use std::path::{Path, PathBuf};
use std::time::SystemTime;

use av_toolbox_filters::testing::{forget_tuning_passes, tuning_passes};
use av_toolbox_gpu::{Backend, Retune, SampleFormat, Tuning, install_compilation_cache_at};
use serde_json::json;

use super::{float_of, noisy_clip, yuv420};
use crate::stream::{Input, Output, Stream};
use crate::support::{backend, built_filter};

/// A build name shaped like the ones installs are given.
const BUILD: &str = "0123456789abcdef";

fn run_tuned(stage: &str, args: serde_json::Value, chunks: Vec<Vec<Input>>) -> Vec<Output> {
    let backend = backend().with_tuning(Tuning::Auto);

    run_on(&backend, stage, args, chunks)
}

fn run_on(backend: &Backend, stage: &str, args: serde_json::Value, chunks: Vec<Vec<Input>>) -> Vec<Output> {
    let format = yuv420(64, 32, SampleFormat::U8);
    let output = float_of(&format);
    let filter = built_filter(backend, stage, format, output, args);
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

/// Every result file in `store` with when it was last written.
fn stored_results(store: &Path) -> Vec<(PathBuf, SystemTime)> {
    let entries = std::fs::read_dir(store).expect("the store exists");

    let mut results = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_none_or(|extension| extension != "json") {
            continue;
        }

        let metadata = entry.metadata().expect("the result has metadata");
        let modified = metadata.modified().expect("the result has a modified time");
        results.push((path, modified));
    }

    results.sort();
    results
}

#[test]
fn auto_tuning_stores_a_result_beside_the_kernel_cache() {
    let cache = tempfile::tempdir().expect("a temp dir");
    // CubeCL's config is process wide, so this relies on nextest running each test in its own process.
    let build_dir = install_compilation_cache_at(cache.path(), BUILD).expect("no client is open yet");

    let format = yuv420(64, 32, SampleFormat::U8);
    let clip = noisy_clip(&format, 4);

    let outputs = run_tuned(
        "av-denoise/nlmeans",
        json!({ "preset": "veryfast" }),
        vec![clip.clone()],
    );

    assert_eq!(outputs.len(), clip.len());

    let store = build_dir.join("av-denoise-tune");
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

#[test]
fn a_retune_tunes_a_stored_spec_again_once_per_process() {
    let cache = tempfile::tempdir().expect("a temp dir");
    // CubeCL's config is process wide, so this relies on nextest running each test in its own process.
    let build_dir = install_compilation_cache_at(cache.path(), BUILD).expect("no client is open yet");
    let store = build_dir.join("av-denoise-tune");

    let format = yuv420(64, 32, SampleFormat::U8);
    let clip = noisy_clip(&format, 4);
    let args = json!({ "preset": "veryfast" });

    run_tuned("av-denoise/nlmeans", args.clone(), vec![clip.clone()]);
    assert_eq!(tuning_passes(), 1);
    let first = stored_results(&store);
    assert!(!first.is_empty(), "nothing was stored in {}", store.display());

    // A fresh process sees the stored result, so auto tuning does not tune again.
    forget_tuning_passes();
    run_tuned("av-denoise/nlmeans", args.clone(), vec![clip.clone()]);
    assert_eq!(tuning_passes(), 0);

    let retuning = backend().with_retune(Retune::Families);
    let outputs = run_on(&retuning, "av-denoise/nlmeans", args.clone(), vec![clip.clone()]);
    assert_eq!(outputs.len(), clip.len());
    assert_eq!(tuning_passes(), 1);
    let retuned = stored_results(&store);
    assert_ne!(retuned, first, "the retune saved nothing new");

    run_on(&retuning, "av-denoise/nlmeans", args, vec![clip.clone()]);
    assert_eq!(tuning_passes(), 1);
    let again = stored_results(&store);
    assert_eq!(again, retuned, "the second retune tuned again");
}
