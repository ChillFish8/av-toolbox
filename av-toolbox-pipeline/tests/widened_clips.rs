#![cfg(all(unix, feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]

use std::fs;

use av_toolbox_core::PipelineRegistry;
use av_toolbox_gpu::{Backend, SampleFormat, test_device};
use av_toolbox_graph::StageSpec;
use av_toolbox_pipeline::{CompiledPipeline, DiskFiles, Pipeline, Supplied};
use serde_json::json;

fn loaded_specs(text: &str) -> Vec<StageSpec> {
    let directory = tempfile::tempdir().expect("a temp dir");
    let path = directory.path().join("pipeline.toml");
    fs::write(&path, text).expect("writes the pipeline");

    let supplied = Supplied::new();
    let loaded = av_toolbox_pipeline::load(&path, &DiskFiles, &supplied).expect("the file should load");

    loaded.specs
}

fn compiled<'r, 'b>(
    text: &str,
    registry: &'r PipelineRegistry,
    backend: &'b Backend,
) -> CompiledPipeline<'r, 'b> {
    let specs = loaded_specs(text);
    let pipeline = Pipeline::build(registry, &specs).expect("should build");

    pipeline.compile(backend).expect("should compile")
}

const SOURCE: &str = r##"
[source.input]
type = "solid/fill"
args.colour = "#808080"
args.width = 64
args.height = 64
args.frames = 2
args.frame_rate = [24, 1]

[mask.edges]
type = "edges/prewitt"
inputs = ["input"]
"##;

#[test]
fn a_narrow_base_widens_to_the_float_overlay() {
    let text = format!(
        r#"{SOURCE}
[filter.float]
type = "resample/depth"
inputs = ["input"]
args.depth = "f32"

[filter.blended]
type = "blend/masked"
inputs = ["input", "float", "edges"]

[sink.output]
type = "y4m/pipe"
inputs = ["blended"]
args.path = "/dev/null"
"#
    );
    let registry = av_toolbox_pipeline::registry().expect("the built-in stages register");
    let backend = Backend::open_default(test_device()).expect("should open a device");
    let compiled = compiled(&text, &registry, &backend);

    let graph = compiled.graph();
    let injected: Vec<_> = graph
        .nodes()
        .iter()
        .filter(|node| node.identifier() == "resample/depth")
        .filter(|node| node.name() != "float")
        .filter(|node| *node.args() == json!({ "depth": "f32" }))
        .collect();
    assert_eq!(injected.len(), 1);

    let widen = injected[0];
    let formats = compiled.formats().get(widen.id());
    let input = formats.inputs()[0].expect("the widen has an input");
    let output = formats.output().expect("the widen has an output");
    assert_eq!(input.sample(), SampleFormat::U8);
    assert_eq!(output.sample(), SampleFormat::F32);

    let blended = graph
        .nodes()
        .iter()
        .find(|node| node.name() == "blended")
        .expect("the blend node");
    let base_edge = &blended.inputs()[0];
    assert_eq!(base_edge.source(), widen.id());
}

#[test]
fn one_producer_in_both_slots_solves() {
    let text = format!(
        r#"{SOURCE}
[filter.float]
type = "resample/depth"
inputs = ["input"]
args.depth = "f32"

[filter.blended]
type = "blend/masked"
inputs = ["float", "float", "edges"]

[sink.output]
type = "y4m/pipe"
inputs = ["blended"]
args.path = "/dev/null"
"#
    );
    let registry = av_toolbox_pipeline::registry().expect("the built-in stages register");
    let backend = Backend::open_default(test_device()).expect("should open a device");
    let compiled = compiled(&text, &registry, &backend);

    let graph = compiled.graph();
    let injected = graph
        .nodes()
        .iter()
        .filter(|node| node.identifier() == "resample/depth")
        .filter(|node| *node.args() == json!({ "depth": "f32" }))
        .count();
    assert_eq!(injected, 1);
}
