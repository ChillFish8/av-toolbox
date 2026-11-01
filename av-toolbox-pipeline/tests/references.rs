#![cfg(all(unix, feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]

use std::fs;

use av_toolbox_core::PipelineRegistry;
use av_toolbox_gpu::{Backend, test_device};
use av_toolbox_graph::{DiagnosticCode, Node, Span, StageSpec};
use av_toolbox_pipeline::{CompiledPipeline, DiskFiles, Pipeline, PipelineError, Supplied};
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

fn compile_failure(specs: &[StageSpec]) -> PipelineError {
    let registry = av_toolbox_pipeline::registry().expect("the built-in stages register");
    let backend = Backend::open_default(test_device()).expect("should open a device");
    let pipeline = Pipeline::build(&registry, specs).expect("should build");

    let result = pipeline.compile(&backend);
    let Err(error) = result else {
        panic!("the filled depth should be rejected");
    };

    error
}

/// Where the `deep` stage wrote its depth template.
fn depth_template_span(specs: &[StageSpec]) -> Span {
    let deep = specs
        .iter()
        .find(|spec| spec.name.value == "deep")
        .expect("the depth spec");

    deep.arg_spans
        .narrowest("/depth")
        .expect("the template's span is recorded")
}

fn node_named<'a>(compiled: &'a CompiledPipeline<'_, '_>, name: &str) -> Option<&'a Node> {
    compiled.graph().nodes().iter().find(|node| node.name() == name)
}

const CLIP: &str = r##"
[source.clip]
type = "solid/fill"
args.colour = "#808080"
args.width = 32
args.height = 24
args.frames = 2
args.frame_rate = [24, 1]
"##;

const SIZED_FROM_REFERENCE: &str = r##"
[source.reference]
type = "solid/fill"
args.colour = "#808080"
args.width = 64
args.height = 48
args.frames = 2
args.frame_rate = [24, 1]

[filter.resized]
type = "resize/bilinear"
inputs = ["clip"]
args.width = "{{ reference.width }}"
args.height = "{{ reference.height }}"

[sink.output]
type = "y4m/pipe"
inputs = ["resized"]
args.path = "/dev/null"
"##;

fn depth_pipeline(template: &str) -> String {
    format!(
        r#"{CLIP}
[filter.deep]
type = "resample/depth"
inputs = ["clip"]
args.depth = "{template}"

[sink.output]
type = "y4m/pipe"
inputs = ["deep"]
args.path = "/dev/null"
"#
    )
}

#[test]
fn a_resize_takes_its_size_from_another_source() {
    let text = format!("{CLIP}{SIZED_FROM_REFERENCE}");
    let registry = av_toolbox_pipeline::registry().expect("the built-in stages register");
    let backend = Backend::open_default(test_device()).expect("should open a device");
    let pipeline = compiled(&text, &registry, &backend);

    let resized = node_named(&pipeline, "resized").expect("the resize node");
    let output = pipeline
        .formats()
        .output(resized.id())
        .expect("the resize has an output");
    let frames = output.as_frames().expect("the resize emits frames");

    assert_eq!(frames.width, 64);
    assert_eq!(frames.height, 48);
}

#[test]
fn a_stage_only_referenced_is_dropped_before_the_run() {
    let text = format!("{CLIP}{SIZED_FROM_REFERENCE}");
    let registry = av_toolbox_pipeline::registry().expect("the built-in stages register");
    let backend = Backend::open_default(test_device()).expect("should open a device");
    let pipeline = compiled(&text, &registry, &backend);

    assert!(node_named(&pipeline, "reference").is_none());
}

#[test]
fn max_of_the_natural_depth_raises_an_eight_bit_clip() {
    let text = depth_pipeline("{{ max(10, clip.bit_depth) }}");
    let registry = av_toolbox_pipeline::registry().expect("the built-in stages register");
    let backend = Backend::open_default(test_device()).expect("should open a device");
    let pipeline = compiled(&text, &registry, &backend);

    let deep = node_named(&pipeline, "deep").expect("the depth node");

    assert_eq!(deep.args()["depth"], json!(10));
}

#[test]
fn a_rejected_filled_value_points_at_the_template() {
    let text = depth_pipeline("{{ clip.bit_depth * 40 }}");
    let specs = loaded_specs(&text);
    let error = compile_failure(&specs);

    let template_span = depth_template_span(&specs);
    let invalid: Vec<_> = error
        .diagnostics()
        .iter()
        .filter(|diagnostic| diagnostic.code() == DiagnosticCode::InvalidArgs)
        .collect();

    assert_eq!(invalid.len(), 1);
    assert_eq!(invalid[0].span(), template_span);
}

#[test]
fn a_rejected_depth_notes_its_template() {
    let text = depth_pipeline("{{ clip.bit_depth * 4 }}");
    let specs = loaded_specs(&text);
    let error = compile_failure(&specs);

    let template_span = depth_template_span(&specs);
    let diagnostics = error.diagnostics();
    let noted = diagnostics.iter().any(|diagnostic| {
        diagnostic
            .notes()
            .iter()
            .any(|note| note.span == template_span && note.message == "filled from this template")
    });

    assert!(noted, "{diagnostics:?}");
}

const CHUNKED_AFTER_A_LATE_JOIN: &str = r##"
[source.map]
type = "solid/fill"
args.colour = "#404040"
args.format = "gray"
args.width = 32
args.height = 24
args.frames = 2
args.frame_rate = [24, 1]

[filter.float_map]
type = "resample/depth"
inputs = ["map"]
args.depth = "f32"

[filter.painted]
type = "blend/heatmap"
inputs = { clip = "clip", map = "float_map" }
args.scale = "{{ clip.width }}"

[filter.deinterlaced]
type = "deinterlace/bwdif"
inputs = ["painted"]
args.order = "top"

[sink.output]
type = "y4m/pipe"
inputs = ["deinterlaced"]
args.path = "/dev/null"
"##;

#[test]
fn a_chunk_stage_after_a_join_with_a_reference_compiles() {
    let text = format!("{CLIP}{CHUNKED_AFTER_A_LATE_JOIN}");
    let registry = av_toolbox_pipeline::registry().expect("the built-in stages register");
    let backend = Backend::open_default(test_device()).expect("should open a device");
    let pipeline = compiled(&text, &registry, &backend);

    let deinterlaced = node_named(&pipeline, "deinterlaced").expect("the deinterlace node");

    assert_eq!(deinterlaced.demands().len(), 1);
}

#[test]
fn a_late_none_leaves_the_arg_out_so_the_stage_takes_its_default() {
    let text = format!(
        r#"{CLIP}
[filter.deep]
type = "resample/depth"
inputs = ["clip"]
args.depth = 8
args.at_least = "{{{{ if clip.width > 100000 then true else none }}}}"

[sink.output]
type = "y4m/pipe"
inputs = ["deep"]
args.path = "/dev/null"
"#
    );
    let registry = av_toolbox_pipeline::registry().expect("the built-in stages register");
    let backend = Backend::open_default(test_device()).expect("should open a device");
    let pipeline = compiled(&text, &registry, &backend);

    let deep = node_named(&pipeline, "deep").expect("the depth node");
    let args = deep.args().as_object().expect("object args");

    assert!(!args.contains_key("at_least"));
    assert_eq!(args["depth"], json!(8));
}
