use std::path::Path;

use av_toolbox_gpu::{Backend, test_device};
use tempfile::TempDir;

use crate::warm_key::pipeline_key;
use crate::{PipelineInput, PipelineSource, steps};

const PIPELINE: &str = r#"
[variable.image]
type = "path"
default = "image.png"

[variable.format]
type = "string"
default = "yuv444p"

[variable.width]
type = "integer"
default = 32

[variable.output]
type = "path"
default = "output.y4m"

[source.image]
type = "image/file"
args.path = "{{ image }}"
args.frames = 2
args.format = "{{ format }}"
args.frame_rate = [24, 1]

[filter.resize]
type = "resize/lanczos"
inputs = ["image"]
args.width = "{{ width }}"
args.height = 16

[sink.output]
type = "y4m/pipe"
inputs = ["resize"]
args.path = "{{ output }}"
"#;

fn write_image(directory: &TempDir, name: &str) -> String {
    let path = directory.path().join(name);
    let picture = image::RgbImage::from_pixel(16, 16, image::Rgb([200, 100, 50]));
    picture.save(&path).expect("should write the image");

    path_text(&path)
}

fn path_text(path: &Path) -> String {
    path.to_str().expect("a UTF-8 path").to_owned()
}

/// The key of [PIPELINE] with `vars` set, on `backend`.
fn key_with(backend: &Backend, vars: &[(&str, &str)]) -> u128 {
    let input = PipelineInput::Text {
        name: "pipeline.toml".to_owned(),
        text: PIPELINE.to_owned(),
    };
    let vars: Vec<(String, String)> = vars
        .iter()
        .map(|(name, value)| (name.to_string(), value.to_string()))
        .collect();

    let source = PipelineSource::read(input, &vars).expect("should read");
    let pipeline = source.build().expect("should build");
    let compiled = steps::compile(pipeline, backend).expect("should compile");

    pipeline_key(&compiled, backend)
}

fn backend() -> Backend {
    Backend::open_default(test_device()).expect("should open")
}

#[test]
fn two_input_files_share_a_key() {
    let directory = tempfile::tempdir().expect("should make a directory");
    let first = write_image(&directory, "first.png");
    let second = write_image(&directory, "second.png");
    let backend = backend();

    let first_key = key_with(&backend, &[("image", &first)]);
    let second_key = key_with(&backend, &[("image", &second)]);

    assert_eq!(first_key, second_key);
}

#[test]
fn a_filter_arg_changes_the_key() {
    let directory = tempfile::tempdir().expect("should make a directory");
    let image = write_image(&directory, "image.png");
    let backend = backend();

    let narrow_key = key_with(&backend, &[("image", &image), ("width", "32")]);
    let wide_key = key_with(&backend, &[("image", &image), ("width", "48")]);

    assert_ne!(narrow_key, wide_key);
}

#[test]
fn a_different_output_format_changes_the_key() {
    let directory = tempfile::tempdir().expect("should make a directory");
    let image = write_image(&directory, "image.png");
    let backend = backend();

    let full_key = key_with(&backend, &[("image", &image), ("format", "yuv444p")]);
    let subsampled_key = key_with(&backend, &[("image", &image), ("format", "yuv420p")]);

    assert_ne!(full_key, subsampled_key);
}

#[test]
fn two_output_paths_share_a_key() {
    let directory = tempfile::tempdir().expect("should make a directory");
    let image = write_image(&directory, "image.png");
    let first = directory.path().join("first.y4m");
    let second = directory.path().join("second.y4m");
    let first_path = path_text(&first);
    let second_path = path_text(&second);
    let backend = backend();

    let first_key = key_with(&backend, &[("image", &image), ("output", &first_path)]);
    let second_key = key_with(&backend, &[("image", &image), ("output", &second_path)]);

    assert_eq!(first_key, second_key);
}
