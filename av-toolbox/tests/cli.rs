use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn toolbox(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_av-toolbox"))
        .args(["--color", "never"])
        .args(args)
        .output()
        .expect("the binary runs")
}

fn example(name: &str) -> PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    root.join("..").join("examples").join(name)
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

fn written(content: &str) -> tempfile::NamedTempFile {
    let file = tempfile::Builder::new()
        .suffix(".toml")
        .tempfile()
        .expect("a temp file");
    std::fs::write(file.path(), content).expect("written");
    file
}

#[test]
fn every_example_explains_without_a_device() {
    let names = [
        "passthrough.toml",
        "scenes.toml",
        "chunked.toml",
        "metadata.toml",
        "resize.toml",
        "solid.toml",
        "image.toml",
        "recolour-lines.toml",
    ];

    for name in names {
        let path = example(name);
        let path = path.to_str().expect("utf-8 path");

        let output = toolbox(&["explain", path, "--structure-only"]);
        let stderr = text(&output.stderr);

        assert!(output.status.success(), "{name}: {stderr}");
        assert!(text(&output.stdout).contains("output"), "{name}");
    }
}

#[test]
fn explain_writes_html_to_a_path() {
    let directory = tempfile::tempdir().expect("a temp dir");
    let page = directory.path().join("graph.html");
    let example = example("passthrough.toml");

    let output = toolbox(&[
        "explain",
        example.to_str().expect("utf-8 path"),
        "--structure-only",
        "--html",
        "--path",
        page.to_str().expect("utf-8 path"),
    ]);
    let stderr = text(&output.stderr);

    assert!(output.status.success(), "{stderr}");
    let html = std::fs::read_to_string(&page).expect("the page was written");
    assert!(html.contains("<html"));
}

#[test]
fn explain_writes_plan_html_in_the_working_directory_by_default() {
    let directory = tempfile::tempdir().expect("a temp dir");
    let example = example("passthrough.toml");

    let output = Command::new(env!("CARGO_BIN_EXE_av-toolbox"))
        .current_dir(directory.path())
        .args(["--color", "never", "explain"])
        .arg(&example)
        .args(["--structure-only", "--html"])
        .output()
        .expect("the binary runs");
    let stderr = text(&output.stderr);

    assert!(output.status.success(), "{stderr}");
    let page = directory.path().join("plan.html");
    let html = std::fs::read_to_string(&page).expect("the page was written");
    assert!(html.contains("<html"));
}

#[test]
fn a_path_without_html_is_a_usage_error() {
    let example = example("passthrough.toml");

    let output = toolbox(&[
        "explain",
        example.to_str().expect("utf-8 path"),
        "--structure-only",
        "--path",
        "graph.html",
    ]);

    assert_eq!(output.status.code(), Some(2));
}

#[test]
fn an_unwritable_html_path_exits_one() {
    let example = example("passthrough.toml");

    let output = toolbox(&[
        "explain",
        example.to_str().expect("utf-8 path"),
        "--structure-only",
        "--html",
        "--path",
        "/nonexistent-directory/graph.html",
    ]);
    let stderr = text(&output.stderr);

    assert_eq!(output.status.code(), Some(1));
    assert!(stderr.contains("cannot write"));
}

#[test]
fn list_stages_names_the_built_in_stages() {
    let output = toolbox(&["list-stages"]);

    assert!(output.status.success());
    assert!(text(&output.stdout).contains("y4m/stdout"));
}

#[test]
fn list_stages_describes_one_type() {
    let output = toolbox(&["list-stages", "y4m/stdout"]);

    assert!(output.status.success());
    assert!(text(&output.stdout).contains("frame_rate"));
}

#[test]
fn an_unknown_stage_type_exits_one_with_a_suggestion() {
    let output = toolbox(&["list-stages", "y4m/stdot"]);
    let stderr = text(&output.stderr);

    assert_eq!(output.status.code(), Some(1));
    assert!(stderr.contains("did you mean y4m/stdout?"));
}

#[test]
fn a_broken_file_exits_one_with_rendered_diagnostics() {
    let file = written("[sources.input]\ntype = \"y4m/stdin\"\n");
    let path = file.path().to_str().expect("utf-8 path");

    let output = toolbox(&["explain", path, "--structure-only"]);
    let stderr = text(&output.stderr);

    assert_eq!(output.status.code(), Some(1));
    assert!(stderr.contains("error[invalid-definition]"));
    assert!(stderr.contains("did you mean `source`?"));
    assert!(stderr.contains(":1:2"));
}

#[test]
fn an_unknown_reference_is_underlined() {
    let content = "[source.input]\ntype = \"y4m/stdin\"\n\n[sink.output]\ntype = \"y4m/stdout\"\ninputs = [\"inptu\"]\n";
    let file = written(content);
    let path = file.path().to_str().expect("utf-8 path");

    let output = toolbox(&["explain", path, "--structure-only"]);
    let stderr = text(&output.stderr);

    assert_eq!(output.status.code(), Some(1));
    assert!(stderr.contains("error[unknown-stage-reference]"));
    assert!(stderr.contains("did you mean input"));
    assert!(stderr.contains("^^^^^^^"));
}

#[test]
fn an_empty_file_reports_no_sinks() {
    let file = written("");
    let path = file.path().to_str().expect("utf-8 path");

    let output = toolbox(&["explain", path, "--structure-only"]);
    let stderr = text(&output.stderr);

    assert_eq!(output.status.code(), Some(1));
    assert!(stderr.contains("error[no-sinks]"));
}

#[test]
fn a_missing_file_exits_one() {
    let output = toolbox(&[
        "explain",
        "/nonexistent-directory/pipeline.toml",
        "--structure-only",
    ]);
    let stderr = text(&output.stderr);

    assert_eq!(output.status.code(), Some(1));
    assert!(stderr.contains("cannot read"));
}

#[test]
fn a_usage_error_exits_two() {
    let output = toolbox(&["explain"]);

    assert_eq!(output.status.code(), Some(2));
}

#[test]
fn explain_falls_back_to_the_structure_for_a_live_stream() {
    let path = example("passthrough.toml");
    let path = path.to_str().expect("utf-8 path");

    let output = Command::new(env!("CARGO_BIN_EXE_av-toolbox"))
        .args(["--color", "never", "explain", path])
        .env("RUST_LOG", "info")
        .output()
        .expect("the binary runs");
    let stderr = text(&output.stderr);
    let stdout = text(&output.stdout);

    assert!(output.status.success(), "{stderr}");
    assert!(stdout.contains("output"));
    assert!(stderr.contains("reads a live stream"));
}

const SPEED_PIPELINE: &str = r#"
[variable.speed]
type = "string"
choices = ["standard", "fast"]

[source.input]
type = "y4m/stdin"

[transform.scenes]
type = "scene/detect"
inputs = ["input"]
args.mode = "natural"
args.speed = "{{ speed }}"

[sink.output]
type = "y4m/stdout"
inputs = ["scenes"]
"#;

#[test]
fn explain_shows_the_prefixed_names_of_called_stages() {
    let path = example("chunked.toml");
    let path = path.to_str().expect("utf-8 path");

    let output = toolbox(&["explain", path, "--structure-only"]);
    let stderr = text(&output.stderr);

    assert!(output.status.success(), "{stderr}");
    assert!(text(&output.stdout).contains("scenes.detect"));
}

#[test]
fn a_var_flag_sets_a_variable() {
    let file = written(SPEED_PIPELINE);
    let path = file.path().to_str().expect("utf-8 path");

    let output = toolbox(&["explain", path, "--structure-only", "--var", "speed=fast"]);
    let stderr = text(&output.stderr);

    assert!(output.status.success(), "{stderr}");
}

#[test]
fn an_environment_variable_sets_a_variable() {
    let file = written(SPEED_PIPELINE);
    let path = file.path().to_str().expect("utf-8 path");

    let output = Command::new(env!("CARGO_BIN_EXE_av-toolbox"))
        .args(["--color", "never", "explain", path, "--structure-only"])
        .env("AVTB_VAR_SPEED", "fast")
        .output()
        .expect("the binary runs");
    let stderr = text(&output.stderr);

    assert!(output.status.success(), "{stderr}");
}

#[test]
fn a_missing_variable_exits_one_with_how_to_set_it() {
    let file = written(SPEED_PIPELINE);
    let path = file.path().to_str().expect("utf-8 path");

    let output = Command::new(env!("CARGO_BIN_EXE_av-toolbox"))
        .args(["--color", "never", "explain", path, "--structure-only"])
        .env_remove("AVTB_VAR_SPEED")
        .output()
        .expect("the binary runs");
    let stderr = text(&output.stderr);

    assert_eq!(output.status.code(), Some(1));
    assert!(stderr.contains("error[missing-variable]"));
    assert!(stderr.contains("--var speed="));
}

#[test]
fn a_value_outside_the_choices_exits_one() {
    let file = written(SPEED_PIPELINE);
    let path = file.path().to_str().expect("utf-8 path");

    let output = toolbox(&["explain", path, "--structure-only", "--var", "speed=slow"]);
    let stderr = text(&output.stderr);

    assert_eq!(output.status.code(), Some(1));
    assert!(stderr.contains("error[invalid-variable]"));
}

#[test]
fn an_error_in_a_called_file_renders_that_file() {
    let directory = tempfile::tempdir().expect("a temp dir");
    let main = r#"
[source.input]
type = "y4m/stdin"

[call.part]
path = "part.toml"
inputs = ["input"]

[sink.output]
type = "y4m/stdout"
inputs = ["part"]
"#;
    let part = r#"
[input.clip]

[filter.broken]
type = "no/such-stage"
inputs = ["clip"]

[output.done]
inputs = ["broken"]
"#;
    let main_path = directory.path().join("main.toml");
    std::fs::write(&main_path, main).expect("written");
    std::fs::write(directory.path().join("part.toml"), part).expect("written");
    let main_path = main_path.to_str().expect("utf-8 path");

    let output = toolbox(&["explain", main_path, "--structure-only"]);
    let stderr = text(&output.stderr);

    assert_eq!(output.status.code(), Some(1));
    assert!(stderr.contains("part.toml:"), "{stderr}");
    assert!(stderr.contains("called from here"), "{stderr}");
}

#[test]
fn a_var_without_equals_exits_two() {
    let path = example("passthrough.toml");
    let path = path.to_str().expect("utf-8 path");

    let output = toolbox(&["explain", path, "--var", "speed"]);

    assert_eq!(output.status.code(), Some(2));
}

#[test]
fn pull_refuses_an_unknown_pack_without_downloading() {
    let models = tempfile::tempdir().expect("a temp dir");

    let output = Command::new(env!("CARGO_BIN_EXE_av-toolbox"))
        .env("AVTB_MODELS", models.path())
        .args(["--color", "never", "pull", "nothing"])
        .output()
        .expect("the binary runs");
    let stderr = text(&output.stderr);

    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(
        stderr.contains("`nothing`") && stderr.contains("artcnn"),
        "{stderr}"
    );
}

#[cfg(all(unix, feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
mod device {
    use std::io::Write;
    use std::process::{Command, Output, Stdio};
    use std::thread;

    use super::{example, text};

    const FRAME_SIZE: usize = 64;

    const FRAME_COUNT: u64 = 5;

    /// Frames in the textured metadata stream.
    const METADATA_FRAME_COUNT: u64 = 8;

    fn y4m_stream() -> Vec<u8> {
        let mut bytes = Vec::new();
        let ratio = y4m::Ratio::new(30000, 1001);
        let range = y4m::VendorExtensionString::new(b"COLORRANGE=FULL".to_vec()).expect("valid");
        let mut encoder = y4m::encode(FRAME_SIZE, FRAME_SIZE, ratio)
            .append_vendor_extension(range)
            .write_header(&mut bytes)
            .expect("writes the header");

        let luma_size = FRAME_SIZE * FRAME_SIZE;
        let chroma_size = (FRAME_SIZE / 2) * (FRAME_SIZE / 2);
        for index in 0..FRAME_COUNT {
            let luma = vec![index as u8; luma_size];
            let u_plane = vec![100u8; chroma_size];
            let v_plane = vec![150u8; chroma_size];
            let frame = y4m::Frame::new([&luma, &u_plane, &v_plane], None);
            encoder.write_frame(&frame).expect("writes a frame");
        }

        bytes
    }

    /// A luma plane of repeatable noise, with a different pattern in the second half of the
    /// stream.
    fn noisy_luma(index: u64) -> Vec<u8> {
        let half = if index < METADATA_FRAME_COUNT / 2 {
            0u32
        } else {
            1u32
        };
        let mut luma = vec![0u8; FRAME_SIZE * FRAME_SIZE];

        for row in 0..FRAME_SIZE {
            for column in 0..FRAME_SIZE {
                let hash = (row as u32).wrapping_mul(2_654_435_761)
                    ^ (column as u32).wrapping_mul(40_503)
                    ^ half.wrapping_mul(0x9E37_79B9)
                    ^ (index as u32);
                luma[row * FRAME_SIZE + column] = (hash % 256) as u8;
            }
        }

        luma
    }

    fn textured_y4m_stream() -> Vec<u8> {
        let mut bytes = Vec::new();
        let ratio = y4m::Ratio::new(30000, 1001);
        let range = y4m::VendorExtensionString::new(b"COLORRANGE=FULL".to_vec()).expect("valid");
        let mut encoder = y4m::encode(FRAME_SIZE, FRAME_SIZE, ratio)
            .append_vendor_extension(range)
            .write_header(&mut bytes)
            .expect("writes the header");

        let chroma_size = (FRAME_SIZE / 2) * (FRAME_SIZE / 2);
        for index in 0..METADATA_FRAME_COUNT {
            let luma = noisy_luma(index);
            let u_plane = vec![100u8; chroma_size];
            let v_plane = vec![150u8; chroma_size];
            let frame = y4m::Frame::new([&luma, &u_plane, &v_plane], None);
            encoder.write_frame(&frame).expect("writes a frame");
        }

        bytes
    }

    #[test]
    fn executing_the_passthrough_example_copies_stdin_to_stdout() {
        let path = example("passthrough.toml");
        let path = path.to_str().expect("utf-8 path");

        let mut child = Command::new(env!("CARGO_BIN_EXE_av-toolbox"))
            .args(["--color", "never", "execute", path])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("the binary runs");

        let mut stdin = child.stdin.take().expect("stdin is piped");
        let input = y4m_stream();
        let expected = input.clone();
        let feeder = thread::spawn(move || {
            stdin.write_all(&input).expect("writes the stream");
        });

        let output = child.wait_with_output().expect("the binary finishes");
        feeder.join().expect("the feeder thread finishes");
        let stderr = text(&output.stderr);

        assert!(output.status.success(), "{stderr}");
        assert_eq!(output.stdout, expected);
    }

    #[test]
    fn explain_compiles_a_non_live_pipeline_on_a_device() {
        let directory = tempfile::tempdir().expect("a temp dir");
        let input_path = directory.path().join("input.y4m");
        let output_path = directory.path().join("output.y4m");
        let input = y4m_stream();
        std::fs::write(&input_path, input).expect("writes the input");

        let input_path = input_path.to_str().expect("utf-8 path");
        let output_path = output_path.to_str().expect("utf-8 path");
        let content = format!(
            r#"
[source.input]
type = "y4m/pipe"
args.path = "{input_path}"

[sink.output]
type = "y4m/pipe"
inputs = ["input"]
args.path = "{output_path}"
"#
        );
        let pipeline_path = directory.path().join("pipeline.toml");
        std::fs::write(&pipeline_path, content).expect("writes the pipeline");
        let pipeline_path = pipeline_path.to_str().expect("utf-8 path");

        let output = Command::new(env!("CARGO_BIN_EXE_av-toolbox"))
            .args(["--color", "never", "explain", pipeline_path])
            .output()
            .expect("the binary runs");
        let stderr = text(&output.stderr);
        let stdout = text(&output.stdout);

        assert!(output.status.success(), "{stderr}");
        assert!(stdout.contains("output"));
        assert!(!stderr.contains("reads a live stream"));
    }

    #[test]
    fn executing_the_metadata_example_logs_one_line_per_frame() {
        let directory = tempfile::tempdir().expect("a temp dir");
        let log_path = directory.path().join("metadata.ndjson");
        let log_path = log_path.to_str().expect("utf-8 path");
        let path = example("metadata.toml");
        let path = path.to_str().expect("utf-8 path");
        let metadata_var = format!("metadata={log_path}");

        let mut child = Command::new(env!("CARGO_BIN_EXE_av-toolbox"))
            .args(["--color", "never", "execute", path])
            .args(["--var", &metadata_var])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("the binary runs");

        let mut stdin = child.stdin.take().expect("stdin is piped");
        let input = textured_y4m_stream();
        let feeder = thread::spawn(move || {
            stdin.write_all(&input).expect("writes the stream");
        });

        let output = child.wait_with_output().expect("the binary finishes");
        feeder.join().expect("the feeder thread finishes");
        let stderr = text(&output.stderr);

        assert!(output.status.success(), "{stderr}");

        let logged = std::fs::read_to_string(log_path).expect("reads the log");
        let lines: Vec<serde_json::Value> = logged
            .lines()
            .map(|line| serde_json::from_str(line).expect("valid json"))
            .collect();
        assert_eq!(lines.len() as u64, METADATA_FRAME_COUNT);

        for (position, line) in lines.iter().enumerate() {
            assert_eq!(line["index"], position as u64, "{lines:?}");
        }

        let has_a_real_score = lines.iter().any(|line| line["scene_score"].is_number());
        assert!(has_a_real_score, "{lines:?}");
    }

    #[test]
    fn executing_the_chunked_example_runs_through_the_called_file() {
        let path = example("chunked.toml");
        let path = path.to_str().expect("utf-8 path");

        let mut child = Command::new(env!("CARGO_BIN_EXE_av-toolbox"))
            .args(["--color", "never", "execute", path])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("the binary runs");

        let mut stdin = child.stdin.take().expect("stdin is piped");
        let input = y4m_stream();
        let feeder = thread::spawn(move || {
            stdin.write_all(&input).expect("writes the stream");
        });

        let output = child.wait_with_output().expect("the binary finishes");
        feeder.join().expect("the feeder thread finishes");
        let stderr = text(&output.stderr);

        assert!(output.status.success(), "{stderr}");
        assert!(output.stdout.starts_with(b"YUV4MPEG2"));
    }

    /// Runs an example with `vars`, feeding it `input` on standard input.
    fn execute(name: &str, vars: &[&str], input: Vec<u8>) -> Output {
        let path = example(name);
        let path = path.to_str().expect("utf-8 path");
        let var_args = vars.iter().flat_map(|var| ["--var", var]);

        let mut child = Command::new(env!("CARGO_BIN_EXE_av-toolbox"))
            .args(["--color", "never", "execute", path])
            .args(var_args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("the binary runs");

        let mut stdin = child.stdin.take().expect("stdin is piped");
        let feeder = thread::spawn(move || {
            stdin.write_all(&input).expect("writes the stream");
        });

        let output = child.wait_with_output().expect("the binary finishes");
        feeder.join().expect("the feeder thread finishes");

        output
    }

    /// The width, height and frame count of a y4m stream.
    fn shape_of(stream: &[u8]) -> (usize, usize, u64) {
        let mut decoder = y4m::decode(stream).expect("a y4m stream");
        let width = decoder.get_width();
        let height = decoder.get_height();

        let mut frames = 0;
        while decoder.read_frame().is_ok() {
            frames += 1;
        }

        (width, height, frames)
    }

    #[test]
    fn executing_the_resize_example_writes_the_new_size() {
        let input = y4m_stream();

        let output = execute("resize.toml", &["width=32", "height=24"], input);
        let stderr = text(&output.stderr);

        assert!(output.status.success(), "{stderr}");
        assert_eq!(shape_of(&output.stdout), (32, 24, FRAME_COUNT));
    }

    #[test]
    fn executing_the_solid_example_writes_its_frames() {
        let vars = ["width=64", "height=48", "frames=3"];

        let output = execute("solid.toml", &vars, Vec::new());
        let stderr = text(&output.stderr);

        assert!(output.status.success(), "{stderr}");
        assert_eq!(shape_of(&output.stdout), (64, 48, 3));
    }

    #[test]
    fn executing_the_image_example_writes_the_resized_image() {
        let directory = tempfile::tempdir().expect("a temp dir");
        let image_path = directory.path().join("orange.png");
        let orange = image::RgbImage::from_pixel(10, 6, image::Rgb([255, 128, 0]));
        orange.save(&image_path).expect("writes the image");
        let image_path = image_path.to_str().expect("utf-8 path");
        let image_var = format!("image={image_path}");
        let vars = [image_var.as_str(), "frames=2", "width=20", "height=12"];

        let output = execute("image.toml", &vars, Vec::new());
        let stderr = text(&output.stderr);

        assert!(output.status.success(), "{stderr}");
        assert_eq!(shape_of(&output.stdout), (20, 12, 2));
    }

    #[test]
    fn executing_the_recolour_lines_example_keeps_the_input_shape() {
        let input = textured_y4m_stream();

        let output = execute("recolour-lines.toml", &[], input);
        let stderr = text(&output.stderr);

        assert!(output.status.success(), "{stderr}");
        assert_eq!(
            shape_of(&output.stdout),
            (FRAME_SIZE, FRAME_SIZE, METADATA_FRAME_COUNT)
        );
    }
}
