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
        "artcnn.toml",
        "deblock-dpir.toml",
        "deblock-h264.toml",
        "deblock-qed.toml",
        "deblock-pp7.toml",
        "nl4d-grain-table.toml",
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
    let stdout = text(&output.stdout);

    assert!(output.status.success(), "{stderr}");
    let html = std::fs::read_to_string(&page).expect("the page was written");
    assert!(html.contains("<html"));

    let announced = format!("wrote {}", page.display());
    assert!(stdout.contains(&announced), "{stdout}");
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
    let stdout = text(&output.stdout);

    assert!(output.status.success(), "{stderr}");
    let page = directory.path().join("plan.html");
    let html = std::fs::read_to_string(&page).expect("the page was written");
    assert!(html.contains("<html"));
    assert!(stdout.contains("wrote plan.html"), "{stdout}");
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
fn list_stages_shows_each_providers_kind_and_side() {
    let output = toolbox(&["list-stages"]);
    let stdout = text(&output.stdout);

    let columns_of = |identifier: &str| -> Vec<&str> {
        let row = stdout
            .lines()
            .find(|line| line.starts_with(identifier))
            .expect("the provider is listed");

        row.split_whitespace().collect()
    };

    assert_eq!(
        columns_of("provider/scene-split"),
        ["provider/scene-split", "scenes", "meta"]
    );
    assert_eq!(
        columns_of("provider/scene-score"),
        ["provider/scene-score", "scene-score", "host"]
    );
}

#[test]
fn list_stages_describes_one_type() {
    let output = toolbox(&["list-stages", "y4m/stdout"]);

    assert!(output.status.success());
    assert!(text(&output.stdout).contains("frame_rate"));
}

#[test]
fn list_stages_shows_the_nl4d_grain_table() {
    let output = toolbox(&["list-stages", "av-denoise/nl4d"]);
    let stdout = text(&output.stdout);

    assert!(output.status.success());
    assert!(stdout.contains("grain_table  opaque  .tbl"), "{stdout}");
    assert!(stdout.contains("film grain table"), "{stdout}");
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

[sink.output]
type = "y4m/stdout"
inputs = ["input"]

[sink.log]
type = "metadata/ndjson"
inputs = ["input"]
args.path = "scenes.ndjson"
args.scenes = { breaks = { mode = "natural" }, speed = "{{ speed }}" }
"#;

const CALLED_PART: &str = r#"
[input.clip]

[filter.shrink]
type = "resize/bilinear"
inputs = ["clip"]
args.width = 32
args.height = 32

[output.shrunk]
inputs = ["shrink"]
"#;

const CALLING_PIPELINE: &str = r#"
[source.input]
type = "y4m/stdin"

[call.parts]
path = "part.toml"
inputs = ["input"]

[sink.output]
type = "y4m/stdout"
inputs = ["parts"]
"#;

#[test]
fn explain_shows_the_prefixed_names_of_called_stages() {
    let directory = tempfile::tempdir().expect("a temp dir");
    let part_path = directory.path().join("part.toml");
    let pipeline_path = directory.path().join("pipeline.toml");
    std::fs::write(&part_path, CALLED_PART).expect("writes the part");
    std::fs::write(&pipeline_path, CALLING_PIPELINE).expect("writes the pipeline");
    let path = pipeline_path.to_str().expect("utf-8 path");

    let output = toolbox(&["explain", path, "--structure-only"]);
    let stderr = text(&output.stderr);

    assert!(output.status.success(), "{stderr}");
    assert!(text(&output.stdout).contains("parts.shrink"));
}

#[test]
fn explain_shows_the_provider_a_chunk_stage_implies() {
    let path = example("denoise-nl4d.toml");
    let path = path.to_str().expect("utf-8 path");

    let output = toolbox(&["explain", path, "--structure-only"]);
    let stderr = text(&output.stderr);
    let stdout = text(&output.stdout);

    assert!(output.status.success(), "{stderr}");
    assert!(stdout.contains("scenes · implied"), "{stdout}");
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

#[test]
fn execute_refuses_a_missing_profile() {
    let pipeline = example("passthrough.toml");
    let pipeline = pipeline.to_str().expect("utf-8 path");

    let output = toolbox(&["execute", pipeline, "--profile", "/nowhere/clip.profile.toml"]);
    let stderr = text(&output.stderr);

    assert!(!output.status.success());
    assert!(stderr.contains("cannot read the profile"), "{stderr}");
}

#[test]
fn explain_warns_about_a_profile_naming_no_stage() {
    let directory = tempfile::tempdir().expect("a temp dir");
    let profile_path = directory.path().join("passthrough.profile.toml");
    let profile = r#"
pipeline = "passthrough.toml"
device = "vulkan discrete:1"
frames = 4

[workers]
"filter.missing" = 2
"#;
    std::fs::write(&profile_path, profile).expect("writes the profile");
    let pipeline = example("passthrough.toml");
    let pipeline = pipeline.to_str().expect("utf-8 path");
    let profile_path = profile_path.to_str().expect("utf-8 path");

    let output = toolbox(&["explain", pipeline, "--structure-only", "--profile", profile_path]);
    let stderr = text(&output.stderr);

    assert!(output.status.success(), "{stderr}");
    assert!(stderr.contains("filter.missing"), "{stderr}");
}

#[test]
fn profile_refuses_a_live_source() {
    let pipeline = example("passthrough.toml");
    let pipeline = pipeline.to_str().expect("utf-8 path");

    let output = toolbox(&["profile", pipeline]);
    let stderr = text(&output.stderr);

    assert!(!output.status.success());
    assert!(stderr.contains("y4m/stdin"), "{stderr}");
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
    fn explain_draws_the_widths_a_profile_gives() {
        let directory = tempfile::tempdir().expect("a temp dir");
        let input_path = directory.path().join("input.y4m");
        let output_path = directory.path().join("output.y4m");
        let profile_path = directory.path().join("pipeline.profile.toml");
        let input = y4m_stream();
        std::fs::write(&input_path, input).expect("writes the input");

        let input_path = input_path.to_str().expect("utf-8 path");
        let output_path = output_path.to_str().expect("utf-8 path");
        let content = format!(
            r#"
[source.input]
type = "y4m/pipe"
args.path = "{input_path}"

[filter.resize]
type = "resize/bilinear"
inputs = ["input"]
args.width = 32
args.height = 32

[sink.output]
type = "y4m/pipe"
inputs = ["resize"]
args.path = "{output_path}"
"#
        );
        let profile = r#"
pipeline = "pipeline.toml"
device = "vulkan discrete:1"
frames = 4

[workers]
"filter.resize" = 3
"#;
        let pipeline_path = directory.path().join("pipeline.toml");
        std::fs::write(&pipeline_path, content).expect("writes the pipeline");
        std::fs::write(&profile_path, profile).expect("writes the profile");
        let pipeline_path = pipeline_path.to_str().expect("utf-8 path");
        let profile_path = profile_path.to_str().expect("utf-8 path");

        let output = Command::new(env!("CARGO_BIN_EXE_av-toolbox"))
            .args([
                "--color",
                "never",
                "explain",
                pipeline_path,
                "--profile",
                profile_path,
            ])
            .output()
            .expect("the binary runs");
        let stderr = text(&output.stderr);
        let stdout = text(&output.stdout);

        assert!(output.status.success(), "{stderr}");
        assert!(stdout.contains("×3"), "{stdout}");
    }

    #[test]
    fn executing_the_metadata_example_logs_one_line_per_frame() {
        let directory = tempfile::tempdir().expect("a temp dir");
        let log_path = directory.path().join("metadata.ndjson");
        let log_path = log_path.to_str().expect("utf-8 path");
        let metadata_var = format!("metadata={log_path}");

        let run = execute("metadata.toml", &[&metadata_var], Some(textured_y4m_stream()));
        let stderr = text(&run.output.stderr);

        assert!(run.output.status.success(), "{stderr}");

        let logged = std::fs::read_to_string(log_path).expect("reads the log");
        let lines: Vec<serde_json::Value> = logged
            .lines()
            .map(|line| serde_json::from_str(line).expect("valid json"))
            .collect();
        assert_eq!(lines.len() as u64, METADATA_FRAME_COUNT);

        for (position, line) in lines.iter().enumerate() {
            assert_eq!(line["index"], position as u64, "{lines:?}");
            assert!(line["scene_score"].is_null(), "{lines:?}");
        }
    }

    #[test]
    fn executing_the_scenes_example_logs_every_frames_scene() {
        let directory = tempfile::tempdir().expect("a temp dir");
        let log_path = directory.path().join("scenes.ndjson");
        let log_path = log_path.to_str().expect("utf-8 path");
        let scenes_var = format!("scenes={log_path}");

        let run = execute("scenes.toml", &[&scenes_var], Some(textured_y4m_stream()));
        let stderr = text(&run.output.stderr);

        assert!(run.output.status.success(), "{stderr}");

        let logged = std::fs::read_to_string(log_path).expect("reads the log");
        let lines: Vec<serde_json::Value> = logged
            .lines()
            .map(|line| serde_json::from_str(line).expect("valid json"))
            .collect();
        assert_eq!(lines.len() as u64, METADATA_FRAME_COUNT);

        for (position, line) in lines.iter().enumerate() {
            assert_eq!(line["index"], position as u64, "{lines:?}");
            assert!(line["chunk"].is_u64(), "{lines:?}");
        }

        let has_a_real_score = lines.iter().any(|line| line["scene_score"].is_number());
        assert!(has_a_real_score, "{lines:?}");
    }

    #[test]
    fn executing_the_chunked_example_denoises_on_two_workers() {
        let run = execute("chunked.toml", &[], Some(y4m_stream()));
        let stderr = text(&run.output.stderr);

        assert!(run.output.status.success(), "{stderr}");
        assert!(run.stream.starts_with(b"YUV4MPEG2"));
    }

    /// Frames in each scene of the grainy stream.
    const GRAINY_SCENE_LENGTH: usize = 20;

    /// Repeatable noise from `seed`, one byte per sample.
    fn pattern(seed: u32, samples: usize) -> Vec<u8> {
        let mut state = seed.max(1);
        let mut bytes = Vec::with_capacity(samples);

        for _ in 0..samples {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            bytes.push((state & 0xff) as u8);
        }

        bytes
    }

    /// A 192x128 clip of three flat luma bands under fresh grain, swapping sides every scene.
    fn grainy_y4m_stream(frames: usize) -> Vec<u8> {
        let width = 192;
        let height = 128;
        let mut bytes = Vec::new();
        let ratio = y4m::Ratio::new(25, 1);
        let mut encoder = y4m::encode(width, height, ratio)
            .with_colorspace(y4m::Colorspace::C420)
            .write_header(&mut bytes)
            .expect("writes the header");
        let chroma = vec![128u8; width * height / 4];

        for index in 0..frames {
            let scene = index / GRAINY_SCENE_LENGTH;
            let levels: [i16; 3] = if scene.is_multiple_of(2) {
                [60, 120, 180]
            } else {
                [180, 120, 60]
            };
            let seed = index as u32 * 7919 + 12345;
            let grain = pattern(seed, width * height);
            let luma: Vec<u8> = grain
                .iter()
                .enumerate()
                .map(|(pixel, sample)| {
                    let band = (pixel % width) * 3 / width;
                    let wobble = i16::from(*sample % 9) - 4;
                    (levels[band] + wobble) as u8
                })
                .collect();
            let frame = y4m::Frame::new([&luma, &chroma, &chroma], None);
            encoder.write_frame(&frame).expect("writes a frame");
        }

        bytes
    }

    #[test]
    fn executing_the_grain_table_example_writes_a_table() {
        let directory = tempfile::tempdir().expect("a temp dir");
        let grain_path = directory.path().join("grain");
        let grain_text = grain_path.to_str().expect("utf-8 path");
        let grain_var = format!("grain={grain_text}");

        let run = execute(
            "nl4d-grain-table.toml",
            &[&grain_var],
            Some(grainy_y4m_stream(60)),
        );
        let stderr = text(&run.output.stderr);

        assert!(run.output.status.success(), "{stderr}");

        let table_path = grain_path.join("filter.denoise.grain_table.tbl");
        let table = std::fs::read_to_string(table_path).expect("the table was written");

        assert!(table.starts_with("filmgrn1\n"), "{table}");
        assert!(table.contains("\nE "), "{table}");
    }

    #[test]
    fn profiling_writes_no_artefacts() {
        let directory = tempfile::tempdir().expect("a temp dir");
        let pipeline_path = directory.path().join("pipeline.toml");
        let profile_path = directory.path().join("pipeline.profile.toml");
        let grain_path = directory.path().join("grain");
        let grain_text = grain_path.to_str().expect("utf-8 path");
        let content = format!(
            r##"
[source.fill]
type = "solid/fill"
args.colour = "#202020"
args.width = 64
args.height = 64
args.frames = 12
args.frame_rate = [24, 1]

[filter.denoise]
type = "av-denoise/nl4d"
inputs = ["fill"]

[sink.output]
type = "y4m/stdout"
inputs = ["denoise"]

[sink.grain]
type = "artefacts/files"
args.directory = "{}"
"##,
            grain_text
        );
        std::fs::write(&pipeline_path, content).expect("writes the pipeline");
        let pipeline_path = pipeline_path.to_str().expect("utf-8 path");
        let profile_path = profile_path.to_str().expect("utf-8 path");

        let profiled = Command::new(env!("CARGO_BIN_EXE_av-toolbox"))
            .args(["--color", "never", "profile", pipeline_path])
            .args(["--frames", "4", "--warmup", "2", "--output", profile_path])
            .output()
            .expect("the binary runs");
        let stderr = text(&profiled.stderr);
        let stdout = text(&profiled.stdout);

        assert!(profiled.status.success(), "{stderr}");
        assert!(stdout.contains("filter.denoise"), "{stdout}");
        assert!(!grain_path.exists());
    }

    /// An example run, with the y4m stream it wrote to its `output` file.
    struct Run {
        output: Output,
        stream: Vec<u8>,
    }

    /// Runs an example with `vars`, writing `input` to a file for its `input` variable and reading
    /// the stream back from its `output` variable.
    fn execute(name: &str, vars: &[&str], input: Option<Vec<u8>>) -> Run {
        let path = example(name);
        let path = path.to_str().expect("utf-8 path");
        let directory = tempfile::tempdir().expect("a temp dir");
        let input_path = directory.path().join("input.y4m");
        let output_path = directory.path().join("output.y4m");
        let output_var = format!("output={}", output_path.to_str().expect("utf-8 path"));
        let var_args = vars.iter().flat_map(|var| ["--var", var]);

        let mut command = Command::new(env!("CARGO_BIN_EXE_av-toolbox"));
        command
            .args(["--color", "never", "execute", path])
            .args(var_args)
            .args(["--var", &output_var]);

        if let Some(bytes) = input {
            std::fs::write(&input_path, bytes).expect("writes the input");
            let input_var = format!("input={}", input_path.to_str().expect("utf-8 path"));
            command.args(["--var", &input_var]);
        }

        let output = command.output().expect("the binary runs");
        let stream = std::fs::read(&output_path).unwrap_or_default();

        Run { output, stream }
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

        let run = execute("resize.toml", &["width=32", "height=24"], Some(input));
        let stderr = text(&run.output.stderr);

        assert!(run.output.status.success(), "{stderr}");
        assert_eq!(shape_of(&run.stream), (32, 24, FRAME_COUNT));
    }

    #[test]
    fn executing_the_solid_example_writes_its_frames() {
        let vars = ["width=64", "height=48", "frames=3"];

        let run = execute("solid.toml", &vars, None);
        let stderr = text(&run.output.stderr);

        assert!(run.output.status.success(), "{stderr}");
        assert_eq!(shape_of(&run.stream), (64, 48, 3));
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

        let run = execute("image.toml", &vars, None);
        let stderr = text(&run.output.stderr);

        assert!(run.output.status.success(), "{stderr}");
        assert_eq!(shape_of(&run.stream), (20, 12, 2));
    }

    #[test]
    fn executing_the_recolour_lines_example_keeps_the_input_shape() {
        let input = textured_y4m_stream();

        let run = execute("recolour-lines.toml", &[], Some(input));
        let stderr = text(&run.output.stderr);

        assert!(run.output.status.success(), "{stderr}");
        assert_eq!(
            shape_of(&run.stream),
            (FRAME_SIZE, FRAME_SIZE, METADATA_FRAME_COUNT)
        );
    }

    #[test]
    fn profiling_writes_a_profile_that_execute_runs_with() {
        let directory = tempfile::tempdir().expect("a temp dir");
        let pipeline_path = directory.path().join("pipeline.toml");
        let profile_path = directory.path().join("pipeline.profile.toml");
        let content = r##"
[source.fill]
type = "solid/fill"
args.colour = "#202020"
args.width = 64
args.height = 64
args.frames = 12
args.frame_rate = [24, 1]

[filter.resize]
type = "resize/bilinear"
inputs = ["fill"]
args.width = 32
args.height = 32

[sink.output]
type = "y4m/stdout"
inputs = ["resize"]
"##;
        std::fs::write(&pipeline_path, content).expect("writes the pipeline");
        let pipeline_path = pipeline_path.to_str().expect("utf-8 path");
        let profile_path = profile_path.to_str().expect("utf-8 path");

        let profiled = Command::new(env!("CARGO_BIN_EXE_av-toolbox"))
            .args(["--color", "never", "profile", pipeline_path])
            .args(["--frames", "4", "--warmup", "2", "--output", profile_path])
            .output()
            .expect("the binary runs");
        let stderr = text(&profiled.stderr);
        let stdout = text(&profiled.stdout);

        assert!(profiled.status.success(), "{stderr}");
        assert!(stderr.contains("pass 1 of 3"), "{stderr}");
        assert!(stderr.contains("pass 3 of 3"), "{stderr}");
        assert!(stdout.contains("filter.resize"), "{stdout}");
        assert!(
            stdout.contains("sinks replaced by null/discard: sink.output"),
            "{stdout}"
        );

        let written = std::fs::read_to_string(profile_path).expect("the profile was written");
        assert!(written.contains("\"filter.resize\""), "{written}");

        let executed = Command::new(env!("CARGO_BIN_EXE_av-toolbox"))
            .args([
                "--color",
                "never",
                "execute",
                pipeline_path,
                "--profile",
                profile_path,
            ])
            .output()
            .expect("the binary runs");
        let stderr = text(&executed.stderr);

        assert!(executed.status.success(), "{stderr}");
        assert!(!executed.stdout.is_empty());
    }
}

#[test]
fn inspect_prints_json_whose_hash_matches_inspect_hash() {
    let pipeline = example("resize.toml");
    let path = pipeline.to_str().expect("utf-8 path");

    let exported = toolbox(&["inspect", path]);
    let hashed = toolbox(&["inspect", path, "--hash"]);
    let stdout = text(&exported.stdout);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).expect("the export is JSON");
    let printed = text(&hashed.stdout);

    assert!(exported.status.success(), "{}", text(&exported.stderr));
    assert!(hashed.status.success());
    assert_eq!(parsed["valid"], serde_json::json!(true));
    assert_eq!(Some(printed.trim()), parsed["hash"].as_str());
}

#[test]
fn inspect_hashes_a_pipeline_the_same_however_its_path_is_spelt() {
    let spelt_with_dot_dot = example("resize.toml");
    let canonical = spelt_with_dot_dot.canonicalize().expect("the example exists");
    let dotted_path = spelt_with_dot_dot.to_str().expect("utf-8 path");
    let canonical_path = canonical.to_str().expect("utf-8 path");

    let dotted = toolbox(&["inspect", dotted_path, "--hash"]);
    let plain = toolbox(&["inspect", canonical_path, "--hash"]);

    assert!(dotted_path.contains(".."), "{dotted_path}");
    assert!(dotted.status.success(), "{}", text(&dotted.stderr));
    assert_eq!(text(&dotted.stdout), text(&plain.stdout));
}

#[test]
fn inspect_of_a_broken_file_exits_one_with_json_diagnostics() {
    let file = written("[sources.input]\ntype = \"y4m/stdin\"\n");
    let path = file.path().to_str().expect("utf-8 path");

    let output = toolbox(&["inspect", path]);
    let stdout = text(&output.stdout);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).expect("the export is JSON");

    assert_eq!(output.status.code(), Some(1));
    assert_eq!(parsed["valid"], serde_json::json!(false));
    assert_eq!(
        parsed["diagnostics"][0]["code"],
        serde_json::json!("invalid-definition")
    );
    assert!(text(&output.stderr).contains("error[invalid-definition]"));
}

#[test]
fn inspect_hash_of_a_broken_file_prints_nothing() {
    let file = written("[sources.input]\ntype = \"y4m/stdin\"\n");
    let path = file.path().to_str().expect("utf-8 path");

    let output = toolbox(&["inspect", path, "--hash"]);

    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
}
