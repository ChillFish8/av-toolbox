use std::path::Path;
use std::process::{Command, Output};

const BROKEN: &str = r##"[source.fill]
type = "solid/fil"
args.colour = "#202020"

[sink.output]
type = "y4m/pipe"
inputs = ["fil"]
"##;

const BROKEN_DIAGNOSTICS: &str = r#"error[unknown-stage]: no source stage is registered as solid/fil
  ┌─ broken.toml:2:8
  │
2 │ type = "solid/fil"
  │        ^^^^^^^^^^^

error[unknown-stage-reference]: no stage named fil is defined, did you mean fill
  ┌─ broken.toml:7:11
  │
7 │ inputs = ["fil"]
  │           ^^^^^

error: pipeline has 2 errors
"#;

/// Runs `execute` in `directory` with no `RUST_LOG` or `AVTB_VAR_*` reaching it, so its stderr is the same anywhere.
fn execute_in(directory: &Path, args: &[&str]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_av-toolbox"));
    command.current_dir(directory).env_remove("RUST_LOG");

    let pipeline_vars = std::env::vars_os()
        .map(|(name, _)| name)
        .filter(|name| name.to_string_lossy().starts_with("AVTB_VAR_"));
    for name in pipeline_vars {
        command.env_remove(name);
    }

    command
        .args(["--color", "never", "execute"])
        .args(args)
        .output()
        .expect("the binary runs")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

#[test]
fn execute_prints_the_same_diagnostics_for_a_broken_pipeline() {
    let directory = tempfile::tempdir().expect("a temp dir");
    let pipeline = directory.path().join("broken.toml");
    std::fs::write(&pipeline, BROKEN).expect("writes the pipeline");

    let output = execute_in(directory.path(), &["broken.toml"]);
    let stderr = text(&output.stderr);

    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert_eq!(stderr, BROKEN_DIAGNOSTICS);
    assert!(output.stdout.is_empty());
}

#[cfg(all(unix, feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
mod device {
    use std::path::{Path, PathBuf};

    use super::{execute_in, text};

    /// The length of the clip `execute_writes_the_same_y4m_bytes` writes.
    const WRITTEN_LENGTH: usize = 27_718;

    /// The hash of that clip.
    const WRITTEN_HASH: u64 = 0x0400_c52b_ff22_34f8;

    fn example(name: &str) -> PathBuf {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        root.join("..").join("examples").join(name)
    }

    /// The FNV-1a hash of `bytes`, which stays the same across Rust releases.
    fn fnv1a(bytes: &[u8]) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }

        hash
    }

    #[test]
    fn execute_writes_the_same_y4m_bytes() {
        let directory = tempfile::tempdir().expect("a temp dir");
        let pipeline = example("solid.toml");
        let pipeline = pipeline.to_str().expect("utf-8 path");

        let output = execute_in(
            directory.path(),
            &[
                pipeline,
                "--var",
                "colour=#4080c0",
                "--var",
                "width=64",
                "--var",
                "height=48",
                "--var",
                "frames=3",
                "--var",
                "output=solid.y4m",
            ],
        );
        let stderr = text(&output.stderr);

        assert!(output.status.success(), "{stderr}");
        assert!(stderr.contains("the run finished"), "{stderr}");

        let clip = directory.path().join("solid.y4m");
        let written = std::fs::read(clip).expect("the clip was written");
        assert_eq!(written.len(), WRITTEN_LENGTH);
        assert_eq!(fnv1a(&written), WRITTEN_HASH);
    }
}
