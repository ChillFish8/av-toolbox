#![cfg(all(unix, feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]

use std::fs;
use std::path::{Path, PathBuf};

use av_toolbox_gpu::{Backend, Device};
use av_toolbox_pipeline::{DiskFiles, Pipeline, Supplied};
use av_toolbox_runtime::Settings;

const SIZE: usize = 64;

const FRAMES: u64 = 3;

const FLAT_U: u8 = 90;

const FLAT_V: u8 = 180;

/// One decoded frame's planes.
struct DecodedFrame {
    luma: Vec<u8>,
    u_plane: Vec<u8>,
    v_plane: Vec<u8>,
}

/// A decoded y4m stream's colourspace and the planes of each of its frames.
struct Decoded {
    colorspace: y4m::Colorspace,
    frames: Vec<DecodedFrame>,
}

/// The luma plane a `gradient_clip` writes, the same horizontal gradient row repeated down every row.
fn gradient_luma() -> Vec<u8> {
    let row: Vec<u8> = (0..SIZE).map(|column| column as u8 * 4).collect();
    row.repeat(SIZE)
}

/// A C420 y4m clip whose luma is `gradient_luma`, with flat chroma.
fn gradient_clip() -> Vec<u8> {
    let mut bytes = Vec::new();
    let ratio = y4m::Ratio::new(24, 1);
    let mut encoder = y4m::encode(SIZE, SIZE, ratio)
        .write_header(&mut bytes)
        .expect("writes the header");

    let luma = gradient_luma();
    let chroma_samples = (SIZE / 2) * (SIZE / 2);
    let u_plane = vec![FLAT_U; chroma_samples];
    let v_plane = vec![FLAT_V; chroma_samples];
    for _ in 0..FRAMES {
        let frame = y4m::Frame::new([&luma, &u_plane, &v_plane], None);
        encoder.write_frame(&frame).expect("writes a frame");
    }

    bytes
}

fn decode(stream: &[u8]) -> Decoded {
    let mut decoder = y4m::decode(stream).expect("reads the header");
    let colorspace = decoder.get_colorspace();

    let mut frames = Vec::new();
    while let Ok(frame) = decoder.read_frame() {
        let luma = frame.get_y_plane().to_vec();
        let u_plane = frame.get_u_plane().to_vec();
        let v_plane = frame.get_v_plane().to_vec();
        frames.push(DecodedFrame {
            luma,
            u_plane,
            v_plane,
        });
    }

    Decoded { colorspace, frames }
}

fn utf8(path: &Path) -> &str {
    path.to_str().expect("a utf-8 path")
}

fn example(name: &str) -> PathBuf {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../examples")).join(name)
}

/// Writes `gradient_clip` as the input, runs `name` with `extra_flags`, and decodes the output.
fn run_example(name: &str, extra_flags: &[(&str, &str)]) -> Decoded {
    let folder = tempfile::tempdir().expect("a temp dir");
    let input = folder.path().join("input.y4m");
    let output = folder.path().join("output.y4m");

    fs::write(&input, gradient_clip()).expect("writes the input");

    let input_path = utf8(&input);
    let output_path = utf8(&output);
    let mut supplied = Supplied::new();
    supplied.set_flag("input", input_path);
    supplied.set_flag("output", output_path);
    for (key, value) in extra_flags {
        supplied.set_flag(*key, *value);
    }

    let example_path = example(name);
    let loaded = av_toolbox_pipeline::load(&example_path, &DiskFiles, &supplied).expect("the example loads");
    let registry = av_toolbox_pipeline::registry().expect("the built-in stages register");
    let pipeline = Pipeline::build(&registry, &loaded.specs).expect("should build");
    let backend = Backend::open_default(Device::Default).expect("should open a device");
    let compiled = pipeline.compile(&backend).expect("should compile");
    let settings = Settings::default();
    let engine = compiled.engine(settings).expect("should build the engine");
    let report = engine.run().expect("should run");

    assert_eq!(report.frames, FRAMES);

    let written = fs::read(&output).expect("reads the output");
    decode(&written)
}

#[test]
fn the_resample_example_round_trips_a_flat_clip() {
    let decoded = run_example("resample.toml", &[]);
    let expected_luma = gradient_luma();

    assert!(matches!(decoded.colorspace, y4m::Colorspace::C420));
    for frame in &decoded.frames {
        assert_eq!(
            frame.luma, expected_luma,
            "luma should round trip 8 to 10 to 8 exactly"
        );
        assert!(frame.u_plane.iter().all(|sample| sample.abs_diff(FLAT_U) <= 1));
        assert!(frame.v_plane.iter().all(|sample| sample.abs_diff(FLAT_V) <= 1));
    }
}

#[test]
fn the_planes_example_writes_444() {
    let decoded = run_example("planes.toml", &[("width", "64"), ("height", "64")]);
    let expected_luma = gradient_luma();

    assert!(matches!(decoded.colorspace, y4m::Colorspace::C444));
    assert_eq!(decoded.frames.len(), FRAMES as usize);
    for frame in &decoded.frames {
        assert_eq!(frame.luma, expected_luma, "luma should pass through untouched");
    }
}
