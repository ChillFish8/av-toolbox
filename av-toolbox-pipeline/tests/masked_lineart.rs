#![cfg(all(unix, feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]

use std::fs;
use std::path::Path;

use av_toolbox_gpu::{Backend, Device};
use av_toolbox_pipeline::{DiskFiles, Pipeline, Supplied};
use av_toolbox_runtime::Settings;

const WIDTH: usize = 64;

const HEIGHT: usize = 48;

const FRAMES: u64 = 2;

const PLANE_NAMES: [&str; 3] = ["y", "u", "v"];

/// One vertical line, the column its profile starts at and its samples across on the 8-bit scale.
struct Line {
    left: usize,
    profile: &'static [u16],
}

const LUMA_FIELD: u16 = 220;

/// Anti-aliased lines of different widths and depths, some off the sample grid.
const LUMA_LINES: [Line; 5] = [
    Line {
        left: 4,
        profile: &[185, 60, 40, 80, 200],
    },
    Line {
        left: 16,
        profile: &[148, 40, 40, 184],
    },
    Line {
        left: 28,
        profile: &[160, 70, 50, 70, 160],
    },
    Line {
        left: 40,
        profile: &[200, 120, 60, 120, 200],
    },
    Line {
        left: 52,
        profile: &[170, 50, 170],
    },
];

const CHROMA_FIELD: u16 = 160;

const U_LINES: [Line; 1] = [Line {
    left: 8,
    profile: &[140, 90, 70, 110, 150],
}];

const V_LINES: [Line; 1] = [Line {
    left: 20,
    profile: &[150, 100, 80, 120, 150],
}];

/// The samples of a line below this are its core, which odd rows lighten so the line is combed.
const CORE_BELOW: u16 = 130;

const COMB: u16 = 16;

/// The most samples of a plane that may sit a code from `blend/masked`. It measured 24, on the U plane
/// of darken at 8 bits, where darken alone is rounded before the blend.
const OFF_BY_ONE_LIMIT: usize = 24;

/// The lineart stages under test, and the arguments that make each one alone what a mask merges.
///
/// A bound mask replaces thin's own mask, so thin alone runs with `mthr = 0` to warp every sample.
const STAGES: [(&str, &str); 4] = [
    ("lineart/vinverse", ""),
    ("lineart/darken", ""),
    ("lineart/toon", ""),
    ("lineart/thin", "args.mthr = 0.0"),
];

/// A plane of `field` with `lines` down it, on the 8-bit scale times `scale`.
fn lined_plane(width: usize, height: usize, field: u16, lines: &[Line], scale: u16) -> Vec<u16> {
    let mut plane = Vec::with_capacity(width * height);

    for row in 0..height {
        let mut samples = vec![field; width];
        for line in lines {
            for (offset, value) in line.profile.iter().enumerate() {
                let is_core = *value < CORE_BELOW;
                let comb = if is_core && row % 2 == 1 { COMB } else { 0 };
                samples[line.left + offset] = value + comb;
            }
        }

        let scaled = samples.iter().map(|sample| sample * scale);
        plane.extend(scaled);
    }

    plane
}

/// The luma, U and V planes of the lined clip at `bits`.
fn lined_planes(bits: u32) -> [Vec<u16>; 3] {
    let scale = 1u16 << (bits - 8);
    let chroma_width = WIDTH / 2;
    let chroma_height = HEIGHT / 2;

    let luma = lined_plane(WIDTH, HEIGHT, LUMA_FIELD, &LUMA_LINES, scale);
    let u_plane = lined_plane(chroma_width, chroma_height, CHROMA_FIELD, &U_LINES, scale);
    let v_plane = lined_plane(chroma_width, chroma_height, CHROMA_FIELD, &V_LINES, scale);

    [luma, u_plane, v_plane]
}

fn colorspace_of(bits: u32) -> y4m::Colorspace {
    match bits {
        8 => y4m::Colorspace::C420jpeg,
        _ => y4m::Colorspace::C420p10,
    }
}

fn plane_bytes(plane: &[u16], bits: u32) -> Vec<u8> {
    match bits {
        8 => plane.iter().map(|sample| *sample as u8).collect(),
        _ => plane.iter().flat_map(|sample| sample.to_le_bytes()).collect(),
    }
}

fn plane_codes(bytes: &[u8], bits: u32) -> Vec<u16> {
    match bits {
        8 => bytes.iter().map(|sample| u16::from(*sample)).collect(),
        _ => {
            let (pairs, _) = bytes.as_chunks::<2>();
            pairs.iter().map(|pair| u16::from_le_bytes(*pair)).collect()
        },
    }
}

/// A 4:2:0 y4m clip of `FRAMES` frames of `planes` at `bits`.
fn clip(planes: &[Vec<u16>; 3], bits: u32) -> Vec<u8> {
    let mut bytes = Vec::new();
    let ratio = y4m::Ratio::new(24, 1);
    let colorspace = colorspace_of(bits);
    let mut encoder = y4m::encode(WIDTH, HEIGHT, ratio)
        .with_colorspace(colorspace)
        .write_header(&mut bytes)
        .expect("writes the header");

    let luma = plane_bytes(&planes[0], bits);
    let u_plane = plane_bytes(&planes[1], bits);
    let v_plane = plane_bytes(&planes[2], bits);
    for _ in 0..FRAMES {
        let frame = y4m::Frame::new([&luma, &u_plane, &v_plane], None);
        encoder.write_frame(&frame).expect("writes a frame");
    }

    bytes
}

/// The planes of every frame of a y4m stream at `bits`.
fn decode(stream: &[u8], bits: u32) -> Vec<[Vec<u16>; 3]> {
    let mut decoder = y4m::decode(stream).expect("reads the header");
    let mut frames = Vec::new();

    while let Ok(frame) = decoder.read_frame() {
        let luma = plane_codes(frame.get_y_plane(), bits);
        let u_plane = plane_codes(frame.get_u_plane(), bits);
        let v_plane = plane_codes(frame.get_v_plane(), bits);
        frames.push([luma, u_plane, v_plane]);
    }

    frames
}

fn utf8(path: &Path) -> &str {
    path.to_str().expect("a utf-8 path")
}

/// A pipeline that writes `stage` merged through a Prewitt mask to `masked`, and `stage` alone
/// blended over the input by the same mask to `blended`.
fn pipeline_text(stage: &str, alone_args: &str, input: &str, masked: &str, blended: &str) -> String {
    format!(
        r#"
[source.input]
type = "y4m/pipe"
args.path = "{input}"

[mask.edges]
type = "edges/prewitt"
inputs = ["input"]

[filter.alone]
type = "{stage}"
inputs = ["input"]
args.planes = ["y", "u", "v"]
{alone_args}

[filter.masked]
type = "{stage}"
inputs = ["input", "edges"]
args.planes = ["y", "u", "v"]

[filter.blended]
type = "blend/masked"
inputs = ["input", "alone", "edges"]

[sink.masked_output]
type = "y4m/pipe"
inputs = ["masked"]
args.path = "{masked}"

[sink.blended_output]
type = "y4m/pipe"
inputs = ["blended"]
args.path = "{blended}"
"#
    )
}

/// The first frame `stage` writes when merged through a mask, and when blended by `blend/masked`.
fn masked_and_blended(stage: &str, alone_args: &str, bits: u32) -> ([Vec<u16>; 3], [Vec<u16>; 3]) {
    let folder = tempfile::tempdir().expect("a temp dir");
    let input = folder.path().join("input.y4m");
    let masked = folder.path().join("masked.y4m");
    let blended = folder.path().join("blended.y4m");
    let pipeline_path = folder.path().join("pipeline.toml");

    let planes = lined_planes(bits);
    let input_clip = clip(&planes, bits);
    fs::write(&input, input_clip).expect("writes the input");

    let input_path = utf8(&input);
    let masked_path = utf8(&masked);
    let blended_path = utf8(&blended);
    let text = pipeline_text(stage, alone_args, input_path, masked_path, blended_path);
    fs::write(&pipeline_path, &text).expect("writes the pipeline");

    let supplied = Supplied::new();
    let loaded =
        av_toolbox_pipeline::load(&pipeline_path, &DiskFiles, &supplied).expect("the file should load");
    let registry = av_toolbox_pipeline::registry().expect("the built-in stages register");
    let pipeline = Pipeline::build(&registry, &loaded.specs).expect("should build");
    let backend = Backend::open_default(Device::Default).expect("should open a device");
    let compiled = pipeline.compile(&backend).expect("should compile");
    let settings = Settings::default();
    let engine = compiled.engine(settings).expect("should build the engine");
    let report = engine.run().expect("should run");

    assert_eq!(report.frames, FRAMES * 2, "each sink counts every frame it wrote");

    let masked_stream = fs::read(&masked).expect("reads the masked output");
    let blended_stream = fs::read(&blended).expect("reads the blended output");
    let mut masked_frames = decode(&masked_stream, bits);
    let mut blended_frames = decode(&blended_stream, bits);

    assert_eq!(masked_frames.len(), FRAMES as usize);
    assert_eq!(blended_frames.len(), FRAMES as usize);

    let masked_first = masked_frames.swap_remove(0);
    let blended_first = blended_frames.swap_remove(0);

    (masked_first, blended_first)
}

fn changed(plane: &[u16], other: &[u16]) -> usize {
    plane
        .iter()
        .zip(other)
        .filter(|(left, right)| left != right)
        .count()
}

fn largest_difference(plane: &[u16], other: &[u16]) -> u16 {
    plane
        .iter()
        .zip(other)
        .map(|(left, right)| left.abs_diff(*right))
        .max()
        .unwrap_or(0)
}

/// Every stage and depth is measured before any is checked, so a failure reports them all.
#[test]
fn a_mask_merges_each_lineart_filter_as_blend_masked_does() {
    let mut failures = Vec::new();

    for bits in [8, 10] {
        let input = lined_planes(bits);

        for (stage, alone_args) in STAGES {
            let (masked, blended) = masked_and_blended(stage, alone_args, bits);

            for (plane, name) in PLANE_NAMES.iter().enumerate() {
                let largest = largest_difference(&masked[plane], &blended[plane]);
                let differing = changed(&masked[plane], &blended[plane]);
                let moved = changed(&masked[plane], &input[plane]);
                let samples = input[plane].len();
                let label = format!("{stage} {bits}-bit {name}");

                println!("{label}: largest {largest}, {differing} of {samples} differ, {moved} moved");

                if moved == 0 {
                    failures.push(format!("{label} never moved a sample"));
                }

                if largest > 1 {
                    failures.push(format!("{label} differs by {largest}"));
                }

                if differing > OFF_BY_ONE_LIMIT {
                    failures.push(format!("{label} differs at {differing} samples"));
                }
            }
        }
    }

    assert!(failures.is_empty(), "{failures:#?}");
}
