#![cfg(all(unix, feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]

use std::fs;
use std::path::Path;

use av_toolbox_gpu::{Backend, test_device};
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

/// Where a run of the pipeline writes each of its outputs.
struct Outputs<'a> {
    input: &'a str,
    masked: &'a str,
    blended: &'a str,
}

/// Every frame of the planes a run wrote.
struct Run {
    masked: Vec<[Vec<u16>; 3]>,
    blended: Vec<[Vec<u16>; 3]>,
}

/// A pipeline that writes `stage` merged through a Prewitt mask to `masked`, and `stage` alone
/// blended over the input by the same mask to `blended`. Both are written as `bits` deep codes.
fn pipeline_text(stage: &str, alone_args: &str, bits: u32, outputs: &Outputs) -> String {
    let Outputs {
        input,
        masked,
        blended,
    } = outputs;

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

[filter.masked_codes]
type = "resample/depth"
inputs = ["masked"]
args.depth = {bits}
args.dither = "none"

[sink.masked_output]
type = "y4m/pipe"
inputs = ["masked_codes"]
args.path = "{masked}"

[filter.blended_codes]
type = "resample/depth"
inputs = ["blended"]
args.depth = {bits}
args.dither = "none"

[sink.blended_output]
type = "y4m/pipe"
inputs = ["blended_codes"]
args.path = "{blended}"
"#
    )
}

/// Every frame `stage` writes when merged through a mask, and when blended by `blend/masked`.
fn run_stage(stage: &str, alone_args: &str, bits: u32) -> Run {
    let folder = tempfile::tempdir().expect("a temp dir");
    let input = folder.path().join("input.y4m");
    let masked = folder.path().join("masked.y4m");
    let blended = folder.path().join("blended.y4m");
    let pipeline_path = folder.path().join("pipeline.toml");

    let planes = lined_planes(bits);
    let input_clip = clip(&planes, bits);
    fs::write(&input, input_clip).expect("writes the input");

    let outputs = Outputs {
        input: utf8(&input),
        masked: utf8(&masked),
        blended: utf8(&blended),
    };
    let text = pipeline_text(stage, alone_args, bits, &outputs);
    fs::write(&pipeline_path, &text).expect("writes the pipeline");

    let supplied = Supplied::new();
    let loaded =
        av_toolbox_pipeline::load(&pipeline_path, &DiskFiles, &supplied).expect("the file should load");
    let specs = loaded.specs;

    let registry = av_toolbox_pipeline::registry().expect("the built-in stages register");
    let pipeline = Pipeline::build(&registry, &specs).expect("should build");
    let backend = Backend::open_default(test_device()).expect("should open a device");
    let compiled = pipeline.compile(&backend).expect("should compile");
    let settings = Settings::default();
    let engine = compiled.engine(settings).expect("should build the engine");
    let report = engine.run().expect("should run");

    assert_eq!(report.frames, FRAMES * 2, "each sink counts every frame it wrote");

    let masked_stream = fs::read(&masked).expect("reads the masked output");
    let blended_stream = fs::read(&blended).expect("reads the blended output");
    let run = Run {
        masked: decode(&masked_stream, bits),
        blended: decode(&blended_stream, bits),
    };

    assert_eq!(run.masked.len(), FRAMES as usize);
    assert_eq!(run.blended.len(), FRAMES as usize);

    run
}

/// How one plane of the masked output sits against the `blend/masked` output.
struct Tally {
    largest: u16,
    differing: usize,
}

fn tally(masked: &[u16], blended: &[u16]) -> Tally {
    let mut tally = Tally {
        largest: 0,
        differing: 0,
    };

    for (masked_sample, blended_sample) in masked.iter().zip(blended) {
        let difference = masked_sample.abs_diff(*blended_sample);
        if difference == 0 {
            continue;
        }

        tally.largest = tally.largest.max(difference);
        tally.differing += 1;
    }

    tally
}

fn changed(plane: &[u16], other: &[u16]) -> usize {
    plane
        .iter()
        .zip(other)
        .filter(|(left, right)| left != right)
        .count()
}

/// Every stage and depth is measured before any is checked, so a failure reports them all.
#[test]
fn a_mask_merges_each_lineart_filter_as_blend_masked_does() {
    let mut failures = Vec::new();

    for bits in [8, 10] {
        let input = lined_planes(bits);

        for (stage, alone_args) in STAGES {
            let run = run_stage(stage, alone_args, bits);

            for (plane, name) in PLANE_NAMES.iter().enumerate() {
                let label = format!("{stage} {bits}-bit {name}");
                let samples = input[plane].len();

                for frame_index in 0..FRAMES as usize {
                    let masked = &run.masked[frame_index][plane];
                    let blended = &run.blended[frame_index][plane];
                    let tally = tally(masked, blended);
                    let moved = changed(masked, &input[plane]);

                    println!(
                        "{label} frame {frame_index}: largest {}, {} of {samples} differ, {moved} moved",
                        tally.largest, tally.differing
                    );

                    if moved == 0 {
                        failures.push(format!("{label} frame {frame_index} never moved a sample"));
                    }

                    if tally.differing > 0 {
                        failures.push(format!(
                            "{label} frame {frame_index} has {} samples off, largest {}",
                            tally.differing, tally.largest
                        ));
                    }
                }
            }
        }
    }

    assert!(failures.is_empty(), "{failures:#?}");
}
