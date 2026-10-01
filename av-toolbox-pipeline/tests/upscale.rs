#![cfg(all(unix, feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]

use std::fs;
use std::path::Path;

use av_toolbox_cubek::artcnn::{Model, Weights};
use av_toolbox_cubek::resize::{Kernel, taps};
use av_toolbox_cubek::testing::{reference_artcnn, reference_resize_wide};
use av_toolbox_gpu::{Backend, test_device};
use av_toolbox_pipeline::{DiskFiles, Pipeline, Supplied};
use av_toolbox_runtime::Settings;

const FRAMES: u64 = 3;

const FLAT_U: u16 = 90;

const FLAT_V: u16 = 180;

const PEAK_8: u16 = 255;

const PEAK_10: u16 = 1023;

/// Catrom, the kernel the stage resizes chroma with.
const CATROM: Kernel = Kernel::Bicubic { b: 0.0, c: 0.5 };

/// How far an output luma code may sit from the rounded reference at f16.
///
/// It measured 1 code on an AMD R9700, so one code of headroom is kept.
const F16_LUMA_BOUND: f64 = 2.0;

/// How far an output luma code may sit from the rounded reference at f32.
///
/// It measured 0 codes on an AMD R9700, so one code of headroom is kept.
const F32_LUMA_BOUND: f64 = 1.0;

/// The largest share of output luma codes that may differ from the reference at all.
///
/// The most measured on an AMD R9700 was 29 of 1120 codes, about 2.6%, at 10 bits and f16.
const DIFFERING_SHARE: f64 = 0.05;

/// A clip whose every frame holds the same planes, as code values.
struct Clip {
    width: usize,
    height: usize,
    colorspace: y4m::Colorspace,
    luma: Vec<u16>,
    u_plane: Vec<u16>,
    v_plane: Vec<u16>,
}

/// One decoded frame's planes, as code values.
struct DecodedFrame {
    luma: Vec<u16>,
    u_plane: Vec<u16>,
    v_plane: Vec<u16>,
}

/// A decoded y4m stream's header and the planes of each of its frames.
struct Decoded {
    width: usize,
    height: usize,
    colorspace: y4m::Colorspace,
    frames: Vec<DecodedFrame>,
}

/// A horizontal ramp over a gentler vertical one, with a sharp step up halfway across.
fn edged_gradient(width: usize, height: usize, peak: u16) -> Vec<u16> {
    let peak = f64::from(peak);
    let mut plane = Vec::with_capacity(width * height);

    for row in 0..height {
        for column in 0..width {
            let across = column as f64 / width as f64 * 0.4;
            let down = row as f64 / height as f64 * 0.1;
            let step = if column >= width / 2 { 0.4 } else { 0.0 };
            let value = (across + down + step) * peak + 0.05 * peak;
            plane.push(value.round() as u16);
        }
    }

    plane
}

/// A 10-bit ramp with a ripple, so Catrom lands between codes rather than on halves.
fn chroma_gradient(column: usize, row: usize) -> u16 {
    let ripple = (column * row) % 13;
    let value = 100 + column * 31 + row * 17 + ripple;

    value as u16
}

/// An 8-bit ramp with a ripple, so Catrom lands between codes rather than on halves.
fn small_chroma_gradient(column: usize, row: usize) -> u16 {
    let ripple = (column * row) % 5;
    let value = 40 + column * 9 + row * 7 + ripple;

    value as u16
}

/// A plane of `width` by `height` samples, all `value`.
fn flat(width: usize, height: usize, value: u16) -> Vec<u16> {
    vec![value; width * height]
}

/// An 8-bit 4:2:0 clip whose luma is an `edged_gradient`, with flat chroma.
fn clip_420(width: usize, height: usize) -> Clip {
    let chroma_width = width.div_ceil(2);
    let chroma_height = height.div_ceil(2);
    let luma = edged_gradient(width, height, PEAK_8);
    let u_plane = flat(chroma_width, chroma_height, FLAT_U);
    let v_plane = flat(chroma_width, chroma_height, FLAT_V);

    Clip {
        width,
        height,
        colorspace: y4m::Colorspace::C420,
        luma,
        u_plane,
        v_plane,
    }
}

fn samples_as_bytes(plane: &[u16], bytes_per_sample: usize) -> Vec<u8> {
    if bytes_per_sample == 1 {
        return plane.iter().map(|sample| *sample as u8).collect();
    }

    plane.iter().flat_map(|sample| sample.to_le_bytes()).collect()
}

fn bytes_as_samples(bytes: &[u8], bytes_per_sample: usize) -> Vec<u16> {
    if bytes_per_sample == 1 {
        return bytes.iter().map(|byte| u16::from(*byte)).collect();
    }

    let (pairs, _) = bytes.as_chunks::<2>();

    pairs.iter().map(|pair| u16::from_le_bytes(*pair)).collect()
}

/// `clip` as a y4m stream of `FRAMES` frames.
fn encode(clip: &Clip) -> Vec<u8> {
    let mut bytes = Vec::new();
    let ratio = y4m::Ratio::new(24, 1);
    let mut encoder = y4m::encode(clip.width, clip.height, ratio)
        .with_colorspace(clip.colorspace)
        .write_header(&mut bytes)
        .expect("writes the header");

    let bytes_per_sample = clip.colorspace.get_bytes_per_sample();
    let luma = samples_as_bytes(&clip.luma, bytes_per_sample);
    let u_plane = samples_as_bytes(&clip.u_plane, bytes_per_sample);
    let v_plane = samples_as_bytes(&clip.v_plane, bytes_per_sample);
    for _ in 0..FRAMES {
        let frame = y4m::Frame::new([&luma, &u_plane, &v_plane], None);
        encoder.write_frame(&frame).expect("writes a frame");
    }

    bytes
}

fn decode(stream: &[u8]) -> Decoded {
    let mut decoder = y4m::decode(stream).expect("reads the header");
    let width = decoder.get_width();
    let height = decoder.get_height();
    let colorspace = decoder.get_colorspace();
    let bytes_per_sample = colorspace.get_bytes_per_sample();

    let mut frames = Vec::new();
    while let Ok(frame) = decoder.read_frame() {
        let luma = bytes_as_samples(frame.get_y_plane(), bytes_per_sample);
        let u_plane = bytes_as_samples(frame.get_u_plane(), bytes_per_sample);
        let v_plane = bytes_as_samples(frame.get_v_plane(), bytes_per_sample);
        frames.push(DecodedFrame {
            luma,
            u_plane,
            v_plane,
        });
    }

    Decoded {
        width,
        height,
        colorspace,
        frames,
    }
}

fn utf8(path: &Path) -> &str {
    path.to_str().expect("a utf-8 path")
}

/// Runs `clip` through `upscale/artcnn` at `precision` and decodes what it wrote.
fn upscale(clip: &Clip, precision: &str) -> Decoded {
    let folder = tempfile::tempdir().expect("a temp dir");
    let input = folder.path().join("input.y4m");
    let output = folder.path().join("output.y4m");
    let pipeline_file = folder.path().join("upscale.toml");

    let stream = encode(clip);
    fs::write(&input, stream).expect("writes the input");

    let input_path = utf8(&input);
    let output_path = utf8(&output);
    let text = format!(
        r#"
[source.input]
type = "y4m/pipe"
args.path = "{input_path}"

[filter.upscaled]
type = "upscale/artcnn"
inputs = ["input"]
args.precision = "{precision}"

[sink.output]
type = "y4m/pipe"
inputs = ["upscaled"]
args.path = "{output_path}"
"#
    );
    fs::write(&pipeline_file, text).expect("writes the pipeline");

    let supplied = Supplied::new();
    let loaded =
        av_toolbox_pipeline::load(&pipeline_file, &DiskFiles, &supplied).expect("the pipeline loads");
    let registry = av_toolbox_pipeline::registry().expect("the built-in stages register");
    let pipeline = Pipeline::build(&registry, &loaded.specs).expect("should build");
    let backend = Backend::open_default(test_device()).expect("should open a device");
    let compiled = pipeline.compile(&backend).expect("should compile");
    let settings = Settings::default();
    let engine = compiled.engine(settings).expect("should build the engine");
    let report = engine.run().expect("should run");

    assert_eq!(report.frames, FRAMES);

    let written = fs::read(&output).expect("reads the output");
    decode(&written)
}

/// The builtin R8F64 network's doubled `luma`, as rounded codes out of `peak`.
fn reference_luma(luma: &[u16], width: usize, height: usize, peak: u16) -> Vec<f64> {
    let model = Model::R8F64;
    let bytes = av_toolbox_models::builtin("artcnn", model.file_name()).expect("R8F64 is built in");
    let weights = Weights::parse(bytes, model.features(), model.blocks()).expect("the builtin weights parse");
    let scale = f32::from(peak);
    let unit: Vec<f32> = luma.iter().map(|code| f32::from(*code) / scale).collect();
    let doubled = reference_artcnn(&unit, width as u32, height as u32, &weights);
    let peak = f64::from(peak);

    doubled
        .iter()
        .map(|value| (value * peak).round().clamp(0.0, peak))
        .collect()
}

/// The largest difference between `found` and `expected`, and how many samples differ at all.
fn differences(found: &[u16], expected: &[f64]) -> (f64, usize) {
    assert_eq!(found.len(), expected.len(), "the planes differ in size");

    let mut largest = 0.0f64;
    let mut differing = 0;
    for (sample, reference) in found.iter().zip(expected) {
        let difference = (f64::from(*sample) - reference).abs();
        largest = largest.max(difference);
        if difference > 0.0 {
            differing += 1;
        }
    }

    (largest, differing)
}

/// Checks every frame's luma against `expected` at `precision`, printing what it measured under `label`.
fn assert_luma_close(decoded: &Decoded, expected: &[f64], precision: &str, label: &str) {
    let bound = if precision == "f32" {
        F32_LUMA_BOUND
    } else {
        F16_LUMA_BOUND
    };
    let allowed = expected.len() as f64 * DIFFERING_SHARE;

    for (index, frame) in decoded.frames.iter().enumerate() {
        let (largest, differing) = differences(&frame.luma, expected);
        println!("{label} frame {index}: largest difference {largest}, {differing} codes differ");

        assert!(
            largest <= bound,
            "{label} frame {index}: luma is {largest} codes from the reference"
        );
        assert!(
            differing as f64 <= allowed,
            "{label} frame {index}: {differing} luma codes differ from the reference"
        );
    }
}

#[test]
fn luma_matches_the_reference_at_both_precisions() {
    let clip = clip_420(24, 18);
    let expected = reference_luma(&clip.luma, clip.width, clip.height, PEAK_8);

    for precision in ["f16", "f32"] {
        let decoded = upscale(&clip, precision);

        assert_eq!((decoded.width, decoded.height), (48, 36));
        assert_eq!(decoded.frames.len(), FRAMES as usize);
        // On an AMD R9700 f16 measured at most 1 code out, with 11 of 1728 codes differing, and f32
        // matched exactly.
        assert_luma_close(&decoded, &expected, precision, precision);
    }
}

#[test]
fn flat_chroma_stays_flat_and_doubles() {
    let clip = clip_420(24, 18);

    let decoded = upscale(&clip, "f16");

    assert!(matches!(decoded.colorspace, y4m::Colorspace::C420));

    for frame in &decoded.frames {
        assert_eq!(frame.u_plane.len(), 24 * 18);
        assert_eq!(frame.v_plane.len(), 24 * 18);
        assert!(frame.u_plane.iter().all(|sample| *sample == FLAT_U));
        assert!(frame.v_plane.iter().all(|sample| *sample == FLAT_V));
    }
}

#[test]
fn chroma_follows_catrom_at_its_siting() {
    let width = 20;
    let height = 14;
    let luma = edged_gradient(width, height, PEAK_10);
    let u_plane: Vec<u16> = (0..height)
        .flat_map(|row| (0..width).map(move |column| chroma_gradient(column, row)))
        .collect();
    let v_plane = flat(width, height, 512);
    let clip = Clip {
        width,
        height,
        colorspace: y4m::Colorspace::C444p10,
        luma,
        u_plane,
        v_plane,
    };

    // A 2x window reads output sample `i` at `0.5 * i - 0.25`, and 4:4:4 chroma is sited like luma.
    let across = taps(40, 0.5, -0.25, &CATROM);
    let down = taps(28, 0.5, -0.25, &CATROM);
    let expected_u = reference_resize_wide(&clip.u_plane, 20, 14, &across, &down, PEAK_10);
    let expected_luma = reference_luma(&clip.luma, width, height, PEAK_10);

    let decoded = upscale(&clip, "f16");

    assert!(matches!(decoded.colorspace, y4m::Colorspace::C444p10));
    assert_eq!((decoded.width, decoded.height), (40, 28));

    let expected_codes: Vec<f64> = expected_u.iter().map(|code| f64::from(*code)).collect();
    for (index, frame) in decoded.frames.iter().enumerate() {
        let (largest, differing) = differences(&frame.u_plane, &expected_codes);
        println!("u frame {index}: largest difference {largest}, {differing} codes differ");

        // On an AMD R9700 every one of the 1120 codes matched, so none may differ.
        assert_eq!(
            largest, 0.0,
            "u frame {index}: chroma is {largest} codes from Catrom"
        );
        assert!(frame.v_plane.iter().all(|sample| *sample == 512));
    }

    // On an AMD R9700 this measured at most 1 code out, with 29 of 1120 codes differing.
    assert_luma_close(&decoded, &expected_luma, "f16", "10-bit f16");
}

#[test]
fn grey_doubles_luma_only() {
    let width = 16;
    let height = 12;
    let luma = edged_gradient(width, height, PEAK_8);
    let clip = Clip {
        width,
        height,
        colorspace: y4m::Colorspace::Cmono,
        luma,
        u_plane: Vec::new(),
        v_plane: Vec::new(),
    };
    let expected = reference_luma(&clip.luma, width, height, PEAK_8);

    let decoded = upscale(&clip, "f16");

    assert!(matches!(decoded.colorspace, y4m::Colorspace::Cmono));
    assert_eq!((decoded.width, decoded.height), (32, 24));

    for frame in &decoded.frames {
        assert!(frame.u_plane.is_empty());
        assert!(frame.v_plane.is_empty());
    }

    // On an AMD R9700 every code matched.
    assert_luma_close(&decoded, &expected, "f16", "grey f16");
}

#[test]
fn an_odd_sized_420_clip_gets_the_chroma_its_doubled_size_needs() {
    let clip = clip_420(15, 9);

    let decoded = upscale(&clip, "f16");

    assert_eq!(clip.u_plane.len(), 8 * 5);
    assert_eq!((decoded.width, decoded.height), (30, 18));

    for frame in &decoded.frames {
        assert_eq!(frame.luma.len(), 30 * 18);
        assert_eq!(frame.u_plane.len(), 15 * 9);
        assert_eq!(frame.v_plane.len(), 15 * 9);
        assert!(frame.u_plane.iter().all(|sample| *sample == FLAT_U));
        assert!(frame.v_plane.iter().all(|sample| *sample == FLAT_V));
    }
}

#[test]
fn subsampled_chroma_follows_catrom_at_its_left_siting() {
    let chroma_width = 12;
    let chroma_height = 9;
    let base = clip_420(24, 18);
    let u_plane: Vec<u16> = (0..chroma_height)
        .flat_map(|row| (0..chroma_width).map(move |column| small_chroma_gradient(column, row)))
        .collect();
    let clip = Clip { u_plane, ..base };

    // Plain C420 is left sited. Across, chroma sits on luma sample 0, so output sample `i` reads
    // `0.5 * i - 0.125`. Down, it sits halfway between two luma rows, so it reads `0.5 * i - 0.25`.
    let across = taps(24, 0.5, -0.125, &CATROM);
    let down = taps(18, 0.5, -0.25, &CATROM);
    let expected_u = reference_resize_wide(&clip.u_plane, 12, 9, &across, &down, PEAK_8);
    let expected_codes: Vec<f64> = expected_u.iter().map(|code| f64::from(*code)).collect();

    let decoded = upscale(&clip, "f16");

    assert!(matches!(decoded.colorspace, y4m::Colorspace::C420));
    assert_eq!((decoded.width, decoded.height), (48, 36));

    for (index, frame) in decoded.frames.iter().enumerate() {
        let (largest, differing) = differences(&frame.u_plane, &expected_codes);
        println!("420 u frame {index}: largest difference {largest}, {differing} codes differ");

        // On an AMD R9700 every one of the 432 codes matched, so none may differ.
        assert_eq!(
            largest, 0.0,
            "u frame {index}: chroma is {largest} codes from Catrom"
        );
        assert!(frame.v_plane.iter().all(|sample| *sample == FLAT_V));
    }
}
