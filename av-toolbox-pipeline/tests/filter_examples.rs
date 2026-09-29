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
    let luma = gradient_luma();

    clip_with_luma(&luma)
}

/// A C420 y4m clip of `FRAMES` frames whose luma is `luma`, with flat chroma.
fn clip_with_luma(luma: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::new();
    let ratio = y4m::Ratio::new(24, 1);
    let mut encoder = y4m::encode(SIZE, SIZE, ratio)
        .write_header(&mut bytes)
        .expect("writes the header");

    let chroma_samples = (SIZE / 2) * (SIZE / 2);
    let u_plane = vec![FLAT_U; chroma_samples];
    let v_plane = vec![FLAT_V; chroma_samples];
    for _ in 0..FRAMES {
        let frame = y4m::Frame::new([luma, &u_plane, &v_plane], None);
        encoder.write_frame(&frame).expect("writes a frame");
    }

    bytes
}

const LINE_FIELD: u8 = 220;

const LINE_LEFT: usize = 30;

/// The line's samples across. It is anti-aliased and off the sample grid, so it has no centre sample.
const LINE_PROFILE: [u8; 5] = [185, 60, 40, 80, 200];

/// The samples of the profile below this are the line's core.
const LINE_CORE_BELOW: u8 = 130;

/// How much lighter the samples of the line's core are on odd rows.
const LINE_COMB: u8 = 24;

/// Halfway between the field and the line's darkest sample, so a sample below it is part of the line.
const LINE_MIDPOINT: f64 = 130.0;

/// The most of its input row-to-row difference the comb may keep.
const COMB_KEPT: f64 = 0.25;

/// The columns a measurement of the line covers, with a sample of field either side.
fn line_columns() -> std::ops::Range<usize> {
    (LINE_LEFT - 1)..(LINE_LEFT + LINE_PROFILE.len() + 1)
}

/// The luma of a light field with a faded, anti-aliased vertical line about 3 samples wide. Odd rows lighten
/// the core of the line, so the line is combed.
fn combed_faded_line_luma() -> Vec<u8> {
    (0..SIZE)
        .flat_map(|row| {
            let mut samples = vec![LINE_FIELD; SIZE];
            for (offset, value) in LINE_PROFILE.iter().enumerate() {
                let is_core = *value < LINE_CORE_BELOW;
                let comb = if is_core && row % 2 == 1 { LINE_COMB } else { 0 };
                samples[LINE_LEFT + offset] = value + comb;
            }

            samples
        })
        .collect()
}

/// The smallest sample in the line's columns.
fn line_minimum(plane: &[u8]) -> u8 {
    let columns = line_columns();

    columns_of(plane, columns)
        .into_iter()
        .min()
        .expect("the line has samples")
}

/// How many samples of a row lie below `LINE_MIDPOINT`, counting the part of a step that crosses it.
fn row_width_below_midpoint(row: &[u8]) -> f64 {
    let mut width = 0.0;
    for pair in row.windows(2) {
        let first = f64::from(pair[0]);
        let second = f64::from(pair[1]);
        let lower = first.min(second);
        let higher = first.max(second);
        if higher < LINE_MIDPOINT {
            width += 1.0;
        } else if lower < LINE_MIDPOINT {
            width += (LINE_MIDPOINT - lower) / (higher - lower);
        }
    }

    width
}

/// The mean over every row of the line's width below `LINE_MIDPOINT`, with steps interpolated.
fn line_width(plane: &[u8]) -> f64 {
    let columns = line_columns();
    let total: f64 = plane
        .chunks(SIZE)
        .map(|row| row_width_below_midpoint(&row[columns.clone()]))
        .sum();

    total / SIZE as f64
}

/// The mean absolute difference between neighbouring rows, over the line's columns.
fn line_comb(plane: &[u8]) -> f64 {
    let rows: Vec<&[u8]> = plane.chunks(SIZE).collect();
    let columns = line_columns();
    let mut total = 0u32;
    for pair in rows.windows(2) {
        for column in columns.clone() {
            total += u32::from(pair[0][column].abs_diff(pair[1][column]));
        }
    }

    let pairs = (SIZE - 1) * columns.len();

    f64::from(total) / pairs as f64
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
    let clip = gradient_clip();

    run_example_on(name, extra_flags, &clip)
}

/// Writes `clip` as the input, runs `name` with `extra_flags`, and decodes the output.
fn run_example_on(name: &str, extra_flags: &[(&str, &str)], clip: &[u8]) -> Decoded {
    let folder = tempfile::tempdir().expect("a temp dir");
    let input = folder.path().join("input.y4m");
    let output = folder.path().join("output.y4m");

    fs::write(&input, clip).expect("writes the input");

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

/// The mean of each row of a plane `SIZE` samples wide.
fn row_means(plane: &[u8]) -> Vec<f64> {
    plane
        .chunks(SIZE)
        .map(|row| {
            let total: u32 = row.iter().map(|sample| u32::from(*sample)).sum();
            let total_as_f64 = f64::from(total);
            let row_width = SIZE as f64;
            total_as_f64 / row_width
        })
        .collect()
}

#[test]
fn the_deband_example_keeps_the_ramp_and_moves_its_grain() {
    let decoded = run_example("deband.toml", &[]);
    let expected_luma = gradient_luma();
    let expected_means = row_means(&expected_luma);

    assert!(matches!(decoded.colorspace, y4m::Colorspace::C420));
    assert_eq!(decoded.frames.len(), FRAMES as usize);
    for frame in &decoded.frames {
        let luma = &frame.luma;
        let means = row_means(luma);
        for (found, expected) in means.iter().zip(&expected_means) {
            assert!(
                (found - expected).abs() <= 1.0,
                "a row mean moved from {expected} to {found}"
            );
        }

        assert!(frame.u_plane.iter().all(|sample| *sample == FLAT_U));
        assert!(frame.v_plane.iter().all(|sample| *sample == FLAT_V));
    }

    assert_ne!(
        decoded.frames[0].luma, decoded.frames[1].luma,
        "the grain should move between frames"
    );
}

#[test]
fn the_warp_example_leaves_an_even_ramp_alone() {
    // An even ramp has the same edge strength everywhere, so the mask is flat and nothing moves.
    let decoded = run_example("warp.toml", &[]);
    let expected_luma = gradient_luma();

    assert!(matches!(decoded.colorspace, y4m::Colorspace::C420));
    assert_eq!(decoded.frames.len(), FRAMES as usize);
    for frame in &decoded.frames {
        assert_eq!(frame.luma, expected_luma);
        assert!(frame.u_plane.iter().all(|sample| *sample == FLAT_U));
        assert!(frame.v_plane.iter().all(|sample| *sample == FLAT_V));
    }
}

#[test]
fn the_dehalo_example_leaves_an_even_ramp_alone() {
    // An even ramp's edges are far below every mask threshold, so neither stage changes it.
    let decoded = run_example("dehalo.toml", &[]);
    let expected_luma = gradient_luma();

    assert!(matches!(decoded.colorspace, y4m::Colorspace::C420));
    assert_eq!(decoded.frames.len(), FRAMES as usize);
    for frame in &decoded.frames {
        assert_eq!(frame.luma, expected_luma);
        assert!(frame.u_plane.iter().all(|sample| *sample == FLAT_U));
        assert!(frame.v_plane.iter().all(|sample| *sample == FLAT_V));
    }
}

/// The luma of a clip whose left half is a gradient banded into 8 steps of 4 columns each, and whose
/// right half is a busy texture.
fn gradient_and_texture_luma() -> Vec<u8> {
    let half = SIZE / 2;

    (0..SIZE)
        .flat_map(|row| {
            (0..SIZE).map(move |column| {
                if column < half {
                    return (100 + column / 4) as u8;
                }

                ((row * 37 + column * 17 + row * column) % 251) as u8
            })
        })
        .collect()
}

/// The samples of `plane` in the `columns` of every row.
fn columns_of(plane: &[u8], columns: std::ops::Range<usize>) -> Vec<u8> {
    plane
        .chunks(SIZE)
        .flat_map(|row| row[columns.clone()].iter().copied())
        .collect()
}

/// The mean of each row's samples in the `columns`.
fn row_means_over(plane: &[u8], columns: std::ops::Range<usize>) -> Vec<f64> {
    let width = columns.len() as f64;

    plane
        .chunks(SIZE)
        .map(|row| {
            let total: u32 = row[columns.clone()].iter().map(|sample| u32::from(*sample)).sum();

            f64::from(total) / width
        })
        .collect()
}

/// The indices where `debanded` differs from `input`.
fn debanded_moves(input: &[u8], debanded: &[u8]) -> Vec<usize> {
    input
        .iter()
        .zip(debanded)
        .enumerate()
        .filter(|(_, (before, after))| before != after)
        .map(|(index, _)| index)
        .collect()
}

/// The least share of the gradient samples the deband example moves that the detail example must move
/// the same way.
const DEBANDED_SHARE: f64 = 0.75;

#[test]
fn the_detail_example_keeps_texture_and_debands_the_rest() {
    let half = SIZE / 2;
    let luma = gradient_and_texture_luma();
    let clip = clip_with_luma(&luma);
    let decoded = run_example_on("detail.toml", &[], &clip);
    let deband_only = run_example_on("deband.toml", &[], &clip);

    let input_texture = columns_of(&luma, half..SIZE);
    let input_gradient = columns_of(&luma, 0..half);
    let input_means = row_means_over(&luma, 0..half);

    assert!(matches!(decoded.colorspace, y4m::Colorspace::C420));
    assert_eq!(decoded.frames.len(), FRAMES as usize);
    assert_eq!(deband_only.frames.len(), FRAMES as usize);
    for (frame, debanded_frame) in decoded.frames.iter().zip(&deband_only.frames) {
        let texture = columns_of(&frame.luma, half..SIZE);
        assert_eq!(texture, input_texture, "the texture half changed");

        let gradient = columns_of(&frame.luma, 0..half);
        let changed = gradient
            .iter()
            .zip(&input_gradient)
            .filter(|(found, expected)| found != expected)
            .count();
        println!("gradient samples debanded: {changed} of {}", gradient.len());
        assert!(changed > 0, "the gradient half was not debanded");

        let debanded_gradient = columns_of(&debanded_frame.luma, 0..half);
        let moved = debanded_moves(&input_gradient, &debanded_gradient);
        let followed = moved
            .iter()
            .filter(|index| gradient[**index] == debanded_gradient[**index])
            .count();
        let debanded_share = followed as f64 / moved.len() as f64;
        println!(
            "gradient samples the deband moves: {}, taken from the deband: {followed}, a share of {debanded_share}",
            moved.len()
        );
        assert!(
            debanded_share >= DEBANDED_SHARE,
            "only {debanded_share} of the gradient's debanded samples came through"
        );

        let means = row_means_over(&frame.luma, 0..half);
        let largest = means
            .iter()
            .zip(&input_means)
            .map(|(found, expected)| (found - expected).abs())
            .fold(0.0, f64::max);
        println!("largest gradient row mean move: {largest}");
        assert!(largest <= 1.0, "a gradient row mean moved by {largest}");
    }
}

#[test]
fn the_lineart_cleanup_example_darkens_and_thins_a_line() {
    let luma = combed_faded_line_luma();
    let clip = clip_with_luma(&luma);
    let decoded = run_example_on("lineart-cleanup.toml", &[], &clip);

    let input_minimum = line_minimum(&luma);
    let input_width = line_width(&luma);
    let input_comb = line_comb(&luma);
    let far_columns = 0..(LINE_LEFT - 12);
    let far_input = columns_of(&luma, far_columns.clone());

    assert!(matches!(decoded.colorspace, y4m::Colorspace::C420));
    assert_eq!(decoded.frames.len(), FRAMES as usize);

    for frame in &decoded.frames {
        let minimum = line_minimum(&frame.luma);
        println!("line minimum: {input_minimum} to {minimum}");
        assert!(
            minimum < input_minimum,
            "the line's minimum went from {input_minimum} to {minimum}"
        );

        let width = line_width(&frame.luma);
        println!("line width below {LINE_MIDPOINT}: {input_width} to {width}");
        assert!(
            width < input_width,
            "the line's width went from {input_width} to {width}"
        );

        let comb = line_comb(&frame.luma);
        println!("line comb: {input_comb} to {comb}");
        assert!(
            comb <= input_comb * COMB_KEPT,
            "the comb went from {input_comb} to {comb}"
        );

        let far = columns_of(&frame.luma, far_columns.clone());
        let largest_far_move = far
            .iter()
            .zip(&far_input)
            .map(|(found, expected)| found.abs_diff(*expected))
            .max()
            .expect("the region has samples");
        println!("largest move far from the line: {largest_far_move}");
        assert!(largest_far_move <= 1, "the field moved by {largest_far_move}");

        assert!(frame.u_plane.iter().all(|sample| sample.abs_diff(FLAT_U) <= 1));
        assert!(frame.v_plane.iter().all(|sample| sample.abs_diff(FLAT_V) <= 1));
    }
}
