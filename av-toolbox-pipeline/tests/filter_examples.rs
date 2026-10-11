#![cfg(all(unix, feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]

use std::fs;
use std::path::{Path, PathBuf};

use av_toolbox_cubek::artcnn::{Model, Weights};
use av_toolbox_cubek::deblock::h264::thresholds;
use av_toolbox_cubek::deblock::pp7::Pp7Mode;
use av_toolbox_cubek::dpir::{PACK, Variant};
use av_toolbox_cubek::matrix::MatrixSamples;
use av_toolbox_cubek::testing::{
    FixtureLayout,
    H264Plane,
    Pp7Case,
    QedCase,
    QedChroma,
    reference_artcnn,
    reference_h264,
    reference_pp7,
    reference_qed,
};
use av_toolbox_gpu::{Backend, test_device};
use av_toolbox_models::Store;
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
    let lumas = vec![luma.to_vec(); FRAMES as usize];

    clip_with_lumas(&lumas)
}

/// A C420 y4m clip with one frame per entry of `lumas`, each with flat chroma.
fn clip_with_lumas(lumas: &[Vec<u8>]) -> Vec<u8> {
    let mut bytes = Vec::new();
    let ratio = y4m::Ratio::new(24, 1);
    let mut encoder = y4m::encode(SIZE, SIZE, ratio)
        .write_header(&mut bytes)
        .expect("writes the header");

    let chroma_samples = (SIZE / 2) * (SIZE / 2);
    let u_plane = vec![FLAT_U; chroma_samples];
    let v_plane = vec![FLAT_V; chroma_samples];
    for luma in lumas {
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
    let (frames, decoded) = run_example_counted(name, extra_flags, clip);

    assert_eq!(frames, FRAMES);

    decoded
}

/// Runs `name` like `run_example_on`, handing back the frame count the run reported with the decoded output.
fn run_example_counted(name: &str, extra_flags: &[(&str, &str)], clip: &[u8]) -> (u64, Decoded) {
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
    let backend = Backend::open_default(test_device()).expect("should open a device");
    let compiled = pipeline.compile(&backend).expect("should compile");
    let settings = Settings::default();
    let engine = compiled.engine(settings).expect("should build the engine");
    let report = engine.run().expect("should run");

    let written = fs::read(&output).expect("reads the output");

    (report.frames, decode(&written))
}

#[cfg(feature = "bestsource")]
#[test]
fn the_bestsource_example_decodes_the_fixture() {
    let fixture = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../av-toolbox-sources/tests/media/yuv420p.mkv"
    );
    let folder = tempfile::tempdir().expect("a temp dir");
    let media = folder.path().join("yuv420p.mkv");
    fs::copy(fixture, &media).expect("copies the fixture");

    let flags = [("input", utf8(&media))];
    let (frames, decoded) = run_example_counted("bestsource.toml", &flags, &[]);

    assert_eq!(frames, 8);
    assert_eq!(decoded.frames.len(), 8);
    for frame in &decoded.frames {
        assert_eq!(frame.luma.len(), SIZE * SIZE);
    }
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

/// A gradient picture between black bars 8 rows tall at the top and bottom.
fn letterboxed_luma() -> Vec<u8> {
    let gradient = gradient_luma();
    let bar_rows = 8;
    let mut luma = Vec::with_capacity(gradient.len());

    for (row, line) in gradient.chunks(SIZE).enumerate() {
        if row < bar_rows || row >= SIZE - bar_rows {
            luma.extend(vec![16u8; SIZE]);
            continue;
        }

        let bright = line.iter().map(|value| (*value).max(40));
        luma.extend(bright);
    }

    luma
}

#[test]
fn the_crop_example_removes_the_letterbox() {
    let luma = letterboxed_luma();
    let clip = clip_with_luma(&luma);

    let decoded = run_example_on("crop.toml", &[], &clip);

    let picture_rows = &luma[8 * SIZE..(SIZE - 8) * SIZE];
    for frame in &decoded.frames {
        assert_eq!(
            frame.luma, picture_rows,
            "only the picture between the bars is kept"
        );
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

/// How far an output luma code may sit from the rounded reference.
///
/// The stage measured 1 code out at f16 on an AMD R9700, so one code of headroom is kept.
const ARTCNN_LUMA_BOUND: f64 = 2.0;

/// The builtin R8F64 network's doubled `luma`, as rounded 8-bit codes.
fn reference_artcnn_luma(luma: &[u8]) -> Vec<f64> {
    let model = Model::R8F64;
    let bytes = av_toolbox_models::builtin("artcnn", model.file_name()).expect("R8F64 is built in");
    let weights = Weights::parse(bytes, model.features(), model.blocks()).expect("the builtin weights parse");
    let unit: Vec<f32> = luma.iter().map(|code| f32::from(*code) / 255.0).collect();
    let doubled = reference_artcnn(&unit, SIZE as u32, SIZE as u32, &weights);

    doubled
        .iter()
        .map(|value| (value * 255.0).round().clamp(0.0, 255.0))
        .collect()
}

#[test]
fn the_artcnn_example_doubles_the_clip() {
    let decoded = run_example("artcnn.toml", &[]);
    let expected = reference_artcnn_luma(&gradient_luma());
    let doubled_chroma = SIZE * SIZE;

    assert!(matches!(decoded.colorspace, y4m::Colorspace::C420));
    assert_eq!(decoded.frames.len(), FRAMES as usize);
    for (index, frame) in decoded.frames.iter().enumerate() {
        assert_eq!(frame.luma.len(), SIZE * 2 * SIZE * 2);
        assert_eq!(frame.u_plane.len(), doubled_chroma);
        assert_eq!(frame.v_plane.len(), doubled_chroma);
        assert!(frame.u_plane.iter().all(|sample| *sample == FLAT_U));
        assert!(frame.v_plane.iter().all(|sample| *sample == FLAT_V));

        // The clip repeats one frame, so every frame is compared against the same reference.
        let largest = frame
            .luma
            .iter()
            .zip(&expected)
            .map(|(found, reference)| (f64::from(*found) - reference).abs())
            .fold(0.0, f64::max);
        println!("frame {index}: largest luma difference from the reference: {largest}");
        assert!(
            largest <= ARTCNN_LUMA_BOUND,
            "frame {index}: luma is {largest} codes from the reference"
        );
    }
}

/// The field and the line of `staircase_luma`.
const STAIRCASE_FIELD: u8 = 40;

const STAIRCASE_LINE: u8 = 220;

/// The row the staircase's line starts on, and how many columns each of its runs lasts.
const STAIRCASE_TOP: usize = 16;

const STAIRCASE_RUN: usize = 6;

/// The luma of a dark field with a light one-sample line falling one row every `STAIRCASE_RUN` columns,
/// drawn as a staircase of flat runs.
fn staircase_luma() -> Vec<u8> {
    (0..SIZE * SIZE)
        .map(|index| {
            let column = index % SIZE;
            let row = index / SIZE;
            let line_row = STAIRCASE_TOP + column / STAIRCASE_RUN;

            if row == line_row {
                STAIRCASE_LINE
            } else {
                STAIRCASE_FIELD
            }
        })
        .collect()
}

/// The largest difference between two vertically adjacent samples in `columns` of `plane`.
fn largest_vertical_step(plane: &[u8], columns: std::ops::Range<usize>) -> u8 {
    let rows: Vec<&[u8]> = plane.chunks(SIZE).collect();

    rows.windows(2)
        .flat_map(|pair| {
            columns
                .clone()
                .map(move |column| pair[0][column].abs_diff(pair[1][column]))
        })
        .max()
        .expect("the plane has rows")
}

#[test]
fn the_anti_aliasing_example_softens_a_staircase() {
    let luma = staircase_luma();
    let clip = clip_with_luma(&luma);
    let decoded = run_example_on("anti-aliasing.toml", &[], &clip);

    // The line's first and last runs keep their full step at the plane's sides, so only the middle counts.
    let middle = 8..(SIZE - 8);
    let input_step = largest_vertical_step(&luma, middle.clone());
    let far_rows = 0..(STAIRCASE_TOP - 6) * SIZE;

    assert!(matches!(decoded.colorspace, y4m::Colorspace::C420));
    assert_eq!(decoded.frames.len(), FRAMES as usize);

    for frame in &decoded.frames {
        let step = largest_vertical_step(&frame.luma, middle.clone());
        println!("largest step in the middle: {input_step} to {step}");
        assert!(step < input_step, "the step went from {input_step} to {step}");

        assert_eq!(frame.luma[far_rows.clone()], luma[far_rows.clone()]);
        assert!(frame.u_plane.iter().all(|sample| *sample == FLAT_U));
        assert!(frame.v_plane.iter().all(|sample| *sample == FLAT_V));
    }
}

/// How far noise moves a sample of `noisy_gradient_lumas` either way.
const NOISE_SPREAD: i32 = 12;

/// `gradient_luma` with deterministic noise of up to `NOISE_SPREAD` either way, drawn afresh for every frame.
fn noisy_gradient_lumas() -> Vec<Vec<u8>> {
    let clean = gradient_luma();
    let span = 2 * NOISE_SPREAD as u32 + 1;

    (0..FRAMES)
        .map(|frame| {
            clean
                .iter()
                .enumerate()
                .map(|(index, sample)| {
                    let seed = (index as u32).wrapping_mul(2_654_435_761) ^ (frame as u32).wrapping_mul(97);
                    let noise = (seed % span) as i32 - NOISE_SPREAD;
                    let noisy = i32::from(*sample) + noise;
                    noisy.clamp(0, 255) as u8
                })
                .collect()
        })
        .collect()
}

/// The mean absolute difference between two planes.
fn mean_error(found: &[u8], wanted: &[u8]) -> f64 {
    let total: u32 = found
        .iter()
        .zip(wanted)
        .map(|(sample, expected)| u32::from(sample.abs_diff(*expected)))
        .sum();

    f64::from(total) / found.len() as f64
}

/// Runs the denoise example `name` on a noisy ramp and checks every frame's luma moved toward the clean ramp.
fn assert_denoises(name: &str) {
    let lumas = noisy_gradient_lumas();
    let clip = clip_with_lumas(&lumas);
    let decoded = run_example_on(name, &[], &clip);
    let clean = gradient_luma();

    assert!(matches!(decoded.colorspace, y4m::Colorspace::C420));
    assert_eq!(decoded.frames.len(), FRAMES as usize);

    for (frame, noisy) in decoded.frames.iter().zip(&lumas) {
        let before = mean_error(noisy, &clean);
        let after = mean_error(&frame.luma, &clean);
        println!("{name}: mean error {before:.3} to {after:.3}");
        assert!(
            after < before,
            "{name} moved the mean error from {before:.3} to {after:.3}"
        );

        assert!(frame.u_plane.iter().all(|sample| sample.abs_diff(FLAT_U) <= 1));
        assert!(frame.v_plane.iter().all(|sample| sample.abs_diff(FLAT_V) <= 1));
    }
}

#[test]
fn the_denoise_nlmeans_example_cleans_a_noisy_ramp() {
    assert_denoises("denoise-nlmeans.toml");
}

#[test]
fn the_denoise_nlmeans_hq_example_cleans_a_noisy_ramp() {
    assert_denoises("denoise-nlmeans-hq.toml");
}

#[test]
fn the_denoise_nl4d_example_cleans_a_noisy_ramp() {
    assert_denoises("denoise-nl4d.toml");
}

/// How far an output code of the DPIR example may sit from the smooth input it deblocks.
///
/// Every plane measured at most 1 code out, so one code of headroom is kept.
const DPIR_CODE_BOUND: u8 = 2;

/// Whether the DPIR colour weights are installed, printing why the test does not run when they are not.
fn dpir_installed() -> bool {
    let store = Store::from_environment();
    let file = Variant::Colour.file_name();
    let found = store.find(PACK, file);

    match found {
        Ok(_) => true,
        Err(missing) => {
            println!("{file} is not installed, so the example is not run: {missing}");
            false
        },
    }
}

/// A gentle horizontal gradient whose every sample stays inside the RGB gamut with the flat chroma.
fn in_gamut_luma() -> Vec<u8> {
    let row: Vec<u8> = (0..SIZE).map(|column| 90 + column as u8).collect();
    row.repeat(SIZE)
}

fn largest_move(found: &[u8], expected: &[u8]) -> u8 {
    found
        .iter()
        .zip(expected)
        .map(|(found_code, expected_code)| found_code.abs_diff(*expected_code))
        .max()
        .expect("the plane has samples")
}

#[test]
fn the_dpir_example_keeps_a_smooth_clip() {
    if !dpir_installed() {
        return;
    }

    let expected_luma = in_gamut_luma();
    let clip = clip_with_luma(&expected_luma);

    let decoded = run_example_on("deblock-dpir.toml", &[], &clip);

    assert!(matches!(decoded.colorspace, y4m::Colorspace::C420));
    assert_eq!(decoded.frames.len(), FRAMES as usize);
    for (index, frame) in decoded.frames.iter().enumerate() {
        let flat_u = vec![FLAT_U; frame.u_plane.len()];
        let flat_v = vec![FLAT_V; frame.v_plane.len()];

        let luma_move = largest_move(&frame.luma, &expected_luma);
        let u_move = largest_move(&frame.u_plane, &flat_u);
        let v_move = largest_move(&frame.v_plane, &flat_v);
        println!("frame {index}: largest move luma {luma_move}, u {u_move}, v {v_move}");

        assert_eq!(frame.luma.len(), SIZE * SIZE);
        assert!(
            luma_move <= DPIR_CODE_BOUND,
            "frame {index}: luma moved by {luma_move}"
        );
        assert!(u_move <= DPIR_CODE_BOUND, "frame {index}: u moved by {u_move}");
        assert!(v_move <= DPIR_CODE_BOUND, "frame {index}: v moved by {v_move}");
    }
}

/// The `quant` that `deblock-h264.toml` sets.
const H264_EXAMPLE_QUANT: i32 = 25;

/// A luma of flat 8 by 8 blocks 3 to 9 codes apart, blocking that quant 25 smooths across every edge.
fn blocky_luma() -> Vec<u8> {
    (0..SIZE)
        .flat_map(|row| {
            (0..SIZE).map(move |column| {
                let block = row / 8 + 2 * (column / 8);

                110 + (block % 4) as u8 * 3
            })
        })
        .collect()
}

/// The planes of the test clip as the references take them, `luma` and then the flat chroma.
fn yuv_inputs(luma: &[u8]) -> [Vec<f64>; 3] {
    let chroma_samples = (SIZE / 2) * (SIZE / 2);
    let luma_values: Vec<f64> = luma.iter().map(|code| f64::from(*code)).collect();
    let flat_u_values = vec![f64::from(FLAT_U); chroma_samples];
    let flat_v_values = vec![f64::from(FLAT_V); chroma_samples];

    [luma_values, flat_u_values, flat_v_values]
}

/// Checks every frame has `expected_luma` in place of `input_luma`, with the flat chroma untouched.
fn assert_luma_deblocked(decoded: &Decoded, input_luma: &[u8], expected_luma: &[u8]) {
    assert!(matches!(decoded.colorspace, y4m::Colorspace::C420));
    assert_eq!(decoded.frames.len(), FRAMES as usize);

    for (index, frame) in decoded.frames.iter().enumerate() {
        let flat_u = vec![FLAT_U; frame.u_plane.len()];
        let flat_v = vec![FLAT_V; frame.v_plane.len()];

        assert_ne!(frame.luma, input_luma, "frame {index} came back unchanged");
        assert_eq!(
            frame.luma, expected_luma,
            "frame {index} differs from the reference"
        );
        assert_eq!(frame.u_plane, flat_u, "frame {index}: flat U changed");
        assert_eq!(frame.v_plane, flat_v, "frame {index}: flat V changed");
    }
}

#[test]
fn the_h264_example_deblocks_like_the_reference() {
    let luma = blocky_luma();
    let clip = clip_with_luma(&luma);

    let decoded = run_example_on("deblock-h264.toml", &[], &clip);

    let values: Vec<f64> = luma.iter().map(|code| f64::from(*code)).collect();
    let example_thresholds = thresholds(H264_EXAMPLE_QUANT, 0, 0, MatrixSamples::U8);
    let plane = H264Plane {
        values: &values,
        width: SIZE as u32,
        height: SIZE as u32,
        padded_width: SIZE as u32,
        padded_height: SIZE as u32,
    };
    let reference = reference_h264(plane, MatrixSamples::U8, &example_thresholds);
    let expected_luma: Vec<u8> = reference.iter().map(|value| *value as u8).collect();

    assert_luma_deblocked(&decoded, &luma, &expected_luma);
}

/// The case `deblock-qed.toml` runs, the stage's default arguments on the `SIZE` by `SIZE` 4:2:0 test clip.
fn qed_example_case() -> QedCase {
    QedCase {
        width: SIZE as u32,
        height: SIZE as u32,
        layout: FixtureLayout::Yuv { chroma_shift: (1, 1) },
        samples: MatrixSamples::U8,
        quant: [24, 26],
        aoffset: [1, 1],
        boffset: [2, 2],
        chroma: QedChroma::Qed,
    }
}

#[test]
fn the_qed_example_deblocks_like_the_reference() {
    let luma = blocky_luma();
    let clip = clip_with_luma(&luma);

    let decoded = run_example_on("deblock-qed.toml", &[], &clip);

    let case = qed_example_case();
    let inputs = yuv_inputs(&luma);
    let reference = reference_qed(&case, &inputs);
    let expected_luma: Vec<u8> = reference.output[0].iter().map(|value| *value as u8).collect();

    assert_luma_deblocked(&decoded, &luma, &expected_luma);
}

/// The case `deblock-pp7.toml` runs, qp 2 and hard on the `SIZE` by `SIZE` 4:2:0 test clip.
fn pp7_example_case() -> Pp7Case {
    Pp7Case {
        width: SIZE as u32,
        height: SIZE as u32,
        layout: FixtureLayout::Yuv { chroma_shift: (1, 1) },
        samples: MatrixSamples::U8,
        qp: 2.0,
        mode: Pp7Mode::Hard,
    }
}

#[test]
fn the_pp7_example_deblocks_like_the_reference() {
    let luma = blocky_luma();
    let clip = clip_with_luma(&luma);

    let decoded = run_example_on("deblock-pp7.toml", &[], &clip);

    let case = pp7_example_case();
    let inputs = yuv_inputs(&luma);
    let reference = reference_pp7(&case, &inputs);
    let expected_luma: Vec<u8> = reference[0].iter().map(|value| *value as u8).collect();

    assert_luma_deblocked(&decoded, &luma, &expected_luma);
}
