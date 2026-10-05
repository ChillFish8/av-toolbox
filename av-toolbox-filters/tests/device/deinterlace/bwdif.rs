use std::time::Duration;

use av_toolbox_cubek::aa::{sangnom_aaf, sangnom_max};
use av_toolbox_cubek::deinterlace::BwdifField;
use av_toolbox_cubek::testing::{
    BwdifReferencePlanes,
    Eedi2Parameters,
    Eedi3Clips,
    Eedi3Parameters,
    reference_bwdif,
    reference_eedi2_field,
    reference_eedi3_field,
    reference_sangnom_field,
};
use av_toolbox_gpu::{
    ChromaLocation,
    FieldOrder,
    FrameFormat,
    FrameLayout,
    FrameMeta,
    FrameRate,
    SampleFormat,
    Subsampling,
};
use serde_json::json;

use crate::stream::{Input, Output, Stream};
use crate::support::{
    assert_within_one,
    backend,
    built_filter,
    element_of,
    frame,
    from_bytes,
    peak_of,
    ten_bit,
    to_bytes,
};

const STAGE: &str = "deinterlace/bwdif";

fn interlaced(width: u32, height: u32, sample: SampleFormat) -> FrameFormat {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };

    FrameFormat {
        frame_rate: FrameRate::new(25, 1),
        field_order: Some(FieldOrder::TopFirst),
        ..frame(width, height, layout, sample, ChromaLocation::Left)
    }
}

/// The format `deinterlace/bwdif` emits for `input`, at twice its rate when `double` is set.
fn deinterlaced(input: FrameFormat, double: bool) -> FrameFormat {
    let frame_rate = match (double, input.frame_rate) {
        (true, Some(rate)) => FrameRate::new(rate.numerator() * 2, rate.denominator()),
        (_, unchanged) => unchanged,
    };

    FrameFormat {
        frame_rate,
        field_order: Some(FieldOrder::Progressive),
        ..input
    }
}

fn run(
    args: serde_json::Value,
    input: FrameFormat,
    output: FrameFormat,
    chunks: Vec<Vec<Input>>,
) -> Vec<Output> {
    let backend = backend();
    let filter = built_filter(&backend, STAGE, input, output, args);
    let stream = Stream {
        filter,
        input,
        output,
        chunks,
    };

    backend.build(stream)
}

/// A deterministic hash of a sample's position, for noise.
fn scramble(column: u32, row: u32, frame: u32) -> u32 {
    let mut state =
        column.wrapping_mul(0x9e37_79b1) ^ row.wrapping_mul(0x85eb_ca77) ^ frame.wrapping_mul(0xc2b2_ae3d);
    state ^= state >> 15;
    state = state.wrapping_mul(0x2c1b_3c6d);
    state ^= state >> 12;

    state
}

/// The code values of a moving plane at one bit depth.
#[derive(Debug, Clone, Copy)]
struct Levels {
    /// The background.
    low: u32,
    /// The bar, on odd rows only.
    high: u32,
    /// The noise added to both lies below this.
    noise: u32,
}

const EIGHT_BIT_LEVELS: Levels = Levels {
    low: 40,
    high: 220,
    noise: 17,
};

/// Levels whose bar reaches the 10-bit peak of 1023.
const TEN_BIT_LEVELS: Levels = Levels {
    low: 160,
    high: 960,
    noise: 64,
};

/// A plane of combed noise with a bar that moves 2 samples right each frame, wrapping every 32 columns.
fn moving_plane(width: u32, height: u32, frame: u32, levels: Levels) -> Vec<u16> {
    let shift = (frame * 2) % 32;
    let mut plane = Vec::with_capacity((width * height) as usize);
    for row in 0..height {
        for column in 0..width {
            let inside = (column + 32 - shift) % 32 < 6;
            let base = if inside && row % 2 == 1 {
                levels.high
            } else {
                levels.low
            };
            let noise = scramble(column, row, frame) % levels.noise;
            plane.push((base + noise) as u16);
        }
    }

    plane
}

/// Frame `index` of a moving 4:2:0 clip of `format`, at 25 frames per second.
fn moving_frame(format: &FrameFormat, index: u64) -> Input {
    let frame = index as u32;
    let levels = match format.sample {
        SampleFormat::U8 => EIGHT_BIT_LEVELS,
        _ => TEN_BIT_LEVELS,
    };
    let element = element_of(format.sample);

    let luma_values = moving_plane(format.width, format.height, frame, levels);
    let chroma_values = moving_plane(format.width / 2, format.height / 2, frame + 7, levels);
    let luma = to_bytes(&luma_values, element);
    let chroma = to_bytes(&chroma_values, element);
    let timestamp = Duration::from_millis(40 * index);
    let meta = FrameMeta {
        timestamp: Some(timestamp),
        ..FrameMeta::at(index)
    };

    Input {
        meta,
        planes: vec![luma, chroma.clone(), chroma],
    }
}

fn moving_clip(format: &FrameFormat, indices: std::ops::Range<u64>) -> Vec<Input> {
    indices.map(|index| moving_frame(format, index)).collect()
}

fn widened(bytes: &[u8]) -> Vec<f32> {
    bytes.iter().map(|byte| f32::from(*byte)).collect()
}

/// Reference values rounded and clamped to 8 bits, as the stage narrows them.
fn narrowed(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .map(|value| value.round().clamp(0.0, 255.0) as u8)
        .collect()
}

/// The 8-bit luma bwdif's reference gives for `field` of `current`, rounded and clamped as the stage narrows.
fn reference_luma(
    format: &FrameFormat,
    previous: &Input,
    current: &Input,
    next: &Input,
    field: BwdifField,
) -> Vec<u8> {
    let previous_values = widened(&previous.planes[0]);
    let current_values = widened(&current.planes[0]);
    let next_values = widened(&next.planes[0]);
    let planes = BwdifReferencePlanes {
        previous: &previous_values,
        current: &current_values,
        next: &next_values,
        spatial: None,
    };

    let values = reference_bwdif(&planes, format.width, format.height, field);

    values
        .iter()
        .map(|value| value.round().clamp(0.0, 255.0) as u8)
        .collect()
}

#[test]
fn a_static_clip_comes_out_unchanged_at_double_rate() {
    let input = interlaced(16, 8, SampleFormat::U8);
    let output = deinterlaced(input, true);
    let still = moving_frame(&input, 0);
    let clip: Vec<Input> = (0..3)
        .map(|index| Input {
            meta: FrameMeta::at(index),
            ..still.clone()
        })
        .collect();

    let outputs = run(json!({}), input, output, vec![clip]);

    assert_eq!(outputs.len(), 6);
    for (index, emitted) in outputs.iter().enumerate() {
        assert_eq!(emitted.meta.index, index as u64);
        assert_eq!(emitted.planes, still.planes, "output {index}");
    }
}

#[test]
fn double_rate_numbers_fields_and_steps_timestamps_by_half_a_frame() {
    let input = interlaced(16, 8, SampleFormat::U8);
    let output = deinterlaced(input, true);
    let clip = moving_clip(&input, 0..4);

    let outputs = run(json!({}), input, output, vec![clip]);

    let indices: Vec<u64> = outputs.iter().map(|emitted| emitted.meta.index).collect();
    let timestamps: Vec<Option<Duration>> = outputs.iter().map(|emitted| emitted.meta.timestamp).collect();
    let expected_timestamps: Vec<Option<Duration>> = (0..8)
        .map(|field| Some(Duration::from_millis(20 * field)))
        .collect();

    assert_eq!(indices, (0..8).collect::<Vec<u64>>());
    assert_eq!(timestamps, expected_timestamps);
}

#[test]
fn double_rate_without_a_frame_rate_leaves_the_second_field_without_a_timestamp() {
    let input = FrameFormat {
        frame_rate: None,
        ..interlaced(16, 8, SampleFormat::U8)
    };
    let output = deinterlaced(input, true);
    let clip = moving_clip(&input, 0..2);

    let outputs = run(json!({}), input, output, vec![clip]);

    assert_eq!(outputs[0].meta.timestamp, Some(Duration::ZERO));
    assert_eq!(outputs[1].meta.timestamp, None);
    assert_eq!(outputs[2].meta.timestamp, Some(Duration::from_millis(40)));
    assert_eq!(outputs[3].meta.timestamp, None);
}

#[test]
fn single_rate_keeps_every_frame_and_its_metadata() {
    let input = interlaced(16, 8, SampleFormat::U8);
    let output = deinterlaced(input, false);
    let mut clip = moving_clip(&input, 10..14);
    for input in &mut clip {
        input.meta.chunk = 3;
    }

    let outputs = run(json!({ "rate": "single" }), input, output, vec![clip.clone()]);

    let found: Vec<&FrameMeta> = outputs.iter().map(|emitted| &emitted.meta).collect();
    let expected: Vec<&FrameMeta> = clip.iter().map(|input| &input.meta).collect();
    assert_eq!(found, expected);
}

#[test]
fn every_field_matches_the_reference_on_luma() {
    let input = interlaced(32, 16, SampleFormat::U8);
    let output = deinterlaced(input, true);
    let clip = moving_clip(&input, 0..3);

    let outputs = run(json!({}), input, output, vec![clip.clone()]);

    let first = BwdifField {
        rebuilt: 1,
        first: true,
    };
    let second = BwdifField {
        rebuilt: 0,
        first: false,
    };
    let windows = [(0, 0, 1), (0, 1, 2), (1, 2, 2)];
    for (frame, (previous, current, next)) in windows.into_iter().enumerate() {
        for (field_index, field) in [first, second].into_iter().enumerate() {
            let expected = reference_luma(&input, &clip[previous], &clip[current], &clip[next], field);
            let found = &outputs[frame * 2 + field_index].planes[0];

            let label = format!("frame {frame}, field {field_index}");
            assert_within_one(found, &expected, &label);
        }
    }
}

/// Asserts the first output's luma keeps the odd rows of the first input's luma.
fn assert_odd_rows_kept_first(format: &FrameFormat, outputs: &[Output], clip: &[Input]) {
    let width = format.width as usize;
    let kept_rows = outputs[0].planes[0]
        .chunks(width)
        .zip(clip[0].planes[0].chunks(width));
    for (row, (found, original)) in kept_rows.enumerate() {
        if row % 2 == 1 {
            assert_eq!(found, original, "row {row}");
        }
    }
}

#[test]
fn bottom_first_keeps_the_odd_rows_first() {
    let input = FrameFormat {
        field_order: Some(FieldOrder::BottomFirst),
        ..interlaced(32, 16, SampleFormat::U8)
    };
    let output = deinterlaced(input, true);
    let clip = moving_clip(&input, 0..2);

    let outputs = run(json!({}), input, output, vec![clip.clone()]);

    assert_odd_rows_kept_first(&input, &outputs, &clip);
}

#[test]
fn an_order_of_bottom_on_a_top_first_stream_keeps_the_odd_rows_first() {
    let input = interlaced(32, 16, SampleFormat::U8);
    let output = deinterlaced(input, true);
    let clip = moving_clip(&input, 0..2);

    let outputs = run(json!({ "order": "bottom" }), input, output, vec![clip.clone()]);

    assert_odd_rows_kept_first(&input, &outputs, &clip);
}

#[test]
fn each_chunk_is_deinterlaced_on_its_own() {
    let input = interlaced(16, 8, SampleFormat::U8);
    let output = deinterlaced(input, true);
    let opening = moving_clip(&input, 0..3);
    let closing = moving_clip(&input, 3..5);

    let together = run(json!({}), input, output, vec![opening, closing.clone()]);
    let alone = run(json!({}), input, output, vec![closing]);

    assert_eq!(together.len(), 10);
    assert_eq!(&together[6..], &alone[..]);
}

#[test]
fn a_one_frame_chunk_uses_its_frame_for_every_neighbour() {
    let input = interlaced(16, 8, SampleFormat::U8);
    let output = deinterlaced(input, true);
    let only = moving_frame(&input, 0);

    let outputs = run(json!({}), input, output, vec![vec![only.clone()]]);

    assert_eq!(outputs.len(), 2);
    let first = BwdifField {
        rebuilt: 1,
        first: true,
    };
    let expected = reference_luma(&input, &only, &only, &only, first);
    assert_within_one(&outputs[0].planes[0], &expected, "the first field");
}

#[test]
fn an_unselected_plane_is_the_input() {
    let input = interlaced(16, 8, SampleFormat::U8);
    let output = deinterlaced(input, false);
    let clip = moving_clip(&input, 0..3);

    let outputs = run(
        json!({ "rate": "single", "planes": ["y"] }),
        input,
        output,
        vec![clip.clone()],
    );

    for (emitted, original) in outputs.iter().zip(&clip) {
        assert_eq!(emitted.planes[1], original.planes[1]);
        assert_eq!(emitted.planes[2], original.planes[2]);
    }
}

#[test]
fn a_static_clip_comes_out_unchanged_with_every_spatial_kind() {
    let input = interlaced(32, 32, SampleFormat::U8);
    let output = deinterlaced(input, true);
    let still = moving_frame(&input, 0);
    let kinds = ["eedi2", "eedi3", "sangnom"];

    for kind in kinds {
        let clip: Vec<Input> = (0..2)
            .map(|index| Input {
                meta: FrameMeta::at(index),
                ..still.clone()
            })
            .collect();

        let outputs = run(json!({ "spatial": { "kind": kind } }), input, output, vec![clip]);

        assert_eq!(outputs.len(), 4, "{kind}");

        for emitted in &outputs {
            assert_eq!(emitted.planes, still.planes, "{kind}");
        }
    }
}

#[test]
fn sangnom_prediction_matches_the_reference_on_luma() {
    let input = interlaced(32, 32, SampleFormat::U8);
    let output = deinterlaced(input, false);
    let clip = moving_clip(&input, 0..3);

    let args = json!({ "rate": "single", "planes": ["y"], "spatial": { "kind": "sangnom" } });
    let outputs = run(args, input, output, vec![clip.clone()]);

    let field = BwdifField {
        rebuilt: 1,
        first: true,
    };
    let current = widened(&clip[1].planes[0]);
    let kept = 1 - field.rebuilt;
    let aaf = sangnom_aaf(48, 8);
    let max = sangnom_max(8);
    let spatial = reference_sangnom_field(&current, input.width, input.height, kept, aaf, max);
    let previous = widened(&clip[0].planes[0]);
    let next = widened(&clip[2].planes[0]);
    let planes = BwdifReferencePlanes {
        previous: &previous,
        current: &current,
        next: &next,
        spatial: Some(&spatial),
    };

    let values = reference_bwdif(&planes, input.width, input.height, field);
    let expected: Vec<u8> = values
        .iter()
        .map(|value| value.round().clamp(0.0, 255.0) as u8)
        .collect();

    assert_within_one(&outputs[1].planes[0], &expected, "frame 1 luma");
}

#[test]
fn sangnom_prediction_matches_the_reference_for_both_fields_at_double_rate() {
    let input = interlaced(32, 32, SampleFormat::U8);
    let output = deinterlaced(input, true);
    let clip = moving_clip(&input, 0..3);

    let args = json!({ "planes": ["y"], "spatial": { "kind": "sangnom" } });
    let outputs = run(args, input, output, vec![clip.clone()]);

    let first = BwdifField {
        rebuilt: 1,
        first: true,
    };
    let second = BwdifField {
        rebuilt: 0,
        first: false,
    };
    let current = widened(&clip[1].planes[0]);
    let previous = widened(&clip[0].planes[0]);
    let next = widened(&clip[2].planes[0]);
    let aaf = sangnom_aaf(48, 8);
    let max = sangnom_max(8);
    for (field_index, field) in [first, second].into_iter().enumerate() {
        let kept = 1 - field.rebuilt;
        let spatial = reference_sangnom_field(&current, input.width, input.height, kept, aaf, max);
        let planes = BwdifReferencePlanes {
            previous: &previous,
            current: &current,
            next: &next,
            spatial: Some(&spatial),
        };

        let values = reference_bwdif(&planes, input.width, input.height, field);
        let expected: Vec<u8> = values
            .iter()
            .map(|value| value.round().clamp(0.0, 255.0) as u8)
            .collect();

        let label = format!("frame 1, field {field_index}");
        assert_within_one(&outputs[2 + field_index].planes[0], &expected, &label);
    }
}

/// Asserts each of `found` lies within one code value of the same sample of `expected`.
fn assert_codes_within_one(found: &[u16], expected: &[u16], label: &str) {
    assert_eq!(found.len(), expected.len(), "{label}");

    for (index, (got, want)) in found.iter().zip(expected).enumerate() {
        assert!(
            got.abs_diff(*want) <= 1,
            "{label}: sample {index} was {got}, expected {want}"
        );
    }
}

#[test]
fn every_field_matches_the_reference_on_ten_bit_luma() {
    let input = interlaced(32, 16, ten_bit());
    let output = deinterlaced(input, true);
    let clip = moving_clip(&input, 0..3);
    let element = element_of(input.sample);
    let peak = peak_of(input.sample);
    let luma: Vec<Vec<u16>> = clip
        .iter()
        .map(|frame| from_bytes(&frame.planes[0], element))
        .collect();

    let brightest = luma.iter().flatten().max();
    assert_eq!(
        brightest,
        Some(&(peak as u16)),
        "the clip reaches the 10-bit peak"
    );

    let outputs = run(json!({}), input, output, vec![clip.clone()]);

    let first = BwdifField {
        rebuilt: 1,
        first: true,
    };
    let second = BwdifField {
        rebuilt: 0,
        first: false,
    };
    let windows = [(0, 0, 1), (0, 1, 2), (1, 2, 2)];
    for (frame, (previous, current, next)) in windows.into_iter().enumerate() {
        let previous_values: Vec<f32> = luma[previous].iter().map(|code| f32::from(*code)).collect();
        let current_values: Vec<f32> = luma[current].iter().map(|code| f32::from(*code)).collect();
        let next_values: Vec<f32> = luma[next].iter().map(|code| f32::from(*code)).collect();
        let planes = BwdifReferencePlanes {
            previous: &previous_values,
            current: &current_values,
            next: &next_values,
            spatial: None,
        };

        for (field_index, field) in [first, second].into_iter().enumerate() {
            let values = reference_bwdif(&planes, input.width, input.height, field);
            let expected: Vec<u16> = values
                .iter()
                .map(|value| value.round().clamp(0.0, peak as f32) as u16)
                .collect();
            let emitted = &outputs[frame * 2 + field_index].planes[0];
            let found = from_bytes(emitted, element);

            let label = format!("frame {frame}, field {field_index}");
            assert_codes_within_one(&found, &expected, &label);
        }
    }
}

/// An 8-bit plane of a shallow and a steep line, each two samples thick, over a faint texture.
///
/// The shallow line falls one row every 2 columns and the steep line moves one column every 2 rows.
/// Both move one sample each frame.
fn aliased_lines_plane(width: u32, height: u32, frame: u32) -> Vec<u8> {
    let mut plane = Vec::with_capacity((width * height) as usize);
    for row in 0..height {
        for column in 0..width {
            let shallow_row = height / 4 + frame + column / 2;
            let steep_column = width / 4 + frame + row / 2;
            let on_shallow = row >= shallow_row && row < shallow_row + 2;
            let on_steep = column >= steep_column && column < steep_column + 2;
            let texture = (column * 7 + row * 3) % 11;
            let value = match (on_shallow, on_steep) {
                (true, _) => 220,
                (false, true) => 180,
                (false, false) => 40 + texture,
            };
            plane.push(value as u8);
        }
    }

    plane
}

/// Frame `index` of an 8-bit 4:2:0 clip of `format` with moving aliased lines on every plane.
fn aliased_lines_frame(format: &FrameFormat, index: u64) -> Input {
    let frame = index as u32;
    let luma = aliased_lines_plane(format.width, format.height, frame);
    let chroma = aliased_lines_plane(format.width / 2, format.height / 2, frame);

    Input {
        meta: FrameMeta::at(index),
        planes: vec![luma, chroma.clone(), chroma],
    }
}

/// The largest difference between two samples at the same place in `first` and `second`.
fn largest_difference(first: &[u8], second: &[u8]) -> u8 {
    first
        .iter()
        .zip(second)
        .map(|(left, right)| left.abs_diff(*right))
        .max()
        .unwrap_or(0)
}

/// EEDI2's subsampling shift `ssw` for plane `plane_index` of a 4:2:0 clip, whose chroma is half as wide.
fn eedi2_shift(plane_index: usize) -> u32 {
    if plane_index == 0 { 0 } else { 1 }
}

#[test]
fn eedi2_prediction_matches_the_reference_on_every_plane_at_double_rate() {
    let input = interlaced(48, 32, SampleFormat::U8);
    let output = deinterlaced(input, true);
    let clip: Vec<Input> = (0..3).map(|index| aliased_lines_frame(&input, index)).collect();

    let args = json!({ "spatial": { "kind": "eedi2" } });
    let outputs = run(args, input, output, vec![clip.clone()]);

    let first = BwdifField {
        rebuilt: 1,
        first: true,
    };
    let second = BwdifField {
        rebuilt: 0,
        first: false,
    };
    let channels = input.channels();
    for (plane_index, channel) in channels.iter().enumerate() {
        let (width, height) = input
            .channel_dimensions(*channel)
            .expect("the format carries this channel");
        let parameters = Eedi2Parameters {
            ssw: eedi2_shift(plane_index),
            ..Eedi2Parameters::default()
        };
        let previous = widened(&clip[0].planes[plane_index]);
        let current = widened(&clip[1].planes[plane_index]);
        let next = widened(&clip[2].planes[plane_index]);

        for (field_index, field) in [first, second].into_iter().enumerate() {
            let kept = 1 - field.rebuilt;
            let spatial = reference_eedi2_field(&current, width, height, kept, &parameters);
            let planes = BwdifReferencePlanes {
                previous: &previous,
                current: &current,
                next: &next,
                spatial: Some(&spatial),
            };
            let cubic_planes = BwdifReferencePlanes {
                spatial: None,
                ..planes
            };

            let values = reference_bwdif(&planes, width, height, field);
            let expected = narrowed(&values);
            let cubic_values = reference_bwdif(&cubic_planes, width, height, field);
            let cubic = narrowed(&cubic_values);

            let label = format!("plane {channel}, frame 1, field {field_index}");
            assert_ne!(expected, cubic, "{label}: the prediction changes nothing");

            if plane_index > 0 {
                let unshifted = Eedi2Parameters { ssw: 0, ..parameters };
                let misread_spatial = reference_eedi2_field(&current, width, height, kept, &unshifted);
                let misread_planes = BwdifReferencePlanes {
                    spatial: Some(&misread_spatial),
                    ..planes
                };
                let misread_values = reference_bwdif(&misread_planes, width, height, field);
                let misread = narrowed(&misread_values);

                let largest_gap = largest_difference(&expected, &misread);
                assert!(largest_gap > 1, "{label} reads the same without the shift");
            }

            let found = &outputs[2 + field_index].planes[plane_index];
            assert_within_one(found, &expected, &label);
        }
    }
}

/// Asserts `found` is within one of `expected` wherever `touched` is unset, giving how many samples it sets.
///
/// A touched sample is one where a close decision of EEDI3 can change its prediction.
fn assert_within_one_where_untouched(found: &[u8], expected: &[u8], touched: &[bool], label: &str) -> usize {
    assert_eq!(found.len(), expected.len(), "{label}");
    assert_eq!(touched.len(), expected.len(), "{label}");

    let mut touched_count = 0;
    for (index, (got, want)) in found.iter().zip(expected).enumerate() {
        if touched[index] {
            touched_count += 1;
            continue;
        }

        assert!(
            got.abs_diff(*want) <= 1,
            "{label}: sample {index} was {got}, expected {want}"
        );
    }

    touched_count
}

#[test]
fn eedi3_prediction_matches_the_reference_for_both_fields_at_double_rate() {
    let input = interlaced(64, 32, SampleFormat::U8);
    let output = deinterlaced(input, true);
    let clip = moving_clip(&input, 0..3);

    let args = json!({ "planes": ["y"], "spatial": { "kind": "eedi3" } });
    let outputs = run(args, input, output, vec![clip.clone()]);

    let first = BwdifField {
        rebuilt: 1,
        first: true,
    };
    let second = BwdifField {
        rebuilt: 0,
        first: false,
    };
    let parameters = Eedi3Parameters::default();
    let clips = Eedi3Clips::default();
    let previous = widened(&clip[0].planes[0]);
    let current = widened(&clip[1].planes[0]);
    let next = widened(&clip[2].planes[0]);

    let mut compared = 0;
    let mut touched = 0;
    for (field_index, field) in [first, second].into_iter().enumerate() {
        let kept = 1 - field.rebuilt;
        let pass = reference_eedi3_field(&current, input.width, input.height, kept, &parameters, clips);
        let planes = BwdifReferencePlanes {
            previous: &previous,
            current: &current,
            next: &next,
            spatial: Some(&pass.output),
        };
        let cubic_planes = BwdifReferencePlanes {
            spatial: None,
            ..planes
        };

        let values = reference_bwdif(&planes, input.width, input.height, field);
        let expected = narrowed(&values);
        let cubic_values = reference_bwdif(&cubic_planes, input.width, input.height, field);
        let cubic = narrowed(&cubic_values);

        let label = format!("frame 1, field {field_index}");
        assert_ne!(expected, cubic, "{label}: the prediction changes nothing");

        let found = &outputs[2 + field_index].planes[0];
        compared += expected.len();
        touched += assert_within_one_where_untouched(found, &expected, &pass.touched, &label);
    }

    assert!(
        touched * 50 <= compared,
        "{touched} of {compared} samples are touched, more than 2 %"
    );
}
