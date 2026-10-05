use av_toolbox_gpu::{FieldOrder, Subsampling};
use serde_json::json;

use super::{
    Clip,
    assert_matches,
    cubic,
    depth_of,
    expected,
    interlaced,
    peak_code,
    reference_defaults,
    run_clip,
};
use crate::support::{element_of, from_bytes};

/// The furthest a kept row's sample may move from the input, in 8-bit code values.
const KEPT_LARGEST_8_BIT: u32 = 48;

/// The furthest a kept row may move from the input on average, in 8-bit code values.
const KEPT_MEAN_8_BIT: f64 = 2.0;

/// The U and V planes made from `luma`, U at half its contrast around the middle code and V inverted.
fn luma_chroma(luma: &[i32], bits: u32) -> [Vec<i32>; 2] {
    let peak = (1 << bits) - 1;
    let middle = 1 << (bits - 1);

    let u_plane = luma.iter().map(|sample| (sample + middle + 1) / 2).collect();
    let v_plane = luma.iter().map(|sample| peak - sample).collect();

    [u_plane, v_plane]
}

/// `plane`, `width` samples across, with each horizontal pair of samples averaged on every row.
fn paired(plane: &[i32], width: usize) -> Vec<i32> {
    let mut halved = Vec::with_capacity(plane.len().div_ceil(2));
    for row in plane.chunks(width) {
        for pair in row.chunks(2) {
            let count = pair.len() as i32;
            let total: i32 = pair.iter().sum();
            halved.push((total + count / 2) / count);
        }
    }

    halved
}

/// `clip` at 4:4:4, with chroma made from its luma.
fn at_444(clip: &Clip) -> Clip {
    let frames = clip
        .frames
        .iter()
        .map(|planes| {
            let luma = planes[0].clone();
            let [u_plane, v_plane] = luma_chroma(&luma, clip.bits);
            [luma, u_plane, v_plane]
        })
        .collect();
    let name = format!("{} at 4:4:4", clip.name);

    Clip {
        name,
        subsampling: Subsampling::Yuv444,
        frames,
        ..clip.clone()
    }
}

/// `clip` at 4:2:2, with chroma made from its luma a horizontal pair at a time, on every row.
fn at_422(clip: &Clip) -> Clip {
    let full = at_444(clip);
    let frames = full
        .frames
        .iter()
        .map(|planes| {
            let u_plane = paired(&planes[1], clip.width);
            let v_plane = paired(&planes[2], clip.width);
            [planes[0].clone(), u_plane, v_plane]
        })
        .collect();
    let name = format!("{} at 4:2:2", clip.name);

    Clip {
        name,
        subsampling: Subsampling::Yuv422,
        frames,
        ..clip.clone()
    }
}

/// `clip`, a 10-bit clip, lifted to 16 bits.
fn at_16_bit(clip: &Clip) -> Clip {
    assert_eq!(clip.bits, 10, "{} is not a 10-bit clip", clip.name);

    let frames = clip
        .frames
        .iter()
        .map(|planes| {
            planes
                .each_ref()
                .map(|plane| plane.iter().map(|sample| sample << 6).collect())
        })
        .collect();
    let name = format!("{} at 16 bits", clip.name);

    Clip {
        name,
        bits: 16,
        frames,
        ..clip.clone()
    }
}

/// The top left `width` by `height` samples of `plane`, `plane_width` samples across.
fn cropped_plane(plane: &[i32], plane_width: usize, width: usize, height: usize) -> Vec<i32> {
    plane
        .chunks(plane_width)
        .take(height)
        .flat_map(|row| row[..width].iter().copied())
        .collect()
}

/// The top left `width` by `height` luma samples of `clip`, a 4:2:0 clip, with the chroma they cover.
fn cropped(clip: &Clip, width: usize, height: usize) -> Clip {
    assert_eq!(
        clip.subsampling,
        Subsampling::Yuv420,
        "{} is not 4:2:0",
        clip.name
    );

    let chroma_width = clip.width.div_ceil(2);
    let kept_width = width.div_ceil(2);
    let kept_height = height.div_ceil(2);
    let frames = clip
        .frames
        .iter()
        .map(|planes| {
            let luma = cropped_plane(&planes[0], clip.width, width, height);
            let u_plane = cropped_plane(&planes[1], chroma_width, kept_width, kept_height);
            let v_plane = cropped_plane(&planes[2], chroma_width, kept_width, kept_height);
            [luma, u_plane, v_plane]
        })
        .collect();
    let name = format!("{} cropped to {width}x{height}", clip.name);

    Clip {
        name,
        width,
        height,
        frames,
        ..clip.clone()
    }
}

/// Asserts `clip` run with `spatial: cubic` matches the reference.
fn assert_cubic_matches(clip: &Clip) {
    let run = run_clip(json!({ "spatial": { "kind": "cubic" } }), clip);

    let settings = reference_defaults();
    let wanted = expected(clip, &settings, &cubic);
    assert_matches(&run, &wanted, clip, &clip.name);
}

/// How far the rows of `parity` in `found` moved from `input`, both `width` samples across, as the mean
/// and the largest difference.
fn kept_row_drift(found: &[u16], input: &[i32], width: usize, parity: usize) -> (f64, u32) {
    let kept_rows = found
        .chunks(width)
        .zip(input.chunks(width))
        .skip(parity)
        .step_by(2);

    let mut total = 0u64;
    let mut count = 0u64;
    let mut largest = 0u32;
    for (found_row, input_row) in kept_rows {
        for (found_sample, input_sample) in found_row.iter().zip(input_row) {
            let difference = (i32::from(*found_sample) - input_sample).unsigned_abs();
            largest = largest.max(difference);
            total += u64::from(difference);
            count += 1;
        }
    }

    (total as f64 / count as f64, largest)
}

/// Runs `clip` with the `spatial` named and asserts every output has the input's size, every sample is in
/// range, and each field keeps its own rows of the input within tolerance.
///
/// Outputs are integer codes, so every sample is finite once it is in range.
fn assert_spatial_runs(clip: &Clip, spatial: &str) {
    let run = run_clip(json!({ "spatial": { "kind": spatial } }), clip);
    assert_eq!(
        run.outputs.len(),
        clip.field_count(),
        "{spatial}: one output per field"
    );

    let format = interlaced(clip, FieldOrder::TopFirst);
    let element = element_of(format.sample);
    let peak = peak_code(format.sample);
    let scale = 1u32 << (depth_of(format.sample) - 8);
    let channels = format.channels();

    let mut worst_mean = 0.0f64;
    let mut worst_largest = 0u32;
    for (field, output) in run.outputs.iter().enumerate() {
        assert_eq!(
            output.planes.len(),
            channels.len(),
            "{spatial}: field {field} plane count"
        );

        let kept_parity = field % 2;
        let source = &clip.frames[field / 2];
        for (plane, channel) in channels.iter().enumerate() {
            let (width, height) = format
                .channel_dimensions(*channel)
                .expect("the format carries the channel");
            let codes = from_bytes(&output.planes[plane], element);
            let label = format!("{spatial}: field {field} plane {plane}");
            assert_eq!(codes.len(), (width * height) as usize, "{label} has another size");

            let above = codes.iter().find(|code| **code > peak);
            assert!(above.is_none(), "{label} has a sample {above:?} over {peak}");

            let (mean, largest) = kept_row_drift(&codes, &source[plane], width as usize, kept_parity);
            worst_mean = worst_mean.max(mean);
            worst_largest = worst_largest.max(largest);
        }
    }

    println!("{spatial}: kept rows move {worst_mean:.3} codes on average at worst, {worst_largest} at most");
    let mean_allowed = KEPT_MEAN_8_BIT * f64::from(scale);
    let largest_allowed = KEPT_LARGEST_8_BIT * scale;
    assert!(
        worst_mean <= mean_allowed,
        "{spatial}: a plane's kept rows move {worst_mean:.3} codes on average, over {mean_allowed}"
    );
    assert!(
        worst_largest <= largest_allowed,
        "{spatial}: a kept sample moves {worst_largest} codes, over {largest_allowed}"
    );
}

#[test]
fn the_stage_matches_the_reference_at_4_2_2_10_bit() {
    let clip = Clip::named("field_noisy_10");
    let subsampled = at_422(&clip);

    assert_cubic_matches(&subsampled);
}

#[test]
fn the_stage_matches_the_reference_at_4_4_4_8_bit() {
    let clip = Clip::named("field_objects_8");
    let full = at_444(&clip);

    assert_cubic_matches(&full);
}

#[test]
fn the_stage_matches_the_reference_at_4_2_0_16_bit() {
    let clip = Clip::named("field_noisy_10");
    let lifted = at_16_bit(&clip);

    assert_cubic_matches(&lifted);
}

#[test]
fn a_field_of_exactly_one_block_matches_the_reference() {
    let clip = Clip::named("field_objects_8");
    let smallest = cropped(&clip, 16, 32);

    assert_cubic_matches(&smallest);
}

#[test]
fn an_odd_luma_width_matches_the_reference() {
    let clip = Clip::named("field_objects_8");
    let odd = cropped(&clip, 127, 64);

    assert_cubic_matches(&odd);
}

#[test]
fn spatial_eedi2_runs_and_keeps_each_fields_rows() {
    let clip = Clip::named("field_objects_8");

    assert_spatial_runs(&clip, "eedi2");
}

#[test]
fn spatial_sangnom_runs_and_keeps_each_fields_rows() {
    let clip = Clip::named("field_objects_8");

    assert_spatial_runs(&clip, "sangnom");
}
