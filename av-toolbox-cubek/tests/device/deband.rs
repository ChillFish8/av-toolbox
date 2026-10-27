use av_toolbox_cubek::deband::{DebandParams, DebandPlanes, deband};
use av_toolbox_cubek::depth::{Element, Source};
use av_toolbox_cubek::testing::{reference_deband, reference_deband_codes};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use crate::support::{Sizing, backend, empty, plane, read_f32, to_bytes, upload_u8};

struct Deband {
    input: Vec<u16>,
    width: u32,
    height: u32,
    params: DebandParams,
    salt: u32,
}

/// `values` laid out as `source` samples.
fn source_bytes(values: &[u16], source: Source) -> Vec<u8> {
    match source {
        Source::U8 => to_bytes(values, Element::U8),
        Source::U16 => to_bytes(values, Element::U16),
        Source::F32 => values
            .iter()
            .flat_map(|value| f32::from(*value).to_le_bytes())
            .collect(),
    }
}

impl BuildTask for Deband {
    type Output = Vec<f32>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let bytes = source_bytes(&self.input, self.params.element);

        let input = upload_u8(client, &bytes);
        let output_bytes = self.input.len() * size_of::<f32>();
        let output = empty(client, output_bytes);
        let input_plane = plane(&input, self.width, self.height);
        let output_plane = plane(&output, self.width, self.height);
        let planes = DebandPlanes {
            input: input_plane,
            output: output_plane,
        };
        deband(client, &sizing, planes, &self.params, self.salt);

        read_f32(client, &output)
    }
}

fn params(element: Source, peak: f32) -> DebandParams {
    DebandParams {
        element,
        iterations: 2,
        radius: 16.0,
        threshold: 6.0,
        grain: 4.0,
        neutral: 0.0,
        peak,
    }
}

fn on_device(input: &[u16], width: u32, height: u32, params: DebandParams, salt: u32) -> Vec<f32> {
    backend().build(Deband {
        input: input.to_vec(),
        width,
        height,
        params,
        salt,
    })
}

/// A smooth diagonal ramp, wrapped at `peak`.
fn ramp(width: u32, height: u32, peak: u32) -> Vec<u16> {
    (0..width * height)
        .map(|index| {
            let column = index % width;
            let row = index / width;
            (((column * 3 + row * 5) / 8) % (peak + 1)) as u16
        })
        .collect()
}

/// Checks at least 99.9% of samples are within 1 code of the reference.
fn assert_mostly_within_one(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());

    let close = actual
        .iter()
        .zip(expected)
        .filter(|(got, want)| (*got - *want).abs() <= 1.0)
        .count();
    let needed = (expected.len() as f64 * 0.999).ceil() as usize;

    assert!(close >= needed, "{close} of {} samples within 1", expected.len());
}

/// Code values rounded to the nearest whole code.
fn whole(codes: &[f32]) -> Vec<u16> {
    codes.iter().map(|code| code.round() as u16).collect()
}

#[test]
fn eight_and_ten_bit_match_the_reference() {
    for (element, peak) in [(Source::U8, 255u32), (Source::U16, 1023)] {
        let input = ramp(128, 64, peak);
        let params = params(element, peak as f32);

        let actual = on_device(&input, 128, 64, params, 17);
        let expected = reference_deband_codes(&input, 128, 64, &params, 17);

        assert_mostly_within_one(&actual, &expected);
    }
}

#[test]
fn a_radius_past_the_plane_clamps_to_the_edge() {
    let input = ramp(8, 8, 255);
    let params = DebandParams {
        iterations: 16,
        radius: 64.0,
        ..params(Source::U8, 255.0)
    };

    let actual = on_device(&input, 8, 8, params, 3);
    let expected = reference_deband_codes(&input, 8, 8, &params, 3);

    assert_mostly_within_one(&actual, &expected);
}

#[test]
fn a_banded_ramp_gains_steps_between_its_bands() {
    // 10-bit bands 4 codes apart, 16 samples wide, the way an 8-bit gradient looks at 10 bits.
    let width = 256;
    let height = 16;
    let input: Vec<u16> = (0..width * height)
        .map(|index| (256 + 4 * ((index % width) / 16)) as u16)
        .collect();
    let params = DebandParams {
        iterations: 4,
        grain: 0.0,
        threshold: 3.0,
        ..params(Source::U16, 1023.0)
    };

    let debanded = on_device(&input, width, height, params, 5);

    let bands: Vec<u16> = (0..16).map(|index| 256 + index * 4).collect();

    let mut levels = whole(&debanded);
    levels.sort_unstable();
    levels.dedup();

    for neighbours in bands.windows(2) {
        let band = neighbours[0];
        let midpoint = band + 2;

        assert!(
            levels.contains(&midpoint),
            "the output is missing the midpoint {midpoint}"
        );
    }
}

#[test]
fn an_edge_past_the_threshold_survives_without_grain() {
    let width = 64;
    let height = 32;
    let input: Vec<u16> = (0..width * height)
        .map(|index| if index % width < width / 2 { 20 } else { 220 })
        .collect();
    let params = DebandParams {
        iterations: 4,
        grain: 0.0,
        ..params(Source::U8, 255.0)
    };

    let debanded = on_device(&input, width, height, params, 5);

    assert_eq!(whole(&debanded), input);
}

#[test]
fn the_same_salt_repeats_and_another_salt_moves_the_grain() {
    let input = vec![512u16; 64 * 32];
    let params = params(Source::U16, 1023.0);

    let first = on_device(&input, 64, 32, params, 40);
    let again = on_device(&input, 64, 32, params, 40);
    let next = on_device(&input, 64, 32, params, 41);

    assert_eq!(first, again);
    assert_ne!(first, next);
}

#[test]
fn grain_at_the_peak_never_wraps() {
    let input = vec![65535u16; 64 * 32];
    let params = DebandParams {
        grain: 50.0,
        ..params(Source::U16, 65535.0)
    };

    let grained = on_device(&input, 64, 32, params, 2);

    let lowest = grained.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(lowest > 65535.0 - 2000.0, "a sample fell to {lowest}");
}

#[test]
fn rounding_the_codes_gives_the_rounded_reference() {
    let input = ramp(128, 64, 255);
    let params = params(Source::U8, 255.0);

    let actual = on_device(&input, 128, 64, params, 17);
    let expected = reference_deband(&input, 128, 64, &params, 17);
    let rounded: Vec<f32> = actual.iter().map(|code| code.round().clamp(0.0, 255.0)).collect();
    let widened: Vec<f32> = expected.iter().map(|code| f32::from(*code)).collect();

    assert_mostly_within_one(&rounded, &widened);
}

#[test]
fn float_codes_match_sixteen_bit_codes() {
    let input = ramp(128, 64, 65535);
    let sixteen_bit = params(Source::U16, 65535.0);
    let float = params(Source::F32, 65535.0);

    let from_integers = on_device(&input, 128, 64, sixteen_bit, 17);
    let from_floats = on_device(&input, 128, 64, float, 17);

    assert_eq!(from_integers, from_floats);
}

#[test]
fn grain_leaves_codes_between_whole_values() {
    let input = vec![128u16; 64 * 32];
    let params = params(Source::U8, 255.0);

    let grained = on_device(&input, 64, 32, params, 9);

    let fractional = grained.iter().filter(|code| code.fract() != 0.0).count();
    assert!(
        fractional > grained.len() / 2,
        "{fractional} of {} fractional",
        grained.len()
    );
}
