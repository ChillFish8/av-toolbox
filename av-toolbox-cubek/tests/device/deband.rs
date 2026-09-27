use av_toolbox_cubek::deband::{DebandParams, DebandPlanes, deband};
use av_toolbox_cubek::depth::Element;
use av_toolbox_cubek::testing::reference_deband;
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use crate::support::{Sizing, backend, empty, plane, read_u8, upload_u8};

struct Deband {
    input: Vec<u16>,
    width: u32,
    height: u32,
    params: DebandParams,
    salt: u32,
}

impl BuildTask for Deband {
    type Output = Vec<u16>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let bytes = to_bytes(&self.input, self.params.element);

        let input = upload_u8(client, &bytes);
        let byte_count = bytes.len();
        let output = empty(client, byte_count);
        let input_plane = plane(&input, self.width, self.height);
        let output_plane = plane(&output, self.width, self.height);
        let planes = DebandPlanes {
            input: input_plane,
            output: output_plane,
        };
        deband(client, &sizing, planes, &self.params, self.salt);

        let read = read_u8(client, &output);
        from_bytes(&read, self.params.element)
    }
}

fn to_bytes(values: &[u16], element: Element) -> Vec<u8> {
    match element {
        Element::U8 => values.iter().map(|value| *value as u8).collect(),
        Element::U16 => values.iter().flat_map(|value| value.to_le_bytes()).collect(),
    }
}

fn from_bytes(bytes: &[u8], element: Element) -> Vec<u16> {
    match element {
        Element::U8 => bytes.iter().map(|value| u16::from(*value)).collect(),
        Element::U16 => {
            let (pairs, _) = bytes.as_chunks::<2>();
            pairs.iter().map(|pair| u16::from_le_bytes(*pair)).collect()
        },
    }
}

fn params(element: Element, peak: f32) -> DebandParams {
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

fn on_device(input: &[u16], width: u32, height: u32, params: DebandParams, salt: u32) -> Vec<u16> {
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

/// Checks at least 99.9% of samples are within 1 of the reference.
fn assert_mostly_within_one(actual: &[u16], expected: &[u16]) {
    assert_eq!(actual.len(), expected.len());

    let close = actual
        .iter()
        .zip(expected)
        .filter(|(got, want)| got.abs_diff(**want) <= 1)
        .count();
    let needed = (expected.len() as f64 * 0.999).ceil() as usize;

    assert!(close >= needed, "{close} of {} samples within 1", expected.len());
}

#[test]
fn eight_and_ten_bit_match_the_reference() {
    for (element, peak) in [(Element::U8, 255u32), (Element::U16, 1023)] {
        let input = ramp(128, 64, peak);
        let params = params(element, peak as f32);

        let actual = on_device(&input, 128, 64, params, 17);
        let expected = reference_deband(&input, 128, 64, &params, 17);

        assert_mostly_within_one(&actual, &expected);
    }
}

#[test]
fn a_radius_past_the_plane_clamps_to_the_edge() {
    let input = ramp(8, 8, 255);
    let params = DebandParams {
        iterations: 16,
        radius: 64.0,
        ..params(Element::U8, 255.0)
    };

    let actual = on_device(&input, 8, 8, params, 3);
    let expected = reference_deband(&input, 8, 8, &params, 3);

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
        ..params(Element::U16, 1023.0)
    };

    let debanded = on_device(&input, width, height, params, 5);

    let bands: Vec<u16> = (0..16).map(|index| 256 + index * 4).collect();

    let mut levels = debanded.clone();
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
        ..params(Element::U8, 255.0)
    };

    let debanded = on_device(&input, width, height, params, 5);

    assert_eq!(debanded, input);
}

#[test]
fn the_same_salt_repeats_and_another_salt_moves_the_grain() {
    let input = vec![512u16; 64 * 32];
    let params = params(Element::U16, 1023.0);

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
        ..params(Element::U16, 65535.0)
    };

    let grained = on_device(&input, 64, 32, params, 2);

    let lowest = grained.iter().min().copied().unwrap_or_default();
    assert!(lowest > 65535 - 2000, "a sample fell to {lowest}");
}
