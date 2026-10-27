use av_toolbox_cubek::depth::{Conversion, Element, convert_to_float_sample};
use av_toolbox_cubek::matrix::SampleRange;
use av_toolbox_cubek::resize::{
    Kernel,
    ResizePlanes,
    Samples,
    Taps,
    apply,
    apply_float,
    centred_taps,
    taps,
    upload,
};
use av_toolbox_cubek::testing::{reference_resize, reference_resize_float, reference_resize_wide};
use av_toolbox_cubek::values::FloatCodes;
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use crate::neighbourhood::busy;
use crate::support::{Sizing, assert_close, backend, empty, plane, read_f32, read_u8, upload_f32, upload_u8};

const EVERY: [Kernel; 8] = [
    Kernel::Point,
    Kernel::Bilinear,
    Kernel::Bicubic {
        b: 1.0 / 3.0,
        c: 1.0 / 3.0,
    },
    Kernel::Bicubic { b: 0.0, c: 0.5 },
    Kernel::Lanczos { taps: 3 },
    Kernel::Spline16,
    Kernel::Spline36,
    Kernel::Spline64,
];

struct Resize {
    /// The input plane's bytes, little endian for U16.
    input: Vec<u8>,
    width: u32,
    height: u32,
    across: Taps,
    down: Taps,
    samples: Samples,
}

impl BuildTask for Resize {
    type Output = Vec<u8>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let sample_bytes = match self.samples {
            Samples::U8 => 1,
            Samples::U16 { .. } => 2,
            Samples::F32 | Samples::IntegerToFloat { .. } => 4,
        };
        let out_width = self.across.starts.len() as u32;
        let out_height = self.down.starts.len() as u32;
        let scratch_bytes = (out_width * self.height) as usize * size_of::<f32>();
        let output_bytes = (out_width * out_height) as usize * sample_bytes;

        let input = upload_u8(client, &self.input);
        let scratch = empty(client, scratch_bytes);
        let output = empty(client, output_bytes);
        let across = upload(client, &self.across);
        let down = upload(client, &self.down);
        let input_plane = plane(&input, self.width, self.height);
        let scratch_plane = plane(&scratch, out_width, self.height);
        let output_plane = plane(&output, out_width, out_height);
        let planes = ResizePlanes {
            input: input_plane,
            scratch: scratch_plane,
            output: output_plane,
        };
        apply(client, &sizing, planes, &across, &down, self.samples);

        read_u8(client, &output)
    }
}

/// Every chroma location at 4:2:0 and 4:2:2, as `(scale_x, scale_y, offset_x, offset_y)`.
const CHROMA_SHAPES: [(f32, f32, f32, f32); 8] = [
    (2.0, 2.0, 0.0, 0.5),
    (2.0, 2.0, 0.5, 0.5),
    (2.0, 2.0, 0.0, 0.0),
    (2.0, 2.0, 0.5, 0.0),
    (2.0, 2.0, 0.0, 1.0),
    (2.0, 2.0, 0.5, 1.0),
    (2.0, 1.0, 0.0, 0.0),
    (2.0, 1.0, 0.5, 0.0),
];

/// Centre-aligned taps taking `input_len` samples to `output_len`.
fn centred(input_len: u32, output_len: u32, kernel: &Kernel) -> Taps {
    let scale = input_len as f32 / output_len as f32;
    let offset = 0.5 * scale - 0.5;

    taps(output_len, scale, offset, kernel)
}

fn largest_difference(found: &[u16], expected: &[u16]) -> u16 {
    found
        .iter()
        .zip(expected)
        .map(|(got, want)| got.abs_diff(*want))
        .max()
        .unwrap_or(0)
}

fn widen(samples: &[u16]) -> Vec<u8> {
    samples.iter().flat_map(|sample| sample.to_le_bytes()).collect()
}

fn unwiden(bytes: &[u8]) -> Vec<u16> {
    let (pairs, _) = bytes.as_chunks::<2>();

    pairs.iter().map(|pair| u16::from_le_bytes(*pair)).collect()
}

fn check(input: &[u8], width: u32, height: u32, across: Taps, down: Taps) {
    let expected = reference_resize(input, width, height, &across, &down);

    let found = backend().build(Resize {
        input: input.to_vec(),
        width,
        height,
        across,
        down,
        samples: Samples::U8,
    });

    let found_wide: Vec<u16> = found.iter().map(|sample| u16::from(*sample)).collect();
    let expected_wide: Vec<u16> = expected.iter().map(|sample| u16::from(*sample)).collect();
    let largest = largest_difference(&found_wide, &expected_wide);
    assert_eq!(found.len(), expected.len());
    assert!(largest <= 1, "differed by {largest}");
}

/// Resizes a 10-bit plane on the device, checks it against the reference and hands back what the device made.
fn check_ten_bit(input: &[u16], width: u32, height: u32, across: Taps, down: Taps) -> Vec<u16> {
    let expected = reference_resize_wide(input, width, height, &across, &down, 1023);

    let bytes = backend().build(Resize {
        input: widen(input),
        width,
        height,
        across,
        down,
        samples: Samples::U16 { largest: 1023 },
    });

    let found = unwiden(&bytes);
    let largest = largest_difference(&found, &expected);
    assert_eq!(found.len(), expected.len());
    assert!(largest <= 1, "differed by {largest}");

    found
}

fn check_chroma_shapes(input: &[u8], width: u32, height: u32) {
    for (scale_x, scale_y, offset_x, offset_y) in CHROMA_SHAPES {
        let out_width = width.div_ceil(scale_x as u32);
        let out_height = height.div_ceil(scale_y as u32);
        let across = taps(out_width, scale_x, offset_x, &Kernel::Bilinear);
        let down = taps(out_height, scale_y, offset_y, &Kernel::Bilinear);

        check(input, width, height, across, down);
    }
}

#[test]
fn a_ramp_of_odd_size_downscales_at_every_chroma_location() {
    let input: Vec<u8> = (0..35).map(|index| (index * 7) as u8).collect();

    check_chroma_shapes(&input, 7, 5);
}

#[test]
fn a_hard_mask_downscales_with_the_bilinear_edge_weights() {
    let input: Vec<u8> = (0..35)
        .map(|index| if index % 7 >= 3 { 255 } else { 0 })
        .collect();

    check_chroma_shapes(&input, 7, 5);
}

#[test]
fn a_plane_doubles_with_bilinear() {
    let input: Vec<u8> = (0..12).map(|index| (index * 20) as u8).collect();
    let across = taps(8, 0.5, 0.0, &Kernel::Bilinear);
    let down = taps(6, 0.5, 0.0, &Kernel::Bilinear);

    check(&input, 4, 3, across, down);
}

#[test]
fn every_kernel_matches_the_reference_up_and_down_in_u8() {
    let input: Vec<u8> = (0..176).map(|index| (index * 37 % 256) as u8).collect();

    for kernel in EVERY {
        let up_across = centred(16, 23, &kernel);
        let up_down = centred(11, 17, &kernel);
        check(&input, 16, 11, up_across, up_down);

        let down_across = centred(16, 7, &kernel);
        let down_down = centred(11, 5, &kernel);
        check(&input, 16, 11, down_across, down_down);
    }
}

#[test]
fn every_kernel_matches_the_reference_up_and_down_in_ten_bit() {
    let input: Vec<u16> = (0..176).map(|index| (index * 149 % 1024) as u16).collect();

    for kernel in EVERY {
        let up_across = centred(16, 23, &kernel);
        let up_down = centred(11, 17, &kernel);
        check_ten_bit(&input, 16, 11, up_across, up_down);

        let down_across = centred(16, 7, &kernel);
        let down_down = centred(11, 5, &kernel);
        check_ten_bit(&input, 16, 11, down_across, down_down);
    }
}

#[test]
fn a_ten_bit_edge_through_lanczos_stays_within_its_depth() {
    let input: Vec<u16> = (0..64)
        .map(|index| if index % 8 >= 4 { 1023 } else { 0 })
        .collect();
    let kernel = Kernel::Lanczos { taps: 4 };
    let across = centred(8, 29, &kernel);
    let down = centred(8, 8, &kernel);

    let found = check_ten_bit(&input, 8, 8, across, down);

    let brightest = found.iter().max().copied().unwrap_or(0);
    assert_eq!(brightest, 1023);
}

#[test]
fn lanczos_ringing_clamps_at_both_ends_of_u8() {
    let input: Vec<u8> = (0..64)
        .map(|index| if index % 8 >= 4 { 255 } else { 0 })
        .collect();
    let kernel = Kernel::Lanczos { taps: 4 };
    let across = centred(8, 29, &kernel);
    let down = centred(8, 8, &kernel);

    check(&input, 8, 8, across, down);
}

#[test]
fn a_plane_shrinks_to_one_sample_with_lanczos() {
    let input: Vec<u8> = (0..64).map(|index| (index * 4) as u8).collect();
    let kernel = Kernel::Lanczos { taps: 3 };
    let across = centred(64, 1, &kernel);
    let down = centred(1, 1, &kernel);

    check(&input, 64, 1, across, down);
}

#[test]
fn a_single_column_widens() {
    let input: Vec<u8> = vec![10, 60, 110, 160, 210];
    let kernel = Kernel::Spline36;
    let across = centred(1, 4, &kernel);
    let down = centred(5, 5, &kernel);

    check(&input, 1, 5, across, down);
}

struct ResizeFloat {
    input: Vec<f32>,
    width: u32,
    height: u32,
    across: Taps,
    down: Taps,
}

impl BuildTask for ResizeFloat {
    type Output = Vec<f32>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let out_width = self.across.starts.len() as u32;
        let out_height = self.down.starts.len() as u32;
        let scratch_bytes = (out_width * self.height) as usize * size_of::<f32>();
        let output_bytes = (out_width * out_height) as usize * size_of::<f32>();

        let input = upload_f32(client, &self.input);
        let scratch = empty(client, scratch_bytes);
        let output = empty(client, output_bytes);
        let across = upload(client, &self.across);
        let down = upload(client, &self.down);
        let input_plane = plane(&input, self.width, self.height);
        let scratch_plane = plane(&scratch, out_width, self.height);
        let output_plane = plane(&output, out_width, out_height);
        let planes = ResizePlanes {
            input: input_plane,
            scratch: scratch_plane,
            output: output_plane,
        };
        apply_float(client, &sizing, planes, &across, &down);

        read_f32(client, &output)
    }
}

#[test]
fn apply_float_matches_the_reference_in_f32() {
    let input = busy(40, 24);
    let bicubic = Kernel::Bicubic {
        b: 1.0 / 3.0,
        c: 1.0 / 3.0,
    };
    let across = centred_taps(40, 16, &bicubic);
    let down = centred_taps(24, 10, &bicubic);
    let expected = reference_resize_float(&input, 40, 24, &across, &down);

    let task = ResizeFloat {
        input,
        width: 40,
        height: 24,
        across,
        down,
    };
    let actual = backend().build(task);

    assert_close(&actual, &expected, 1e-3);
}

#[test]
fn apply_float_upscales_and_matches_the_reference_in_f32() {
    let input = busy(16, 10);
    let bicubic = Kernel::Bicubic {
        b: 1.0 / 3.0,
        c: 1.0 / 3.0,
    };
    let across = centred_taps(16, 40, &bicubic);
    let down = centred_taps(10, 24, &bicubic);
    let expected = reference_resize_float(&input, 16, 10, &across, &down);

    let task = ResizeFloat {
        input,
        width: 16,
        height: 10,
        across,
        down,
    };
    let actual = backend().build(task);

    assert_close(&actual, &expected, 1e-3);
}

fn float_values(bytes: &[u8]) -> Vec<f32> {
    let (words, _) = bytes.as_chunks::<4>();

    words.iter().map(|word| f32::from_le_bytes(*word)).collect()
}

#[test]
fn integer_rows_resize_into_float_samples() {
    let input: Vec<u8> = (0..16 * 11).map(|index| (16 + index * 37 % 220) as u8).collect();
    let kernel = Kernel::Spline36;

    for chroma in [false, true] {
        let conversion = Conversion::to_float(8, SampleRange::Limited, chroma);
        let floats: Vec<f32> = input
            .iter()
            .map(|code| convert_to_float_sample(f32::from(*code), &conversion))
            .collect();
        let float_task = ResizeFloat {
            input: floats,
            width: 16,
            height: 11,
            across: centred(16, 23, &kernel),
            down: centred(11, 5, &kernel),
        };
        let expected = backend().build(float_task);

        let codes = FloatCodes::new(8, SampleRange::Limited, chroma);
        let integer_task = Resize {
            input: input.clone(),
            width: 16,
            height: 11,
            across: centred(16, 23, &kernel),
            down: centred(11, 5, &kernel),
            samples: Samples::IntegerToFloat {
                element: Element::U8,
                codes,
            },
        };
        let bytes = backend().build(integer_task);
        let found = float_values(&bytes);

        assert_close(&found, &expected, 1e-6);
    }
}
