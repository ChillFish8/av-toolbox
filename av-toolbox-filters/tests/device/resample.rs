use av_toolbox_cubek::depth::{Conversion, Thresholds, plane_shift};
use av_toolbox_cubek::matrix::{Coefficients, Direction, MatrixConversion, MatrixSamples, SampleRange};
use av_toolbox_cubek::resize::{Kernel, taps};
use av_toolbox_cubek::testing::{
    decode_samples,
    encode_samples,
    reference_depth,
    reference_float_depth,
    reference_matrix,
    reference_resize,
};
use av_toolbox_gpu::{
    Channel,
    ChromaLocation,
    ColourInfo,
    ColourRange,
    Depth,
    FrameFormat,
    FrameLayout,
    Matrix as StreamMatrix,
    SampleFormat,
    Subsampling,
};
use serde_json::json;

use crate::support::{
    assert_within_one,
    every_plane_is_shared,
    frame,
    luma_is_shared,
    passes_through,
    patterned_planes,
    run_once,
};

/// Every plane of `format`, each sample `(index * 13) % 1024`.
fn ten_bit_values(format: &FrameFormat) -> Vec<Vec<u16>> {
    format
        .channels()
        .iter()
        .map(|channel| {
            let samples = format.channel_samples(*channel).expect("a listed channel");
            (0..samples).map(|index| (index * 13 % 1024) as u16).collect()
        })
        .collect()
}

fn widen(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn widen_u8(bytes: &[u8]) -> Vec<u16> {
    bytes.iter().map(|sample| u16::from(*sample)).collect()
}

fn unwiden(bytes: &[u8]) -> Vec<u16> {
    let (pairs, _) = bytes.as_chunks::<2>();

    pairs.iter().map(|pair| u16::from_le_bytes(*pair)).collect()
}

/// Every plane of `format`, each sample `(index * 13) % 256`.
fn eight_bit_values(format: &FrameFormat) -> Vec<Vec<u8>> {
    format
        .channels()
        .iter()
        .map(|channel| {
            let samples = format.channel_samples(*channel).expect("a listed channel");
            (0..samples).map(|index| (index * 13 % 256) as u8).collect()
        })
        .collect()
}

#[test]
fn lowering_a_ten_bit_frame_matches_the_reference() {
    let ten_bit = SampleFormat::U16 { depth: Depth::TEN };
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let input = frame(32, 16, layout, ten_bit, ChromaLocation::Left);
    let output = FrameFormat {
        sample: SampleFormat::U8,
        ..input
    };
    let args = json!({ "depth": 8, "dither": "ordered" });

    let values = ten_bit_values(&input);
    let planes: Vec<Vec<u8>> = values.iter().map(|plane| widen(plane)).collect();
    let converted = run_once("resample/depth", input, output, args, &planes);

    let conversion = Conversion::new(10, 8, SampleRange::Limited, false);
    let thresholds = Thresholds::ordered();
    for (index, channel) in input.channels().iter().enumerate() {
        let (width, _) = input.channel_dimensions(*channel).expect("carried");
        let shift = plane_shift(index as u32);
        let expected = reference_depth(&values[index], width, &conversion, &thresholds, shift);
        let actual = widen_u8(&converted[index]);

        assert_eq!(actual, expected, "{channel}");
    }
}

/// Every plane of `format`, luma between 0 and 1 and chroma between -0.5 and 0.5.
fn float_values(format: &FrameFormat) -> Vec<Vec<f32>> {
    format
        .channels()
        .iter()
        .map(|channel| {
            let samples = format.channel_samples(*channel).expect("a listed channel");
            let low = match channel {
                Channel::U | Channel::V => -0.5,
                _ => 0.0,
            };

            (0..samples)
                .map(|index| (index * 13 % 256) as f32 / 255.0 + low)
                .collect()
        })
        .collect()
}

fn float_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

#[test]
fn a_float_frame_centres_its_chroma() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let float = frame(32, 16, layout, SampleFormat::F32, ChromaLocation::Left);
    let colour = ColourInfo {
        range: ColourRange::Full,
        ..float.colour
    };
    let input = FrameFormat { colour, ..float };
    let output = FrameFormat {
        sample: SampleFormat::U8,
        ..input
    };
    let args = json!({ "depth": 8, "dither": "none" });

    let values = float_values(&input);
    let planes: Vec<Vec<u8>> = values.iter().map(|plane| float_bytes(plane)).collect();
    let converted = run_once("resample/depth", input, output, args, &planes);

    let thresholds = Thresholds::rounding();
    for (index, channel) in input.channels().iter().enumerate() {
        let (width, _) = input.channel_dimensions(*channel).expect("carried");
        let chroma = matches!(channel, Channel::U | Channel::V);
        let conversion = Conversion::from_float(8, SampleRange::Full, chroma);
        let shift = plane_shift(index as u32);
        let expected = reference_float_depth(&values[index], width, &conversion, &thresholds, shift);
        let expected: Vec<u8> = expected.iter().map(|value| *value as u8).collect();
        let label = format!("{channel}");

        assert_within_one(&converted[index], &expected, &label);
    }
}

fn limited_grey(sample: SampleFormat) -> FrameFormat {
    let grey = frame(16, 16, FrameLayout::Grey, sample, ChromaLocation::Left);
    let colour = ColourInfo {
        range: ColourRange::Limited,
        ..grey.colour
    };

    FrameFormat { colour, ..grey }
}

#[test]
fn an_eight_bit_round_trip_through_float_matches_a_direct_shift() {
    let eight_bit = limited_grey(SampleFormat::U8);
    let float = limited_grey(SampleFormat::F32);
    let ten_bit = limited_grey(SampleFormat::U16 { depth: Depth::TEN });
    let raised = json!({ "depth": "f32" });
    let lowered = json!({ "depth": 10, "dither": "none" });

    let levels = [16u8, 128, 235];
    let plane: Vec<u8> = (0..16 * 16).map(|index| levels[index % 3]).collect();
    let planes = vec![plane];

    let floats = run_once("resample/depth", eight_bit, float, raised, &planes);
    let round_trip = run_once("resample/depth", float, ten_bit, lowered.clone(), &floats);
    let direct = run_once("resample/depth", eight_bit, ten_bit, lowered, &planes);

    let round_trip = unwiden(&round_trip[0]);
    let direct = unwiden(&direct[0]);

    assert_eq!(round_trip, direct);
    assert_eq!(&round_trip[..3], [64, 512, 940]);
}

fn yuv420_in(range: ColourRange, sample: SampleFormat) -> FrameFormat {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let yuv = frame(16, 16, layout, sample, ChromaLocation::Left);
    let colour = ColourInfo { range, ..yuv.colour };

    FrameFormat { colour, ..yuv }
}

fn read_floats(bytes: &[u8]) -> Vec<f32> {
    let (chunks, _) = bytes.as_chunks::<4>();

    chunks.iter().map(|chunk| f32::from_le_bytes(*chunk)).collect()
}

#[test]
fn limited_yuv_becomes_full_range_floats_with_centred_chroma() {
    let input = yuv420_in(ColourRange::Limited, SampleFormat::U8);
    let output = yuv420_in(ColourRange::Limited, SampleFormat::F32);
    let args = json!({ "depth": "f32" });

    let luma_levels = [16u8, 235];
    let luma: Vec<u8> = (0..16 * 16).map(|index| luma_levels[index % 2]).collect();
    let chroma = vec![128u8; 8 * 8];
    let planes = vec![luma, chroma.clone(), chroma];

    let converted = run_once("resample/depth", input, output, args, &planes);

    let luma = read_floats(&converted[0]);
    assert!(luma[0].abs() <= 1e-6, "{}", luma[0]);
    assert!((luma[1] - 1.0).abs() <= 1e-6, "{}", luma[1]);
    for plane in &converted[1..] {
        let chroma = read_floats(plane);
        assert!(chroma.iter().all(|value| value.abs() <= 1e-6), "{chroma:?}");
    }
}

#[test]
fn full_range_chroma_through_float_matches_a_direct_change() {
    let eight_bit = yuv420_in(ColourRange::Full, SampleFormat::U8);
    let float = yuv420_in(ColourRange::Full, SampleFormat::F32);
    let ten_bit = yuv420_in(ColourRange::Full, SampleFormat::U16 { depth: Depth::TEN });
    let raised = json!({ "depth": "f32" });
    let lowered = json!({ "depth": 10, "dither": "none" });

    let planes = eight_bit_values(&eight_bit);
    let floats = run_once("resample/depth", eight_bit, float, raised, &planes);
    let round_trip = run_once("resample/depth", float, ten_bit, lowered.clone(), &floats);
    let direct = run_once("resample/depth", eight_bit, ten_bit, lowered, &planes);

    for (index, channel) in eight_bit.channels().iter().enumerate() {
        let through_float = unwiden(&round_trip[index]);
        let changed = unwiden(&direct[index]);

        assert_eq!(through_float, changed, "{channel}");
    }

    let neutral = vec![128u8; 8 * 8];
    let planes = vec![planes[0].clone(), neutral.clone(), neutral];
    let direct = run_once(
        "resample/depth",
        eight_bit,
        ten_bit,
        json!({ "depth": 10 }),
        &planes,
    );
    for plane in &direct[1..] {
        let chroma = unwiden(plane);
        assert!(chroma.iter().all(|value| *value == 512), "{chroma:?}");
    }
}

#[test]
fn build_converts_to_the_settled_sample() {
    let input = limited_grey(SampleFormat::U8);
    let twelve_bit = Depth::new(12).expect("valid");
    let output = limited_grey(SampleFormat::U16 { depth: twelve_bit });
    let args = json!({ "depth": 10, "at_least": true });

    let planes = eight_bit_values(&input);
    let converted = run_once("resample/depth", input, output, args, &planes);

    let expected: Vec<u16> = planes[0].iter().map(|value| u16::from(*value) << 4).collect();
    let actual = unwiden(&converted[0]);

    assert_eq!(actual, expected);
}

#[test]
fn raising_the_depth_ignores_the_dither() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let narrow = frame(32, 16, layout, SampleFormat::U8, ChromaLocation::Left);
    let colour = ColourInfo {
        range: ColourRange::Full,
        ..narrow.colour
    };
    let input = FrameFormat { colour, ..narrow };
    let ten_bit = SampleFormat::U16 { depth: Depth::TEN };
    let output = FrameFormat {
        sample: ten_bit,
        ..input
    };
    let args = json!({ "depth": 10, "dither": "ordered" });

    let planes = eight_bit_values(&input);
    let converted = run_once("resample/depth", input, output, args, &planes);

    let thresholds = Thresholds::rounding();
    for (index, channel) in input.channels().iter().enumerate() {
        let (width, _) = input.channel_dimensions(*channel).expect("carried");
        let chroma = matches!(channel, Channel::U | Channel::V);
        let conversion = Conversion::new(8, 10, SampleRange::Full, chroma);
        let shift = plane_shift(index as u32);
        let values = widen_u8(&planes[index]);
        let expected = reference_depth(&values, width, &conversion, &thresholds, shift);
        let actual = unwiden(&converted[index]);

        assert_eq!(actual, expected, "{channel}");
    }
}

#[test]
fn an_unchanged_depth_hands_back_the_input() {
    let layout = FrameLayout::Grey;
    let input = frame(16, 16, layout, SampleFormat::U8, ChromaLocation::Left);
    let args = json!({ "depth": 8 });

    let same = passes_through("resample/depth", input, input, args);

    assert!(same);
}

#[test]
fn raising_to_a_minimum_depth_matches_the_reference_in_either_range() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let narrow = frame(32, 16, layout, SampleFormat::U8, ChromaLocation::Left);
    let ten_bit = SampleFormat::U16 { depth: Depth::TEN };

    for range in [ColourRange::Limited, ColourRange::Full] {
        let colour = ColourInfo {
            range,
            ..narrow.colour
        };
        let input = FrameFormat { colour, ..narrow };
        let output = FrameFormat {
            sample: ten_bit,
            ..input
        };
        let args = json!({ "depth": 10 });

        let planes = eight_bit_values(&input);
        let raised = run_once("resample/min_depth", input, output, args, &planes);

        let sample_range = match range {
            ColourRange::Full => SampleRange::Full,
            _ => SampleRange::Limited,
        };
        let thresholds = Thresholds::rounding();
        for (index, channel) in input.channels().iter().enumerate() {
            let (width, _) = input.channel_dimensions(*channel).expect("carried");
            let chroma = matches!(channel, Channel::U | Channel::V);
            let conversion = Conversion::new(8, 10, sample_range, chroma);
            let shift = plane_shift(index as u32);
            let values = widen_u8(&planes[index]);
            let expected = reference_depth(&values, width, &conversion, &thresholds, shift);
            let actual = unwiden(&raised[index]);

            assert_eq!(actual, expected, "{range:?} {channel}");
        }
    }
}

#[test]
fn a_depth_at_or_above_the_minimum_hands_back_the_input() {
    let twelve_bit = Depth::new(12).expect("valid");
    let kept = [
        SampleFormat::U16 { depth: Depth::TEN },
        SampleFormat::U16 { depth: twelve_bit },
    ];

    for sample in kept {
        let input = frame(16, 16, FrameLayout::Grey, sample, ChromaLocation::Left);
        let args = json!({ "depth": 10 });

        let same = passes_through("resample/min_depth", input, input, args);

        assert!(same, "{sample:?}");
    }
}

#[test]
fn every_kernel_matches_the_reference_between_subsamplings() {
    let kernels = [
        ("point", Kernel::Point),
        ("bilinear", Kernel::Bilinear),
        (
            "bicubic",
            Kernel::Bicubic {
                b: 1.0 / 3.0,
                c: 1.0 / 3.0,
            },
        ),
        ("lanczos", Kernel::Lanczos { taps: 3 }),
        ("spline16", Kernel::Spline16),
        ("spline36", Kernel::Spline36),
        ("spline64", Kernel::Spline64),
    ];
    let changes = [
        (Subsampling::Yuv420, Subsampling::Yuv444),
        (Subsampling::Yuv444, Subsampling::Yuv420),
        (Subsampling::Yuv422, Subsampling::Yuv420),
        (Subsampling::Yuv420, Subsampling::Yuv422),
        (Subsampling::Yuv422, Subsampling::Yuv444),
        (Subsampling::Yuv444, Subsampling::Yuv422),
    ];

    for (name, kernel) in kernels {
        for (from, to) in changes {
            let input = frame(
                32,
                16,
                FrameLayout::Yuv { subsampling: from },
                SampleFormat::U8,
                ChromaLocation::Left,
            );
            let output = FrameFormat {
                layout: FrameLayout::Yuv { subsampling: to },
                ..input
            };
            let target = json!(to.to_string());
            let args = json!({ "subsampling": target, "kernel": name });

            let planes = patterned_planes(&input);
            let resampled = run_once("resample/chroma", input, output, args, &planes);

            let (in_width, in_height) = input.channel_dimensions(Channel::U).expect("carried");
            let (out_width, out_height) = output.channel_dimensions(Channel::U).expect("carried");
            let (factor_in_x, factor_in_y) = from.factors();
            let (factor_out_x, factor_out_y) = to.factors();
            let (siting_in_x, siting_in_y) = ChromaLocation::Left.offsets(factor_in_x, factor_in_y);
            let (siting_out_x, siting_out_y) = ChromaLocation::Left.offsets(factor_out_x, factor_out_y);
            let scale_x = factor_out_x as f32 / factor_in_x as f32;
            let scale_y = factor_out_y as f32 / factor_in_y as f32;
            let offset_x = (siting_out_x - siting_in_x) / factor_in_x as f32;
            let offset_y = (siting_out_y - siting_in_y) / factor_in_y as f32;
            let across = taps(out_width, scale_x, offset_x, &kernel);
            let down = taps(out_height, scale_y, offset_y, &kernel);

            assert_eq!(resampled[0], planes[0], "{name}: luma must be untouched");
            for index in [1, 2] {
                let expected = reference_resize(&planes[index], in_width, in_height, &across, &down);
                assert_within_one(&resampled[index], &expected, name);
            }
        }
    }
}

#[test]
fn the_luma_plane_is_shared_not_copied() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let input = frame(32, 16, layout, SampleFormat::U8, ChromaLocation::Left);
    let output = FrameFormat {
        layout: FrameLayout::Yuv {
            subsampling: Subsampling::Yuv444,
        },
        ..input
    };
    let args = json!({ "subsampling": "4:4:4" });

    let shared = luma_is_shared("resample/chroma", input, output, args);

    assert!(shared);
}

#[test]
fn a_vertical_only_siting_change_at_422_relabels_every_plane_instead_of_resampling() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv422,
    };
    let input = frame(32, 16, layout, SampleFormat::U8, ChromaLocation::Left);
    let colour = ColourInfo {
        chroma_location: ChromaLocation::TopLeft,
        ..input.colour
    };
    let output = FrameFormat { colour, ..input };
    let args = json!({ "subsampling": "4:2:2", "location": "top_left" });

    let shared = every_plane_is_shared("resample/chroma", input, output, args);

    assert!(shared);
}

#[test]
fn a_new_siting_at_444_relabels_every_plane_instead_of_resampling() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv444,
    };
    let input = frame(32, 16, layout, SampleFormat::U8, ChromaLocation::Left);
    let colour = ColourInfo {
        chroma_location: ChromaLocation::Center,
        ..input.colour
    };
    let output = FrameFormat { colour, ..input };
    let args = json!({ "subsampling": "4:4:4", "location": "center" });

    let shared = every_plane_is_shared("resample/chroma", input, output, args);

    assert!(shared);
}

fn coloured(
    layout: FrameLayout,
    sample: SampleFormat,
    matrix: StreamMatrix,
    range: ColourRange,
) -> FrameFormat {
    let input = frame(16, 8, layout, sample, ChromaLocation::Left);
    let colour = ColourInfo {
        matrix,
        range,
        ..input.colour
    };

    FrameFormat { colour, ..input }
}

/// Three planes of `samples` in `0..=peak`, each a different pattern, as f64 code values.
///
/// Integer formats get whole code values.
fn patterned_values(samples: usize, peak: f64, integer: bool) -> [Vec<f64>; 3] {
    [7usize, 11, 13].map(|step| {
        (0..samples)
            .map(|index| ((index * step) % 97) as f64 / 96.0 * peak)
            .map(|value| if integer { value.round() } else { value })
            .collect()
    })
}

fn run_matrix(
    input: FrameFormat,
    output: FrameFormat,
    args: serde_json::Value,
    values: &[Vec<f64>; 3],
    samples: MatrixSamples,
) -> Vec<f64> {
    let planes: Vec<Vec<u8>> = values
        .iter()
        .map(|plane| encode_samples(plane, samples))
        .collect();
    let converted = run_once("resample/matrix", input, output, args, &planes);
    let bytes = converted.concat();

    decode_samples(&bytes, samples)
}

#[test]
fn yuv_to_rgb_matches_the_reference_and_relabels_the_frame() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv444,
    };
    let input = coloured(
        layout,
        SampleFormat::U8,
        StreamMatrix::Bt709,
        ColourRange::Limited,
    );
    let output = coloured(
        FrameLayout::Rgb,
        SampleFormat::U8,
        StreamMatrix::Identity,
        ColourRange::Full,
    );
    let values = patterned_values(16 * 8, 255.0, true);
    let conversion = MatrixConversion {
        coefficients: Coefficients::Bt709,
        direction: Direction::ToRgb,
        samples: MatrixSamples::U8,
        input_range: SampleRange::Limited,
        output_range: SampleRange::Full,
    };

    let found = run_matrix(input, output, json!({ "to": "rgb" }), &values, MatrixSamples::U8);

    let expected = reference_matrix(&conversion, [&values[0], &values[1], &values[2]]).concat();
    for (value, wanted) in found.iter().zip(&expected) {
        assert!((value - wanted).abs() <= 1.0, "{value} against {wanted}");
    }
}

#[test]
fn a_ten_bit_yuv_frame_converts_with_ten_bit_offsets() {
    let ten_bit = SampleFormat::U16 { depth: Depth::TEN };
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv444,
    };
    let input = coloured(layout, ten_bit, StreamMatrix::Bt2020Ncl, ColourRange::Limited);
    let output = coloured(
        FrameLayout::Rgb,
        ten_bit,
        StreamMatrix::Identity,
        ColourRange::Full,
    );
    let samples = MatrixSamples::U16 { bits: 10 };
    let mut values = patterned_values(16 * 8, 1023.0, true);
    values[0][0] = 64.0;
    values[1][0] = 512.0;
    values[2][0] = 512.0;
    values[0][1] = 940.0;
    values[1][1] = 512.0;
    values[2][1] = 512.0;

    let found = run_matrix(input, output, json!({ "to": "rgb" }), &values, samples);

    let plane = 16 * 8;
    for channel in 0..3 {
        assert_eq!(found[channel * plane], 0.0, "black in channel {channel}");
        assert_eq!(found[channel * plane + 1], 1023.0, "white in channel {channel}");
    }
}

#[test]
fn rgb_to_yuv_takes_its_matrix_and_range_from_the_arguments() {
    let input = coloured(
        FrameLayout::Rgb,
        SampleFormat::U8,
        StreamMatrix::Identity,
        ColourRange::Full,
    );
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv444,
    };
    let output = coloured(layout, SampleFormat::U8, StreamMatrix::Bt601, ColourRange::Full);
    let values = patterned_values(16 * 8, 255.0, true);
    let conversion = MatrixConversion {
        coefficients: Coefficients::Bt601,
        direction: Direction::ToYuv,
        samples: MatrixSamples::U8,
        input_range: SampleRange::Full,
        output_range: SampleRange::Full,
    };
    let args = json!({ "to": "yuv", "matrix": "bt601", "range": "full" });

    let found = run_matrix(input, output, args, &values, MatrixSamples::U8);

    let expected = reference_matrix(&conversion, [&values[0], &values[1], &values[2]]).concat();
    for (value, wanted) in found.iter().zip(&expected) {
        assert!((value - wanted).abs() <= 1.0, "{value} against {wanted}");
    }
}

#[test]
fn a_float_round_trip_through_both_stages_returns_the_input() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv444,
    };
    let yuv = coloured(layout, SampleFormat::F32, StreamMatrix::Bt709, ColourRange::Full);
    let rgb = coloured(
        FrameLayout::Rgb,
        SampleFormat::F32,
        StreamMatrix::Identity,
        ColourRange::Full,
    );
    let samples = MatrixSamples::F32;
    let [luma, blue, red] = patterned_values(16 * 8, 1.0, false);
    let centred = |plane: Vec<f64>| -> Vec<f64> { plane.iter().map(|value| value - 0.5).collect() };
    let values = [luma, centred(blue), centred(red)];

    let rgb_values = run_matrix(yuv, rgb, json!({ "to": "rgb" }), &values, samples);
    let plane = 16 * 8;
    let rgb_planes = [0, 1, 2].map(|index| rgb_values[index * plane..(index + 1) * plane].to_vec());
    let args = json!({ "to": "yuv", "matrix": "bt709", "range": "full" });
    let back = run_matrix(rgb, yuv, args, &rgb_planes, samples);

    let expected = values.concat();
    for (value, wanted) in back.iter().zip(&expected) {
        assert!((value - wanted).abs() <= 1e-5, "{value} against {wanted}");
    }
}
