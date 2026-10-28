use av_toolbox_cubek::depth::Element;
use av_toolbox_cubek::matrix::SampleRange;
use av_toolbox_cubek::testing::reference_narrow;
use av_toolbox_cubek::values::{
    FloatCodes,
    codes_to_float,
    float_to_codes,
    float_to_whole_codes,
    narrow,
    offset,
    unit_to_float,
    widen,
};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use crate::support::{
    Sizing,
    assert_close,
    backend,
    empty,
    from_bytes,
    plane,
    read_f32,
    read_u8,
    to_bytes,
    upload_f32,
    upload_u8,
};

struct Widen {
    input: Vec<u16>,
    element: Element,
}

impl BuildTask for Widen {
    type Output = Vec<f32>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let bytes = to_bytes(&self.input, self.element);
        let samples = self.input.len() as u32;
        let output_bytes = self.input.len() * size_of::<f32>();

        let input = upload_u8(client, &bytes);
        let output = empty(client, output_bytes);
        let input_plane = plane(&input, samples, 1);
        let output_plane = plane(&output, samples, 1);
        widen(client, &sizing, input_plane, output_plane, self.element);

        read_f32(client, &output)
    }
}

struct Narrow {
    input: Vec<f32>,
    element: Element,
    peak: f32,
}

impl BuildTask for Narrow {
    type Output = Vec<u16>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let samples = self.input.len() as u32;
        let sample_bytes = match self.element {
            Element::U8 => 1,
            Element::U16 => 2,
        };
        let output_bytes = self.input.len() * sample_bytes;

        let input = upload_f32(client, &self.input);
        let output = empty(client, output_bytes);
        let input_plane = plane(&input, samples, 1);
        let output_plane = plane(&output, samples, 1);
        narrow(
            client,
            &sizing,
            input_plane,
            output_plane,
            self.element,
            self.peak,
        );

        let read = read_u8(client, &output);
        from_bytes(&read, self.element)
    }
}

#[test]
fn widening_keeps_every_code_value() {
    let eight_bit: Vec<u16> = (0..256).collect();
    let ten_bit: Vec<u16> = (0..1024).collect();

    for (values, element) in [(eight_bit, Element::U8), (ten_bit, Element::U16)] {
        let task = Widen {
            input: values.clone(),
            element,
        };
        let widened = backend().build(task);
        let expected: Vec<f32> = values.iter().map(|value| f32::from(*value)).collect();

        assert_eq!(widened, expected);
    }
}

#[test]
fn narrowing_rounds_and_clamps_like_the_reference() {
    let values = vec![-3.0, 0.0, 0.49, 0.5, 1.5, 254.4, 254.5, 255.0, 300.0, 1022.6];

    for (element, peak) in [(Element::U8, 255.0), (Element::U16, 1023.0)] {
        let task = Narrow {
            input: values.clone(),
            element,
            peak,
        };
        let narrowed = backend().build(task);
        let expected = reference_narrow(&values, peak);

        assert_eq!(narrowed, expected, "peak {peak}");
    }
}

struct Offset {
    input: Vec<f32>,
    amount: f32,
}

impl BuildTask for Offset {
    type Output = Vec<f32>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let samples = self.input.len() as u32;
        let output_bytes = self.input.len() * size_of::<f32>();

        let input = upload_f32(client, &self.input);
        let output = empty(client, output_bytes);
        let input_plane = plane(&input, samples, 1);
        let output_plane = plane(&output, samples, 1);
        offset(client, &sizing, input_plane, output_plane, self.amount);

        read_f32(client, &output)
    }
}

#[test]
fn offsetting_adds_the_amount_to_every_sample() {
    let values = vec![-0.5, -0.25, 0.0, 0.125, 0.5];

    for amount in [0.5, -0.5] {
        let task = Offset {
            input: values.clone(),
            amount,
        };
        let offsetted = backend().build(task);
        let expected: Vec<f32> = values.iter().map(|value| value + amount).collect();

        assert_eq!(offsetted, expected, "amount {amount}");
    }
}

/// Which way a [MapCodes] task maps its samples.
#[derive(Clone, Copy)]
enum Mapping {
    Codes,
    WholeCodes,
    Float,
    Unit,
}

struct MapCodes {
    input: Vec<f32>,
    codes: FloatCodes,
    mapping: Mapping,
}

impl BuildTask for MapCodes {
    type Output = Vec<f32>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let samples = self.input.len() as u32;
        let output_bytes = self.input.len() * size_of::<f32>();

        let input = upload_f32(client, &self.input);
        let output = empty(client, output_bytes);
        let input_plane = plane(&input, samples, 1);
        let output_plane = plane(&output, samples, 1);
        match self.mapping {
            Mapping::Codes => float_to_codes(client, &sizing, input_plane, output_plane, self.codes),
            Mapping::WholeCodes => {
                float_to_whole_codes(client, &sizing, input_plane, output_plane, self.codes);
            },
            Mapping::Float => codes_to_float(client, &sizing, input_plane, output_plane, self.codes),
            Mapping::Unit => unit_to_float(client, &sizing, input_plane, output_plane, self.codes),
        }

        read_f32(client, &output)
    }
}

fn mapped(input: Vec<f32>, codes: FloatCodes, mapping: Mapping) -> Vec<f32> {
    let task = MapCodes {
        input,
        codes,
        mapping,
    };

    backend().build(task)
}

fn float_codes(input: Vec<f32>, codes: FloatCodes) -> Vec<f32> {
    mapped(input, codes, Mapping::Codes)
}

fn codes_floats(input: Vec<f32>, codes: FloatCodes) -> Vec<f32> {
    mapped(input, codes, Mapping::Float)
}

#[test]
fn float_samples_land_on_their_code_values() {
    let limited_luma = FloatCodes::new(16, SampleRange::Limited, false);
    let full_chroma = FloatCodes::new(16, SampleRange::Full, true);

    let luma = float_codes(vec![0.0, 1.0, 0.5], limited_luma);
    let chroma = float_codes(vec![-0.5, 0.0], full_chroma);

    assert_close(&luma, &[4096.0, 60160.0, 32128.0], 0.01);
    // Full range chroma spans 65535 codes around 32768, so -0.5 lands half a code above zero.
    assert_close(&chroma, &[0.5, 32768.0], 0.01);
}

#[test]
fn float_samples_outside_the_range_are_held_to_it() {
    let codes = FloatCodes::new(16, SampleRange::Full, false);

    let held = float_codes(vec![-0.25, 1.5, f32::NAN], codes);

    assert_close(&held, &[0.0, 65535.0, 0.0], 0.01);
}

#[test]
fn codes_return_to_their_float_samples_unrounded() {
    let codes = FloatCodes::new(16, SampleRange::Limited, false);

    let samples = codes_floats(vec![4096.5, 70000.0], codes);

    assert!((samples[0] - 0.5 / 56064.0).abs() < 1e-7, "was {}", samples[0]);
    assert!(samples[1] > 1.0, "was {}", samples[1]);
}

#[test]
fn a_round_trip_keeps_every_sixteen_bit_code() {
    let every_code: Vec<f32> = (0..=65535u32).map(|code| code as f32).collect();

    for range in [SampleRange::Limited, SampleRange::Full] {
        for chroma in [false, true] {
            let codes = FloatCodes::new(16, range, chroma);

            let samples = codes_floats(every_code.clone(), codes);
            let returned = float_codes(samples, codes);

            assert_close(&returned, &every_code, 0.01);
        }
    }
}

#[test]
fn whole_codes_round_to_the_nearest_code() {
    let codes = FloatCodes::new(16, SampleRange::Limited, true);
    let spread: Vec<f32> = (0..=65535u32).step_by(7).map(|code| code as f32).collect();
    let samples: Vec<f32> = spread.iter().map(|code| (code - 32768.0) / 57344.0).collect();
    let between = vec![
        (1000.3 - 32768.0) / 57344.0,
        (1000.7 - 32768.0) / 57344.0,
        f32::NAN,
    ];

    let whole = mapped(samples, codes, Mapping::WholeCodes);
    let rounded = mapped(between, codes, Mapping::WholeCodes);

    assert_eq!(whole, spread);
    // A NaN reads as the sample 0, which is chroma's centre.
    assert_eq!(rounded, vec![1000.0, 1001.0, 32768.0]);
}

#[test]
fn unit_floats_land_on_their_float_samples() {
    let luma = FloatCodes::new(8, SampleRange::Limited, false);
    let chroma = FloatCodes::new(8, SampleRange::Limited, true);
    let units = vec![16.0 / 255.0, 235.0 / 255.0, 0.0, 1.0];

    let luma_samples = mapped(units, luma, Mapping::Unit);
    let chroma_samples = mapped(vec![128.0 / 255.0, 16.0 / 255.0], chroma, Mapping::Unit);

    assert_close(&luma_samples, &[0.0, 1.0, -16.0 / 219.0, 239.0 / 219.0], 1e-6);
    assert_close(&chroma_samples, &[0.0, -0.5], 1e-6);
}
