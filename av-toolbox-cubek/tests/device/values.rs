use av_toolbox_cubek::depth::Element;
use av_toolbox_cubek::testing::reference_narrow;
use av_toolbox_cubek::values::{narrow, offset, widen};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use crate::support::{
    Sizing,
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
