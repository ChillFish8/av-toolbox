use av_toolbox_cubek::depth::Element;
use av_toolbox_cubek::words::copy_plane;
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use crate::support::{Sizing, backend, from_bytes, plane, read_u8, to_bytes, upload_u8};

struct Copy {
    input: Vec<u16>,
    output_samples: usize,
    element: Element,
}

impl BuildTask for Copy {
    type Output = Vec<u16>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let sample_bytes = match self.element {
            Element::U8 => 1,
            Element::U16 => 2,
        };

        let bytes = to_bytes(&self.input, self.element);
        let input = upload_u8(client, &bytes);

        // Filled with a marker first, so a sample the kernel skipped shows up.
        let marker = vec![0xAB; self.output_samples * sample_bytes];
        let output = upload_u8(client, &marker);

        let input_samples = self.input.len() as u32;
        let output_samples = self.output_samples as u32;
        let input_plane = plane(&input, input_samples, 1);
        let output_plane = plane(&output, output_samples, 1);
        copy_plane(client, &sizing, input_plane, output_plane, self.element);

        let read = read_u8(client, &output);
        from_bytes(&read, self.element)
    }
}

fn copied(input: Vec<u16>, output_samples: usize, element: Element) -> Vec<u16> {
    let task = Copy {
        input,
        output_samples,
        element,
    };

    backend().build(task)
}

#[test]
fn padding_a_u8_plane_zeroes_the_tail() {
    let input: Vec<u16> = (1..=9).collect();

    let output = copied(input.clone(), 12, Element::U8);

    assert_eq!(&output[..9], &input[..]);
    assert_eq!(&output[9..], &[0, 0, 0]);
}

#[test]
fn padding_a_u16_plane_zeroes_the_tail() {
    let input: Vec<u16> = vec![1023, 7, 512];

    let output = copied(input.clone(), 4, Element::U16);

    assert_eq!(output, vec![1023, 7, 512, 0]);
}

#[test]
fn unpadding_keeps_only_the_leading_samples() {
    let input: Vec<u16> = (10..22).collect();

    let output = copied(input.clone(), 9, Element::U8);

    assert_eq!(output, input[..9].to_vec());
}
