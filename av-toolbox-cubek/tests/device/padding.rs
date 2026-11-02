use av_toolbox_cubek::padding::{crop, pad};
use av_toolbox_cubek::testing::{reference_crop, reference_pad};
use av_toolbox_gpu::{BuildContext, BuildTask};

use crate::support::{Sizing, assert_close, backend, empty, plane, read_f32, upload_f32};

struct Pad {
    input: Vec<f32>,
    width: u32,
    height: u32,
    amount: u32,
}

impl BuildTask for Pad {
    type Output = Vec<f32>;

    fn run(self, build: &BuildContext) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let output_width = self.width + 2 * self.amount;
        let output_height = self.height + 2 * self.amount;
        let output_bytes = output_width as usize * output_height as usize * size_of::<f32>();

        let input = upload_f32(client, &self.input);
        let output = empty(client, output_bytes);
        let input_plane = plane(&input, self.width, self.height);
        let output_plane = plane(&output, output_width, output_height);
        pad(client, &sizing, input_plane, output_plane, self.amount);

        read_f32(client, &output)
    }
}

struct Crop {
    input: Vec<f32>,
    width: u32,
    height: u32,
    amount: u32,
}

impl BuildTask for Crop {
    type Output = Vec<f32>;

    fn run(self, build: &BuildContext) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let output_width = self.width - 2 * self.amount;
        let output_height = self.height - 2 * self.amount;
        let output_bytes = output_width as usize * output_height as usize * size_of::<f32>();

        let input = upload_f32(client, &self.input);
        let output = empty(client, output_bytes);
        let input_plane = plane(&input, self.width, self.height);
        let output_plane = plane(&output, output_width, output_height);
        crop(client, &sizing, input_plane, output_plane, self.amount);

        read_f32(client, &output)
    }
}

fn check_pad(input: Vec<f32>, width: u32, height: u32, amount: u32) {
    let expected = reference_pad(&input, width, height, amount);

    let padded = backend().build(Pad {
        input,
        width,
        height,
        amount,
    });

    assert_close(&padded, &expected, 0.0);
}

fn check_crop(input: Vec<f32>, width: u32, height: u32, amount: u32) {
    let expected = reference_crop(&input, width, height, amount);

    let cropped = backend().build(Crop {
        input,
        width,
        height,
        amount,
    });

    assert_close(&cropped, &expected, 0.0);
}

#[test]
fn padding_by_six_matches_the_reference() {
    let big: Vec<f32> = (0..37 * 23).map(|index| index as f32).collect();
    let single = vec![7.0];
    let small: Vec<f32> = (0..2 * 5).map(|index| index as f32).collect();

    check_pad(big, 37, 23, 6);
    check_pad(single, 1, 1, 6);
    check_pad(small, 2, 5, 6);
}

#[test]
fn cropping_by_six_matches_the_reference() {
    let big: Vec<f32> = (0..49 * 35).map(|index| index as f32).collect();
    let single = vec![7.0; 13 * 13];
    let small: Vec<f32> = (0..14 * 17).map(|index| index as f32).collect();

    check_crop(big, 49, 35, 6);
    check_crop(single, 13, 13, 6);
    check_crop(small, 14, 17, 6);
}
