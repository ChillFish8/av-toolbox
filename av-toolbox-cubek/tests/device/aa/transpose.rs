use av_toolbox_cubek::aa::transpose;
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use crate::aa::transposed_plane;
use crate::support::{Sizing, backend, empty, plane, read_f32, upload_f32};

struct Transpose {
    input: Vec<f32>,
    width: u32,
    height: u32,
}

impl BuildTask for Transpose {
    type Output = (Vec<f32>, Vec<f32>);

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let bytes = self.input.len() * size_of::<f32>();

        let input = upload_f32(client, &self.input);
        let output = empty(client, bytes);
        let round_trip = empty(client, bytes);
        let input_plane = plane(&input, self.width, self.height);
        let output_plane = plane(&output, self.height, self.width);
        let round_trip_plane = plane(&round_trip, self.width, self.height);

        transpose(client, &sizing, input_plane, output_plane);
        transpose(client, &sizing, output_plane, round_trip_plane);
        cubecl::future::block_on(client.sync()).expect("the device should finish");

        (read_f32(client, &output), read_f32(client, &round_trip))
    }
}

fn on_device(input: &[f32], width: u32, height: u32) -> (Vec<f32>, Vec<f32>) {
    let task = Transpose {
        input: input.to_vec(),
        width,
        height,
    };

    backend().build(task)
}

#[test]
fn transpose_swaps_axes() {
    let width = 7;
    let height = 3;
    let mut input = Vec::new();
    for y in 0..height {
        for x in 0..width {
            input.push((x + 10 * y) as f32);
        }
    }

    let (transposed, round_trip) = on_device(&input, width as u32, height as u32);

    let expected = transposed_plane(&input, width as u32, height as u32);
    assert_eq!(transposed, expected);
    assert_eq!(round_trip, input);
}

#[test]
fn transpose_handles_a_single_row_and_column() {
    let input = [1.0, 2.0, 3.0, 4.0, 5.0];

    let (row_transposed, row_round_trip) = on_device(&input, 5, 1);
    let (column_transposed, column_round_trip) = on_device(&input, 1, 5);

    assert_eq!(row_transposed, input);
    assert_eq!(row_round_trip, input);
    assert_eq!(column_transposed, input);
    assert_eq!(column_round_trip, input);
}

#[test]
fn transpose_matches_the_cpu_at_sizes_off_the_tile() {
    let sizes = [
        (1, 1),
        (1, 77),
        (77, 1),
        (31, 33),
        (32, 32),
        (33, 31),
        (64, 65),
        (100, 37),
        (1000, 3),
    ];
    for (width, height) in sizes {
        let input: Vec<f32> = (0..width * height).map(|value| value as f32).collect();
        let expected = transposed_plane(&input, width, height);

        let (found, round_trip) = on_device(&input, width, height);

        assert_eq!(found, expected, "{width}x{height}");
        assert_eq!(round_trip, input, "{width}x{height} round trip");
    }
}

#[test]
fn transpose_matches_the_cpu_on_a_plane_too_wide_to_tile() {
    let width = 2_097_121;
    let input: Vec<f32> = (0..width).map(|value| value as f32).collect();
    let expected = transposed_plane(&input, width, 1);

    let (found, round_trip) = on_device(&input, width, 1);

    assert!(found == expected, "the transpose differs from the CPU");
    assert!(round_trip == input, "the round trip differs from the input");
}
