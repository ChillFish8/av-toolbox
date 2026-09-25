use av_toolbox_cubek::sobel::{SobelPlanes, sobel};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use crate::support::{Sizing, assert_close, backend, empty, plane, read_f32, reflect, upload_f32};

fn reference_sobel(input: &[f32], width: u32, height: u32) -> (Vec<f32>, Vec<f32>) {
    let width = width as i32;
    let height = height as i32;
    let sample = |column: i32, row: i32| {
        let x = reflect(column, width);
        let y = reflect(row, height);
        input[(y * width + x) as usize]
    };

    let mut horizontal = Vec::with_capacity(input.len());
    let mut vertical = Vec::with_capacity(input.len());
    for row in 0..height {
        for column in 0..width {
            let right =
                sample(column + 1, row - 1) + 2.0 * sample(column + 1, row) + sample(column + 1, row + 1);
            let left =
                sample(column - 1, row - 1) + 2.0 * sample(column - 1, row) + sample(column - 1, row + 1);
            let below =
                sample(column - 1, row + 1) + 2.0 * sample(column, row + 1) + sample(column + 1, row + 1);
            let above =
                sample(column - 1, row - 1) + 2.0 * sample(column, row - 1) + sample(column + 1, row - 1);
            horizontal.push(right - left);
            vertical.push(below - above);
        }
    }

    (horizontal, vertical)
}

struct Gradients {
    input: Vec<f32>,
    width: u32,
    height: u32,
}

impl BuildTask for Gradients {
    type Output = (Vec<f32>, Vec<f32>);

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let bytes = self.input.len() * size_of::<f32>();

        let input = upload_f32(client, &self.input);
        let horizontal = empty(client, bytes);
        let vertical = empty(client, bytes);
        let input_plane = plane(&input, self.width, self.height);
        let horizontal_plane = plane(&horizontal, self.width, self.height);
        let vertical_plane = plane(&vertical, self.width, self.height);
        let planes = SobelPlanes {
            input: input_plane,
            horizontal: horizontal_plane,
            vertical: vertical_plane,
        };
        sobel(client, &sizing, planes);

        let across = read_f32(client, &horizontal);
        let down = read_f32(client, &vertical);

        (across, down)
    }
}

fn check(input: Vec<f32>, width: u32, height: u32) -> (Vec<f32>, Vec<f32>) {
    let (expected_across, expected_down) = reference_sobel(&input, width, height);

    let (across, down) = backend().build(Gradients { input, width, height });

    assert_close(&across, &expected_across, 1e-5);
    assert_close(&down, &expected_down, 1e-5);

    (across, down)
}

#[test]
fn a_vertical_step_has_only_a_horizontal_gradient() {
    let input: Vec<f32> = (0..9 * 7)
        .map(|index| if index % 9 < 4 { 0.0 } else { 1.0 })
        .collect();

    let (across, down) = check(input, 9, 7);

    assert!(down.iter().all(|value| *value == 0.0));
    assert_eq!(across[3], 4.0);
}

#[test]
fn a_ramp_of_odd_size_matches_the_reference() {
    let input: Vec<f32> = (0..35u32)
        .map(|index| {
            let x = (index % 7) as f32;
            let y = (index / 7) as f32;
            (x * x + 3.0 * y) / 50.0
        })
        .collect();

    check(input, 7, 5);
}
