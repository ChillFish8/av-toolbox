use av_toolbox_cubek::neighbourhood::{Extreme, NeighbourPlanes, Shape, Smooth, extreme, smooth};
use av_toolbox_cubek::testing::{reference_extreme, reference_smooth};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use crate::support::{Sizing, assert_close, backend, empty, plane, read_f32, upload_f32};

/// Which neighbourhood operation a device run performs.
#[derive(Debug, Clone, Copy)]
pub enum Operation {
    Extreme(Extreme, Shape),
    Smooth(Smooth),
}

struct Neighbours {
    input: Vec<f32>,
    width: u32,
    height: u32,
    operation: Operation,
}

impl BuildTask for Neighbours {
    type Output = Vec<f32>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let bytes = self.input.len() * size_of::<f32>();

        let input = upload_f32(client, &self.input);
        let output = empty(client, bytes);
        let input_plane = plane(&input, self.width, self.height);
        let output_plane = plane(&output, self.width, self.height);
        let planes = NeighbourPlanes {
            input: input_plane,
            output: output_plane,
        };

        match self.operation {
            Operation::Extreme(which, shape) => extreme(client, &sizing, planes, which, shape),
            Operation::Smooth(which) => smooth(client, &sizing, planes, which),
        }

        read_f32(client, &output)
    }
}

pub fn on_device(input: &[f32], width: u32, height: u32, operation: Operation) -> Vec<f32> {
    let task = Neighbours {
        input: input.to_vec(),
        width,
        height,
        operation,
    };

    backend().build(task)
}

/// A busy integer-valued plane.
pub fn busy(width: u32, height: u32) -> Vec<f32> {
    (0..width * height)
        .map(|index| {
            let column = index % width;
            let row = index / width;
            ((column * 37 + row * 11 + (column * row) % 17) % 256) as f32
        })
        .collect()
}

#[test]
fn every_extreme_and_shape_matches_the_reference() {
    let shapes = [Shape::Square, Shape::Plus, Shape::Horizontal, Shape::Vertical];

    for which in [Extreme::Maximum, Extreme::Minimum] {
        for shape in shapes {
            for (width, height) in [(37, 23), (1, 5), (4, 1)] {
                let input = busy(width, height);
                let operation = Operation::Extreme(which, shape);

                let actual = on_device(&input, width, height, operation);
                let expected = reference_extreme(&input, width, height, which, shape);

                assert_eq!(actual, expected, "{which:?} {shape:?} {width}x{height}");
            }
        }
    }
}

#[test]
fn both_blurs_match_the_reference() {
    for which in [Smooth::Binomial, Smooth::Box] {
        let input = busy(37, 23);
        let operation = Operation::Smooth(which);

        let actual = on_device(&input, 37, 23, operation);
        let expected = reference_smooth(&input, 37, 23, which);

        assert_close(&actual, &expected, 1e-3);
    }
}
