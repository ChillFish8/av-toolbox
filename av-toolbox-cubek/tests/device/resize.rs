use av_toolbox_cubek::resize::{BILINEAR_SUPPORT, ResizePlanes, Taps, apply, bilinear, taps, upload};
use av_toolbox_cubek::testing::reference_resize;
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use crate::support::{Sizing, backend, empty, plane, read_u8, upload_u8};

struct Resize {
    input: Vec<u8>,
    width: u32,
    height: u32,
    across: Taps,
    down: Taps,
}

impl BuildTask for Resize {
    type Output = Vec<u8>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let out_width = self.across.starts.len() as u32;
        let out_height = self.down.starts.len() as u32;
        let scratch_bytes = (out_width * self.height) as usize * size_of::<f32>();
        let output_bytes = (out_width * out_height) as usize;

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
        apply(client, &sizing, planes, &across, &down);

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

fn check(input: &[u8], width: u32, height: u32, across: Taps, down: Taps) {
    let expected = reference_resize(input, width, height, &across, &down);

    let found = backend().build(Resize {
        input: input.to_vec(),
        width,
        height,
        across,
        down,
    });

    let largest = found
        .iter()
        .zip(&expected)
        .map(|(got, want)| got.abs_diff(*want))
        .max()
        .unwrap_or(0);
    assert_eq!(found.len(), expected.len());
    assert!(largest <= 1, "differed by {largest}");
}

fn check_chroma_shapes(input: &[u8], width: u32, height: u32) {
    for (scale_x, scale_y, offset_x, offset_y) in CHROMA_SHAPES {
        let out_width = width.div_ceil(scale_x as u32);
        let out_height = height.div_ceil(scale_y as u32);
        let across = taps(out_width, scale_x, offset_x, bilinear, BILINEAR_SUPPORT);
        let down = taps(out_height, scale_y, offset_y, bilinear, BILINEAR_SUPPORT);

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
    let across = taps(8, 0.5, 0.0, bilinear, BILINEAR_SUPPORT);
    let down = taps(6, 0.5, 0.0, bilinear, BILINEAR_SUPPORT);

    check(&input, 4, 3, across, down);
}
