use av_toolbox_cubek::depth::Element;
use av_toolbox_cubek::testing::reference_warp_sharp;
use av_toolbox_cubek::warp::{BlurKind, WarpParams, WarpPlanes, warp_sharp};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use crate::support::{Sizing, backend, empty, from_bytes, plane, read_u8, to_bytes, upload_u8};

struct WarpSharp {
    input: Vec<u16>,
    width: u32,
    height: u32,
    params: WarpParams,
}

impl BuildTask for WarpSharp {
    type Output = Vec<u16>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let bytes = to_bytes(&self.input, self.params.element);
        let samples = (self.width * self.height) as usize;
        let mask_bytes = samples * size_of::<u32>();
        let output_bytes = bytes.len();

        let input = upload_u8(client, &bytes);
        let mask = empty(client, mask_bytes);
        let scratch = empty(client, mask_bytes);
        let output = empty(client, output_bytes);
        let input_plane = plane(&input, self.width, self.height);
        let mask_plane = plane(&mask, self.width, self.height);
        let scratch_plane = plane(&scratch, self.width, self.height);
        let output_plane = plane(&output, self.width, self.height);
        let planes = WarpPlanes {
            input: input_plane,
            mask: mask_plane,
            scratch: scratch_plane,
            output: output_plane,
        };

        warp_sharp(client, &sizing, planes, &self.params);

        let read = read_u8(client, &output);
        from_bytes(&read, self.params.element)
    }
}

fn on_device(input: &[u16], width: u32, height: u32, params: WarpParams) -> Vec<u16> {
    let task = WarpSharp {
        input: input.to_vec(),
        width,
        height,
        params,
    };

    backend().build(task)
}

/// A busy plane with edges in every direction, wrapped at `peak`.
fn patterned(width: u32, height: u32, peak: i32) -> Vec<u16> {
    (0..width * height)
        .map(|index| {
            let column = (index % width) as i32;
            let row = (index / width) as i32;
            let value = column * 37 + row * 11 + (column * row) % 17;
            (value % (peak + 1)) as u16
        })
        .collect()
}

fn params(element: Element, bits: u32, blur: BlurKind, depth: i32) -> WarpParams {
    let passes = match blur {
        BlurKind::R6 => 2,
        BlurKind::R2 => 3,
    };

    WarpParams {
        element,
        bits,
        thresh: 128,
        blur,
        passes,
        depth,
    }
}

fn matches_reference(width: u32, height: u32, params: WarpParams) {
    let peak = params.peak();
    let input = patterned(width, height, peak);

    let actual = on_device(&input, width, height, params);
    let expected = reference_warp_sharp(&input, width, height, &params);

    assert_eq!(actual, expected, "{width}x{height} {params:?}");
}

#[test]
fn every_blur_and_depth_matches_the_reference() {
    let depths = [(Element::U8, 8), (Element::U16, 10)];

    for blur in [BlurKind::R6, BlurKind::R2] {
        for (element, bits) in depths {
            for depth in [16, -8] {
                let chosen = params(element, bits, blur, depth);
                matches_reference(40, 24, chosen);
            }
        }
    }
}

#[test]
fn small_planes_clamp_every_read() {
    for blur in [BlurKind::R6, BlurKind::R2] {
        let chosen = params(Element::U8, 8, blur, 16);

        for (width, height) in [(5, 4), (1, 7), (13, 2), (11, 11), (1, 1)] {
            matches_reference(width, height, chosen);
        }
    }
}

/// A hard step, 0 in the left half and 65535 in the right half.
fn stepped(width: u32, height: u32) -> Vec<u16> {
    (0..width * height)
        .map(|index| {
            let column = index % width;

            if column < width / 2 { 0 } else { u16::MAX }
        })
        .collect()
}

#[test]
fn sixteen_bit_extremes_match_the_reference() {
    let input = stepped(40, 24);
    let mut moved_something = false;

    for depth in [-128, 127] {
        let chosen = WarpParams {
            thresh: 255,
            ..params(Element::U16, 16, BlurKind::R6, depth)
        };

        let actual = on_device(&input, 40, 24, chosen);
        let expected = reference_warp_sharp(&input, 40, 24, &chosen);

        assert_eq!(actual, expected, "40x24 {chosen:?}");

        if actual != input {
            moved_something = true;
        }
    }

    assert!(moved_something, "neither depth moved a sample off the input");
}

#[test]
fn a_zero_threshold_leaves_all_but_the_last_row_alone() {
    let input = patterned(40, 24, 255);
    let chosen = WarpParams {
        thresh: 0,
        ..params(Element::U8, 8, BlurKind::R6, 16)
    };

    let warped = on_device(&input, 40, 24, chosen);
    let expected = reference_warp_sharp(&input, 40, 24, &chosen);

    assert_eq!(warped, expected);

    let last_row_start = 40 * 23;
    assert_eq!(warped[..last_row_start], input[..last_row_start]);
}

#[test]
fn a_flat_plane_is_unchanged() {
    let input = vec![300u16; 40 * 24];
    let chosen = params(Element::U16, 10, BlurKind::R6, 16);

    let warped = on_device(&input, 40, 24, chosen);

    assert_eq!(warped, input);
}
