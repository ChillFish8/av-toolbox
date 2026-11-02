use av_toolbox_cubek::depth::Element;
use av_toolbox_cubek::testing::{reference_blur_mask, reference_warp_sharp};
use av_toolbox_cubek::warp::{BlurKind, WarpParams, WarpPlanes, blur_mask, blur_mask_passes, warp_sharp};
use av_toolbox_gpu::{BuildContext, BuildTask};

use crate::support::{
    Sizing,
    backend,
    empty,
    from_bytes,
    plane,
    read_u8,
    read_u32,
    seeded_plane,
    to_bytes,
    upload_u8,
    upload_u32,
};

struct WarpSharp {
    input: Vec<u16>,
    width: u32,
    height: u32,
    params: WarpParams,
}

impl BuildTask for WarpSharp {
    type Output = Vec<u16>;

    fn run(self, build: &BuildContext) -> Self::Output {
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

#[derive(Debug, Clone, Copy)]
enum BlurCall {
    /// One [blur_mask] call, read back from the mask.
    Once,
    /// One [blur_mask_passes] call, read back from the plane it hands back.
    Passes(u32),
}

struct BlurCase {
    mask: Vec<u32>,
    width: u32,
    height: u32,
    kind: BlurKind,
    call: BlurCall,
}

struct BlurMasks(Vec<BlurCase>);

impl BuildTask for BlurMasks {
    type Output = Vec<Vec<u32>>;

    fn run(self, build: &BuildContext) -> Self::Output {
        let client = build.client();
        let sizing = Sizing(build.frame_context());
        let mut blurred = Vec::with_capacity(self.0.len());

        for case in self.0 {
            let mask = upload_u32(client, &case.mask);
            let scratch = empty(client, case.mask.len() * size_of::<u32>());
            let mask_plane = plane(&mask, case.width, case.height);
            let scratch_plane = plane(&scratch, case.width, case.height);

            let result = match case.call {
                BlurCall::Once => {
                    blur_mask(client, &sizing, mask_plane, scratch_plane, case.kind);
                    mask_plane
                },
                BlurCall::Passes(passes) => {
                    blur_mask_passes(client, &sizing, mask_plane, scratch_plane, case.kind, passes)
                },
            };

            let read = read_u32(client, result.handle());
            blurred.push(read);
        }

        blurred
    }
}

fn seeded_mask(width: u32, height: u32) -> Vec<u32> {
    seeded_plane(width, height)
        .iter()
        .map(|value| (value * 65535.0) as u32)
        .collect()
}

/// Checks every pass count in `pass_counts`, in rising order, against the reference at each size.
fn passes_match_the_reference(sizes: &[(u32, u32)], pass_counts: &[u32]) {
    let mut cases = Vec::new();
    let mut expected = Vec::new();

    for kind in [BlurKind::R2, BlurKind::R6] {
        for &(width, height) in sizes {
            let mask = seeded_mask(width, height);
            let mut blurred = mask.clone();
            let mut done = 0;

            for &passes in pass_counts {
                for _ in done..passes {
                    blurred = reference_blur_mask(&blurred, width, height, kind);
                }

                done = passes;

                let case = BlurCase {
                    mask: mask.clone(),
                    width,
                    height,
                    kind,
                    call: BlurCall::Passes(passes),
                };
                cases.push(case);
                expected.push(blurred.clone());
            }
        }
    }

    let labels: Vec<String> = cases
        .iter()
        .map(|case| format!("{}x{} {:?} {:?}", case.width, case.height, case.kind, case.call))
        .collect();
    let actual = backend().build(BlurMasks(cases));

    for ((actual, expected), label) in actual.iter().zip(&expected).zip(&labels) {
        assert!(actual == expected, "{label} differs from the reference");
    }
}

#[test]
fn every_pass_count_matches_the_reference_bit_for_bit() {
    let small = [
        (257, 131),
        (1, 7),
        (7, 1),
        (13, 13),
        (1, 1),
        (2, 3),
        (5, 2),
        (64, 64),
        (53, 41),
    ];
    passes_match_the_reference(&small, &[1, 2, 3, 4, 5, 100]);
    passes_match_the_reference(&[(1921, 1081)], &[1, 2, 3]);
}

#[test]
fn a_plane_wider_than_the_grid_matches_the_reference() {
    passes_match_the_reference(&[(4_194_311, 2), (2, 4_194_311)], &[1, 3]);
}

#[test]
fn one_blur_mask_lands_in_the_mask() {
    let mut cases = Vec::new();
    let mut expected = Vec::new();

    for kind in [BlurKind::R2, BlurKind::R6] {
        let mask = seeded_mask(37, 23);
        let blurred = reference_blur_mask(&mask, 37, 23, kind);
        let case = BlurCase {
            mask,
            width: 37,
            height: 23,
            kind,
            call: BlurCall::Once,
        };
        cases.push(case);
        expected.push(blurred);
    }

    let actual = backend().build(BlurMasks(cases));

    assert!(actual == expected, "a blur_mask call differs from the reference");
}

/// Blocks of 5 by 4 samples, alternating 40 and 200, so the edge mask varies and every blur pass moves it.
fn checkered(width: u32, height: u32) -> Vec<u16> {
    (0..width * height)
        .map(|index| {
            let block_across = index % width / 5;
            let block_down = index / width / 4;

            if (block_across + block_down).is_multiple_of(2) {
                40
            } else {
                200
            }
        })
        .collect()
}

#[test]
fn every_pass_count_warps_from_the_blurred_mask() {
    let input = checkered(37, 23);

    for blur in [BlurKind::R2, BlurKind::R6] {
        for passes in 0..=3 {
            let chosen = WarpParams {
                passes,
                thresh: 255,
                ..params(Element::U8, 8, blur, 16)
            };

            let actual = on_device(&input, 37, 23, chosen);
            let expected = reference_warp_sharp(&input, 37, 23, &chosen);

            assert_eq!(actual, expected, "37x23 {chosen:?}");
        }
    }
}
