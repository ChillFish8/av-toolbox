use av_toolbox_cubek::motion::{
    BlockGrid,
    CompensatePlanes,
    MotionSettings,
    PlaneScale,
    compensate,
    window_weight,
};
use av_toolbox_cubek::testing::{
    MotionClip,
    motion_clips,
    normalised,
    reference_compensate,
    reference_nlm_mc_warp,
};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::*;

use crate::support::{Sizing, backend, empty, plane, read_f32, upload_f32};

const LUMA: PlaneScale = PlaneScale {
    shift_x: 0,
    shift_y: 0,
};
const CHROMA: PlaneScale = PlaneScale {
    shift_x: 1,
    shift_y: 1,
};

/// How far the device may sit from the CPU on a compensated sample.
const OUTPUT_TOLERANCE: f32 = 1e-3;

/// How far the device may sit from the CPU on a window weight.
const WEIGHT_TOLERANCE: f32 = 1e-6;

/// One plane compensated on the device.
struct Case {
    neighbour: Vec<f32>,
    width: u32,
    height: u32,
    vectors: Vec<i32>,
    grid: BlockGrid,
    pel: u32,
    scale: PlaneScale,
}

impl Case {
    fn new(neighbour: Vec<f32>, size: (u32, u32), vectors: Vec<i32>, grid: BlockGrid) -> Self {
        Self {
            neighbour,
            width: size.0,
            height: size.1,
            vectors,
            grid,
            pel: 1,
            scale: LUMA,
        }
    }

    fn pel(self, pel: u32) -> Self {
        Self { pel, ..self }
    }

    fn scale(self, scale: PlaneScale) -> Self {
        Self { scale, ..self }
    }

    fn reference(&self) -> Vec<f32> {
        let size = (self.width as usize, self.height as usize);

        reference_compensate(
            &self.neighbour,
            size,
            &self.vectors,
            &self.grid,
            self.pel,
            self.scale,
        )
    }
}

/// Compensates every case on the device.
struct CompensateRun {
    cases: Vec<Case>,
}

impl BuildTask for CompensateRun {
    type Output = Vec<Vec<f32>>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let sizing = Sizing(build.frame_context());

        let mut outputs = Vec::new();
        for case in &self.cases {
            let neighbour = upload_f32(client, &case.neighbour);
            let vectors = client.create_from_slice(i32::as_bytes(&case.vectors));
            let samples = (case.width * case.height) as usize;
            let output = empty(client, samples * size_of::<f32>());

            let planes = CompensatePlanes {
                neighbour: plane(&neighbour, case.width, case.height),
                vectors: plane(&vectors, 2 * case.grid.across, case.grid.down),
                output: plane(&output, case.width, case.height),
            };
            compensate(client, &sizing, planes, &case.grid, case.pel, case.scale);

            outputs.push(read_f32(client, &output));
        }

        outputs
    }
}

/// The largest difference seen between a device output and its reference.
fn largest_difference(found: &[f32], expected: &[f32], label: &str, tolerance: f32) -> f32 {
    assert_eq!(found.len(), expected.len(), "{label} has the wrong length");

    let mut largest = 0.0f32;
    for (index, (actual, wanted)) in found.iter().zip(expected).enumerate() {
        let difference = (actual - wanted).abs();
        assert!(
            difference <= tolerance,
            "{label} sample {index} was {actual}, expected {wanted} within {tolerance}"
        );
        largest = largest.max(difference);
    }

    largest
}

/// A vector for each block of `grid` that is varied, negative and odd.
fn varied_vectors(grid: &BlockGrid) -> Vec<i32> {
    let mut vectors = Vec::with_capacity(2 * grid.blocks());
    for by in 0..grid.down {
        for bx in 0..grid.across {
            vectors.push((bx * 7 + by * 3) as i32 % 11 - 5);
            vectors.push((bx * 5 + by * 13) as i32 % 9 - 4);
        }
    }

    vectors
}

fn grid_of(blksize: u32, overlap: u32, width: u32, height: u32) -> BlockGrid {
    let settings = MotionSettings {
        blksize,
        overlap,
        ..MotionSettings::default()
    };

    BlockGrid::new(&settings, width, height)
}

/// The luma and first chroma plane of a clip's frame, with their sizes.
fn clip_planes(clip: &MotionClip, frame: usize) -> Vec<(Vec<f32>, (u32, u32), PlaneScale)> {
    let width = clip.width as u32;
    let height = clip.height as u32;

    vec![
        (
            normalised(&clip.frames[frame][0], clip.bits),
            (width, height),
            LUMA,
        ),
        (
            normalised(&clip.frames[frame][1], clip.bits),
            (width / 2, height / 2),
            CHROMA,
        ),
    ]
}

#[test]
fn every_clip_matches_the_reference_at_every_setting() {
    let settings = [(16, 0), (16, 4), (16, 8), (8, 4)];

    let mut cases = Vec::new();
    for clip in motion_clips() {
        for (neighbour, size, scale) in clip_planes(&clip, 1) {
            for (blksize, overlap) in settings {
                let grid = grid_of(blksize, overlap, clip.width as u32, clip.height as u32);
                let vectors = varied_vectors(&grid);

                for pel in [1, 2] {
                    cases.push(
                        Case::new(neighbour.clone(), size, vectors.clone(), grid)
                            .pel(pel)
                            .scale(scale),
                    );
                }
            }
        }
    }

    let expected: Vec<Vec<f32>> = cases.iter().map(Case::reference).collect();
    let found = backend().build(CompensateRun { cases });

    let mut largest = 0.0f32;
    for (index, (output, wanted)) in found.iter().zip(&expected).enumerate() {
        let label = format!("case {index}");
        largest = largest.max(largest_difference(output, wanted, &label, OUTPUT_TOLERANCE));
    }

    println!(
        "largest device against CPU difference over {} cases: {largest:e}",
        found.len()
    );
}

#[test]
fn zero_overlap_is_the_plain_warp() {
    let clip = motion_clips().remove(2);
    let (neighbour, _, _) = clip_planes(&clip, 1).remove(0);
    let width = clip.width;
    let height = clip.height;

    let mut cases = Vec::new();
    let mut expected = Vec::new();
    for blksize in [16, 8] {
        let grid = grid_of(blksize, 0, width as u32, height as u32);
        let mut vectors = varied_vectors(&grid);
        vectors[0] = 1000;
        vectors[1] = -1000;
        let last = vectors.len() - 2;
        vectors[last] = -1000;
        vectors[last + 1] = 1000;

        let warped = reference_nlm_mc_warp(
            &neighbour,
            (width, height),
            &vectors,
            grid.step,
            (grid.across, grid.down),
        );
        expected.push(warped);
        cases.push(Case::new(
            neighbour.clone(),
            (width as u32, height as u32),
            vectors,
            grid,
        ));
    }

    let found = backend().build(CompensateRun { cases });

    assert_eq!(found, expected, "the compensation differs from the plain warp");
}

#[test]
fn an_edge_block_clamps_its_reads() {
    let (width, height) = (20u32, 20u32);
    let neighbour: Vec<f32> = (0..width * height).map(|index| index as f32 / 400.0).collect();
    let grid = grid_of(16, 0, width, height);
    assert_eq!((grid.across, grid.down), (2, 2));

    let mut vectors = vec![0; 8];
    let last = 6;
    vectors[last] = 10;
    vectors[last + 1] = 10;
    let case = Case::new(neighbour.clone(), (width, height), vectors, grid);
    let found = backend().build(CompensateRun { cases: vec![case] }).remove(0);

    let corner = neighbour[(width * height - 1) as usize];
    for y in 16..height {
        for x in 16..width {
            let index = (y * width + x) as usize;
            assert_eq!(found[index], corner, "sample {x},{y} did not clamp to the corner");
        }
    }
}

#[test]
fn a_vector_past_the_frame_compensates_from_its_edge() {
    let clip = motion_clips().remove(0);
    let (neighbour, size, _) = clip_planes(&clip, 1).remove(0);
    let grid = grid_of(16, 8, size.0, size.1);
    let vectors = [1000, -1000].repeat(grid.blocks());
    let top_right = neighbour[size.0 as usize - 1];

    let cases = [1, 2]
        .into_iter()
        .map(|pel| Case::new(neighbour.clone(), size, vectors.clone(), grid).pel(pel))
        .collect();
    let found = backend().build(CompensateRun { cases });

    for (pel, output) in [1, 2].into_iter().zip(&found) {
        let label = format!("pel {pel}");
        let flat = vec![top_right; output.len()];
        largest_difference(output, &flat, &label, WEIGHT_TOLERANCE);
    }
}

/// The device's rising weight along one axis, read back from a blend of blocks.
///
/// The neighbour is 0 along its first column or row and 1 along its last. Blocks alternate between reading
/// the first and the last, so where the first two overlap the output is the second block's share of the
/// weight. Returns the plane, its width and the grid's step.
fn device_ramp(blksize: u32, overlap: u32, transposed: bool) -> (Vec<f32>, u32) {
    let step = blksize - overlap;
    let length = 3 * step;
    let (width, height) = if transposed {
        (step, length)
    } else {
        (length, step)
    };

    let mut neighbour = vec![0.5f32; (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let along = if transposed { y } else { x };
            let last = length - 1;
            let index = (y * width + x) as usize;
            if along == 0 {
                neighbour[index] = 0.0;
            } else if along == last {
                neighbour[index] = 1.0;
            }
        }
    }

    let grid = grid_of(blksize, overlap, width, height);
    assert_eq!((grid.across, grid.down), if transposed { (1, 3) } else { (3, 1) });

    let mut vectors = Vec::new();
    for block in 0..3 {
        let reach = if block % 2 == 0 { -1000 } else { 1000 };
        let vector = if transposed { [0, reach] } else { [reach, 0] };
        vectors.extend(vector);
    }

    let case = Case::new(neighbour, (width, height), vectors, grid);
    let output = backend().build(CompensateRun { cases: vec![case] }).remove(0);

    (output, step)
}

#[test]
fn the_device_window_matches_the_reference_weights() {
    for transposed in [false, true] {
        let (blksize, overlap) = (16, 8);
        let (output, step) = device_ramp(blksize, overlap, transposed);
        let width = if transposed { step } else { 3 * step } as usize;

        let mut largest = 0.0f32;
        for position in 0..overlap {
            let wanted = window_weight(position, blksize, overlap, false, false);
            let along = (step + position) as usize;
            let index = if transposed { along * width } else { along };
            let found = output[index];

            let difference = (found - wanted).abs();
            assert!(
                difference <= WEIGHT_TOLERANCE,
                "transposed {transposed} position {position} weighs {found}, expected {wanted}"
            );
            largest = largest.max(difference);
        }

        println!("largest window weight difference, transposed {transposed}: {largest:e}");
    }
}
