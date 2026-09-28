use av_toolbox_cubek::hysteresis::{HysteresisPlanes, hysteresis};
use av_toolbox_cubek::labels::Settled;
use av_toolbox_cubek::testing::reference_hysteresis;
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use crate::neighbourhood::busy;
use crate::support::{Sizing, backend, empty, plane, read_f32, upload_f32};

/// What running hysteresis on the device gave.
struct Hysteresised {
    output: Vec<f32>,
    settled: Settled,
}

/// Runs hysteresis with `seeds_from` and `region` on the device.
struct Hysteresis {
    seeds_from: Vec<f32>,
    region: Vec<f32>,
    width: u32,
    height: u32,
    peak: f32,
}

impl BuildTask for Hysteresis {
    type Output = Hysteresised;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let samples = self.region.len();

        let seeds_from = upload_f32(client, &self.seeds_from);
        let region = upload_f32(client, &self.region);
        let marks = empty(client, samples);
        let seeds = empty(client, samples);
        let labels = empty(client, samples * size_of::<u32>());
        let changed = empty(client, size_of::<u32>());
        let counts = empty(client, samples * size_of::<u32>());
        let kept = empty(client, samples);
        let output = empty(client, samples * size_of::<f32>());

        let planes = HysteresisPlanes {
            seeds_from: plane(&seeds_from, self.width, self.height),
            region: plane(&region, self.width, self.height),
            marks: plane(&marks, self.width, self.height),
            seeds: plane(&seeds, self.width, self.height),
            labels: plane(&labels, self.width, self.height),
            changed: plane(&changed, 1, 1),
            counts: plane(&counts, self.width, self.height),
            kept: plane(&kept, self.width, self.height),
            output: plane(&output, self.width, self.height),
        };
        let settled = hysteresis(client, &sizing, planes, self.peak)
            .expect("the device should hand back the changed flag");

        let output = read_f32(client, &output);

        Hysteresised { output, settled }
    }
}

fn run(seeds_from: Vec<f32>, region: Vec<f32>, width: u32, height: u32, peak: f32) -> Hysteresised {
    let task = Hysteresis {
        seeds_from,
        region,
        width,
        height,
        peak,
    };

    backend().build(task)
}

const SCENE_WIDTH: u32 = 40;
const SCENE_HEIGHT: u32 = 24;

/// A region with a border touching every edge, a long snaking path through the interior joined
/// into one piece by a connector at every turn, several small blobs from a thresholded busy plane,
/// and one hand-placed component that never touches a seed. The border and the snake carry a seed,
/// the blobs and the hand-placed component do not.
fn scene() -> (Vec<f32>, Vec<f32>, usize, usize) {
    let samples = (SCENE_WIDTH * SCENE_HEIGHT) as usize;
    let mut region = vec![0.0f32; samples];
    let mut seeds_from = vec![0.0f32; samples];

    let noise = busy(SCENE_WIDTH, SCENE_HEIGHT);
    for row in 2..SCENE_HEIGHT - 2 {
        for column in 16..SCENE_WIDTH - 4 {
            let index = (row * SCENE_WIDTH + column) as usize;
            if noise[index] > 220.0 {
                region[index] = 1.0;
            }
        }
    }

    for column in 0..SCENE_WIDTH {
        region[column as usize] = 1.0;
        region[((SCENE_HEIGHT - 1) * SCENE_WIDTH + column) as usize] = 1.0;
    }

    for row in 0..SCENE_HEIGHT {
        region[(row * SCENE_WIDTH) as usize] = 1.0;
        region[(row * SCENE_WIDTH + SCENE_WIDTH - 1) as usize] = 1.0;
    }
    seeds_from[0] = 1.0;

    let mut column = 2i32;
    let mut row = 2i32;
    let mut direction = 1i32;
    let mut far_end = (2 * SCENE_WIDTH + 2) as usize;
    while row < SCENE_HEIGHT as i32 - 2 {
        let index = (row * SCENE_WIDTH as i32 + column) as usize;
        region[index] = 1.0;

        let at_right = direction == 1 && column >= 12;
        let at_left = direction == -1 && column <= 2;
        if at_right || at_left {
            let connector_index = ((row + 1) * SCENE_WIDTH as i32 + column) as usize;
            region[connector_index] = 1.0;
            far_end = connector_index;

            row += 2;
            direction = -direction;
        } else {
            column += direction;
        }
    }

    seeds_from[(2 * SCENE_WIDTH + 2) as usize] = 1.0;

    let dropped_component = (10 * SCENE_WIDTH + 37) as usize;
    let dropped_seed = (10 * SCENE_WIDTH + 38) as usize;
    region[dropped_component] = 1.0;
    seeds_from[dropped_seed] = 1.0;

    (seeds_from, region, far_end, dropped_component)
}

#[test]
fn hysteresis_matches_the_reference() {
    let peak = 255.0;
    let (seeds_from, region, far_end, dropped_component) = scene();
    let seeds_from_input = seeds_from.clone();
    let region_input = region.clone();

    let hysteresised = run(seeds_from_input, region_input, SCENE_WIDTH, SCENE_HEIGHT, peak);

    let expected = reference_hysteresis(&seeds_from, &region, SCENE_WIDTH, SCENE_HEIGHT, peak);
    assert_eq!(hysteresised.output, expected);
    assert!(matches!(hysteresised.settled, Settled::Converged { .. }));

    let kept = expected.iter().filter(|value| **value == peak).count();
    assert!(kept > 0, "no component was kept");
    assert_eq!(expected[far_end], peak, "the far end of the snake should be kept");
    assert_eq!(
        expected[dropped_component], 0.0,
        "the hand-placed component should be dropped"
    );
}

#[test]
fn hysteresis_matches_the_reference_on_a_1x7_plane() {
    let peak = 255.0;
    let region = vec![1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0];
    let seeds_from = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let seeds_from_input = seeds_from.clone();
    let region_input = region.clone();

    let hysteresised = run(seeds_from_input, region_input, 1, 7, peak);

    let expected = reference_hysteresis(&seeds_from, &region, 1, 7, peak);
    assert_eq!(hysteresised.output, expected);
}

#[test]
fn hysteresis_matches_the_reference_on_a_2x2_plane() {
    let peak = 255.0;
    let region = vec![1.0, 1.0, 1.0, 1.0];
    let seeds_from = vec![0.0, 0.0, 0.0, 1.0];
    let seeds_from_input = seeds_from.clone();
    let region_input = region.clone();

    let hysteresised = run(seeds_from_input, region_input, 2, 2, peak);

    let expected = reference_hysteresis(&seeds_from, &region, 2, 2, peak);
    assert_eq!(hysteresised.output, expected);
}
