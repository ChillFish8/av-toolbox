use av_toolbox_cubek::deinterlace::qtgmc::{MAX_AVERAGED, tr0_weights, weighted_sum};
use av_toolbox_cubek::testing::{QtgmcFixture, assert_plane_within, qtgmc_fixture, reference_tr0};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use super::{DEVICE_TOLERANCE, FIXTURE_TOLERANCE, assert_fixture_within, codes, peak};
use crate::support::{Sizing, backend, empty, plane, read_f32, upload_f32};

/// The luma plane of every bob frame of a clip.
fn luma_bobs(fixture: &QtgmcFixture) -> Vec<Vec<f32>> {
    fixture
        .frames
        .iter()
        .map(|planes| codes(&planes[0].samples))
        .collect()
}

fn bobs_by_clip() -> Vec<(QtgmcFixture, Vec<Vec<f32>>)> {
    qtgmc_fixture("bob", "out")
        .into_iter()
        .map(|fixture| {
            let bobs = luma_bobs(&fixture);

            (fixture, bobs)
        })
        .collect()
}

/// The bobs -2..=2 around `centre`, the nearest repeated at either end of the clip.
fn window_around(bobs: &[Vec<f32>], centre: usize) -> Vec<Vec<f32>> {
    let last = bobs.len() as isize - 1;

    (-2..=2isize)
        .map(|offset| bobs[(centre as isize + offset).clamp(0, last) as usize].clone())
        .collect()
}

/// Bobs weighed into one plane on the device.
struct AverageRun {
    window: Vec<Vec<f32>>,
    weights: [f32; MAX_AVERAGED],
    width: u32,
    height: u32,
}

impl BuildTask for AverageRun {
    type Output = Vec<f32>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let sizing = Sizing(build.frame_context());

        let handles: Vec<_> = self.window.iter().map(|bob| upload_f32(client, bob)).collect();
        let planes: Vec<_> = handles
            .iter()
            .map(|handle| plane(handle, self.width, self.height))
            .collect();
        let output = empty(client, self.window[0].len() * size_of::<f32>());

        let output_plane = plane(&output, self.width, self.height);
        weighted_sum(client, &sizing, &planes, &self.weights, output_plane);

        cubecl::future::block_on(client.sync()).expect("the device should finish");

        read_f32(client, &output)
    }
}

#[test]
fn tr0_matches_havsfunc() {
    for (tr0, name) in [(1, "tr1"), (2, "tr2")] {
        let expected = qtgmc_fixture("average", name);

        for ((fixture, bobs), wanted) in bobs_by_clip().iter().zip(&expected) {
            let ceiling = peak(fixture);
            let (width, height) = (fixture.frames[0][0].width, fixture.frames[0][0].height);

            for centre in 0..bobs.len() {
                let found = reference_tr0(bobs, centre, tr0, width, height);
                let target = &wanted.frames[centre][0].samples;

                let label = format!("{} tr0 {tr0} centre {centre}", fixture.clip);
                assert_fixture_within(&found, target, ceiling, |_| FIXTURE_TOLERANCE, &label);
            }
        }
    }
}

#[test]
fn the_device_weighted_sum_matches_the_reference() {
    for (fixture, bobs) in bobs_by_clip() {
        let (width, height) = (fixture.frames[0][0].width, fixture.frames[0][0].height);

        for tr0 in 0..=2 {
            for centre in 0..bobs.len() {
                let expected = reference_tr0(&bobs, centre, tr0, width, height);
                let task = AverageRun {
                    window: window_around(&bobs, centre),
                    weights: tr0_weights(tr0),
                    width: width as u32,
                    height: height as u32,
                };

                let found = backend().build(task);

                let label = format!("{} tr0 {tr0} centre {centre}", fixture.clip);
                assert_plane_within(&found, &expected, DEVICE_TOLERANCE, width as u32, &label);
            }
        }
    }
}

#[test]
fn edge_bobs_repeat_the_nearest() {
    let (fixture, bobs) = bobs_by_clip().remove(0);
    let (width, height) = (fixture.frames[0][0].width, fixture.frames[0][0].height);

    let found = reference_tr0(&bobs, 0, 2, width, height);

    let mut padded = vec![bobs[0].clone(), bobs[0].clone()];
    padded.extend(bobs.iter().cloned());
    let expected = reference_tr0(&padded, 2, 2, width, height);

    assert_plane_within(&found, &expected, 0.0, width as u32, "centre 0 at tr0 2");
}

/// The bob before the cut of `cut_8`. The clip cuts at its frame 4 and the fixture starts at frame 2, so
/// the cut sits between input frames 1 and 2, which is between bobs 3 and 4.
const BEFORE_CUT: usize = 3;

/// The sum of absolute differences between two bobs.
fn bob_difference(first: &[f32], second: &[f32]) -> f32 {
    first
        .iter()
        .zip(second)
        .map(|(first_sample, second_sample)| (first_sample - second_sample).abs())
        .sum()
}

/// The index of the bob whose successor differs from it the most.
fn largest_jump_after(bobs: &[Vec<f32>]) -> usize {
    let jump = |index: usize| bob_difference(&bobs[index], &bobs[index + 1]);

    (0..bobs.len() - 1)
        .max_by(|first, second| jump(*first).total_cmp(&jump(*second)))
        .expect("the clip has bobs")
}

#[test]
fn tr0_averages_across_a_cut_as_qtgmc_does() {
    let (fixture, bobs) = bobs_by_clip()
        .into_iter()
        .find(|(fixture, _)| fixture.clip == "cut_8")
        .expect("cut_8 is a fixture clip");
    let (width, height) = (fixture.frames[0][0].width, fixture.frames[0][0].height);

    assert_eq!(
        largest_jump_after(&bobs),
        BEFORE_CUT,
        "the cut is not where the fixture puts it"
    );

    let found = reference_tr0(&bobs, BEFORE_CUT, 1, width, height);

    let blended: Vec<f32> = (0..width * height)
        .map(|position| {
            let earlier = bobs[BEFORE_CUT - 1][position];
            let centre = bobs[BEFORE_CUT][position];
            let later = bobs[BEFORE_CUT + 1][position];

            0.25 * earlier + 0.5 * centre + 0.25 * later
        })
        .collect();

    assert_plane_within(
        &found,
        &blended,
        DEVICE_TOLERANCE,
        width as u32,
        "the blend across the cut",
    );
}

#[test]
fn tr0_weights_sum_to_one() {
    for tr0 in 0..=2 {
        let total: f32 = tr0_weights(tr0).iter().sum();

        assert!((total - 1.0).abs() < 1e-6, "tr0 {tr0} weights sum to {total}");
    }
}
