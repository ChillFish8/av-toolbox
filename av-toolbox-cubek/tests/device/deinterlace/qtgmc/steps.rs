use av_toolbox_cubek::deinterlace::qtgmc::{
    BackBlendPlanes,
    BobWeights,
    Parity,
    QtgmcTables,
    SearchPlanes,
    StepScratch,
    back_blend,
    havsfunc_scale,
    search_clip_from_repaired,
    smode2_resharpen,
    tr0_taps,
    tr0_weights,
};
use av_toolbox_cubek::testing::{
    assert_plane_within,
    qtgmc_fixture,
    reference_back_blend,
    reference_resharpen,
    reference_search_clip,
    reference_search_inputs,
};
use av_toolbox_gpu::{BuildContext, BuildTask};

use super::pointwise::TIE_GAP;
use super::{DEVICE_TOLERANCE, codes};
use crate::support::{Sizing, backend, empty, plane, read_f32, upload_f32};

/// The most samples of a frame the composite search clip test may leave out as ties.
const MAX_TIE_SKIPS: usize = 16;

/// The strength every resharpen test uses.
pub(super) const STRENGTH: f32 = 1.5;

/// The composite steps a [StepRun] runs.
#[derive(Clone, Copy)]
pub(super) enum Step {
    /// Inputs are the repaired plane and the bob.
    Search,
    /// The input is the plane to sharpen.
    Resharpen,
    /// Inputs are the sharpened and the unsharpened plane.
    BackBlend,
}

/// Frames of one plane size put through a composite step on the device, one set of inputs per frame.
pub(super) struct StepRun {
    pub(super) step: Step,
    pub(super) frames: Vec<Vec<Vec<f32>>>,
    pub(super) size: (usize, usize),
    pub(super) bits: u32,
}

impl BuildTask for StepRun {
    type Output = Vec<Vec<f32>>;

    fn run(self, build: &BuildContext) -> Self::Output {
        let client = build.client();
        let sizing = Sizing(build.frame_context());
        let (width, height) = (self.size.0 as u32, self.size.1 as u32);
        let bytes = self.size.0 * self.size.1 * size_of::<f32>();
        let tables = QtgmcTables::new(client, &[(width, height)]);

        let first = empty(client, bytes);
        let second = empty(client, bytes);
        let output = empty(client, bytes);

        let mut results = Vec::with_capacity(self.frames.len());
        for inputs in &self.frames {
            let handles: Vec<_> = inputs.iter().map(|values| upload_f32(client, values)).collect();
            let input = plane(&handles[0], width, height);
            let scratch = StepScratch {
                first: plane(&first, width, height),
                second: plane(&second, width, height),
            };
            let output_plane = plane(&output, width, height);

            match self.step {
                Step::Search => {
                    let planes = SearchPlanes {
                        repaired: input,
                        bob: plane(&handles[1], width, height),
                        scratch,
                        output: output_plane,
                    };
                    search_clip_from_repaired(client, &sizing, &tables, planes, self.bits);
                },
                Step::Resharpen => {
                    smode2_resharpen(client, &sizing, input, scratch, output_plane, STRENGTH, self.bits);
                },
                Step::BackBlend => {
                    let planes = BackBlendPlanes {
                        sharpened: input,
                        unsharpened: plane(&handles[1], width, height),
                        scratch,
                        output: output_plane,
                    };
                    back_blend(client, &sizing, &tables, planes, self.bits);
                },
            }

            cubecl::future::block_on(client.sync()).expect("the device should finish");

            results.push(read_f32(client, &output));
        }

        results
    }
}

/// Runs `step` over `frames` of `size` on the device.
pub(super) fn run_step(
    step: Step,
    frames: Vec<Vec<Vec<f32>>>,
    size: (usize, usize),
    bits: u32,
) -> Vec<Vec<f32>> {
    let task = StepRun {
        step,
        frames,
        size,
        bits,
    };

    backend().build(task)
}

/// The luma plane of every frame of a fixture clip, as code values.
pub(super) fn luma_frames(step: &str, name: &str) -> Vec<Vec<Vec<f32>>> {
    qtgmc_fixture(step, name)
        .iter()
        .map(|fixture| {
            fixture
                .frames
                .iter()
                .map(|planes| codes(&planes[0].samples))
                .collect()
        })
        .collect()
}

// A sample whose blur and bob sit within `TIE_GAP` of the `scale(7)` threshold is left out, as f32 rounding
// decides which side of it the device and the CPU fall on. No more than `MAX_TIE_SKIPS` may be left out of a
// frame.
#[test]
fn the_composite_search_clip_matches_its_reference() {
    let outputs = qtgmc_fixture("search", "out");
    let repaired = luma_frames("shimmer", "out");
    let bobs = luma_frames("bob", "out");

    let mut most_skipped = 0;
    for ((output, repaired_frames), bob_frames) in outputs.iter().zip(&repaired).zip(&bobs) {
        let size = (output.frames[0][0].width, output.frames[0][0].height);
        let frames = repaired_frames
            .iter()
            .zip(bob_frames)
            .map(|(repair, bob)| vec![repair.clone(), bob.clone()])
            .collect();
        let found = run_step(Step::Search, frames, size, output.bits);

        for (frame, plane_found) in found.iter().enumerate() {
            let (repair, bob) = (&repaired_frames[frame], &bob_frames[frame]);
            let expected = reference_search_clip(repair, bob, size.0, size.1, output.bits);
            let (blur, tweaked) = reference_search_inputs(repair, bob, size.0, size.1, output.bits);
            let threshold = havsfunc_scale(7.0, ((1u32 << output.bits) - 1) as f32);

            let mut compared = plane_found.clone();
            let mut skipped = 0;
            for (position, sample) in compared.iter_mut().enumerate() {
                let gap = ((tweaked[position] - blur[position]).abs() - threshold).abs();
                if gap < TIE_GAP {
                    *sample = expected[position];
                    skipped += 1;
                }
            }

            most_skipped = most_skipped.max(skipped);
            assert!(
                skipped <= MAX_TIE_SKIPS,
                "{} frame {frame}: {skipped} ties skipped, expected at most {MAX_TIE_SKIPS}",
                output.clip
            );

            let label = format!("{} search frame {frame}", output.clip);
            assert_plane_within(&compared, &expected, DEVICE_TOLERANCE, size.0 as u32, &label);
        }
    }

    println!("composite search clip: most ties skipped in a frame {most_skipped}");
}

#[test]
fn the_composite_resharpen_matches_its_reference() {
    for fixture in qtgmc_fixture("average", "tr2_yuv") {
        for plane_index in 0..3 {
            let source = &fixture.frames[0][plane_index];
            let size = (source.width, source.height);
            let frames: Vec<_> = fixture
                .frames
                .iter()
                .map(|planes| vec![codes(&planes[plane_index].samples)])
                .collect();

            let found = run_step(Step::Resharpen, frames.clone(), size, fixture.bits);

            for (frame, plane_found) in found.iter().enumerate() {
                let expected = reference_resharpen(&frames[frame][0], size.0, size.1, STRENGTH, fixture.bits);

                let label = format!("{} resharpen frame {frame} plane {plane_index}", fixture.clip);
                assert_plane_within(plane_found, &expected, DEVICE_TOLERANCE, size.0 as u32, &label);
            }
        }
    }
}

#[test]
fn the_composite_back_blend_matches_its_reference() {
    let outputs = qtgmc_fixture("backblend", "out");
    let sharpened = luma_frames("resharpen", "out");
    let unsharpened = luma_frames("average", "tr2_yuv");

    for ((output, sharp_frames), plain_frames) in outputs.iter().zip(&sharpened).zip(&unsharpened) {
        let size = (output.frames[0][0].width, output.frames[0][0].height);
        let frames = sharp_frames
            .iter()
            .zip(plain_frames)
            .map(|(sharp, plain)| vec![sharp.clone(), plain.clone()])
            .collect();
        let found = run_step(Step::BackBlend, frames, size, output.bits);

        for (frame, plane_found) in found.iter().enumerate() {
            let expected = reference_back_blend(
                &sharp_frames[frame],
                &plain_frames[frame],
                size.0,
                size.1,
                output.bits,
            );

            let label = format!("{} back blend frame {frame}", output.clip);
            assert_plane_within(plane_found, &expected, DEVICE_TOLERANCE, size.0 as u32, &label);
        }
    }
}

#[test]
fn tr0_taps_drop_zero_weights() {
    assert_eq!(tr0_taps(0), vec![(0, 1.0)]);

    for tr0 in [1, 2] {
        let weights = tr0_weights(tr0);
        let taps = tr0_taps(tr0);

        assert!(taps.iter().all(|(_, weight)| *weight != 0.0), "tr0 {tr0}");
        for (offset, weight) in &taps {
            assert_eq!(
                *weight,
                weights[(offset + 2) as usize],
                "tr0 {tr0} offset {offset}"
            );
        }

        let live = weights.iter().filter(|weight| **weight != 0.0).count();
        assert_eq!(taps.len(), live, "tr0 {tr0}");
    }

    let offsets: Vec<i32> = tr0_taps(1).iter().map(|(offset, _)| *offset).collect();
    assert_eq!(offsets, vec![-1, 0, 1]);

    let offsets: Vec<i32> = tr0_taps(2).iter().map(|(offset, _)| *offset).collect();
    assert_eq!(offsets, vec![-2, -1, 0, 1, 2]);
}

/// Builds the tables of planes of `planes` sizes and reads back what each bob table holds.
struct TablesRun {
    planes: Vec<(u32, u32)>,
}

impl BuildTask for TablesRun {
    type Output = Vec<(usize, Parity, Parity, u32)>;

    fn run(self, build: &BuildContext) -> Self::Output {
        let tables = QtgmcTables::new(build.client(), &self.planes);

        let mut found = Vec::new();
        for plane_index in 0..self.planes.len() {
            for parity in [Parity::Top, Parity::Bottom] {
                let weights: &BobWeights = tables.bob(plane_index, parity);
                found.push((plane_index, parity, weights.parity(), weights.field_height()));
            }
        }

        found
    }
}

#[test]
fn tables_hold_a_bob_table_per_plane_and_parity() {
    let task = TablesRun {
        planes: vec![(128, 64), (64, 32), (64, 32)],
    };
    let found = backend().build(task);

    assert_eq!(found.len(), 6);
    for (plane_index, asked, held, field_height) in found {
        let plane_height = if plane_index == 0 { 64 } else { 32 };

        assert_eq!(asked, held, "plane {plane_index}");
        assert_eq!(field_height, plane_height / 2, "plane {plane_index}");
    }
}
