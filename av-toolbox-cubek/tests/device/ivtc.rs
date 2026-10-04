use av_toolbox_cubek::ivtc::{DifferencePlanes, WeavePlanes, difference_sums, segments, weave};
use av_toolbox_cubek::testing::{
    SyntheticClip,
    reference_difference,
    reference_match_weave,
    synthetic_clips,
};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use crate::support::{Sizing, backend, empty, plane, read_f32, read_u32, upload_f32};

fn to_floats(samples: &[i32]) -> Vec<f32> {
    samples.iter().map(|sample| *sample as f32).collect()
}

fn clip_named(name: &str) -> SyntheticClip {
    synthetic_clips()
        .into_iter()
        .find(|clip| clip.name == name)
        .unwrap_or_else(|| panic!("there is no synthetic clip {name}"))
}

/// One weave of `matching` and `other`, both `width` by `height`.
struct Weave {
    matching: Vec<f32>,
    other: Vec<f32>,
    width: u32,
    height: u32,
    matching_parity: u32,
}

impl BuildTask for Weave {
    type Output = Vec<f32>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let sizing = Sizing(build.frame_context());
        let bytes = self.matching.len() * size_of::<f32>();

        let matching = upload_f32(client, &self.matching);
        let other = upload_f32(client, &self.other);
        let output = empty(client, bytes);

        let planes = WeavePlanes {
            matching: plane(&matching, self.width, self.height),
            other: plane(&other, self.width, self.height),
            output: plane(&output, self.width, self.height),
        };
        weave(client, &sizing, planes, self.matching_parity);

        cubecl::future::block_on(client.sync()).expect("the device should finish");

        read_f32(client, &output)
    }
}

/// The frames a VFM match reads its matching and other rows from.
fn match_sources(matched: usize) -> (usize, usize) {
    match matched {
        0 => (1, 0),
        1 => (1, 1),
        2 => (1, 2),
        3 => (0, 1),
        4 => (2, 1),
        _ => panic!("VFM has five matches, not {matched}"),
    }
}

#[test]
fn every_match_weave_matches_the_reference() {
    let clip = clip_named("tff_10");

    for current in 1..=3 {
        for plane_index in 0..3 {
            let frames = [
                clip.frames[current - 1][plane_index].as_slice(),
                clip.frames[current][plane_index].as_slice(),
                clip.frames[current + 1][plane_index].as_slice(),
            ];
            let width = if plane_index == 0 { 64 } else { 32 };
            let height = if plane_index == 0 { 32 } else { 16 };

            for matched in 0..5 {
                for field in 0..2 {
                    let expected = reference_match_weave(frames, width, height, matched, field);
                    let (matching_frame, other_frame) = match_sources(matched);
                    let task = Weave {
                        matching: to_floats(frames[matching_frame]),
                        other: to_floats(frames[other_frame]),
                        width: width as u32,
                        height: height as u32,
                        matching_parity: 1 - field as u32,
                    };

                    let found = backend().build(task);

                    assert_eq!(
                        found,
                        to_floats(&expected),
                        "frame {current} plane {plane_index} match {matched} field {field}",
                    );
                }
            }
        }
    }
}

/// The partial sums of `|first - second|` over planes `width` by `height`.
struct Differences {
    first: Vec<f32>,
    second: Vec<f32>,
    width: u32,
    height: u32,
}

impl BuildTask for Differences {
    type Output = Vec<u32>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let sizing = Sizing(build.frame_context());
        let segment_count = segments(self.width);
        let sum_bytes = self.height as usize * segment_count as usize * size_of::<u32>();

        let first = upload_f32(client, &self.first);
        let second = upload_f32(client, &self.second);
        let sums = empty(client, sum_bytes);

        let planes = DifferencePlanes {
            first: plane(&first, self.width, self.height),
            second: plane(&second, self.width, self.height),
            sums: plane(&sums, segment_count, self.height),
        };
        difference_sums(client, &sizing, planes);

        cubecl::future::block_on(client.sync()).expect("the device should finish");

        read_u32(client, &sums)
    }
}

fn total_of(partial_sums: &[u32]) -> u64 {
    partial_sums.iter().map(|sum| u64::from(*sum)).sum()
}

#[test]
fn difference_sums_add_up_to_the_reference() {
    let clip = clip_named("cut_8");

    for current in 0..clip.frames.len() {
        for before in 0..clip.frames.len() {
            let first = &clip.frames[current][0];
            let second = &clip.frames[before][0];
            let task = Differences {
                first: to_floats(first),
                second: to_floats(second),
                width: 64,
                height: 32,
            };

            let partial_sums = backend().build(task);

            assert_eq!(
                total_of(&partial_sums),
                reference_difference(first, second),
                "frames {current} and {before}",
            );
        }
    }
}

#[test]
fn a_wide_sixteen_bit_difference_fits_its_segments() {
    let width = 7680;
    let height = 2;
    let samples = (width * height) as usize;
    let task = Differences {
        first: vec![65535.0; samples],
        second: vec![0.0; samples],
        width,
        height,
    };

    let partial_sums = backend().build(task);

    assert_eq!(partial_sums.len(), height as usize * segments(width) as usize);
    assert_eq!(total_of(&partial_sums), 7680 * 2 * 65535);
}
