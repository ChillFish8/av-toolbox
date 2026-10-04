use av_toolbox_cubek::ivtc::{
    ChromaMasks,
    CombPlanes,
    DifferencePlanes,
    MicBlocks,
    MicPlanes,
    WeavePlanes,
    chroma_into_luma,
    clear_mics,
    comb_mask,
    difference_sums,
    mic,
    mic_count_len,
    segments,
    weave,
};
use av_toolbox_cubek::testing::{
    MaskFixture,
    SyntheticClip,
    mask_fixtures,
    reference_difference,
    reference_match_weave,
    synthetic_clips,
};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;
use cubecl::server::Handle;

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

const LUMA_WIDTH: u32 = 64;
const LUMA_HEIGHT: u32 = 32;
const BLOCKS: MicBlocks = MicBlocks { x: 16, y: 16 };

/// The three f32 planes of one match weave.
struct Weaves {
    luma: Vec<f32>,
    u: Vec<f32>,
    v: Vec<f32>,
}

/// The luma, U and V comb masks of one weave.
struct Masks {
    luma: Vec<u32>,
    u: Vec<u32>,
    v: Vec<u32>,
}

/// The comb masks of up to five weaves, each counted into the mic slot of its index.
struct CombAndCount {
    weaves: Vec<Weaves>,
    cthresh: i32,
}

/// Builds the comb mask of one plane of a weave into `mask`, handing back the uploaded plane.
fn device_comb_mask<R: Runtime>(
    build: &BuildContext<R>,
    input: &[f32],
    mask: &Handle,
    size: (u32, u32),
    cthresh: i32,
) -> Handle {
    let client = build.client();
    let sizing = Sizing(build.frame_context());
    let (width, height) = size;
    let input = upload_f32(client, input);

    let planes = CombPlanes {
        input: plane(&input, width, height),
        mask: plane(mask, width, height),
    };
    comb_mask(client, &sizing, planes, cthresh);

    input
}

impl BuildTask for CombAndCount {
    type Output = (Vec<Masks>, Vec<u32>);

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let sizing = Sizing(build.frame_context());
        let luma_size = (LUMA_WIDTH, LUMA_HEIGHT);
        let chroma_size = (LUMA_WIDTH / 2, LUMA_HEIGHT / 2);
        let luma_bytes = (LUMA_WIDTH * LUMA_HEIGHT) as usize * size_of::<u32>();
        let chroma_bytes = luma_bytes / 4;
        let count_len = mic_count_len(LUMA_WIDTH, LUMA_HEIGHT, BLOCKS);
        let slots = self.weaves.len() as u32;

        let counts = empty(client, count_len * size_of::<u32>());
        let results = empty(client, slots as usize * size_of::<u32>());
        clear_mics(client, &sizing, plane(&results, slots, 1));

        let mut masks = Vec::new();
        let mut inputs = Vec::new();
        for (slot, weaves) in self.weaves.iter().enumerate() {
            let luma = empty(client, luma_bytes);
            let u = empty(client, chroma_bytes);
            let v = empty(client, chroma_bytes);
            inputs.push(device_comb_mask(
                build,
                &weaves.luma,
                &luma,
                luma_size,
                self.cthresh,
            ));
            inputs.push(device_comb_mask(build, &weaves.u, &u, chroma_size, self.cthresh));
            inputs.push(device_comb_mask(build, &weaves.v, &v, chroma_size, self.cthresh));

            let chroma_masks = ChromaMasks {
                u: plane(&u, chroma_size.0, chroma_size.1),
                v: plane(&v, chroma_size.0, chroma_size.1),
                luma: plane(&luma, LUMA_WIDTH, LUMA_HEIGHT),
            };
            chroma_into_luma(client, &sizing, chroma_masks, 1, 1);

            let mic_planes = MicPlanes {
                mask: plane(&luma, LUMA_WIDTH, LUMA_HEIGHT),
                counts: plane(&counts, count_len as u32, 1),
                results: plane(&results, slots, 1),
            };
            mic(client, &sizing, mic_planes, BLOCKS, slot as u32);

            masks.push((luma, u, v));
        }

        cubecl::future::block_on(client.sync()).expect("the device should finish");

        let masks = masks
            .iter()
            .map(|(luma, u, v)| Masks {
                luma: read_u32(client, luma),
                u: read_u32(client, u),
                v: read_u32(client, v),
            })
            .collect();
        let mics = read_u32(client, &results);

        (masks, mics)
    }
}

/// The f32 planes of `fixture`'s weave, from its clip's synthetic frames.
fn weaves_of(fixture: &MaskFixture) -> Weaves {
    let clip = clip_named(&fixture.clip);
    let mut planes = Vec::new();
    for plane_index in 0..3 {
        let frames = [
            clip.frames[fixture.frame - 1][plane_index].as_slice(),
            clip.frames[fixture.frame][plane_index].as_slice(),
            clip.frames[fixture.frame + 1][plane_index].as_slice(),
        ];
        let width = if plane_index == 0 { 64 } else { 32 };
        let height = if plane_index == 0 { 32 } else { 16 };
        let woven = reference_match_weave(frames, width, height, fixture.matched, fixture.field);
        planes.push(to_floats(&woven));
    }

    let [luma, u, v]: [Vec<f32>; 3] = planes.try_into().expect("three planes");
    Weaves { luma, u, v }
}

fn fixture_name(fixture: &MaskFixture) -> String {
    format!(
        "{} frame {} field {} match {}",
        fixture.clip, fixture.frame, fixture.field, fixture.matched,
    )
}

fn to_bytes(mask: &[u32]) -> Vec<u8> {
    mask.iter().map(|value| *value as u8).collect()
}

/// The fixtures in groups of the five matches of one frame and field, in match order.
fn fixture_groups() -> Vec<Vec<MaskFixture>> {
    let mut groups: Vec<Vec<MaskFixture>> = Vec::new();
    for fixture in mask_fixtures() {
        let starts_group = fixture.matched == 0;
        if starts_group {
            groups.push(Vec::new());
        }

        let group = groups.last_mut().expect("every group starts at match 0");
        assert_eq!(group.len(), fixture.matched, "{}", fixture_name(&fixture));
        group.push(fixture);
    }

    groups
}

fn comb_and_count(group: &[MaskFixture], cthresh: i32) -> (Vec<Masks>, Vec<u32>) {
    let weaves = group.iter().map(weaves_of).collect();
    let task = CombAndCount { weaves, cthresh };

    backend().build(task)
}

fn assert_masks_match(group: &[MaskFixture], masks: &[Masks]) {
    for (fixture, found) in group.iter().zip(masks) {
        let name = fixture_name(fixture);

        assert_eq!(to_bytes(&found.u), fixture.u, "{name} U");
        assert_eq!(to_bytes(&found.v), fixture.v, "{name} V");
        assert_eq!(to_bytes(&found.luma), fixture.luma, "{name} luma");
    }
}

fn scaled_cthresh(clip: &str) -> i32 {
    let bits = clip_named(clip).bits;
    9 << (bits - 8)
}

#[test]
fn every_device_mask_matches_the_harness() {
    for group in fixture_groups() {
        let cthresh = scaled_cthresh(&group[0].clip);
        let (masks, _) = comb_and_count(&group, cthresh);

        assert_masks_match(&group, &masks);
    }
}

#[test]
fn every_device_mic_matches_the_harness() {
    for group in fixture_groups() {
        let cthresh = scaled_cthresh(&group[0].clip);
        let (_, mics) = comb_and_count(&group, cthresh);
        let expected: Vec<u32> = group.iter().map(|fixture| fixture.mic).collect();

        assert_eq!(mics, expected, "{}", fixture_name(&group[0]));
    }
}

#[test]
fn a_ten_bit_mask_scales_cthresh() {
    let groups = fixture_groups()
        .into_iter()
        .filter(|group| group[0].clip == "tff_10");
    for group in groups {
        let (masks, _) = comb_and_count(&group, 9 << 2);

        assert_masks_match(&group, &masks);
    }
}
