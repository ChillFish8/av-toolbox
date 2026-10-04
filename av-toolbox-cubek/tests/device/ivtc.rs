use av_toolbox_cubek::ivtc::{
    ChromaMasks,
    CombPlanes,
    ComparePass,
    ComparePlanes,
    DifferencePlanes,
    FieldSums,
    MicBlocks,
    MicPlanes,
    WeavePlanes,
    chroma_into_luma,
    clear_mics,
    comb_mask,
    compare_fields,
    compare_sums_len,
    difference_sums,
    mic,
    mic_count_len,
    segments,
    weave,
};
use av_toolbox_cubek::testing::{
    IntPlane,
    MapFixture,
    MaskFixture,
    SyntheticClip,
    excluded_map_fixtures,
    map_fixtures,
    mask_fixtures,
    reference_chroma_into_luma,
    reference_comb_mask,
    reference_compare_plane,
    reference_difference,
    reference_match_weave,
    synthetic_clips,
};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;
use cubecl::server::Handle;

use crate::support::{Sizing, backend, empty, plane, read_f32, read_u32, upload_f32, upload_u32};

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

/// One plane's field compare, its previous, current and next frames `width` by `height`.
struct ComparePlane {
    frames: [Vec<f32>; 3],
    width: u32,
    height: u32,
    pass: ComparePass,
}

/// The map and partial sums of one plane's field compare.
struct CompareOutput {
    map: Vec<u32>,
    sums: Vec<u32>,
}

/// The field compares of several planes, each into buffers of its own.
struct FieldCompares {
    planes: Vec<ComparePlane>,
}

/// Uploads one plane's frames and launches its field compare, handing back every buffer it uses.
fn launch_compare<R: Runtime>(build: &BuildContext<R>, compare: &ComparePlane) -> [Handle; 6] {
    let client = build.client();
    let sizing = Sizing(build.frame_context());
    let (width, height) = (compare.width, compare.height);
    let samples = (width * height) as usize;
    let sums_len = compare_sums_len(width, height);

    let previous = upload_f32(client, &compare.frames[0]);
    let current = upload_f32(client, &compare.frames[1]);
    let next = upload_f32(client, &compare.frames[2]);
    let differences = empty(client, samples / 2 * size_of::<u32>());
    let map = empty(client, samples * size_of::<u32>());
    let sums = empty(client, sums_len * size_of::<u32>());

    let planes = ComparePlanes {
        frames: [
            plane(&previous, width, height),
            plane(&current, width, height),
            plane(&next, width, height),
        ],
        differences: plane(&differences, width, height / 2),
        map: plane(&map, width, height),
        sums: plane(&sums, sums_len as u32, 1),
    };
    compare_fields(client, &sizing, planes, compare.pass);

    [previous, current, next, differences, map, sums]
}

impl BuildTask for FieldCompares {
    type Output = Vec<CompareOutput>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let launched: Vec<[Handle; 6]> = self
            .planes
            .iter()
            .map(|compare| launch_compare(build, compare))
            .collect();

        cubecl::future::block_on(client.sync()).expect("the device should finish");

        launched
            .iter()
            .map(|[_, _, _, _, map, sums]| CompareOutput {
                map: read_u32(client, map),
                sums: read_u32(client, sums),
            })
            .collect()
    }
}

/// The three planes of `fixture`'s compare, as integer frames for the reference and their settings.
fn compare_inputs(fixture: &MapFixture) -> Vec<([Vec<i32>; 3], u32, u32, ComparePass)> {
    let clip = clip_named(&fixture.clip);
    let mut inputs = Vec::new();
    for plane_index in 0..3 {
        let chroma = u32::from(plane_index > 0);
        let frames = [
            clip.frames[fixture.frame - 1][plane_index].clone(),
            clip.frames[fixture.frame][plane_index].clone(),
            clip.frames[fixture.frame + 1][plane_index].clone(),
        ];
        let pass = ComparePass {
            first: fixture.first,
            second: fixture.second,
            field: fixture.field,
            exclude_from: fixture.exclude_from >> chroma,
            exclude_to: fixture.exclude_to >> chroma,
            start_x: 8 >> chroma,
            shift: clip.bits - 8,
        };
        inputs.push((frames, LUMA_WIDTH >> chroma, LUMA_HEIGHT >> chroma, pass));
    }

    inputs
}

fn device_compare(fixture: &MapFixture) -> Vec<CompareOutput> {
    let planes = compare_inputs(fixture)
        .into_iter()
        .map(|(frames, width, height, pass)| ComparePlane {
            frames: frames.map(|frame| to_floats(&frame)),
            width,
            height,
            pass,
        })
        .collect();

    backend().build(FieldCompares { planes })
}

fn reference_sums(frames: &[Vec<i32>; 3], width: u32, height: u32, pass: &ComparePass) -> FieldSums {
    let planes = frames.each_ref().map(|frame| IntPlane {
        samples: frame,
        width: width as usize,
        height: height as usize,
    });
    let (sums, _) = reference_compare_plane(planes, pass);

    sums
}

fn map_name(fixture: &MapFixture) -> String {
    format!(
        "{} frame {} field {} matches {} and {} excluding {} to {}",
        fixture.clip,
        fixture.frame,
        fixture.field,
        fixture.first,
        fixture.second,
        fixture.exclude_from,
        fixture.exclude_to,
    )
}

fn every_map_fixture() -> Vec<MapFixture> {
    let mut fixtures = map_fixtures();
    fixtures.extend(excluded_map_fixtures());
    fixtures
}

#[test]
fn every_device_map_matches_the_harness() {
    for fixture in every_map_fixture() {
        let outputs = device_compare(&fixture);
        let expected = [&fixture.luma, &fixture.u, &fixture.v];
        let name = map_name(&fixture);

        for (plane_index, (output, expected)) in outputs.iter().zip(expected).enumerate() {
            assert_eq!(&to_bytes(&output.map), expected, "{name} plane {plane_index}");
        }
    }
}

#[test]
fn device_sums_match_the_reference() {
    for fixture in every_map_fixture() {
        let outputs = device_compare(&fixture);
        let found: FieldSums = outputs
            .iter()
            .map(|output| FieldSums::from_partials(&output.sums))
            .sum();

        let mut expected = FieldSums::default();
        for (frames, width, height, pass) in compare_inputs(&fixture) {
            expected += reference_sums(&frames, width, height, &pass);
        }

        assert_eq!(found, expected, "{}", map_name(&fixture));
    }
}

#[test]
fn a_wide_sixteen_bit_plane_sums_without_overflow() {
    let width = 7680u32;
    let height = 8u32;
    let samples = (width * height) as usize;
    let frames = [vec![65535; samples], vec![0; samples], vec![0; samples]];
    let pass = ComparePass {
        first: 0,
        second: 1,
        field: 0,
        exclude_from: 0,
        exclude_to: 0,
        start_x: 8,
        shift: 8,
    };
    let task = FieldCompares {
        planes: vec![ComparePlane {
            frames: frames.each_ref().map(|frame| to_floats(frame)),
            width,
            height,
            pass,
        }],
    };

    let outputs = backend().build(task);
    let found = FieldSums::from_partials(&outputs[0].sums);
    let expected = reference_sums(&frames, width, height, &pass);

    // Both row pairs see the largest difference, `6 * 65535`, on every summed column of the first match.
    let largest = 2 * (u64::from(width) - 16) * 6 * 65535;
    assert_eq!(expected.first.combing, largest);
    assert_eq!(expected.first.motion, largest);
    assert_eq!(found, expected);
}

/// A pseudo-random mask with about one sample in `sparseness` set, the same for every `seed`.
fn scattered_mask(width: usize, height: usize, sparseness: u32, seed: u32) -> Vec<u8> {
    let mut state = seed.wrapping_mul(0x9e37_79b9) | 1;
    let mut mask = Vec::with_capacity(width * height);
    for _ in 0..width * height {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;

        let is_set = state.is_multiple_of(sparseness);
        mask.push(if is_set { 255 } else { 0 });
    }

    mask
}

fn to_words(mask: &[u8]) -> Vec<u32> {
    mask.iter().map(|value| u32::from(*value)).collect()
}

/// One luma mask with U and V masks marked into it, the chroma subsampled by `shift_x` and `shift_y`.
struct ChromaMarking {
    luma: Vec<u8>,
    u_mask: Vec<u8>,
    v_mask: Vec<u8>,
    shift_x: u32,
    shift_y: u32,
}

impl BuildTask for ChromaMarking {
    type Output = Vec<u8>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let sizing = Sizing(build.frame_context());
        let chroma_width = LUMA_WIDTH >> self.shift_x;
        let chroma_height = LUMA_HEIGHT >> self.shift_y;

        let luma_words = to_words(&self.luma);
        let u_words = to_words(&self.u_mask);
        let v_words = to_words(&self.v_mask);
        let luma = upload_u32(client, &luma_words);
        let u_mask = upload_u32(client, &u_words);
        let v_mask = upload_u32(client, &v_words);

        let masks = ChromaMasks {
            u: plane(&u_mask, chroma_width, chroma_height),
            v: plane(&v_mask, chroma_width, chroma_height),
            luma: plane(&luma, LUMA_WIDTH, LUMA_HEIGHT),
        };
        chroma_into_luma(client, &sizing, masks, self.shift_x, self.shift_y);

        cubecl::future::block_on(client.sync()).expect("the device should finish");

        let marked = read_u32(client, &luma);
        to_bytes(&marked)
    }
}

#[test]
fn chroma_marks_luma_as_the_reference_does_at_4_4_4_and_4_2_2() {
    let luma_width = LUMA_WIDTH as usize;
    let luma_height = LUMA_HEIGHT as usize;

    for (shift_x, shift_y) in [(0, 0), (1, 0)] {
        let chroma_width = luma_width >> shift_x;
        let chroma_height = luma_height >> shift_y;
        let luma = scattered_mask(luma_width, luma_height, 8, 1);
        let u_mask = scattered_mask(chroma_width, chroma_height, 4, 2);
        let v_mask = scattered_mask(chroma_width, chroma_height, 4, 3);

        let mut expected = luma.clone();
        reference_chroma_into_luma(
            &mut expected,
            luma_width,
            &u_mask,
            &v_mask,
            chroma_width,
            chroma_height,
            shift_x,
            shift_y,
        );
        assert_ne!(expected, luma, "shifts {shift_x}, {shift_y} mark nothing");

        let task = ChromaMarking {
            luma,
            u_mask,
            v_mask,
            shift_x,
            shift_y,
        };
        let found = backend().build(task);

        assert_eq!(found, expected, "shifts {shift_x}, {shift_y}");
    }
}

/// A plane `width` by `height` whose odd rows sit well above its even rows, with noise from `seed`.
fn combed_plane(width: usize, height: usize, seed: u32) -> Vec<f32> {
    let noise = scattered_mask(width, height, 3, seed);
    let mut samples = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let gradient = (x * 3 + y * 5) % 120 + 40;
            let comb = if y % 2 == 1 { 60 } else { 0 };
            let speck = usize::from(noise[y * width + x] / 51);
            samples.push((gradient + comb + speck) as f32);
        }
    }

    samples
}

/// The comb masks of several planes, each its own width and height.
struct CombMasks {
    planes: Vec<(Vec<f32>, u32, u32)>,
    cthresh: i32,
}

impl BuildTask for CombMasks {
    type Output = Vec<Vec<u8>>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();

        let mut launched = Vec::new();
        for (samples, width, height) in &self.planes {
            let bytes = samples.len() * size_of::<u32>();
            let mask = empty(client, bytes);
            let input = device_comb_mask(build, samples, &mask, (*width, *height), self.cthresh);
            launched.push((input, mask));
        }

        cubecl::future::block_on(client.sync()).expect("the device should finish");

        launched
            .iter()
            .map(|(_, mask)| {
                let marked = read_u32(client, mask);
                to_bytes(&marked)
            })
            .collect()
    }
}

#[test]
fn a_negative_cthresh_marks_every_plane_shape_as_the_reference_does() {
    // Luma, then 4:4:4, 4:2:2 and 4:2:0 chroma.
    let shapes = [(64, 32), (64, 32), (32, 32), (32, 16)];
    let planes: Vec<(Vec<f32>, u32, u32)> = shapes
        .iter()
        .enumerate()
        .map(|(seed, (width, height))| {
            let samples = combed_plane(*width as usize, *height as usize, seed as u32);
            (samples, *width, *height)
        })
        .collect();

    let mut expected = Vec::new();
    for (samples, width, height) in &planes {
        let integers: Vec<i32> = samples.iter().map(|sample| *sample as i32).collect();
        let plane = IntPlane {
            samples: &integers,
            width: *width as usize,
            height: *height as usize,
        };
        expected.push(reference_comb_mask(plane, -1));
    }

    let task = CombMasks { planes, cthresh: -1 };
    let found = backend().build(task);

    assert_eq!(found, expected);
}

/// Every other column of luma plane of frame `frame` of `clip`, a 4:2:2 shaped chroma plane.
fn halved_columns(clip: &SyntheticClip, frame: usize) -> Vec<i32> {
    let luma = &clip.frames[frame][0];
    let width = LUMA_WIDTH as usize;

    luma.chunks_exact(width)
        .flat_map(|row| row.iter().step_by(2))
        .copied()
        .collect()
}

/// Every field, match pair and excluded band a 4:2:2 chroma compare is tried with.
fn chroma_compare_passes() -> Vec<ComparePass> {
    let pairs = [(0, 1), (1, 2), (0, 2)];
    let bands = [(16, 16), (8, 16)];

    let mut passes = Vec::new();
    for field in 0..2 {
        for (first, second) in pairs {
            for (exclude_from, exclude_to) in bands {
                passes.push(ComparePass {
                    first,
                    second,
                    field,
                    exclude_from,
                    exclude_to,
                    start_x: 4,
                    shift: 0,
                });
            }
        }
    }

    passes
}

#[test]
fn a_4_2_2_chroma_plane_compares_as_the_reference_does() {
    let clip = clip_named("tff_8");
    let width = LUMA_WIDTH / 2;
    let height = LUMA_HEIGHT;

    let mut summed = FieldSums::default();
    for current in 1..=3 {
        let frames = [current - 1, current, current + 1].map(|frame| halved_columns(&clip, frame));
        for pass in chroma_compare_passes() {
            let label = format!("frame {current} {pass:?}");
            let task = FieldCompares {
                planes: vec![ComparePlane {
                    frames: frames.each_ref().map(|frame| to_floats(frame)),
                    width,
                    height,
                    pass,
                }],
            };

            let outputs = backend().build(task);
            let found_sums = FieldSums::from_partials(&outputs[0].sums);
            let planes = frames.each_ref().map(|frame| IntPlane {
                samples: frame,
                width: width as usize,
                height: height as usize,
            });
            let (expected_sums, expected_map) = reference_compare_plane(planes, &pass);

            assert_eq!(to_bytes(&outputs[0].map), expected_map, "{label}");
            assert_eq!(found_sums, expected_sums, "{label}");

            summed += expected_sums;
        }
    }

    assert_ne!(summed.first.combing, 0, "no compare saw any combing");
}
