use av_toolbox_cubek::DevicePlane;
use av_toolbox_cubek::degrain::{DegrainNeighbour, DegrainSettings, degrain};
use av_toolbox_cubek::deinterlace::qtgmc::{
    BackBlendPlanes,
    BobPlanes,
    Parity,
    QtgmcTables,
    SearchPlanes,
    ShimmerPlanes,
    StepScratch,
    back_blend,
    bob,
    clamp_between,
    havsfunc_scale,
    merge,
    search_clip_from_repaired,
    sharpness_strength,
    shimmer,
    smode2_resharpen,
    temporal_limits,
    tr0_taps,
    weighted_sum,
};
use av_toolbox_cubek::motion::{
    BlockGrid,
    CompensatePlanes,
    FieldBuffers,
    MotionSettings,
    PlaneScale,
    SceneFlags,
    build_pyramid,
    compensate,
    normalise,
    scene_change_flags,
    search,
};
use av_toolbox_cubek::testing::{
    FieldClip,
    OUTPUT_TOLERANCE,
    QtgmcFieldSteps,
    QtgmcReferenceSettings,
    ReferenceField,
    assert_plane_within,
    assert_same_plane,
    field_clips,
    reference_bob,
    reference_qtgmc_field,
    reference_search_inputs,
};
use av_toolbox_gpu::{BuildContext, BuildTask};
// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::DEVICE_TOLERANCE;
use super::pointwise::TIE_GAP;
use crate::motion::{CHROMA, DevicePyramid, read_field};
use crate::support::{Sizing, backend, empty, plane, read_f32, read_u32, upload_f32};

/// How many frames of the clip the chain runs over.
const FRAMES: usize = 5;

/// The field compared, a bottom field with every neighbour of the default radii.
const CENTRE: usize = 5;

/// How far a normalised sample may sit from the CPU's, on a scale of 0 to 1.
const NORMALISED_TOLERANCE: f32 = 1e-6;

/// The most search clip samples of the field that may be left out as ties.
const MAX_TIE_SKIPS: usize = 16;

/// QTGMC's TR1 weights at a radius of 2, of the second degrain and then of `edi`.
const TR1_WEIGHTS: (f32, f32) = (0.2, 0.0625);

/// What the device made of each step, every step fed the reference's inputs.
struct DeviceSteps {
    bob: Vec<f32>,
    tr0: Vec<f32>,
    repaired: Vec<f32>,
    search_clip: Vec<f32>,
    normalised: Vec<f32>,
    fields: Vec<ReferenceField>,
    flags: Vec<bool>,
    edi: [Vec<f32>; 3],
    upper: [Vec<f32>; 3],
    lower: [Vec<f32>; 3],
    smoothed: [Vec<f32>; 3],
    resharpened: [Vec<f32>; 3],
    back_blended: Vec<f32>,
    limited: [Vec<f32>; 3],
    stable: [Vec<f32>; 3],
    output: [Vec<f32>; 3],
}

/// One centre of a clip put through the device kernels in the chain's order.
struct ChainRun {
    steps: QtgmcFieldSteps,
    /// The interlaced frame holding the centre field.
    frame: [Vec<f32>; 3],
    /// The bobs TR0 reads, in the order of its taps.
    tr0_bobs: Vec<Vec<f32>>,
    parity: Parity,
    sizes: [(u32, u32); 3],
    bits: u32,
    settings: QtgmcReferenceSettings,
}

/// The device, its tables and the clip's format.
struct Device<'a, R: Runtime> {
    client: &'a ComputeClient<R>,
    sizing: Sizing,
    tables: QtgmcTables,
    sizes: [(u32, u32); 3],
    bits: u32,
}

const SCALES: [PlaneScale; 3] = [PlaneScale::LUMA, CHROMA, CHROMA];

impl<R: Runtime> Device<'_, R> {
    fn empty(&self, plane_index: usize) -> Handle {
        let (width, height) = self.sizes[plane_index];

        empty(self.client, (width * height) as usize * size_of::<f32>())
    }

    fn plane<'h>(&self, handle: &'h Handle, plane_index: usize) -> DevicePlane<'h> {
        let (width, height) = self.sizes[plane_index];

        plane(handle, width, height)
    }

    fn scratch<'h>(&self, handles: &'h [Handle; 2], plane_index: usize) -> StepScratch<'h> {
        StepScratch {
            first: self.plane(&handles[0], plane_index),
            second: self.plane(&handles[1], plane_index),
        }
    }

    fn scratch_handles(&self, plane_index: usize) -> [Handle; 2] {
        [0, 1].map(|_| self.empty(plane_index))
    }

    fn read(&self, handle: &Handle) -> Vec<f32> {
        cubecl::future::block_on(self.client.sync()).expect("the device should finish");

        read_f32(self.client, handle)
    }

    fn bob(&self, frame: &[f32], parity: Parity, plane_index: usize) -> Vec<f32> {
        let frame_handle = upload_f32(self.client, frame);
        let output = self.empty(plane_index);

        let planes = BobPlanes {
            frame: self.plane(&frame_handle, plane_index),
            output: self.plane(&output, plane_index),
        };
        bob(
            self.client,
            &self.sizing,
            planes,
            self.tables.bob(plane_index, parity),
        );

        self.read(&output)
    }

    fn tr0(&self, bobs: &[Vec<f32>], tr0: u32) -> Vec<f32> {
        let handles: Vec<Handle> = bobs
            .iter()
            .map(|values| upload_f32(self.client, values))
            .collect();
        let planes: Vec<_> = handles.iter().map(|handle| self.plane(handle, 0)).collect();
        let weights: Vec<f32> = tr0_taps(tr0).iter().map(|(_, weight)| *weight).collect();
        let output = self.empty(0);

        weighted_sum(
            self.client,
            &self.sizing,
            &planes,
            &weights,
            self.plane(&output, 0),
        );

        self.read(&output)
    }

    fn shimmer(&self, input: &[f32], reference: &[f32], plane_index: usize) -> Vec<f32> {
        let input_handle = upload_f32(self.client, input);
        let reference_handle = upload_f32(self.client, reference);
        let scratch = [0, 1, 2, 3].map(|_| self.empty(plane_index));
        let output = self.empty(plane_index);

        let planes = ShimmerPlanes {
            input: self.plane(&input_handle, plane_index),
            reference: self.plane(&reference_handle, plane_index),
            difference: self.plane(&scratch[0], plane_index),
            first: self.plane(&scratch[1], plane_index),
            positive_choke: self.plane(&scratch[2], plane_index),
            negative_choke: self.plane(&scratch[3], plane_index),
            output: self.plane(&output, plane_index),
        };
        shimmer(self.client, &self.sizing, planes, self.bits);

        self.read(&output)
    }

    fn search_clip(&self, repaired: &[f32], bob_plane: &[f32]) -> Vec<f32> {
        let repaired_handle = upload_f32(self.client, repaired);
        let bob_handle = upload_f32(self.client, bob_plane);
        let scratch = self.scratch_handles(0);
        let output = self.empty(0);

        let planes = SearchPlanes {
            repaired: self.plane(&repaired_handle, 0),
            bob: self.plane(&bob_handle, 0),
            scratch: self.scratch(&scratch, 0),
            output: self.plane(&output, 0),
        };
        search_clip_from_repaired(self.client, &self.sizing, &self.tables, planes, self.bits);

        self.read(&output)
    }

    fn normalise(&self, search_clip: &[f32]) -> Vec<f32> {
        let input = upload_f32(self.client, search_clip);
        let output = self.empty(0);

        normalise(
            self.client,
            &self.sizing,
            self.plane(&input, 0),
            self.plane(&output, 0),
            self.bits,
        );

        self.read(&output)
    }

    fn pyramid(&self, luma: &[f32]) -> DevicePyramid {
        let (width, height) = self.sizes[0];
        let levels = MotionSettings::default().pyramid_levels;

        let pyramid = DevicePyramid::upload(self.client, luma, width, height, levels);
        build_pyramid(self.client, &self.sizing, &pyramid.pyramid());

        pyramid
    }

    /// Searches the centre against each neighbour's normalised clip.
    fn search(&self, centre: &[f32], neighbours: &[&[f32]]) -> Vec<FieldBuffers> {
        let settings = MotionSettings::default();
        let (width, height) = self.sizes[0];
        let grid = BlockGrid::new(&settings, width, height);
        let centre_pyramid = self.pyramid(centre);

        let mut fields = Vec::with_capacity(neighbours.len());
        for neighbour in neighbours {
            let neighbour_pyramid = self.pyramid(neighbour);
            let buffers = FieldBuffers::new(self.client, &grid);
            let field = buffers.field(settings.pel, false);
            search(
                self.client,
                &self.sizing,
                &centre_pyramid.pyramid(),
                &neighbour_pyramid.pyramid(),
                &field,
                &settings,
                None,
            );
            fields.push(buffers);
        }

        fields
    }

    fn flags(&self, fields: &[FieldBuffers]) -> SceneFlags {
        let pel = MotionSettings::default().pel;
        let flags = SceneFlags::new(self.client, fields.len() as u32);
        let device_fields: Vec<_> = fields.iter().map(|buffers| buffers.field(pel, false)).collect();

        scene_change_flags(
            self.client,
            &device_fields,
            flags.plane(),
            QtgmcReferenceSettings::SCENE_CHANGE,
        );

        flags
    }

    /// The temporal limits of `edi` from `neighbours`, each a field index and its plane, compensated onto it.
    fn limits(
        &self,
        edi: &[f32],
        neighbours: &[(usize, &[f32])],
        motion: (&[FieldBuffers], &SceneFlags),
        plane_index: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let (fields, flags) = motion;
        let pel = MotionSettings::default().pel;
        let centre = upload_f32(self.client, edi);

        let mut compensated = Vec::with_capacity(neighbours.len());
        for (index, neighbour) in neighbours {
            let neighbour_handle = upload_f32(self.client, neighbour);
            let output = self.empty(plane_index);
            let planes = CompensatePlanes {
                centre: self.plane(&centre, plane_index),
                neighbour: self.plane(&neighbour_handle, plane_index),
                output: self.plane(&output, plane_index),
            };
            let field = fields[*index].field(pel, false);
            let flag = flags.flag(*index as u32);
            compensate(
                self.client,
                &self.sizing,
                planes,
                &field,
                flag,
                SCALES[plane_index],
            );

            compensated.push(output);
        }

        let upper = self.empty(plane_index);
        let lower = self.empty(plane_index);
        let neighbour_planes: Vec<_> = compensated
            .iter()
            .map(|handle| self.plane(handle, plane_index))
            .collect();
        temporal_limits(
            self.client,
            &self.sizing,
            self.plane(&centre, plane_index),
            &neighbour_planes,
            self.plane(&upper, plane_index),
            self.plane(&lower, plane_index),
        );

        (self.read(&upper), self.read(&lower))
    }

    /// Degrains `centre` with `neighbours`, each a field index and its plane.
    fn degrain(
        &self,
        centre: &[f32],
        neighbours: &[(usize, &[f32])],
        motion: (&[FieldBuffers], &SceneFlags),
        settings: &DegrainSettings,
        plane_index: usize,
    ) -> Vec<f32> {
        let (fields, flags) = motion;
        let pel = MotionSettings::default().pel;
        let centre_handle = upload_f32(self.client, centre);
        let handles: Vec<Handle> = neighbours
            .iter()
            .map(|(_, values)| upload_f32(self.client, values))
            .collect();
        let output = self.empty(plane_index);

        let degrain_neighbours: Vec<DegrainNeighbour<'_>> = neighbours
            .iter()
            .zip(&handles)
            .map(|((index, _), handle)| DegrainNeighbour {
                plane: self.plane(handle, plane_index),
                field: fields[*index].field(pel, false),
                flag: flags.flag(*index as u32),
            })
            .collect();
        degrain(
            self.client,
            &self.sizing,
            self.plane(&centre_handle, plane_index),
            &degrain_neighbours,
            self.plane(&output, plane_index),
            settings,
            SCALES[plane_index],
            plane_index > 0,
        );

        self.read(&output)
    }

    fn merge(&self, first: &[f32], second: &[f32], weight: f32, plane_index: usize) -> Vec<f32> {
        let first_handle = upload_f32(self.client, first);
        let second_handle = upload_f32(self.client, second);
        let output = self.empty(plane_index);

        merge(
            self.client,
            &self.sizing,
            self.plane(&first_handle, plane_index),
            self.plane(&second_handle, plane_index),
            self.plane(&output, plane_index),
            weight,
        );

        self.read(&output)
    }

    fn resharpen(&self, input: &[f32], strength: f32, plane_index: usize) -> Vec<f32> {
        let input_handle = upload_f32(self.client, input);
        let scratch = self.scratch_handles(plane_index);
        let output = self.empty(plane_index);

        smode2_resharpen(
            self.client,
            &self.sizing,
            self.plane(&input_handle, plane_index),
            self.scratch(&scratch, plane_index),
            self.plane(&output, plane_index),
            strength,
            self.bits,
        );

        self.read(&output)
    }

    fn back_blend(&self, sharpened: &[f32], unsharpened: &[f32]) -> Vec<f32> {
        let sharpened_handle = upload_f32(self.client, sharpened);
        let unsharpened_handle = upload_f32(self.client, unsharpened);
        let scratch = self.scratch_handles(0);
        let output = self.empty(0);

        let planes = BackBlendPlanes {
            sharpened: self.plane(&sharpened_handle, 0),
            unsharpened: self.plane(&unsharpened_handle, 0),
            scratch: self.scratch(&scratch, 0),
            output: self.plane(&output, 0),
        };
        back_blend(self.client, &self.sizing, &self.tables, planes, self.bits);

        self.read(&output)
    }

    fn clamp(&self, input: &[f32], upper: &[f32], lower: &[f32], plane_index: usize) -> Vec<f32> {
        let input_handle = upload_f32(self.client, input);
        let upper_handle = upload_f32(self.client, upper);
        let lower_handle = upload_f32(self.client, lower);
        let output = self.empty(plane_index);

        clamp_between(
            self.client,
            &self.sizing,
            self.plane(&input_handle, plane_index),
            self.plane(&upper_handle, plane_index),
            self.plane(&lower_handle, plane_index),
            self.plane(&output, plane_index),
        );

        self.read(&output)
    }
}

/// The neighbours of `steps` whose delta `keep` accepts, as an index into its motion and a plane `pick`
/// reads from the neighbour.
fn chosen<'a>(
    steps: &'a QtgmcFieldSteps,
    keep: impl Fn(i32) -> bool,
    pick: impl Fn(usize) -> &'a [f32],
) -> Vec<(usize, &'a [f32])> {
    steps
        .motion
        .iter()
        .enumerate()
        .filter(|(_, found)| keep(found.delta))
        .map(|(index, _)| (index, pick(index)))
        .collect()
}

impl BuildTask for ChainRun {
    type Output = DeviceSteps;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let device = Device {
            client,
            sizing: Sizing(build.frame_context()),
            tables: QtgmcTables::new(client, &self.sizes),
            sizes: self.sizes,
            bits: self.bits,
        };
        let steps = &self.steps;
        let settings = &self.settings;

        let bob = device.bob(&self.frame[0], self.parity, 0);
        let tr0 = device.tr0(&self.tr0_bobs, settings.tr0);
        let repaired = device.shimmer(&steps.tr0, &steps.bob, 0);
        let search_clip = device.search_clip(&steps.repaired, &steps.bob);
        let normalised = device.normalise(&steps.search_clip);

        let neighbour_clips: Vec<&[f32]> = steps
            .motion
            .iter()
            .map(|found| found.neighbour_normalised.as_slice())
            .collect();
        let fields = device.search(&steps.normalised, &neighbour_clips);
        let flags = device.flags(&fields);
        let motion = (fields.as_slice(), &flags);

        let strength = sharpness_strength(settings.sharpness, settings.tr1, settings.tr2);
        let tr1_settings = settings.tr1_degrain();
        let tr2_settings = settings.tr2_degrain();
        let radius = settings.tr2 as i32;

        let found_fields = fields
            .iter()
            .map(|buffers| read_field(client, buffers, false))
            .collect();
        let flag_values = read_u32(client, flags.plane().handle());
        let found_flags = flag_values.iter().map(|flag| *flag != 0).collect();
        let back_blended = device.back_blend(&steps.resharpened[0], &steps.smoothed[0]);

        let mut found = DeviceSteps {
            bob,
            tr0,
            repaired,
            search_clip,
            normalised,
            fields: found_fields,
            flags: found_flags,
            edi: Default::default(),
            upper: Default::default(),
            lower: Default::default(),
            smoothed: Default::default(),
            resharpened: Default::default(),
            back_blended,
            limited: Default::default(),
            stable: Default::default(),
            output: Default::default(),
        };
        for plane_index in 0..3 {
            let edi = &steps.edi[plane_index];
            let neighbour_edi = |index: usize| steps.motion[index].neighbour_edi[plane_index].as_slice();
            let neighbour_limited =
                |index: usize| steps.motion[index].neighbour_limited[plane_index].as_slice();

            found.edi[plane_index] = device.bob(&self.frame[plane_index], self.parity, plane_index);

            let near = chosen(steps, |delta| delta.abs() == 1, neighbour_edi);
            let (upper, lower) = device.limits(edi, &near, motion, plane_index);
            found.upper[plane_index] = upper;
            found.lower[plane_index] = lower;

            let far = chosen(steps, |delta| delta.abs() == 2, neighbour_edi);
            let first = device.degrain(edi, &near, motion, &tr1_settings, plane_index);
            let second = device.degrain(edi, &far, motion, &tr1_settings, plane_index);
            let blended = device.merge(&first, &second, TR1_WEIGHTS.0, plane_index);
            found.smoothed[plane_index] = device.merge(&blended, edi, TR1_WEIGHTS.1, plane_index);

            let smoothed = &steps.smoothed[plane_index];
            found.resharpened[plane_index] = device.resharpen(smoothed, strength, plane_index);

            let (back_blended, upper, lower) = (
                &steps.back_blended[plane_index],
                &steps.upper[plane_index],
                &steps.lower[plane_index],
            );
            found.limited[plane_index] = device.clamp(back_blended, upper, lower, plane_index);

            let limited = &steps.limited[plane_index];
            let within = chosen(steps, |delta| delta.abs() <= radius, neighbour_limited);
            found.stable[plane_index] = device.degrain(limited, &within, motion, &tr2_settings, plane_index);

            let stable = &steps.stable[plane_index];
            found.output[plane_index] = device.shimmer(stable, edi, plane_index);
        }

        found
    }
}

fn bob_interpolate(plane_values: &[f32], width: usize, height: usize, parity: Parity) -> Vec<f32> {
    reference_bob(plane_values, (width, height), parity)
}

fn code_frames(clip: &FieldClip) -> Vec<[Vec<f32>; 3]> {
    clip.frames
        .iter()
        .take(FRAMES)
        .map(|planes| {
            planes
                .each_ref()
                .map(|values| values.iter().map(|sample| *sample as f32).collect())
        })
        .collect()
}

/// The bob of every field TR0 reads around [CENTRE], the nearest standing in past either end.
fn tr0_bobs(frames: &[[Vec<f32>; 3]], luma: (usize, usize), tr0: u32) -> Vec<Vec<f32>> {
    let last = 2 * frames.len() as i32 - 1;

    tr0_taps(tr0)
        .iter()
        .map(|(offset, _)| {
            let field = (CENTRE as i32 + offset).clamp(0, last) as usize;
            let parity = if field.is_multiple_of(2) {
                Parity::Top
            } else {
                Parity::Bottom
            };
            reference_bob(&frames[field / 2][0], luma, parity)
        })
        .collect()
}

/// Panics where the search clip differs, leaving out samples within [TIE_GAP] of the `scale(7)` threshold.
fn assert_search_clip(found: &[f32], steps: &QtgmcFieldSteps, luma: (usize, usize), bits: u32) {
    let (blur, tweaked) = reference_search_inputs(&steps.repaired, &steps.bob, luma.0, luma.1, bits);
    let threshold = havsfunc_scale(7.0, ((1u32 << bits) - 1) as f32);

    let mut compared = found.to_vec();
    let mut skipped = 0;
    for (position, sample) in compared.iter_mut().enumerate() {
        let gap = ((tweaked[position] - blur[position]).abs() - threshold).abs();
        if gap < TIE_GAP {
            *sample = steps.search_clip[position];
            skipped += 1;
        }
    }

    assert!(
        skipped <= MAX_TIE_SKIPS,
        "{skipped} ties skipped, expected at most {MAX_TIE_SKIPS}"
    );
    assert_plane_within(
        &compared,
        &steps.search_clip,
        DEVICE_TOLERANCE,
        luma.0 as u32,
        "search clip",
    );
}

fn assert_planes(
    found: &[Vec<f32>; 3],
    expected: &[Vec<f32>; 3],
    tolerance: f32,
    widths: [u32; 3],
    step: &str,
) {
    for plane_index in 0..3 {
        let label = format!("{step} plane {plane_index}");
        let (found_plane, expected_plane) = (&found[plane_index], &expected[plane_index]);

        assert_plane_within(
            found_plane,
            expected_plane,
            tolerance,
            widths[plane_index],
            &label,
        );
    }
}

#[test]
fn the_device_steps_match_the_reference_chain() {
    let clips = field_clips();
    let clip = clips
        .iter()
        .find(|clip| clip.name == "field_pan_8")
        .expect("field_pan_8");
    let frames = code_frames(clip);
    let luma = (clip.width, clip.height);
    let settings = QtgmcReferenceSettings::default();

    let steps = reference_qtgmc_field(&frames, luma, clip.bits, &settings, &bob_interpolate, CENTRE);
    let deltas: Vec<i32> = steps.motion.iter().map(|found| found.delta).collect();
    assert_eq!(deltas, vec![1, -1, 2, -2], "the centre has every neighbour");

    let (width, height) = (clip.width as u32, clip.height as u32);
    let chroma = (width / 2, height / 2);
    let run = ChainRun {
        frame: frames[CENTRE / 2].clone(),
        tr0_bobs: tr0_bobs(&frames, luma, settings.tr0),
        steps: steps.clone(),
        parity: Parity::Bottom,
        sizes: [(width, height), chroma, chroma],
        bits: clip.bits,
        settings,
    };
    let found = backend().build(run);
    let widths = [width, chroma.0, chroma.0];

    assert_plane_within(&found.bob, &steps.bob, DEVICE_TOLERANCE, width, "bob");
    assert_plane_within(&found.tr0, &steps.tr0, DEVICE_TOLERANCE, width, "tr0");
    assert_plane_within(&found.repaired, &steps.repaired, DEVICE_TOLERANCE, width, "rep0");
    assert_search_clip(&found.search_clip, &steps, luma, clip.bits);
    assert_plane_within(
        &found.normalised,
        &steps.normalised,
        NORMALISED_TOLERANCE,
        width,
        "normalised",
    );

    let across = BlockGrid::new(&MotionSettings::default(), width, height).across;
    for (device_field, motion) in found.fields.iter().zip(&steps.motion) {
        let label = format!("delta {}", motion.delta);
        assert_same_plane(&device_field.vectors, &motion.field.vectors, 2 * across, &label);
        assert_same_plane(&device_field.sads, &motion.field.sads, across, &label);
    }

    let flagged: Vec<bool> = steps.motion.iter().map(|motion| motion.flagged).collect();
    assert_eq!(found.flags, flagged, "scene-change flags");

    assert_planes(&found.edi, &steps.edi, DEVICE_TOLERANCE, widths, "edi");
    assert_planes(&found.upper, &steps.upper, OUTPUT_TOLERANCE, widths, "upper");
    assert_planes(&found.lower, &steps.lower, OUTPUT_TOLERANCE, widths, "lower");
    assert_planes(&found.smoothed, &steps.smoothed, OUTPUT_TOLERANCE, widths, "tr1");
    assert_planes(
        &found.resharpened,
        &steps.resharpened,
        DEVICE_TOLERANCE,
        widths,
        "resharpen",
    );
    assert_plane_within(
        &found.back_blended,
        &steps.back_blended[0],
        DEVICE_TOLERANCE,
        width,
        "back-blend",
    );
    assert_planes(&found.limited, &steps.limited, DEVICE_TOLERANCE, widths, "limit");
    assert_planes(&found.stable, &steps.stable, OUTPUT_TOLERANCE, widths, "tr2");
    assert_planes(&found.output, &steps.output, DEVICE_TOLERANCE, widths, "rep2");
}
