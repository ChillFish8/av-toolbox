use av_toolbox_cubek::degrain::{DegrainNeighbour, DegrainSettings, degrain, scene_changed};
use av_toolbox_cubek::motion::{
    BlockGrid,
    ConfidenceScores,
    MotionSettings,
    PlaneScale,
    build_pyramid,
    search,
};
use av_toolbox_cubek::testing::{
    MotionClip,
    ReferenceField,
    ReferenceNeighbour,
    motion_clips,
    normalised,
    reference_degrain,
};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::*;

use crate::motion::{DeviceField, DevicePyramid};
use crate::support::{Sizing, backend, empty, plane, read_f32, upload_f32};

const LUMA: PlaneScale = PlaneScale {
    shift_x: 0,
    shift_y: 0,
};
const CHROMA: PlaneScale = PlaneScale {
    shift_x: 1,
    shift_y: 1,
};

/// How far the device may sit from the CPU on a degrained sample.
const OUTPUT_TOLERANCE: f32 = 1e-3;

/// The centre frame of a clip and the frames around it, as degrain reads them.
#[derive(Clone)]
pub(crate) struct Frames {
    /// Each frame's luma and first chroma plane, normalised.
    planes: Vec<[Vec<f32>; 2]>,
    pub(crate) width: u32,
    pub(crate) height: u32,
    centre: usize,
    neighbours: Vec<usize>,
}

impl Frames {
    pub(crate) fn new(clip: &MotionClip, centre: usize, neighbours: &[usize]) -> Self {
        let planes = clip
            .frames
            .iter()
            .map(|frame| [normalised(&frame[0], clip.bits), normalised(&frame[1], clip.bits)])
            .collect();

        Self {
            planes,
            width: clip.width as u32,
            height: clip.height as u32,
            centre,
            neighbours: neighbours.to_vec(),
        }
    }
}

/// What the device produced for one degrain run.
pub(crate) struct Outputs {
    /// Each neighbour's field, centre to neighbour.
    pub(crate) fields: Vec<ReferenceField>,
    pub(crate) luma: Vec<f32>,
    chroma: Vec<f32>,
}

/// Searches each neighbour against the centre and degrains the luma and first chroma plane.
pub(crate) struct DegrainRun {
    frames: Frames,
    motion: MotionSettings,
    settings: DegrainSettings,
    weights_on: Vec<bool>,
}

impl DegrainRun {
    pub(crate) fn new(frames: Frames, motion: MotionSettings, settings: DegrainSettings) -> Self {
        let weights_on = vec![true; frames.neighbours.len()];

        Self {
            frames,
            motion,
            settings,
            weights_on,
        }
    }

    fn weights_on(self, weights_on: &[bool]) -> Self {
        Self {
            weights_on: weights_on.to_vec(),
            ..self
        }
    }
}

impl BuildTask for DegrainRun {
    type Output = Outputs;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let sizing = Sizing(build.frame_context());
        let frames = &self.frames;
        let (width, height) = (frames.width, frames.height);
        let grid = BlockGrid::new(&self.motion, width, height);
        let scores = ConfidenceScores {
            noise_floor: 0.0,
            thsad: 1.0,
        };

        let upload = |frame: usize| {
            let pyramid = DevicePyramid::upload(
                client,
                &frames.planes[frame][0],
                width,
                height,
                self.motion.pyramid_levels,
            );
            build_pyramid(client, &sizing, &pyramid.pyramid());
            pyramid
        };
        let centre_pyramid = upload(frames.centre);

        let mut fields = Vec::new();
        for neighbour in &frames.neighbours {
            let neighbour_pyramid = upload(*neighbour);
            let field = DeviceField::new(client, &grid);
            let planes = field.planes(false);
            search(
                client,
                &sizing,
                &centre_pyramid.pyramid(),
                &neighbour_pyramid.pyramid(),
                &planes,
                &grid,
                &self.motion,
                scores,
            );
            fields.push(field);
        }

        let mut outputs = Vec::new();
        for (index, scale) in [(0, LUMA), (1, CHROMA)] {
            let (plane_width, plane_height) = (width >> scale.shift_x, height >> scale.shift_y);
            let centre = upload_f32(client, &frames.planes[frames.centre][index]);
            let uploaded: Vec<_> = frames
                .neighbours
                .iter()
                .map(|neighbour| upload_f32(client, &frames.planes[*neighbour][index]))
                .collect();
            let output = empty(client, (plane_width * plane_height) as usize * size_of::<f32>());

            let neighbours: Vec<_> = uploaded
                .iter()
                .zip(&fields)
                .map(|(handle, field)| {
                    let planes = field.planes(false);

                    DegrainNeighbour {
                        plane: plane(handle, plane_width, plane_height),
                        vectors: planes.vectors,
                        sads: planes.sads,
                    }
                })
                .collect();
            degrain(
                client,
                &sizing,
                plane(&centre, plane_width, plane_height),
                &neighbours,
                &self.weights_on,
                plane(&output, plane_width, plane_height),
                &grid,
                &self.settings,
                self.motion.pel,
                scale,
                index == 1,
            );

            outputs.push(read_f32(client, &output));
        }

        let chroma = outputs.pop().expect("chroma should be degrained");
        let luma = outputs.pop().expect("luma should be degrained");
        let fields = fields.iter().map(|field| field.read(client, false)).collect();

        Outputs { fields, luma, chroma }
    }
}

/// The largest difference between two planes, panicking past `tolerance`.
fn largest_difference(found: &[f32], expected: &[f32], label: &str) -> f32 {
    assert_eq!(found.len(), expected.len(), "{label} has the wrong length");

    let mut largest = 0.0f32;
    for (index, (actual, wanted)) in found.iter().zip(expected).enumerate() {
        let difference = (actual - wanted).abs();
        assert!(
            difference <= OUTPUT_TOLERANCE,
            "{label} sample {index} was {actual}, expected {wanted} within {OUTPUT_TOLERANCE}"
        );
        largest = largest.max(difference);
    }

    largest
}

fn motion_of(pel: u32) -> MotionSettings {
    MotionSettings {
        pel,
        ..MotionSettings::default()
    }
}

#[test]
fn the_device_matches_the_reference_on_the_noisy_clips() {
    let mut largest = 0.0f32;
    for clip in motion_clips()
        .iter()
        .filter(|clip| clip.name.starts_with("noisy"))
    {
        for (radius, pel) in [(1, 2), (2, 2), (1, 1)] {
            let around: Vec<usize> = (4 - radius..4).chain(5..=4 + radius).collect();
            let frames = Frames::new(clip, 4, &around);
            let settings = DegrainSettings {
                temporal_radius: radius as u32,
                ..DegrainSettings::default()
            };
            let motion = motion_of(pel);
            let grid = BlockGrid::new(&motion, frames.width, frames.height);
            let weights_on = vec![true; around.len()];

            let run = DegrainRun::new(frames.clone(), motion, settings);
            let found = backend().build(run);

            for (index, scale, found_plane) in [(0, LUMA, &found.luma), (1, CHROMA, &found.chroma)] {
                let size = (
                    (frames.width >> scale.shift_x) as usize,
                    (frames.height >> scale.shift_y) as usize,
                );
                let neighbours: Vec<ReferenceNeighbour<'_>> = around
                    .iter()
                    .zip(&found.fields)
                    .map(|(frame, field)| ReferenceNeighbour {
                        plane: &frames.planes[*frame][index],
                        vectors: &field.vectors,
                        sads: &field.sads,
                    })
                    .collect();
                let expected = reference_degrain(
                    &frames.planes[4][index],
                    size,
                    &neighbours,
                    &weights_on,
                    &grid,
                    &settings,
                    pel,
                    scale,
                    index == 1,
                );

                let label = format!("{} radius {radius} pel {pel} plane {index}", clip.name);
                let difference = largest_difference(found_plane, &expected, &label);
                largest = largest.max(difference);

                let unchanged = *found_plane == frames.planes[4][index];
                assert!(!unchanged, "{label} is the centre, so no neighbour contributed");
            }
        }
    }

    println!("largest device against CPU difference: {largest:e}");
}

#[test]
fn a_cut_inside_the_window_weighs_zero() {
    let clip = motion_clips()
        .into_iter()
        .find(|clip| clip.name == "cut_8")
        .expect("cut_8 should be a clip");
    let motion = motion_of(2);
    let settings = DegrainSettings {
        thsad: 1e6,
        thsad_chroma: 1e6,
        ..DegrainSettings::default()
    };
    let around = Frames::new(&clip, 4, &[3, 5]);
    let near_side_only = Frames::new(&clip, 4, &[5]);
    let grid = BlockGrid::new(&motion, around.width, around.height);

    let every = DegrainRun::new(around.clone(), motion, settings);
    let every = backend().build(every);

    let across_cut = scene_changed(&every.fields[0].sads, &grid, &settings);
    let near_side = scene_changed(&every.fields[1].sads, &grid, &settings);
    assert!(across_cut, "neighbour 3 should be a scene change");
    assert!(!near_side, "neighbour 5 should not be a scene change");

    let derived = [!across_cut, !near_side];
    let skipping = DegrainRun::new(around, motion, settings).weights_on(&derived);
    let skipping = backend().build(skipping);

    let alone = DegrainRun::new(near_side_only, motion, settings);
    let alone = backend().build(alone);

    assert_ne!(
        every.luma, alone.luma,
        "the far neighbour should have contributed to the luma"
    );
    assert_eq!(skipping.luma, alone.luma, "the cut weighed in the luma");
    assert_eq!(skipping.chroma, alone.chroma, "the cut weighed in the chroma");
}
