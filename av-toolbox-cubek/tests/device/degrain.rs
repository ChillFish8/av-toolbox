use av_toolbox_cubek::degrain::{DegrainNeighbour, DegrainSettings, degrain};
use av_toolbox_cubek::motion::{
    BlockGrid,
    MotionSettings,
    PlaneScale,
    SceneFlag,
    build_pyramid,
    scene_change_flags,
    search,
};
use av_toolbox_cubek::testing::{
    MotionClip,
    OUTPUT_TOLERANCE,
    ReferenceField,
    ReferenceNeighbour,
    largest_difference,
    motion_clips,
    normalised,
    reference_degrain,
    reference_scene_changed,
};
use av_toolbox_gpu::{BuildContext, BuildTask};
// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::motion::{CHROMA, DeviceField, DevicePyramid};
use crate::support::{Sizing, backend, empty, plane, read_f32, read_u32, upload_f32};

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
    /// Each neighbour's scene-change flag, as the device set it.
    pub(crate) flags: Vec<bool>,
    pub(crate) luma: Vec<f32>,
    chroma: Vec<f32>,
}

/// Searches each neighbour against the centre and degrains the luma and first chroma plane.
pub(crate) struct DegrainRun {
    frames: Frames,
    motion: MotionSettings,
    settings: DegrainSettings,
}

impl DegrainRun {
    pub(crate) fn new(frames: Frames, motion: MotionSettings, settings: DegrainSettings) -> Self {
        Self {
            frames,
            motion,
            settings,
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
            let device_field = field.field(false, self.motion.pel);
            search(
                client,
                &sizing,
                &centre_pyramid.pyramid(),
                &neighbour_pyramid.pyramid(),
                &device_field,
                &self.motion,
                None,
            );
            fields.push(field);
        }

        let flag_handle = empty(client, fields.len() * size_of::<u32>());
        let flag_plane = plane(&flag_handle, fields.len() as u32, 1);
        let device_fields: Vec<_> = fields
            .iter()
            .map(|field| field.field(false, self.motion.pel))
            .collect();
        scene_change_flags(client, &device_fields, flag_plane, self.settings.scene_change());

        let mut outputs = Vec::new();
        for (index, scale) in [(0, PlaneScale::LUMA), (1, CHROMA)] {
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
                .enumerate()
                .map(|(flag, (handle, field))| DegrainNeighbour {
                    plane: plane(handle, plane_width, plane_height),
                    field: field.field(false, self.motion.pel),
                    flag: SceneFlag {
                        flags: flag_plane,
                        index: flag as u32,
                    },
                })
                .collect();
            degrain(
                client,
                &sizing,
                plane(&centre, plane_width, plane_height),
                &neighbours,
                plane(&output, plane_width, plane_height),
                &self.settings,
                scale,
                index == 1,
            );

            outputs.push(read_f32(client, &output));
        }

        let chroma = outputs.pop().expect("chroma should be degrained");
        let luma = outputs.pop().expect("luma should be degrained");
        let fields = fields.iter().map(|field| field.read(client, false)).collect();
        let flags = read_u32(client, &flag_handle)
            .iter()
            .map(|flag| *flag != 0)
            .collect();

        Outputs {
            fields,
            flags,
            luma,
            chroma,
        }
    }
}

/// Whether each field is a scene change, by the CPU rule.
fn flags_of(fields: &[ReferenceField], grid: &BlockGrid, settings: &DegrainSettings) -> Vec<bool> {
    let rule = settings.scene_change();

    fields
        .iter()
        .map(|field| reference_scene_changed(&field.sads, grid, rule))
        .collect()
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
        for (radius, pel) in [(1, 2), (2, 2), (3, 2), (1, 1)] {
            let around: Vec<usize> = (4 - radius..4).chain(5..=4 + radius).collect();
            let frames = Frames::new(clip, 4, &around);
            let settings = DegrainSettings {
                temporal_radius: radius as u32,
                ..DegrainSettings::default()
            };
            let motion = motion_of(pel);
            let grid = BlockGrid::new(&motion, frames.width, frames.height);

            let run = DegrainRun::new(frames.clone(), motion, settings);
            let found = backend().build(run);
            let flagged = flags_of(&found.fields, &grid, &settings);
            assert_eq!(
                found.flags, flagged,
                "{} radius {radius} pel {pel} flags",
                clip.name
            );

            for (index, scale, found_plane) in
                [(0, PlaneScale::LUMA, &found.luma), (1, CHROMA, &found.chroma)]
            {
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
                    &flagged,
                    &grid,
                    &settings,
                    pel,
                    scale,
                    index == 1,
                );

                let label = format!("{} radius {radius} pel {pel} plane {index}", clip.name);
                let plane_width = size.0 as u32;
                let difference =
                    largest_difference(found_plane, &expected, OUTPUT_TOLERANCE, plane_width, &label);
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
    let never_flagged = DegrainSettings {
        thscd2: 255,
        ..settings
    };
    let around = Frames::new(&clip, 4, &[3, 5]);
    let near_side_only = Frames::new(&clip, 4, &[5]);

    let every = DegrainRun::new(around.clone(), motion, never_flagged);
    let every = backend().build(every);
    assert_eq!(
        every.flags,
        [false, false],
        "no field should be flagged at thscd2 255"
    );

    let skipping = DegrainRun::new(around, motion, settings);
    let skipping = backend().build(skipping);
    assert_eq!(
        skipping.flags,
        [true, false],
        "only the field across the cut is a scene change"
    );

    let alone = DegrainRun::new(near_side_only, motion, settings);
    let alone = backend().build(alone);

    assert_ne!(
        every.luma, alone.luma,
        "the far neighbour should have contributed to the luma"
    );
    assert_eq!(skipping.luma, alone.luma, "the cut weighed in the luma");
    assert_eq!(skipping.chroma, alone.chroma, "the cut weighed in the chroma");
}
