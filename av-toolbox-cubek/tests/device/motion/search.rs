use av_toolbox_cubek::motion::{
    BlockGrid,
    ConfidenceScores,
    MotionSettings,
    build_pyramid,
    chain,
    level_dims,
    refine_half_pel,
    refine_seed,
    search,
};
use av_toolbox_cubek::testing::{
    MOTION_FIXTURES,
    MotionClip,
    ReferenceField,
    SettingFixture,
    assert_confidences,
    assert_same_plane,
    fixture_f32s,
    motion_clips,
    normalised,
    reference_chain_seed,
    reference_clip_pyramids,
    reference_refine_half_pel,
    reference_refine_seed,
    reference_search,
};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use super::{DeviceField, DevicePyramid, assert_same_field};
use crate::support::{Sizing, backend};

/// A chained field and the seed it was refined from.
struct Chained {
    seed: Vec<i32>,
    field: ReferenceField,
}

/// Everything the device works out for one clip at one setting.
struct Outputs {
    pyramids: Vec<Vec<Vec<f32>>>,
    /// Frame `t` to `t + 1`, scored.
    direct_1: Vec<ReferenceField>,
    /// Frame `t` to `t + 2`.
    direct_2: Vec<ReferenceField>,
    /// Frame `t` to `t + 2` through `t + 1`, scored.
    chained_2: Vec<Chained>,
    /// Frame `t + 2` to `t` through `t + 1`.
    chained_back_2: Vec<Chained>,
}

/// Runs the device search over every frame of a clip.
struct MotionRun {
    frames: Vec<Vec<f32>>,
    width: u32,
    height: u32,
    settings: MotionSettings,
    refine_radius: u32,
    scores: ConfidenceScores,
}

impl MotionRun {
    fn of_clip(clip: &MotionClip, setting: &SettingFixture) -> Self {
        let frames = clip
            .frames
            .iter()
            .map(|planes| normalised(&planes[0], clip.bits))
            .collect();

        Self {
            frames,
            width: clip.width as u32,
            height: clip.height as u32,
            settings: setting.settings,
            refine_radius: setting.refine_radius,
            scores: setting.scores,
        }
    }
}

impl BuildTask for MotionRun {
    type Output = Outputs;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let sizing = Sizing(build.frame_context());
        let grid = BlockGrid::new(&self.settings, self.width, self.height);
        let levels = self.settings.pyramid_levels;
        let frames = self.frames.len();

        let device_pyramids: Vec<DevicePyramid> = self
            .frames
            .iter()
            .map(|luma| DevicePyramid::upload(client, luma, self.width, self.height, levels))
            .collect();
        for device_pyramid in &device_pyramids {
            build_pyramid(client, &sizing, &device_pyramid.pyramid());
        }

        let pyramids = device_pyramids
            .iter()
            .map(|device_pyramid| device_pyramid.read(client))
            .collect();

        let direct = |centre: usize, neighbour: usize, scored: bool| {
            let field = DeviceField::new(client, &grid);
            let centre_pyramid = device_pyramids[centre].pyramid();
            let neighbour_pyramid = device_pyramids[neighbour].pyramid();
            let planes = field.planes(scored);
            search(
                client,
                &sizing,
                &centre_pyramid,
                &neighbour_pyramid,
                &planes,
                &grid,
                &self.settings,
                self.scores,
            );
            field
        };

        let chained = |centre: usize, neighbour: usize, scored: bool| {
            let middle = (centre + neighbour) / 2;
            let first = direct(centre, middle, false);
            let second = direct(middle, neighbour, false);
            let field = DeviceField::new(client, &grid);
            let planes = field.planes(scored);
            chain(
                client,
                &sizing,
                &first.planes(false),
                &second.planes(false),
                &planes,
                &grid,
            );
            let seed = field.read(client, false).vectors;

            let centre_pyramid = device_pyramids[centre].pyramid();
            let neighbour_pyramid = device_pyramids[neighbour].pyramid();
            refine_seed(
                client,
                &sizing,
                &centre_pyramid,
                &neighbour_pyramid,
                &planes,
                &grid,
                self.refine_radius,
                self.scores,
            );
            if self.settings.pel == 2 {
                refine_half_pel(
                    client,
                    &sizing,
                    &centre_pyramid,
                    &neighbour_pyramid,
                    &planes,
                    &grid,
                );
            }

            let refined = field.read(client, scored);
            Chained { seed, field: refined }
        };

        let direct_1 = (0..frames - 1)
            .map(|centre| direct(centre, centre + 1, true).read(client, true))
            .collect();
        let direct_2 = (0..frames - 2)
            .map(|centre| direct(centre, centre + 2, false).read(client, false))
            .collect();
        let chained_2 = (0..frames - 2)
            .map(|first| chained(first, first + 2, true))
            .collect();
        let chained_back_2 = (0..frames - 2)
            .map(|first| chained(first + 2, first, false))
            .collect();

        Outputs {
            pyramids,
            direct_1,
            direct_2,
            chained_2,
            chained_back_2,
        }
    }
}

/// Every clip, with the device's outputs at each of its fixture settings.
fn every_run() -> Vec<(MotionClip, Vec<(&'static SettingFixture, Outputs)>)> {
    motion_clips()
        .into_iter()
        .zip(MOTION_FIXTURES.iter())
        .map(|(clip, fixture)| {
            assert_eq!(clip.name, fixture.clip, "the fixtures follow the clips' order");

            let runs = fixture
                .settings
                .iter()
                .map(|setting| {
                    let task = MotionRun::of_clip(&clip, setting);
                    (setting, backend().build(task))
                })
                .collect();
            (clip, runs)
        })
        .collect()
}

fn reference_direct(
    clip: &MotionClip,
    pyramids: &[Vec<Vec<f32>>],
    settings: &MotionSettings,
    scores: ConfidenceScores,
    centre: usize,
    neighbour: usize,
) -> ReferenceField {
    let width = clip.width as u32;
    let height = clip.height as u32;

    reference_search(
        &pyramids[centre],
        &pyramids[neighbour],
        width,
        height,
        settings,
        scores,
    )
}

/// The CPU's seed and refined field from frame `centre` to `neighbour`, two frames away either way.
fn reference_chained(
    clip: &MotionClip,
    pyramids: &[Vec<Vec<f32>>],
    setting: &SettingFixture,
    centre: usize,
    neighbour: usize,
) -> Chained {
    let width = clip.width as u32;
    let height = clip.height as u32;
    let settings = &setting.settings;
    let grid = BlockGrid::new(settings, width, height);
    let middle = (centre + neighbour) / 2;

    let first = reference_direct(clip, pyramids, settings, setting.scores, centre, middle);
    let second = reference_direct(clip, pyramids, settings, setting.scores, middle, neighbour);
    let seed = reference_chain_seed(&first.vectors, &second.vectors, &grid, width, height);
    let mut field = reference_refine_seed(
        &pyramids[centre][0],
        &pyramids[neighbour][0],
        width,
        height,
        &seed,
        &grid,
        setting.refine_radius,
        setting.scores,
    );
    if settings.pel == 2 {
        field = reference_refine_half_pel(
            &pyramids[centre][0],
            &pyramids[neighbour][0],
            width,
            height,
            &field,
            &grid,
        );
    }

    Chained { seed, field }
}

#[test]
fn every_device_pyramid_matches_the_reference() {
    for (clip, runs) in every_run() {
        for (setting, outputs) in runs {
            let expected = reference_clip_pyramids(&clip, setting.settings.pyramid_levels);

            for (frame, (found, wanted)) in outputs.pyramids.iter().zip(&expected).enumerate() {
                for (level, (found_level, wanted_level)) in found.iter().zip(wanted).enumerate() {
                    let label = format!("{} {} frame {frame} level {level}", clip.name, setting.label);

                    let (level_width, _) = level_dims(clip.width as u32, clip.height as u32, level as u32);

                    assert_same_plane(found_level, wanted_level, level_width, &label);
                }
            }
        }
    }
}

#[test]
fn every_device_direct_field_matches_the_reference() {
    for (clip, runs) in every_run() {
        for (setting, outputs) in runs {
            let pyramids = reference_clip_pyramids(&clip, setting.settings.pyramid_levels);
            let settings = &setting.settings;
            let across = BlockGrid::new(settings, clip.width as u32, clip.height as u32).across;

            for (centre, found) in outputs.direct_1.iter().enumerate() {
                let expected =
                    reference_direct(&clip, &pyramids, settings, setting.scores, centre, centre + 1);
                let label = format!("{} {} direct_1 field {centre}", clip.name, setting.label);

                assert_same_field(found, &expected, across, &label);
            }

            for (centre, found) in outputs.direct_2.iter().enumerate() {
                let expected =
                    reference_direct(&clip, &pyramids, settings, setting.scores, centre, centre + 2);
                let label = format!("{} {} direct_2 field {centre}", clip.name, setting.label);

                assert_same_field(found, &expected, across, &label);
            }
        }
    }
}

#[test]
fn every_device_chained_field_matches_the_reference() {
    for (clip, runs) in every_run() {
        for (setting, outputs) in runs {
            let pyramids = reference_clip_pyramids(&clip, setting.settings.pyramid_levels);
            let across = BlockGrid::new(&setting.settings, clip.width as u32, clip.height as u32).across;

            for (first, found) in outputs.chained_2.iter().enumerate() {
                let expected = reference_chained(&clip, &pyramids, setting, first, first + 2);
                let seed_label = format!("{} {} chained_2 seed {first}", clip.name, setting.label);
                let field_label = format!("{} {} chained_2 field {first}", clip.name, setting.label);

                assert_same_plane(&found.seed, &expected.seed, 2 * across, &seed_label);
                assert_same_field(&found.field, &expected.field, across, &field_label);
            }

            for (first, found) in outputs.chained_back_2.iter().enumerate() {
                let expected = reference_chained(&clip, &pyramids, setting, first + 2, first);
                let seed_label = format!("{} {} chained_back_2 seed {first}", clip.name, setting.label);
                let field_label = format!("{} {} chained_back_2 field {first}", clip.name, setting.label);

                assert_same_plane(&found.seed, &expected.seed, 2 * across, &seed_label);
                assert_same_field(&found.field, &expected.field, across, &field_label);
            }
        }
    }
}

#[test]
fn every_device_confidence_matches_av_denoise_and_the_reference() {
    for (clip, runs) in every_run() {
        for (setting, outputs) in runs {
            let pyramids = reference_clip_pyramids(&clip, setting.settings.pyramid_levels);
            let settings = &setting.settings;
            let blocks = BlockGrid::new(settings, clip.width as u32, clip.height as u32).blocks();

            let direct: Vec<f32> = outputs
                .direct_1
                .iter()
                .flat_map(|field| field.confidence.iter().copied())
                .collect();
            let reference_direct_scores: Vec<f32> = (0..outputs.direct_1.len())
                .flat_map(|centre| {
                    reference_direct(&clip, &pyramids, settings, setting.scores, centre, centre + 1)
                        .confidence
                })
                .collect();
            let expected_direct = fixture_f32s(setting.confidence_1);
            let direct_label = format!("{} {} confidence_1", clip.name, setting.label);
            assert_confidences(&direct, &expected_direct, blocks, &direct_label);
            assert_confidences(&direct, &reference_direct_scores, blocks, &direct_label);

            let chained: Vec<f32> = outputs
                .chained_2
                .iter()
                .flat_map(|chained| chained.field.confidence.iter().copied())
                .collect();
            let reference_chained_scores: Vec<f32> = (0..outputs.chained_2.len())
                .flat_map(|first| {
                    reference_chained(&clip, &pyramids, setting, first, first + 2)
                        .field
                        .confidence
                })
                .collect();
            let expected_chained = fixture_f32s(setting.confidence_chained_2);
            let chained_label = format!("{} {} confidence_chained_2", clip.name, setting.label);
            assert_confidences(&chained, &expected_chained, blocks, &chained_label);
            assert_confidences(&chained, &reference_chained_scores, blocks, &chained_label);
        }
    }
}

/// The top left `width` by `height` of every frame of `clip`.
fn cropped(clip: &MotionClip, width: usize, height: usize) -> MotionClip {
    let crop = |plane: &Vec<i32>| -> Vec<i32> {
        (0..height)
            .flat_map(|row| plane[row * clip.width..row * clip.width + width].iter().copied())
            .collect()
    };
    let frames = clip
        .frames
        .iter()
        .map(|planes| [crop(&planes[0]), Vec::new(), Vec::new()])
        .collect();

    MotionClip {
        name: clip.name,
        bits: clip.bits,
        width,
        height,
        frames,
        clean: Vec::new(),
        displacement: Vec::new(),
    }
}

/// Runs `settings` over `clip` on the device and checks every output against the CPU.
fn assert_device_matches(clip: &MotionClip, setting: &SettingFixture) {
    let task = MotionRun::of_clip(clip, setting);
    let outputs = backend().build(task);
    let pyramids = reference_clip_pyramids(clip, setting.settings.pyramid_levels);
    let settings = &setting.settings;
    let across = BlockGrid::new(settings, clip.width as u32, clip.height as u32).across;

    for (centre, found) in outputs.direct_1.iter().enumerate() {
        let expected = reference_direct(clip, &pyramids, settings, setting.scores, centre, centre + 1);
        let label = format!("{} direct_1 field {centre}", clip.name);

        assert_same_field(found, &expected, across, &label);
    }

    for (centre, found) in outputs.direct_2.iter().enumerate() {
        let expected = reference_direct(clip, &pyramids, settings, setting.scores, centre, centre + 2);
        let label = format!("{} direct_2 field {centre}", clip.name);

        assert_same_field(found, &expected, across, &label);
    }

    for (first, found) in outputs.chained_2.iter().enumerate() {
        let expected = reference_chained(clip, &pyramids, setting, first, first + 2);
        let label = format!("{} chained_2 field {first}", clip.name);

        assert_same_plane(&found.seed, &expected.seed, 2 * across, &label);
        assert_same_field(&found.field, &expected.field, across, &label);
    }

    for (first, found) in outputs.chained_back_2.iter().enumerate() {
        let expected = reference_chained(clip, &pyramids, setting, first + 2, first);
        let label = format!("{} chained_back_2 field {first}", clip.name);

        assert_same_plane(&found.seed, &expected.seed, 2 * across, &label);
        assert_same_field(&found.field, &expected.field, across, &label);
    }
}

fn pan_8() -> MotionClip {
    motion_clips()
        .into_iter()
        .next()
        .expect("pan_8 is the first clip")
}

#[test]
fn a_single_level_search_matches_the_reference() {
    let base = &MOTION_FIXTURES[0].settings[1];
    let settings = MotionSettings {
        pyramid_levels: 1,
        ..base.settings
    };
    let setting = SettingFixture { settings, ..*base };

    assert_device_matches(&pan_8(), &setting);
}

#[test]
fn a_frame_smaller_than_a_block_has_one_block() {
    let clip = cropped(&pan_8(), 6, 5);

    for levels in [1, 2, 3] {
        let base = &MOTION_FIXTURES[0].settings[0];
        let settings = MotionSettings {
            pyramid_levels: levels,
            ..base.settings
        };
        let grid = BlockGrid::new(&settings, 6, 5);
        assert_eq!((grid.across, grid.down), (1, 1));

        let setting = SettingFixture { settings, ..*base };
        assert_device_matches(&clip, &setting);
    }
}

/// The same fixture setting with half pixel vectors.
fn half_pel(base: &SettingFixture) -> SettingFixture {
    let settings = MotionSettings {
        pel: 2,
        ..base.settings
    };

    SettingFixture { settings, ..*base }
}

#[test]
fn every_device_half_pel_field_matches_the_reference() {
    for clip in motion_clips() {
        for fixture in &MOTION_FIXTURES[0].settings {
            assert_device_matches(&clip, &half_pel(fixture));
        }
    }
}

/// The share of interior blocks, two blocks in from every edge, of `fields` holding `vector`.
fn interior_share(fields: &[ReferenceField], grid: &BlockGrid, vector: (i32, i32)) -> f64 {
    let mut interior = 0usize;
    let mut matching = 0usize;

    for field in fields {
        for by in 2..grid.down - 2 {
            for bx in 2..grid.across - 2 {
                let slot = 2 * (by * grid.across + bx) as usize;

                interior += 1;
                if (field.vectors[slot], field.vectors[slot + 1]) == vector {
                    matching += 1;
                }
            }
        }
    }

    matching as f64 / interior as f64
}

#[test]
fn pel_two_finds_the_half_pixel_pan() {
    let clip = pan_8();
    let setting = half_pel(&MOTION_FIXTURES[0].settings[0]);
    let grid = BlockGrid::new(&setting.settings, clip.width as u32, clip.height as u32);
    let outputs = backend().build(MotionRun::of_clip(&clip, &setting));

    let share = interior_share(&outputs.direct_1, &grid, (3, 1));

    println!(
        "pan_8 interior blocks at (3, 1) half pixels: {:.1}%",
        share * 100.0
    );
    assert!(
        share >= 0.9,
        "only {:.1}% of interior blocks found (3, 1)",
        share * 100.0
    );
}

#[test]
fn a_whole_pixel_motion_keeps_its_vector() {
    let clip = pan_8();
    let setting = half_pel(&MOTION_FIXTURES[0].settings[0]);
    let width = clip.width;
    let centre = normalised(&clip.frames[0][0], clip.bits);
    let shifted: Vec<f32> = (0..clip.height)
        .flat_map(|row| {
            let source = &centre[row * width..(row + 1) * width];
            (0..width).map(move |column| source[column.saturating_sub(2)])
        })
        .collect();
    let grid = BlockGrid::new(&setting.settings, width as u32, clip.height as u32);
    let task = MotionRun {
        frames: vec![centre, shifted],
        width: width as u32,
        height: clip.height as u32,
        settings: setting.settings,
        refine_radius: setting.refine_radius,
        scores: setting.scores,
    };
    let outputs = backend().build(task);

    let share = interior_share(&outputs.direct_1, &grid, (4, 0));

    assert_eq!(share, 1.0, "{:.1}% of interior blocks kept (4, 0)", share * 100.0);
}
