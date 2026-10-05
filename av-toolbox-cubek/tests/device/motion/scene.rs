use av_toolbox_cubek::degrain::DegrainSettings;
use av_toolbox_cubek::motion::{
    BlockGrid,
    FieldPlanes,
    MotionField,
    MotionSettings,
    SceneChange,
    scene_change_flags,
};
use av_toolbox_cubek::testing::{motion_clips, reference_scene_changed};
use av_toolbox_gpu::{BuildContext, BuildTask};
// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::degrain::{DegrainRun, Frames};
use crate::support::{backend, empty, plane, read_u32, upload_f32};

fn settings_of(blksize: u32, overlap: u32) -> MotionSettings {
    MotionSettings {
        blksize,
        overlap,
        ..MotionSettings::default()
    }
}

#[test]
fn the_device_flags_match_the_reference() {
    let around = [1, 2, 3, 5, 6, 7];
    let rules = [
        DegrainSettings::default(),
        DegrainSettings {
            thscd1: 20.0,
            thscd2: 30,
            ..DegrainSettings::default()
        },
    ];

    let mut flagged = 0;
    let mut unflagged = 0;
    for clip in motion_clips()
        .iter()
        .filter(|clip| matches!(clip.name, "cut_8" | "pan_8"))
    {
        for (blksize, overlap) in [(16, 8), (8, 4)] {
            for settings in rules {
                let motion = settings_of(blksize, overlap);
                let frames = Frames::new(clip, 4, &around);
                let grid = BlockGrid::new(&motion, frames.width, frames.height);

                let settings = DegrainSettings {
                    temporal_radius: 3,
                    ..settings
                };
                let run = DegrainRun::new(frames, motion, settings);
                let found = backend().build(run);

                let rule = settings.scene_change();
                for (index, field) in found.fields.iter().enumerate() {
                    let expected = reference_scene_changed(&field.sads, &grid, rule);
                    let label = format!(
                        "{} blksize {blksize} thscd1 {} field {index}",
                        clip.name, settings.thscd1
                    );

                    assert_eq!(found.flags[index], expected, "{label}");
                    if expected {
                        flagged += 1;
                    } else {
                        unflagged += 1;
                    }
                }
            }
        }
    }

    assert!(
        flagged > 0,
        "no field was flagged, so the comparison proved nothing"
    );
    assert!(
        unflagged > 0,
        "every field was flagged, so the comparison proved nothing"
    );
}

/// The SAD of a block that is above the rule's threshold, and one that is not.
const HIGH: f32 = 2.0;
const LOW: f32 = 0.0;

/// The rule whose threshold is 1.0 on a block of 8, with `thscd2` of its own.
fn rule_of(thscd2: u32) -> SceneChange {
    SceneChange {
        thscd1: 255.0,
        thscd2,
    }
}

/// A grid of `blocks` blocks of 8 in one row.
fn row_of(blocks: u32) -> BlockGrid {
    BlockGrid {
        blksize: 8,
        step: 8,
        across: blocks,
        down: 1,
    }
}

/// SADs for `blocks` blocks, the last `high` of them above the threshold.
fn sads_with(blocks: u32, high: u32) -> Vec<f32> {
    (0..blocks)
        .map(|block| if block >= blocks - high { HIGH } else { LOW })
        .collect()
}

/// Flags fields of hand-written SADs on the device.
struct FlagRun {
    grid: BlockGrid,
    fields: Vec<Vec<f32>>,
    rule: SceneChange,
}

impl BuildTask for FlagRun {
    type Output = Vec<bool>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let blocks = self.grid.blocks() as u32;

        let vectors = vec![0.0f32; 2 * blocks as usize];
        let vector_handle = upload_f32(client, &vectors);
        let sad_handles: Vec<_> = self.fields.iter().map(|sads| upload_f32(client, sads)).collect();
        let fields: Vec<_> = sad_handles
            .iter()
            .map(|sads| MotionField {
                planes: FieldPlanes {
                    vectors: plane(&vector_handle, 2 * blocks, 1),
                    sads: plane(sads, blocks, 1),
                    confidence: None,
                },
                grid: self.grid,
                pel: 1,
            })
            .collect();

        let count = fields.len() as u32;
        let flag_handle = empty(client, fields.len() * size_of::<u32>());
        scene_change_flags(client, &fields, plane(&flag_handle, count, 1), self.rule);

        read_u32(client, &flag_handle)
            .iter()
            .map(|flag| *flag != 0)
            .collect()
    }
}

#[test]
fn the_flags_match_the_reference_on_every_count_at_the_boundaries() {
    let mut on_boundary = 0;
    for blocks in [1u32, 4, 35, 65] {
        let grid = row_of(blocks);
        let fields: Vec<Vec<f32>> = (0..=blocks).map(|high| sads_with(blocks, high)).collect();

        for thscd2 in [0, 1, 50, 51, 52, 127, 128, 254, 255] {
            let rule = rule_of(thscd2);
            let run = FlagRun {
                grid,
                fields: fields.clone(),
                rule,
            };
            let found = backend().build(run);

            for (high, flag) in found.iter().enumerate() {
                let expected = reference_scene_changed(&fields[high], &grid, rule);
                assert_eq!(*flag, expected, "{blocks} blocks, {high} high, thscd2 {thscd2}");

                if high as u32 * 255 == thscd2 * blocks {
                    assert!(
                        !flag,
                        "{blocks} blocks, {high} high, thscd2 {thscd2} is on the boundary"
                    );
                    on_boundary += 1;
                }
            }
        }
    }

    assert!(on_boundary >= 5, "only {on_boundary} cases sat on the boundary");
}

#[test]
fn a_share_just_over_the_boundary_is_a_scene_change() {
    let single = FlagRun {
        grid: row_of(1),
        fields: vec![sads_with(1, 1)],
        rule: rule_of(254),
    };
    assert_eq!(backend().build(single), [true]);

    let half = [127, 128].map(|thscd2| FlagRun {
        grid: row_of(4),
        fields: vec![sads_with(4, 2)],
        rule: rule_of(thscd2),
    });
    let [over, under] = half.map(|run| backend().build(run));
    assert_eq!(over, [true], "2 of 4 blocks at thscd2 127");
    assert_eq!(under, [false], "2 of 4 blocks at thscd2 128");
}
