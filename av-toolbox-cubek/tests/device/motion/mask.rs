use av_toolbox_cubek::motion::{
    BlockGrid,
    ConfidenceScores,
    FieldPlanes,
    MotionField,
    MotionSettings,
    PROGRESSIVE_SAD_MASK,
    SadMaskTables,
    SceneFlag,
    sad_mask,
};
use av_toolbox_cubek::testing::{
    MotionClip,
    assert_same_plane,
    motion_clips,
    normalised,
    reference_pyramid,
    reference_sad_mask,
    reference_search,
};
use av_toolbox_gpu::{BuildContext, BuildTask};

use crate::support::{Sizing, backend, empty, plane, read_f32, upload_f32, upload_u32};

/// A SAD on 8-bit, 8x8 blocks well past saturation at `ml = 10`.
const SATURATED: f32 = 1.0;

const SCORES: ConfidenceScores = ConfidenceScores {
    noise_floor: 1.0,
    thsad: 2.0,
};

/// One mask worked out on the device.
#[derive(Clone)]
struct Case {
    sads: Vec<f32>,
    grid: BlockGrid,
    flagged: bool,
    ml: f32,
    width: u32,
    height: u32,
}

impl Case {
    fn new(sads: Vec<f32>, grid: BlockGrid, (width, height): (u32, u32)) -> Self {
        Self {
            sads,
            grid,
            flagged: false,
            ml: PROGRESSIVE_SAD_MASK,
            width,
            height,
        }
    }

    fn reference(&self) -> Vec<f32> {
        let size = (self.width as usize, self.height as usize);

        reference_sad_mask(&self.sads, &self.grid, self.flagged, self.ml, size)
    }
}

/// Works out every case's mask on the device.
struct MaskRun {
    cases: Vec<Case>,
}

impl BuildTask for MaskRun {
    type Output = Vec<Vec<f32>>;

    fn run(self, build: &BuildContext) -> Self::Output {
        let client = build.client();
        let sizing = Sizing(build.frame_context());

        let mut outputs = Vec::with_capacity(self.cases.len());
        for case in &self.cases {
            let (vector_width, vector_height) = case.grid.vector_dims();
            let (sad_width, sad_height) = case.grid.sad_dims();
            let vectors = empty(client, case.grid.blocks() * 2 * size_of::<i32>());
            let sads = upload_f32(client, &case.sads);
            let flags = upload_u32(client, &[u32::from(case.flagged)]);
            let samples = (case.width * case.height) as usize;
            let output = empty(client, samples * size_of::<f32>());

            let field = MotionField {
                planes: FieldPlanes {
                    vectors: plane(&vectors, vector_width, vector_height),
                    sads: plane(&sads, sad_width, sad_height),
                    confidence: None,
                },
                grid: case.grid,
                pel: 1,
            };
            let flag = SceneFlag {
                flags: plane(&flags, 1, 1),
                index: 0,
            };
            let output_plane = plane(&output, case.width, case.height);
            let tables = SadMaskTables::new(client, &case.grid, case.ml, (case.width, case.height));
            sad_mask(client, &sizing, &field, flag, &tables, output_plane);

            outputs.push(read_f32(client, &output));
        }

        outputs
    }
}

fn run_cases(cases: &[Case]) -> Vec<Vec<f32>> {
    let run = MaskRun {
        cases: cases.to_vec(),
    };

    backend().build(run)
}

/// Panics where the device mask of any case differs from the reference.
fn assert_device_matches(cases: &[Case], label: &str) -> Vec<Vec<f32>> {
    let found = run_cases(cases);

    for (index, (case, mask)) in cases.iter().zip(&found).enumerate() {
        let expected = case.reference();
        let case_label = format!("{label} case {index}");

        assert_same_plane(mask, &expected, case.width, &case_label);
    }

    found
}

fn grid_of(blksize: u32, step: u32, across: u32, down: u32) -> BlockGrid {
    BlockGrid {
        blksize,
        step,
        across,
        down,
    }
}

/// `codes` as the mask's stored values.
fn bytes(codes: &[u32]) -> Vec<f32> {
    let inverse_peak = 1.0f32 / 255.0;

    codes.iter().map(|code| *code as f32 * inverse_peak).collect()
}

/// The SADs of `clip`'s frame 0 searched against frame 1, its luma scaled to `bits`.
fn clip_sads(clip: &MotionClip, settings: &MotionSettings, bits: u32) -> Vec<f32> {
    let width = clip.width as u32;
    let height = clip.height as u32;
    let levels = settings.pyramid_levels;
    let shift = bits - clip.bits;

    let pyramids: Vec<Vec<Vec<f32>>> = clip.frames[..2]
        .iter()
        .map(|planes| {
            let codes: Vec<i32> = planes[0].iter().map(|sample| sample << shift).collect();
            let luma = normalised(&codes, bits);
            reference_pyramid(&luma, width, height, levels)
        })
        .collect();
    let field = reference_search(&pyramids[0], &pyramids[1], width, height, settings, SCORES);

    field.sads
}

#[test]
fn the_mask_matches_the_reference() {
    let mut cases = Vec::new();
    for clip in motion_clips()
        .iter()
        .filter(|clip| matches!(clip.name, "pan_8" | "objects_8"))
    {
        for (blksize, overlap) in [(16, 8), (8, 4)] {
            let settings = MotionSettings {
                blksize,
                overlap,
                ..MotionSettings::default()
            };
            let size = (clip.width as u32, clip.height as u32);
            let grid = BlockGrid::new(&settings, size.0, size.1);

            for bits in [8, 10, 16] {
                let sads = clip_sads(clip, &settings, bits);
                cases.push(Case::new(sads, grid, size));
            }
        }
    }

    let found = assert_device_matches(&cases, "clip fields");

    let partial = found
        .iter()
        .flatten()
        .filter(|value| **value > 0.0 && **value < 1.0)
        .count();
    assert!(
        partial > 0,
        "every sample was 0 or 1, so the comparison proved little"
    );
}

#[test]
fn a_block_saturates_at_255() {
    let grid = grid_of(8, 8, 1, 1);
    let cases = [
        Case::new(vec![SATURATED], grid, (8, 8)),
        Case::new(vec![0.7], grid, (8, 8)),
    ];

    let found = assert_device_matches(&cases, "saturated");

    for mask in &found {
        assert!(mask.iter().all(|value| *value == 1.0), "{mask:?}");
    }
}

#[test]
fn a_block_below_saturation_scales_with_ml_and_blksize() {
    // An 8-bit SAD of 100 gives trunc(255 × 100 × 4 / (ml × blksize²)).
    let sad = 100.0f32 / 255.0;
    let worked = [(8, 10.0, 159), (8, 20.0, 79), (16, 10.0, 39), (4, 40.0, 159)];

    let cases: Vec<Case> = worked
        .iter()
        .map(|(blksize, ml, _)| Case {
            ml: *ml,
            ..Case::new(vec![sad], grid_of(*blksize, *blksize, 1, 1), (*blksize, *blksize))
        })
        .collect();
    let found = assert_device_matches(&cases, "below saturation");

    for (mask, (blksize, ml, code)) in found.iter().zip(worked) {
        let expected = bytes(&[code]);

        assert!(
            mask.iter().all(|value| *value == expected[0]),
            "blksize {blksize}, ml {ml}: {mask:?}, expected {code}"
        );
    }
}

#[test]
fn small_grids_resize_as_simple_resize_does() {
    let one = Case::new(vec![100.0 / 255.0], grid_of(8, 8, 1, 1), (8, 8));
    let two = Case::new(vec![0.0, SATURATED], grid_of(8, 4, 2, 1), (12, 8));
    let three = Case::new(vec![SATURATED, 0.0, SATURATED], grid_of(4, 2, 3, 1), (8, 4));
    let three_down = Case::new(vec![SATURATED, 0.0, SATURATED], grid_of(4, 2, 1, 3), (4, 8));
    let cases = [one, two, three, three_down];

    let found = assert_device_matches(&cases, "small grids");

    // `InitTables` from 2 to 12 weighs the second block by 0, 0, 0, 1365, 4096, 6826, 9557, 12288, 15018
    // and then 16384, out of 16384.
    let two_row = bytes(&[0, 0, 0, 21, 64, 106, 149, 191, 234, 255, 255, 255]);
    // From 3 to 8, the weights are 0, 1024, 7168, 13312 on the first pair and 3072, 9216, 15360, 16384 on
    // the second.
    let three_row = bytes(&[255, 239, 143, 48, 48, 143, 239, 255]);

    assert!(
        found[0].iter().all(|value| *value == bytes(&[159])[0]),
        "one block"
    );
    for row in found[1].chunks(12) {
        assert_eq!(row, two_row, "two blocks");
    }
    for row in found[2].chunks(8) {
        assert_eq!(row, three_row, "three blocks");
    }
    for (row, expected) in found[3].chunks(4).zip(&three_row) {
        assert!(row.iter().all(|value| value == expected), "three blocks down");
    }
}

#[test]
fn the_mask_repeats_past_the_covered_area() {
    // MVTools fits 14 by 7 blocks of 16, 8 apart, into 127x70, covering 120x64.
    let grid = grid_of(16, 8, 14, 7);
    let (width, height) = (127usize, 70usize);
    let sads: Vec<f32> = (0..grid.blocks())
        .map(|block| ((block * 37) % 101) as f32 / 255.0)
        .collect();
    let case = Case::new(sads, grid, (width as u32, height as u32));

    let found = assert_device_matches(std::slice::from_ref(&case), "overhang");
    let mask = &found[0];

    for row in mask.chunks(width) {
        let last_covered = row[119];
        assert!(
            row[120..].iter().all(|value| *value == last_covered),
            "a column past 120"
        );
    }

    let last_covered_row = &mask[63 * width..64 * width];
    for row in mask[64 * width..].chunks(width) {
        assert_eq!(row, last_covered_row, "a row past 64");
    }

    let column_119: Vec<f32> = mask.chunks(width).map(|row| row[119]).collect();
    assert!(
        column_119.windows(2).any(|pair| pair[0] != pair[1]),
        "the last covered column is flat, so the repeat proved nothing"
    );
}

#[test]
fn a_missing_neighbour_gives_a_zero_mask() {
    let grid = grid_of(8, 4, 3, 2);
    let case = Case {
        flagged: true,
        ..Case::new(vec![SATURATED; grid.blocks()], grid, (16, 12))
    };

    let found = assert_device_matches(std::slice::from_ref(&case), "flagged");

    assert!(found[0].iter().all(|value| *value == 0.0));
}
