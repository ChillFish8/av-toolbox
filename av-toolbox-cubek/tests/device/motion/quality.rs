use std::path::{Path, PathBuf};

use av_toolbox_cubek::degrain::DegrainSettings;
use av_toolbox_cubek::motion::{BlockGrid, MotionSettings};
use av_toolbox_cubek::testing::{
    MotionClip,
    motion_clips,
    normalised,
    object_mask,
    psnr,
    psnr_within,
    vector_accuracy,
    write_luma_y4m,
};

use super::compensate::{Case, CompensateRun};
use crate::degrain::{DegrainRun, Frames};
use crate::support::backend;

/// The frame the vector and compensation measurements centre on, against its next frame.
const CENTRE: usize = 3;

/// The frames degrain runs on, each with the two before it and the two after it.
const DEGRAINED: std::ops::RangeInclusive<usize> = 2..=5;

fn clip_named(name: &str) -> MotionClip {
    motion_clips()
        .into_iter()
        .find(|clip| clip.name == name)
        .expect("the clip should exist")
}

/// The field searching `CENTRE`'s blocks in the next frame, as vectors, with the grid they are on.
fn centre_field(clip: &MotionClip, motion: MotionSettings) -> (Vec<i32>, BlockGrid) {
    let frames = Frames::new(clip, CENTRE, &[CENTRE + 1]);
    let grid = BlockGrid::new(&motion, frames.width, frames.height);
    let run = DegrainRun::new(frames, motion, DegrainSettings::default());
    let mut outputs = backend().build(run);
    let field = outputs.fields.remove(0);

    (field.vectors, grid)
}

fn vector_accuracy_of(clip: &MotionClip, pel: u32) -> f64 {
    let motion = MotionSettings {
        pel,
        overlap: 8,
        ..MotionSettings::default()
    };
    let (vectors, grid) = centre_field(clip, motion);
    let displacement = &clip.displacement[CENTRE];

    vector_accuracy(&vectors, &grid, displacement, pel, clip.width, clip.height)
}

/// The PSNR of the next frame compensated onto `CENTRE`, against the clean `CENTRE`.
fn compensated_psnr(clip: &MotionClip, pel: u32, overlap: u32) -> f64 {
    let motion = MotionSettings {
        pel,
        overlap,
        ..MotionSettings::default()
    };
    let (vectors, grid) = centre_field(clip, motion);
    let neighbour = normalised(&clip.frames[CENTRE + 1][0], clip.bits);
    let size = (clip.width as u32, clip.height as u32);
    let case = Case::new(neighbour, size, vectors, grid).pel(pel);
    let mut outputs = backend().build(CompensateRun { cases: vec![case] });
    let compensated = outputs.remove(0);
    let truth = normalised(&clip.clean[CENTRE][0], clip.bits);

    psnr(&compensated, &truth, 1.0)
}

/// Degrain's luma over every frame in `DEGRAINED`, against the clip's noisy and clean luma.
struct Degrained {
    frames: Vec<Vec<f32>>,
    noisy: Vec<Vec<f32>>,
    clean: Vec<Vec<f32>>,
    masks: Vec<Vec<bool>>,
}

impl Degrained {
    fn new(clip: &MotionClip, radius: usize, motion: MotionSettings) -> Self {
        let settings = DegrainSettings {
            temporal_radius: radius as u32,
            ..DegrainSettings::default()
        };

        let mut degrained = Self {
            frames: Vec::new(),
            noisy: Vec::new(),
            clean: Vec::new(),
            masks: Vec::new(),
        };
        for centre in DEGRAINED {
            let around: Vec<usize> = (centre - radius..centre)
                .chain(centre + 1..=centre + radius)
                .collect();
            let frames = Frames::new(clip, centre, &around);
            let run = DegrainRun::new(frames, motion, settings);
            let outputs = backend().build(run);

            let noisy = normalised(&clip.frames[centre][0], clip.bits);
            let clean = normalised(&clip.clean[centre][0], clip.bits);
            let mask = object_mask(clip, centre);

            degrained.frames.push(outputs.luma);
            degrained.noisy.push(noisy);
            degrained.clean.push(clean);
            degrained.masks.push(mask);
        }

        degrained
    }

    fn whole_frame(&self) -> (f64, f64) {
        let found = self.frames.concat();
        let noisy = self.noisy.concat();
        let clean = self.clean.concat();

        (psnr(&found, &clean, 1.0), psnr(&noisy, &clean, 1.0))
    }

    fn objects(&self) -> (f64, f64) {
        let found = self.frames.concat();
        let noisy = self.noisy.concat();
        let clean = self.clean.concat();
        let masks = self.masks.concat();

        (
            psnr_within(&found, &clean, &masks, 1.0),
            psnr_within(&noisy, &clean, &masks, 1.0),
        )
    }
}

/// Where the quality tests write their clips and degrain outputs for a comparison by hand, if asked.
fn report_directory() -> Option<PathBuf> {
    let directory = std::env::var_os("AVTB_MOTION_REPORT_DIR")?;
    let path = PathBuf::from(directory);
    std::fs::create_dir_all(&path).expect("the report directory should be created");

    Some(path)
}

fn write_report(path: &Path, clip: &MotionClip, frames: &[Vec<f32>]) {
    write_luma_y4m(path, clip.width, clip.height, clip.bits, frames).expect("the report should be written");
}

/// Writes `degrained`'s frames, which are the clip's frames 2 to 5, as `<clip>_degrain_<label>.y4m`.
fn report_degrained(clip: &MotionClip, label: &str, degrained: &Degrained) {
    let Some(directory) = report_directory() else {
        return;
    };

    let file = format!("{}_degrain_{label}.y4m", clip.name);
    let path = directory.join(file);
    write_report(&path, clip, &degrained.frames);
}

fn luma_frames(clip: &MotionClip, frames: &[[Vec<i32>; 3]]) -> Vec<Vec<f32>> {
    frames
        .iter()
        .map(|planes| normalised(&planes[0], clip.bits))
        .collect()
}

#[test]
fn the_clips_are_written_for_the_report_when_asked() {
    let Some(directory) = report_directory() else {
        return;
    };

    for clip in motion_clips() {
        let clean = luma_frames(&clip, &clip.clean);
        let clean_path = directory.join(format!("{}_clean.y4m", clip.name));
        write_report(&clean_path, &clip, &clean);

        if clip.frames != clip.clean {
            let noisy = luma_frames(&clip, &clip.frames);
            let noisy_path = directory.join(format!("{}_noisy.y4m", clip.name));
            write_report(&noisy_path, &clip, &noisy);
        }
    }
}

#[test]
fn vectors_reach_their_accuracy_floors() {
    // (clip, pel, floor, measured)
    let floors = [
        ("pan_8", 1, 0.95, 1.0),
        ("pan_8", 2, 0.95, 1.0),
        ("pan_10", 1, 0.95, 1.0),
        ("pan_10", 2, 0.95, 1.0),
        ("objects_8", 1, 0.9, 0.9333),
        ("objects_8", 2, 0.8, 0.8571),
    ];

    for (name, pel, floor, measured) in floors {
        let clip = clip_named(name);
        let accuracy = vector_accuracy_of(&clip, pel);

        println!("{name} pel {pel} accuracy {accuracy:.4}, measured {measured}");
        assert!(
            accuracy >= floor,
            "{name} pel {pel} accuracy {accuracy} fell below {floor}"
        );
    }
}

#[test]
fn compensation_reaches_its_psnr_floors() {
    // (clip, pel, overlap, floor in dB, measured)
    let floors = [
        ("pan_8", 1, 0, 31.0, 31.230),
        ("pan_8", 1, 8, 33.0, 33.205),
        ("pan_8", 2, 0, 42.0, 42.458),
        ("pan_8", 2, 8, 42.5, 42.680),
        ("pan_10", 1, 0, 31.0, 31.246),
        ("pan_10", 1, 8, 32.5, 33.017),
        ("pan_10", 2, 0, 42.0, 42.594),
        ("pan_10", 2, 8, 42.5, 42.829),
        ("objects_8", 1, 0, 27.0, 27.434),
        ("objects_8", 1, 8, 30.5, 30.997),
        ("objects_8", 2, 0, 28.0, 28.399),
        ("objects_8", 2, 8, 33.0, 33.312),
    ];

    for (name, pel, overlap, floor, measured) in floors {
        let clip = clip_named(name);
        let value = compensated_psnr(&clip, pel, overlap);

        println!("{name} pel {pel} overlap {overlap} psnr {value:.3}, measured {measured}");
        assert!(
            value >= floor,
            "{name} pel {pel} overlap {overlap} PSNR {value} fell below {floor}"
        );
    }
}

#[test]
fn half_pixel_vectors_compensate_better_than_whole_pixel_ones() {
    // (clip, margin in dB, measured margin)
    let margins = [
        ("pan_8", 9.0, 9.475),
        ("pan_10", 9.5, 9.812),
        ("objects_8", 2.0, 2.315),
    ];

    for (name, margin, measured) in margins {
        let clip = clip_named(name);
        let whole = compensated_psnr(&clip, 1, 8);
        let half = compensated_psnr(&clip, 2, 8);

        let gain = half - whole;

        println!("{name} pel 2 over pel 1: {gain:.3}, measured {measured}");
        assert!(
            gain >= margin,
            "{name} pel 2 beats pel 1 by {gain} dB, expected {margin}"
        );
    }
}

#[test]
fn overlapped_blocks_compensate_better_than_separate_ones() {
    // (clip, pel, margin in dB, measured margin)
    let margins = [
        ("pan_8", 1, 1.5, 1.975),
        ("pan_8", 2, 0.1, 0.222),
        ("pan_10", 1, 1.5, 1.772),
        ("pan_10", 2, 0.1, 0.235),
        ("objects_8", 1, 3.0, 3.563),
        ("objects_8", 2, 4.5, 4.913),
    ];

    for (name, pel, margin, measured) in margins {
        let clip = clip_named(name);
        let separate = compensated_psnr(&clip, pel, 0);
        let overlapped = compensated_psnr(&clip, pel, 8);
        let gain = overlapped - separate;

        println!("{name} pel {pel} overlap 8 over 0: {gain:.3}, measured {measured}");
        assert!(
            gain >= margin,
            "{name} pel {pel} overlap gains {gain} dB, expected {margin}"
        );
    }
}

#[test]
fn degrain_beats_the_noisy_input_over_the_whole_frame() {
    // (clip, radius, floor in dB, measured, margin over the noisy input, measured margin)
    let floors = [
        ("noisy_8", 1, 41.0, 41.555, 3.0, 3.069),
        ("noisy_8", 2, 42.0, 42.220, 3.5, 3.734),
        ("noisy_10", 1, 41.5, 41.714, 3.0, 3.127),
        ("noisy_10", 2, 42.0, 42.382, 3.5, 3.795),
    ];

    for (name, radius, floor, measured, margin, measured_margin) in floors {
        let clip = clip_named(name);
        let degrained = Degrained::new(&clip, radius, MotionSettings::default());
        let (found, noisy) = degrained.whole_frame();
        let gain = found - noisy;

        println!(
            "{name} radius {radius} whole frame {found:.3} (measured {measured}) over noisy {gain:.3} (measured {measured_margin})"
        );
        assert!(
            found >= floor,
            "{name} radius {radius} PSNR {found} fell below {floor}"
        );
        assert!(
            gain >= margin,
            "{name} radius {radius} gains {gain} dB, expected {margin}"
        );

        let label = format!("r{radius}");
        report_degrained(&clip, &label, &degrained);
    }
}

/// Objects are about one default block across, so blocks of 8 are used to sit inside them.
#[test]
fn degrain_beats_the_noisy_input_over_the_objects_with_small_blocks() {
    let motion = MotionSettings {
        blksize: 8,
        overlap: 4,
        ..MotionSettings::default()
    };
    // (clip, radius, floor in dB, measured, margin over the noisy input, measured margin)
    let floors = [
        ("noisy_8", 1, 39.5, 40.020, 1.5, 1.748),
        ("noisy_8", 2, 40.5, 40.990, 2.5, 2.72),
        ("noisy_10", 1, 40.0, 40.233, 1.5, 1.863),
        ("noisy_10", 2, 41.0, 41.239, 2.5, 2.869),
    ];

    for (name, radius, floor, measured, margin, measured_margin) in floors {
        let clip = clip_named(name);
        let degrained = Degrained::new(&clip, radius, motion);
        let (found, noisy) = degrained.objects();
        let gain = found - noisy;

        println!(
            "{name} radius {radius} objects {found:.3} (measured {measured}) over noisy {gain:.3} (measured {measured_margin})"
        );
        assert!(
            found >= floor,
            "{name} radius {radius} PSNR {found} fell below {floor}"
        );
        assert!(
            gain >= margin,
            "{name} radius {radius} gains {gain} dB, expected {margin}"
        );

        let label = format!("r{radius}_blk8");
        report_degrained(&clip, &label, &degrained);
    }
}
