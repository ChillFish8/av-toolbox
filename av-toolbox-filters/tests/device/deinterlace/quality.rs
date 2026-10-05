use std::path::{Path, PathBuf};

use av_toolbox_cubek::testing::{FieldClip, Interlace, field_clips, psnr, write_y4m};
use av_toolbox_gpu::FieldOrder;
use serde_json::json;

use super::bwdif;
use super::qtgmc::{Clip, deinterlaced, inputs, interlaced, run_clip};
use crate::stream::Output;
use crate::support::{element_of, from_bytes};

/// A clip's planes as code values, one entry per frame.
type Frames = Vec<[Vec<i32>; 3]>;

/// The quality QTGMC must keep on one clip, and the margins it must keep over its alternatives.
///
/// Only the noisy clips have a margin over bwdif.
struct Floors {
    clip: &'static str,
    /// The least luma PSNR over every output field, in dB.
    qtgmc_psnr: f64,
    /// How many dB QTGMC must lead bwdif by, on the clips that ask for it.
    bwdif_margin: Option<f64>,
    /// How many dB `tr2 = 1` must lead `tr2 = 0` by, on the clips that ask for it.
    tr2_margin: Option<f64>,
}

const FLOORS: [Floors; 5] = [
    Floors {
        clip: "field_pan_8",
        // Measured 44.495.
        qtgmc_psnr: 44.0,
        bwdif_margin: None,
        tr2_margin: None,
    },
    Floors {
        clip: "field_objects_8",
        // Measured 39.376.
        qtgmc_psnr: 39.0,
        bwdif_margin: None,
        tr2_margin: None,
    },
    Floors {
        clip: "field_noisy_8",
        // Measured 37.449, over bwdif by 0.784 and over tr2 0 by 0.268.
        qtgmc_psnr: 37.0,
        bwdif_margin: Some(0.5),
        tr2_margin: Some(0.2),
    },
    Floors {
        clip: "field_noisy_10",
        // Measured 37.504, over bwdif by 0.743 and over tr2 0 by 0.229.
        qtgmc_psnr: 37.0,
        bwdif_margin: Some(0.5),
        tr2_margin: Some(0.1),
    },
    Floors {
        clip: "field_static_8",
        // Measured 18.730.
        qtgmc_psnr: 18.0,
        bwdif_margin: None,
        tr2_margin: None,
    },
];

/// The most QTGMC's mean luma change between consecutive fields of `field_static_8` may reach.
///
/// Measured 0.0374, against 0.1631 for its bare bob.
const STATIC_SHIMMER_CEILING: f64 = 0.04;

/// Where the report clips go, if `AVTB_QTGMC_REPORT_DIR` is set.
fn report_directory() -> Option<PathBuf> {
    let directory = std::env::var_os("AVTB_QTGMC_REPORT_DIR")?;
    let path = PathBuf::from(directory);
    std::fs::create_dir_all(&path).expect("the report directory should be created");

    Some(path)
}

/// Writes the clip's interlaced input, its truth, its noise-free truth and QTGMC's output.
fn write_report(directory: &Path, clip: &Clip, truth: &FieldClip, output: &Frames) {
    let name = &truth.name;
    let files = [
        ("input", 25, Interlace::TopFirst, &clip.frames),
        ("truth", 50, Interlace::Progressive, &truth.truth),
        ("clean", 50, Interlace::Progressive, &truth.clean),
        ("qtgmc", 50, Interlace::Progressive, output),
    ];

    for (label, rate, interlace, frames) in files {
        let path = directory.join(format!("{name}_{label}.y4m"));
        let size = (clip.width, clip.height);
        write_y4m(&path, frames, size, clip.bits, rate, interlace)
            .expect("the report file should be written");
    }
}

/// The luma of `frames`, one plane per entry, normalised to between 0 and 1.
fn normalised_luma(frames: &Frames, bits: u32) -> Vec<Vec<f32>> {
    let peak = ((1u32 << bits) - 1) as f32;

    frames
        .iter()
        .map(|planes| planes[0].iter().map(|code| *code as f32 / peak).collect())
        .collect()
}

/// A stage's outputs as code values.
fn output_frames(outputs: &[Output], clip: &Clip) -> Frames {
    let element = element_of(clip.sample());

    outputs
        .iter()
        .map(|output| {
            let planes: Vec<Vec<i32>> = output
                .planes
                .iter()
                .map(|bytes| {
                    let codes = from_bytes(bytes, element);
                    codes.into_iter().map(i32::from).collect()
                })
                .collect();

            let mut planes = planes.into_iter();
            [
                planes.next().expect("a luma plane"),
                planes.next().expect("a U plane"),
                planes.next().expect("a V plane"),
            ]
        })
        .collect()
}

/// The PSNR of every field of `found` against `wanted` together, with a peak of 1.
fn field_psnr(found: &[Vec<f32>], wanted: &[Vec<f32>]) -> f64 {
    assert_eq!(found.len(), wanted.len(), "one output per field");

    let found: Vec<f32> = found.concat();
    let wanted: Vec<f32> = wanted.concat();

    psnr(&found, &wanted, 1.0)
}

/// The mean absolute luma change between consecutive fields.
fn shimmer(fields: &[Vec<f32>]) -> f64 {
    let mut total = 0.0f64;
    let mut count = 0usize;

    for pair in fields.windows(2) {
        for (before, after) in pair[0].iter().zip(&pair[1]) {
            total += f64::from((after - before).abs());
            count += 1;
        }
    }

    total / count as f64
}

/// `bwdif` at `spatial: eedi3`, double rate, top field first.
fn run_bwdif(clip: &Clip) -> Frames {
    let input = interlaced(clip, FieldOrder::TopFirst);
    let output = deinterlaced(input, true);
    let chunk = inputs(clip, &input, 0);
    let args = json!({ "rate": "double", "order": "top", "spatial": { "kind": "eedi3" } });

    let outputs = bwdif::run(args, input, output, vec![chunk]);

    output_frames(&outputs, clip)
}

/// QTGMC with `args`.
fn run_qtgmc(clip: &Clip, args: serde_json::Value) -> Frames {
    let run = run_clip(args, clip);

    output_frames(&run.outputs, clip)
}

/// What one clip measured.
struct Measured {
    qtgmc: f64,
    bwdif: f64,
    qtgmc_shimmer: f64,
    bare_shimmer: Option<f64>,
    tr2_off: Option<f64>,
}

fn measure(name: &str) -> Measured {
    let truth = field_clips()
        .into_iter()
        .find(|candidate| candidate.name == name)
        .expect("the clip exists");
    let clip = Clip::named(name);
    let clean = normalised_luma(&truth.clean, clip.bits);

    let qtgmc_frames = run_qtgmc(&clip, json!({}));
    let qtgmc_luma = normalised_luma(&qtgmc_frames, clip.bits);
    let bwdif_frames = run_bwdif(&clip);
    let bwdif_luma = normalised_luma(&bwdif_frames, clip.bits);

    let tr2_off = name.starts_with("field_noisy").then(|| {
        let frames = run_qtgmc(&clip, json!({ "tr2": 0 }));
        let luma = normalised_luma(&frames, clip.bits);
        field_psnr(&luma, &clean)
    });

    let bare_shimmer = (name == "field_static_8").then(|| {
        let args = json!({ "tr0": 0, "tr1": 0, "tr2": 0, "sharpness": 0 });
        let frames = run_qtgmc(&clip, args);
        let luma = normalised_luma(&frames, clip.bits);
        shimmer(&luma)
    });

    if let Some(directory) = report_directory() {
        write_report(&directory, &clip, &truth, &qtgmc_frames);
    }

    Measured {
        qtgmc: field_psnr(&qtgmc_luma, &clean),
        bwdif: field_psnr(&bwdif_luma, &clean),
        qtgmc_shimmer: shimmer(&qtgmc_luma),
        bare_shimmer,
        tr2_off,
    }
}

fn assert_quality(name: &str) {
    let measured = measure(name);
    let floors = FLOORS
        .iter()
        .find(|floors| floors.clip == name)
        .expect("the clip has floors");

    assert!(
        measured.qtgmc >= floors.qtgmc_psnr,
        "{name}: QTGMC {:.3} dB is under its floor of {} dB",
        measured.qtgmc,
        floors.qtgmc_psnr
    );

    if let Some(margin) = floors.bwdif_margin {
        let lead = measured.qtgmc - measured.bwdif;
        assert!(
            lead >= margin,
            "{name}: QTGMC leads bwdif by {lead:.3} dB, under {margin} dB"
        );
    }

    if let (Some(margin), Some(off)) = (floors.tr2_margin, measured.tr2_off) {
        let lead = measured.qtgmc - off;
        assert!(
            lead >= margin,
            "{name}: tr2 1 leads tr2 0 by {lead:.3} dB, under {margin} dB"
        );
    }

    if let Some(bare) = measured.bare_shimmer {
        assert!(
            measured.qtgmc_shimmer < bare,
            "{name}: QTGMC shimmers {:.5}, its bare bob {bare:.5}",
            measured.qtgmc_shimmer
        );
        assert!(
            measured.qtgmc_shimmer <= STATIC_SHIMMER_CEILING,
            "{name}: QTGMC shimmers {:.5}, over {STATIC_SHIMMER_CEILING}",
            measured.qtgmc_shimmer
        );
    }
}

#[test]
fn quality_on_field_pan_8() {
    assert_quality("field_pan_8");
}

#[test]
fn quality_on_field_objects_8() {
    assert_quality("field_objects_8");
}

#[test]
fn quality_on_field_noisy_8() {
    assert_quality("field_noisy_8");
}

#[test]
fn quality_on_field_noisy_10() {
    assert_quality("field_noisy_10");
}

#[test]
fn quality_on_field_static_8() {
    assert_quality("field_static_8");
}
