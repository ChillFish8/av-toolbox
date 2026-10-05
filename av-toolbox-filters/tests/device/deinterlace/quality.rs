use std::path::{Path, PathBuf};

use av_toolbox_cubek::testing::{FieldClip, Interlace, badly_clips, field_clips, psnr, write_y4m};
use av_toolbox_gpu::{FieldOrder, Subsampling};
use serde_json::json;

use super::bwdif;
use super::qtgmc::{
    Clip,
    deinterlaced,
    inputs,
    interlaced,
    mode_args,
    progressive_format,
    run_chunks,
    run_clip,
};
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

/// The quality of `progressive` at its defaults on one clip, and the margins it must keep.
///
/// The defaults resharpen, which brings back noise along with detail, so they have no ordering against
/// the input. The input only has to be beaten with the sharpening off.
struct ProgressiveFloors {
    clip: &'static str,
    /// The least luma PSNR over every output frame, in dB.
    psnr: f64,
    /// How many dB `tr2 = 1` must lead `tr2 = 0` by, on the clips that ask for it.
    tr2_margin: Option<f64>,
    /// How many dB `sharpness = 0` must lead the input by, on the clips that ask for it.
    unsharpened_margin: Option<f64>,
}

const PROGRESSIVE_FLOORS: [ProgressiveFloors; 3] = [
    ProgressiveFloors {
        clip: "field_noisy_8",
        // Measured 38.481, over tr2 0 by 0.243. Without sharpening it leads the input by 0.790.
        psnr: 38.43,
        tr2_margin: Some(0.19),
        unsharpened_margin: Some(0.74),
    },
    ProgressiveFloors {
        clip: "field_noisy_10",
        // Measured 38.538, over tr2 0 by 0.242. Without sharpening it leads the input by 0.720.
        psnr: 38.48,
        tr2_margin: Some(0.19),
        unsharpened_margin: Some(0.67),
    },
    ProgressiveFloors {
        clip: "field_pan_8",
        // Measured 45.262.
        psnr: 45.21,
        tr2_margin: None,
        unsharpened_margin: None,
    },
];

/// The quality of `repair` and `repair_reversed` on one badly deinterlaced clip.
///
/// The margin between the two modes applies to every clip, whether or not it has a reversed counterpart.
struct RepairFloors {
    clip: &'static str,
    /// The least luma PSNR of `repair`, in dB.
    repair_psnr: f64,
    /// The least luma PSNR of `repair_reversed`, in dB.
    reversed_psnr: f64,
    /// How many dB `repair` must lead its input by, on the clips that ask for it.
    input_margin: Option<f64>,
    /// How many dB the mode that matches the clip's field order must lead the other by.
    order_margin: Option<f64>,
}

const REPAIR_FLOORS: [RepairFloors; 5] = [
    RepairFloors {
        clip: "badly_pan_8",
        // Measured 46.101 and 43.543, over the input by 0.958 and over repair_reversed by 2.558.
        repair_psnr: 46.05,
        reversed_psnr: 43.49,
        input_margin: Some(0.9),
        order_margin: Some(2.5),
    },
    RepairFloors {
        clip: "badly_objects_8",
        // Measured 37.711 and 36.319, over the input by 0.205 and over repair_reversed by 1.392.
        // A repair that does nothing lands at 0 or below, so 0.15 still catches it.
        repair_psnr: 37.66,
        reversed_psnr: 36.26,
        input_margin: Some(0.15),
        order_margin: Some(1.34),
    },
    RepairFloors {
        clip: "badly_static_8",
        // Measured 18.106 and 17.537, over the input by 2.072 and over repair_reversed by 0.569.
        repair_psnr: 18.05,
        reversed_psnr: 17.48,
        input_margin: Some(2.02),
        order_margin: Some(0.51),
    },
    RepairFloors {
        clip: "badly_pan_8_reversed",
        // Measured 43.645 and 46.350, with repair_reversed over repair by 2.705.
        repair_psnr: 43.59,
        reversed_psnr: 46.3,
        input_margin: None,
        order_margin: Some(2.65),
    },
    RepairFloors {
        clip: "badly_static_8_reversed",
        // Measured 17.470 and 18.106, with repair_reversed over repair by 0.636.
        repair_psnr: 17.42,
        reversed_psnr: 18.05,
        input_margin: None,
        order_margin: Some(0.58),
    },
];

/// The most `repair`'s mean luma change between consecutive frames of `badly_static_8` may reach.
///
/// Measured 0.0575, against 0.1629 for its input.
const REPAIR_SHIMMER_CEILING: f64 = 0.06;

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

/// What `progressive` and the repair modes may be asked to restore.
struct Restoration {
    name: &'static str,
    bits: u32,
    width: usize,
    height: usize,
    frames: Frames,
    clean: Frames,
}

/// Runs `mode` on the progressive frames of `restoration`.
fn run_progressive(restoration: &Restoration, args: serde_json::Value) -> Frames {
    let clip = Clip {
        name: restoration.name.to_string(),
        bits: restoration.bits,
        width: restoration.width,
        height: restoration.height,
        subsampling: Subsampling::Yuv420,
        frames: restoration.frames.clone(),
    };
    let input = progressive_format(&clip);
    let output = deinterlaced(input, false);
    let chunk = inputs(&clip, &input, 0);

    let mut runs = run_chunks(args, input, output, vec![chunk]);
    let run = runs.pop().expect("one chunk was run");

    output_frames(&run.outputs, &clip)
}

/// The noisy field clips and the clean pan, rendered as progressive frames.
fn noisy_restorations() -> Vec<Restoration> {
    let names = ["field_noisy_8", "field_noisy_10", "field_pan_8"];

    field_clips()
        .into_iter()
        .filter(|clip| names.contains(&clip.name))
        .map(|clip| Restoration {
            name: clip.name,
            bits: clip.bits,
            width: clip.width,
            height: clip.height,
            frames: clip.truth,
            clean: clip.clean,
        })
        .collect()
}

/// Every badly deinterlaced clip.
fn badly_restorations() -> Vec<Restoration> {
    badly_clips()
        .into_iter()
        .map(|clip| Restoration {
            name: clip.name,
            bits: clip.bits,
            width: clip.width,
            height: clip.height,
            frames: clip.frames,
            clean: clip.clean,
        })
        .collect()
}

fn restoration_psnr(frames: &Frames, restoration: &Restoration) -> f64 {
    let found = normalised_luma(frames, restoration.bits);
    let clean = normalised_luma(&restoration.clean, restoration.bits);

    field_psnr(&found, &clean)
}

/// Writes the clip's input, its clean truth and the repaired output.
fn write_repair_report(directory: &Path, restoration: &Restoration, output: &Frames) {
    let name = restoration.name;
    let size = (restoration.width, restoration.height);
    let files = [
        ("input", &restoration.frames),
        ("clean", &restoration.clean),
        ("output", output),
    ];

    for (label, frames) in files {
        let path = directory.join(format!("{name}_{label}.y4m"));
        write_y4m(&path, frames, size, restoration.bits, 25, Interlace::Progressive)
            .expect("the report file should be written");
    }
}

fn find_restoration(restorations: Vec<Restoration>, name: &str) -> Restoration {
    restorations
        .into_iter()
        .find(|restoration| restoration.name == name)
        .expect("the clip exists")
}

fn assert_progressive_quality(name: &str) {
    let restoration = find_restoration(noisy_restorations(), name);
    let floors = PROGRESSIVE_FLOORS
        .iter()
        .find(|floors| floors.clip == name)
        .expect("the clip has floors");

    let defaults = run_progressive(&restoration, mode_args("progressive", 1));
    let defaults_psnr = restoration_psnr(&defaults, &restoration);
    assert!(
        defaults_psnr >= floors.psnr,
        "{name}: progressive {defaults_psnr:.3} dB is under its floor of {} dB",
        floors.psnr
    );

    if let Some(margin) = floors.tr2_margin {
        let off = run_progressive(&restoration, mode_args("progressive", 0));
        let off_psnr = restoration_psnr(&off, &restoration);
        let lead = defaults_psnr - off_psnr;
        assert!(
            lead >= margin,
            "{name}: tr2 1 leads tr2 0 by {lead:.3} dB, under {margin} dB"
        );
    }

    if let Some(margin) = floors.unsharpened_margin {
        let args = json!({ "input": "progressive", "sharpness": 0 });
        let unsharpened = run_progressive(&restoration, args);
        let input_psnr = restoration_psnr(&restoration.frames, &restoration);
        let unsharpened_psnr = restoration_psnr(&unsharpened, &restoration);
        let lead = unsharpened_psnr - input_psnr;
        assert!(
            lead >= margin,
            "{name}: unsharpened progressive leads its input by {lead:.3} dB, under {margin} dB"
        );
    }
}

fn assert_repair_quality(name: &str) {
    let restoration = find_restoration(badly_restorations(), name);
    let floors = REPAIR_FLOORS
        .iter()
        .find(|floors| floors.clip == name)
        .expect("the clip has floors");
    let reversed_clip = name.ends_with("_reversed");

    let repaired = run_progressive(&restoration, mode_args("repair", 1));
    let reversed = run_progressive(&restoration, mode_args("repair_reversed", 1));
    let repair_psnr = restoration_psnr(&repaired, &restoration);
    let reversed_psnr = restoration_psnr(&reversed, &restoration);
    let input_psnr = restoration_psnr(&restoration.frames, &restoration);

    if let Some(directory) = report_directory() {
        let matching = if reversed_clip { &reversed } else { &repaired };
        write_repair_report(&directory, &restoration, matching);
    }

    assert!(
        repair_psnr >= floors.repair_psnr,
        "{name}: repair {repair_psnr:.3} dB is under its floor of {} dB",
        floors.repair_psnr
    );
    assert!(
        reversed_psnr >= floors.reversed_psnr,
        "{name}: repair_reversed {reversed_psnr:.3} dB is under its floor of {} dB",
        floors.reversed_psnr
    );

    if let Some(margin) = floors.input_margin {
        let lead = repair_psnr - input_psnr;
        assert!(
            lead >= margin,
            "{name}: repair leads its input by {lead:.3} dB, under {margin} dB"
        );
    }

    if let Some(margin) = floors.order_margin {
        let lead = if reversed_clip {
            reversed_psnr - repair_psnr
        } else {
            repair_psnr - reversed_psnr
        };
        assert!(
            lead >= margin,
            "{name}: the mode matching the field order leads by {lead:.3} dB, under {margin} dB"
        );
    }

    if name == "badly_static_8" {
        let input_shimmer = shimmer(&normalised_luma(&restoration.frames, restoration.bits));
        let repair_shimmer = shimmer(&normalised_luma(&repaired, restoration.bits));
        assert!(
            repair_shimmer < input_shimmer,
            "{name}: repair shimmers {repair_shimmer:.5}, its input {input_shimmer:.5}"
        );
        assert!(
            repair_shimmer <= REPAIR_SHIMMER_CEILING,
            "{name}: repair shimmers {repair_shimmer:.5}, over {REPAIR_SHIMMER_CEILING}"
        );
    }
}

#[test]
fn progressive_quality_on_field_noisy_8() {
    assert_progressive_quality("field_noisy_8");
}

#[test]
fn progressive_quality_on_field_noisy_10() {
    assert_progressive_quality("field_noisy_10");
}

#[test]
fn progressive_quality_on_field_pan_8() {
    assert_progressive_quality("field_pan_8");
}

#[test]
fn repair_quality_on_badly_pan_8() {
    assert_repair_quality("badly_pan_8");
}

#[test]
fn repair_quality_on_badly_objects_8() {
    assert_repair_quality("badly_objects_8");
}

#[test]
fn repair_quality_on_badly_static_8() {
    assert_repair_quality("badly_static_8");
}

#[test]
fn repair_quality_on_badly_pan_8_reversed() {
    assert_repair_quality("badly_pan_8_reversed");
}

#[test]
fn repair_quality_on_badly_static_8_reversed() {
    assert_repair_quality("badly_static_8_reversed");
}
