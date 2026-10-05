use av_toolbox_cubek::testing::{FieldClip, field_clips, psnr};
use av_toolbox_gpu::{Depth, FrameFormat, FrameMeta, SampleFormat};
use serde_json::json;

use super::{run, yuv420};
use crate::stream::{Input, Output};

/// How many dB a stage must gain over the noisy input on one clip.
struct Floor {
    stage: &'static str,
    clip: &'static str,
    gain: f64,
}

const FLOORS: [Floor; 6] = [
    Floor {
        // Measured 0.678.
        stage: "av-denoise/nlmeans",
        clip: "field_noisy_8",
        gain: 0.25,
    },
    Floor {
        // Measured 1.452.
        stage: "av-denoise/nlmeans-hq",
        clip: "field_noisy_8",
        gain: 1.0,
    },
    Floor {
        // Measured 4.894.
        stage: "av-denoise/nl4d",
        clip: "field_noisy_8",
        gain: 4.5,
    },
    Floor {
        // Measured 0.602.
        stage: "av-denoise/nlmeans",
        clip: "field_noisy_10",
        gain: 0.25,
    },
    Floor {
        // Measured 1.488.
        stage: "av-denoise/nlmeans-hq",
        clip: "field_noisy_10",
        gain: 1.0,
    },
    Floor {
        // Measured 5.183.
        stage: "av-denoise/nl4d",
        clip: "field_noisy_10",
        gain: 4.75,
    },
];

fn clip_format(clip: &FieldClip) -> FrameFormat {
    let sample = match clip.bits {
        8 => SampleFormat::U8,
        bits => SampleFormat::U16 {
            depth: Depth::new(bits as u8).expect("a supported depth"),
        },
    };

    yuv420(clip.width as u32, clip.height as u32, sample)
}

fn encoded_plane(plane: &[i32], bits: u32) -> Vec<u8> {
    let mut bytes = Vec::new();
    for code in plane {
        match bits {
            8 => bytes.push(*code as u8),
            _ => bytes.extend_from_slice(&(*code as u16).to_le_bytes()),
        }
    }

    bytes
}

fn inputs(clip: &FieldClip) -> Vec<Input> {
    clip.truth
        .iter()
        .enumerate()
        .map(|(index, planes)| Input {
            meta: FrameMeta::at(index as u64),
            planes: planes
                .iter()
                .map(|plane| encoded_plane(plane, clip.bits))
                .collect(),
        })
        .collect()
}

fn decoded_luma(bytes: &[u8], bits: u32) -> Vec<f32> {
    match bits {
        8 => bytes.iter().map(|byte| f32::from(*byte)).collect(),
        _ => {
            let (pairs, _) = bytes.as_chunks::<2>();
            pairs
                .iter()
                .map(|pair| f32::from(u16::from_le_bytes(*pair)))
                .collect()
        },
    }
}

fn mean_luma_psnr(frames: &[Vec<f32>], clean: &[[Vec<i32>; 3]], peak: f32) -> f64 {
    let total: f64 = frames
        .iter()
        .zip(clean)
        .map(|(found, truth)| {
            let wanted: Vec<f32> = truth[0].iter().map(|code| *code as f32).collect();
            psnr(found, &wanted, peak)
        })
        .sum();

    total / frames.len() as f64
}

fn output_luma(outputs: &[Output], bits: u32) -> Vec<Vec<f32>> {
    outputs
        .iter()
        .map(|output| decoded_luma(&output.planes[0], bits))
        .collect()
}

fn noisy_luma(clip: &FieldClip) -> Vec<Vec<f32>> {
    clip.truth
        .iter()
        .map(|planes| planes[0].iter().map(|code| *code as f32).collect())
        .collect()
}

#[test]
fn every_stage_beats_the_noisy_input() {
    let clips = field_clips();

    for floor in &FLOORS {
        let clip = clips
            .iter()
            .find(|clip| clip.name == floor.clip)
            .expect("the clip exists");
        let format = clip_format(clip);
        let peak = ((1u32 << clip.bits) - 1) as f32;

        let outputs = run(floor.stage, json!({}), format, vec![inputs(clip)]);
        let denoised = output_luma(&outputs, clip.bits);
        let noisy = noisy_luma(clip);

        let denoised_psnr = mean_luma_psnr(&denoised, &clip.clean, peak);
        let noisy_psnr = mean_luma_psnr(&noisy, &clip.clean, peak);
        let gain = denoised_psnr - noisy_psnr;

        println!(
            "{} on {}: {denoised_psnr:.3} dB, noisy {noisy_psnr:.3} dB, gain {gain:.3}",
            floor.stage, floor.clip
        );
        assert!(
            gain >= floor.gain,
            "{} on {} gained {gain:.3} dB, needs {}",
            floor.stage,
            floor.clip,
            floor.gain
        );
    }
}
