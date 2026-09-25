use av_toolbox_gpu::{FrameFormat, StreamFormat};
use serde_json::{Value, json};

use crate::reference::{XdogSettings, normalise_u8, stroke_on_gradient, xdog};
use crate::support::{
    SelectFrames,
    assert_masks_close,
    backend,
    built_mask,
    grey,
    mask_format,
    ten_bit_grey,
    wide_bytes,
    yuv420,
};

const WIDTH: u32 = 32;
const HEIGHT: u32 = 24;

fn run(args: Value, format: FrameFormat, frames: Vec<Vec<Vec<u8>>>) -> Vec<Vec<u8>> {
    let backend = backend();
    let output = mask_format(format.width, format.height);
    let mask = built_mask(
        &backend,
        "lines/xdog",
        args,
        &[StreamFormat::Frames(format)],
        output,
    );

    backend.build(SelectFrames { mask, format, frames })
}

#[test]
fn xdog_matches_the_reference_with_fixed_parameters() {
    let luma = stroke_on_gradient(WIDTH, HEIGHT);
    let settings = XdogSettings {
        eps: Some(0.01),
        phi: Some(200.0),
        ..XdogSettings::default()
    };
    let normalised = normalise_u8(&luma);
    let expected = xdog(&normalised, WIDTH, HEIGHT, &settings);

    let args = json!({ "eps": 0.01, "phi": 200.0 });
    let format = grey(WIDTH, HEIGHT);
    let frames = vec![vec![luma]];
    let masks = run(args, format, frames);

    assert_masks_close(&masks[0], &expected, 2, 0.0);
}

#[test]
fn xdog_matches_the_reference_with_automatic_parameters() {
    let luma = stroke_on_gradient(WIDTH, HEIGHT);
    let normalised = normalise_u8(&luma);
    let default_settings = XdogSettings::default();
    let expected = xdog(&normalised, WIDTH, HEIGHT, &default_settings);

    let format = grey(WIDTH, HEIGHT);
    let frames = vec![vec![luma]];
    let masks = run(json!({}), format, frames);

    assert_masks_close(&masks[0], &expected, 3, 0.01);
    assert!(
        masks[0][(HEIGHT / 2 * WIDTH + WIDTH / 2) as usize] > 128,
        "the stroke was not selected"
    );
}

#[test]
fn xdog_selects_nothing_on_a_flat_frame() {
    let luma = vec![128u8; (WIDTH * HEIGHT) as usize];

    let format = grey(WIDTH, HEIGHT);
    let frames = vec![vec![luma]];
    let masks = run(json!({}), format, frames);

    assert!(masks[0].iter().all(|sample| *sample == 0));
}

#[test]
fn xdog_reads_ten_bit_like_eight_bit() {
    let luma = stroke_on_gradient(WIDTH, HEIGHT);
    let wide: Vec<u16> = luma
        .iter()
        .map(|sample| (f32::from(*sample) * 1023.0 / 255.0).round() as u16)
        .collect();

    let narrow_format = grey(WIDTH, HEIGHT);
    let narrow_frames = vec![vec![luma]];
    let narrow_masks = run(json!({}), narrow_format, narrow_frames);

    let wide_format = ten_bit_grey(WIDTH, HEIGHT);
    let wide_plane = wide_bytes(&wide);
    let wide_frames = vec![vec![wide_plane]];
    let wide_masks = run(json!({}), wide_format, wide_frames);

    assert_masks_close(&wide_masks[0], &narrow_masks[0], 3, 0.01);
}

#[test]
fn xdog_reads_only_the_luma_of_a_yuv_frame() {
    let width = 9;
    let height = 7;
    let luma = stroke_on_gradient(width, height);
    let chroma_u: Vec<u8> = (0..20).map(|index| (index * 13) as u8).collect();
    let chroma_v: Vec<u8> = (0..20).map(|index| (index * 29) as u8).collect();

    let grey_format = grey(width, height);
    let luma_for_grey = luma.clone();
    let grey_frames = vec![vec![luma_for_grey]];
    let grey_masks = run(json!({}), grey_format, grey_frames);

    let yuv_format = yuv420(width, height);
    let yuv_frames = vec![vec![luma, chroma_u, chroma_v]];
    let yuv_masks = run(json!({}), yuv_format, yuv_frames);

    assert_eq!(yuv_masks[0], grey_masks[0]);
}

#[test]
fn xdog_gives_the_same_mask_when_a_frame_comes_round_again() {
    let first = stroke_on_gradient(WIDTH, HEIGHT);
    let second: Vec<u8> = first.iter().map(|sample| 255 - sample).collect();

    let format = grey(WIDTH, HEIGHT);
    let first_again = first.clone();
    let frames = vec![vec![first_again], vec![second], vec![first]];
    let masks = run(json!({}), format, frames);

    assert_eq!(masks[0], masks[2]);
}
