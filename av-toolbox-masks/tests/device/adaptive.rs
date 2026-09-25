use av_toolbox_gpu::{FrameFormat, StreamFormat};
use serde_json::{Value, json};

use crate::reference::{adaptive, normalise_u8, stroke_on_gradient};
use crate::support::{SelectFrames, assert_masks_close, backend, built_mask, grey, mask_format};

const WIDTH: u32 = 32;
const HEIGHT: u32 = 24;

fn run(args: Value, format: FrameFormat, frames: Vec<Vec<Vec<u8>>>) -> Vec<Vec<u8>> {
    let backend = backend();
    let output = mask_format(format.width, format.height);
    let mask = built_mask(
        &backend,
        "lines/adaptive",
        args,
        &[StreamFormat::Frames(format)],
        output,
    );

    backend.build(SelectFrames { mask, format, frames })
}

#[test]
fn adaptive_matches_the_reference() {
    let luma = stroke_on_gradient(WIDTH, HEIGHT);
    let normalised = normalise_u8(&luma);
    let expected = adaptive(&normalised, WIDTH, HEIGHT, 15, 7.0);

    let format = grey(WIDTH, HEIGHT);
    let frames = vec![vec![luma]];
    let masks = run(json!({}), format, frames);

    assert_masks_close(&masks[0], &expected, 0, 0.01);
    assert_eq!(
        masks[0][(HEIGHT / 2 * WIDTH + WIDTH / 2) as usize],
        255,
        "the stroke was not selected"
    );
}

#[test]
fn adaptive_honours_a_smaller_block_and_offset() {
    let luma = stroke_on_gradient(WIDTH, HEIGHT);
    let normalised = normalise_u8(&luma);
    let expected = adaptive(&normalised, WIDTH, HEIGHT, 5, 3.0);

    let format = grey(WIDTH, HEIGHT);
    let frames = vec![vec![luma]];
    let masks = run(json!({ "block_size": 5, "c": 3.0 }), format, frames);

    assert_masks_close(&masks[0], &expected, 0, 0.01);
}

#[test]
fn adaptive_selects_nothing_on_a_flat_frame() {
    let luma = vec![77u8; (WIDTH * HEIGHT) as usize];

    let format = grey(WIDTH, HEIGHT);
    let frames = vec![vec![luma]];
    let masks = run(json!({}), format, frames);

    assert!(masks[0].iter().all(|sample| *sample == 0));
}
