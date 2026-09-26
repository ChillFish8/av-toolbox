use av_toolbox_gpu::{FrameFormat, StreamFormat};
use serde_json::{Value, json};

use crate::reference::{
    ShadowSettings,
    adaptive,
    normalise_u8,
    recover_shadows,
    shadowed_scene,
    stroke_on_gradient,
};
use crate::support::{SelectFrames, assert_masks_close, backend, built_mask, grey, mask_format};

const WIDTH: u32 = 32;
const HEIGHT: u32 = 24;
const SCENE: u32 = 48;

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

fn recovered(args: Value, frames: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
    let format = grey(SCENE, SCENE);
    let frames = frames.into_iter().map(|luma| vec![luma]).collect();

    run(args, format, frames)
}

#[test]
fn recovering_shadows_matches_the_reference() {
    let luma = shadowed_scene(SCENE, SCENE);
    let normalised = normalise_u8(&luma);
    let settings = ShadowSettings::default();
    let expected = recover_shadows(&normalised, SCENE, SCENE, &settings);

    let masks = recovered(json!({ "recover_shadows": true }), vec![luma]);

    assert_masks_close(&masks[0], &expected, 0, 0.02);
}

#[test]
fn recovering_shadows_finds_the_stroke_in_the_dark_half() {
    let luma = shadowed_scene(SCENE, SCENE);
    let dark_stroke = (SCENE / 2 * SCENE + SCENE / 2 + 11) as usize;

    let masks = recovered(json!({ "recover_shadows": true }), vec![luma]);

    assert_eq!(masks[0][dark_stroke], 255);
}

fn small_speck() -> impl Iterator<Item = usize> {
    (20..22).flat_map(|row| (5..7).map(move |column| (row * SCENE + column) as usize))
}

fn large_speck() -> impl Iterator<Item = usize> {
    let columns = SCENE / 2 + 17..SCENE / 2 + 20;

    (20..23).flat_map(move |row| columns.clone().map(move |column| (row * SCENE + column) as usize))
}

#[test]
fn recovering_shadows_drops_specks_below_min_speck_and_keeps_larger_ones() {
    let luma = shadowed_scene(SCENE, SCENE);

    let masks = recovered(json!({ "recover_shadows": true, "min_speck": 6 }), vec![luma]);

    let mut small = small_speck();
    assert!(
        small.all(|index| masks[0][index] == 0),
        "the 4 sample speck survived"
    );

    let mut large = large_speck();
    assert!(
        large.any(|index| masks[0][index] == 255),
        "the 9 sample speck was dropped"
    );
}

#[test]
fn recovering_shadows_keeps_specks_at_min_speck() {
    let luma = shadowed_scene(SCENE, SCENE);

    let masks = recovered(json!({ "recover_shadows": true, "min_speck": 4 }), vec![luma]);

    let mut small = small_speck();
    assert!(
        small.all(|index| masks[0][index] == 255),
        "the 4 sample speck was dropped"
    );
}

#[test]
fn recovering_shadows_gives_the_same_mask_when_a_frame_comes_round_again() {
    let first = shadowed_scene(SCENE, SCENE);
    let second: Vec<u8> = first.iter().map(|sample| sample / 2).collect();

    let masks = recovered(
        json!({ "recover_shadows": true }),
        vec![first.clone(), second, first],
    );

    assert_eq!(masks[0], masks[2]);
}

/// A single column of bright samples over dark ones, with one mid-tone sample at the step.
const STEPPED_COLUMN: [u8; 9] = [200, 200, 200, 100, 30, 30, 30, 30, 30];

fn recovered_with(luma: Vec<u8>, width: u32, height: u32, min_speck: u32) -> (Vec<u8>, Vec<u8>) {
    let normalised = normalise_u8(&luma);
    let settings = ShadowSettings {
        min_speck,
        ..ShadowSettings::default()
    };
    let expected = recover_shadows(&normalised, width, height, &settings);

    let format = grey(width, height);
    let frames = vec![vec![luma]];
    let args = json!({ "recover_shadows": true, "min_speck": min_speck });
    let masks = run(args, format, frames);
    let found = masks
        .into_iter()
        .next()
        .expect("one frame goes in, so one mask comes out");

    (found, expected)
}

fn recovered_at(width: u32, height: u32) -> (Vec<u8>, Vec<u8>) {
    let luma = shadowed_scene(width, height);
    let min_speck = ShadowSettings::default().min_speck;

    recovered_with(luma, width, height, min_speck)
}

#[test]
fn recovering_shadows_matches_the_reference_when_tiles_do_not_divide_the_frame() {
    let (found, expected) = recovered_at(45, 37);

    assert_masks_close(&found, &expected, 0, 0.005);
}

/// Tiles one sample across and down give the same tables on any grid, so this only checks that a
/// tiny frame runs.
#[test]
fn recovering_shadows_runs_and_matches_the_reference_on_a_frame_narrower_than_the_grid() {
    let (found, expected) = recovered_at(3, 5);

    assert_masks_close(&found, &expected, 0, 0.02);
}

/// `min_speck` is lowered so the short line groups a one sample wide frame gives are kept.
#[test]
fn recovering_shadows_matches_the_reference_on_a_frame_one_sample_wide() {
    let luma = STEPPED_COLUMN.to_vec();
    let (found, expected) = recovered_with(luma, 1, 9, 2);

    assert_masks_close(&found, &expected, 0, 0.02);
}
