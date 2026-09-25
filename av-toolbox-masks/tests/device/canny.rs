use av_toolbox_gpu::{FrameFormat, StreamFormat};
use serde_json::{Value, json};

use crate::reference::{canny, normalise_u8, square};
use crate::support::{SelectFrames, assert_masks_close, backend, built_mask, grey, mask_format};

const JOINED_ISOLATED_WIDTH: u32 = 32;
const JOINED_ISOLATED_HEIGHT: u32 = 40;

fn run(args: Value, format: FrameFormat, frames: Vec<Vec<Vec<u8>>>) -> Vec<Vec<u8>> {
    let backend = backend();
    let output = mask_format(format.width, format.height);
    let mask = built_mask(
        &backend,
        "lines/canny",
        args,
        &[StreamFormat::Frames(format)],
        output,
    );

    backend.build(SelectFrames { mask, format, frames })
}

/// A vertical edge down the middle whose contrast fades from strong at the top to weak by the
/// bottom, plus a separate stripe on the left whose contrast makes it a candidate but never a
/// strong edge, and which never touches the fading edge's component.
///
/// The fading edge stays a single 8-connected line, so its weak tail only survives hysteresis
/// because it touches the strong pixels above it, not because it clears `high` on its own. The
/// stripe, despite clearing `low`, only ever anchors its own component, so hysteresis drops it.
fn joined_and_isolated() -> Vec<u8> {
    (0..JOINED_ISOLATED_HEIGHT)
        .flat_map(|row| {
            (0..JOINED_ISOLATED_WIDTH).map(move |column| {
                if column >= 16 {
                    (255 - 8 * row as i32).clamp(0, 255) as u8
                } else if (2..10).contains(&column) {
                    48
                } else {
                    0
                }
            })
        })
        .collect()
}

#[test]
fn canny_matches_the_reference_on_a_square() {
    let luma = square(24, 24);
    let normalised = normalise_u8(&luma);
    let expected = canny(&normalised, 24, 24, 50.0, 150.0);

    let format = grey(24, 24);
    let masks = run(json!({}), format, vec![vec![luma]]);

    // A corner mixes two gradient directions into a near tie between a candidate and its
    // neighbours, so the device and the reference round a handful of them to opposite sides.
    assert_masks_close(&masks[0], &expected, 0, 0.02);
    assert!(masks[0].contains(&255), "the square's edges were not found");
}

/// Whether each of the square's four straight sides has a kept pixel near its middle.
fn square_sides_found(mask: &[u8]) -> [bool; 4] {
    let width = 24;
    let top = (7..=8).any(|row| mask[row * width + 12] == 255);
    let bottom = (15..=16).any(|row| mask[row * width + 12] == 255);
    let left = (7..=8).any(|column| mask[12 * width + column] == 255);
    let right = (15..=16).any(|column| mask[12 * width + column] == 255);
    [top, bottom, left, right]
}

#[test]
fn canny_finds_the_straight_sides_of_a_square() {
    let luma = square(24, 24);
    let normalised = normalise_u8(&luma);
    let expected = canny(&normalised, 24, 24, 50.0, 150.0);

    let format = grey(24, 24);
    let masks = run(json!({}), format, vec![vec![luma]]);

    let reference_sides = square_sides_found(&expected);
    assert!(
        reference_sides.into_iter().all(|found| found),
        "the reference lost a straight side of the square"
    );

    let device_sides = square_sides_found(&masks[0]);
    assert!(
        device_sides.into_iter().all(|found| found),
        "the device lost a straight side of the square"
    );
}

#[test]
fn canny_finds_nothing_on_a_flat_frame() {
    let luma = vec![90u8; 24 * 24];

    let format = grey(24, 24);
    let masks = run(json!({}), format, vec![vec![luma]]);

    assert!(masks[0].iter().all(|sample| *sample == 0));
}

#[test]
fn canny_keeps_weak_edges_joined_to_strong_ones_and_drops_the_rest() {
    let luma = joined_and_isolated();
    let normalised = normalise_u8(&luma);
    let expected = canny(
        &normalised,
        JOINED_ISOLATED_WIDTH,
        JOINED_ISOLATED_HEIGHT,
        50.0,
        150.0,
    );

    let format = grey(JOINED_ISOLATED_WIDTH, JOINED_ISOLATED_HEIGHT);
    let masks = run(json!({}), format, vec![vec![luma]]);

    assert_masks_close(&masks[0], &expected, 0, 0.01);
    let width = JOINED_ISOLATED_WIDTH as usize;
    let joined = (28..=30).any(|row| masks[0][row * width + 16] == 255);
    let isolated = (0..JOINED_ISOLATED_HEIGHT as usize).any(|row| masks[0][row * width + 5] == 255);
    assert!(
        joined,
        "the weak tail of the edge joined to its strong part was dropped"
    );
    assert!(
        !isolated,
        "the stripe's own weak candidates were kept despite anchoring nothing strong"
    );
}

#[test]
fn canny_gives_the_same_mask_when_a_frame_comes_round_again() {
    let first = square(24, 24);
    let second = joined_and_isolated()[..24 * 24].to_vec();

    let format = grey(24, 24);
    let masks = run(
        json!({}),
        format,
        vec![vec![first.clone()], vec![second], vec![first]],
    );

    assert_eq!(masks[0], masks[2]);
    assert!(masks[0].contains(&255), "the square's edges were not found");
}
