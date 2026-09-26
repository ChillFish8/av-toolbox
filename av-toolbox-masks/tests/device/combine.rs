use av_toolbox_gpu::{ChromaLocation, MaskChroma, MaskFormat, StreamFormat, Subsampling};
use serde_json::{Value, json};

use crate::support::{SelectMasks, Selected, backend, built_mask, mask_format};

fn ramp(samples: usize, step: usize) -> Vec<u8> {
    (0..samples).map(|index| (index * step % 256) as u8).collect()
}

#[test]
fn threshold_hardens_at_the_cutoff_and_leaves_its_input_alone() {
    let backend = backend();
    let format = mask_format(7, 5);
    let mask = built_mask(
        &backend,
        "mask/threshold",
        json!({ "cutoff": 100 }),
        &[StreamFormat::Mask(format)],
        format,
    );
    let input = ramp(35, 7);

    let selected = backend.build(SelectMasks {
        mask,
        format,
        inputs: vec![input.clone()],
    });

    let expected: Vec<u8> = input
        .iter()
        .map(|sample| if *sample >= 100 { 255 } else { 0 })
        .collect();
    assert_eq!(selected.output, expected);
    assert_eq!(selected.inputs_after, vec![input]);
}

fn three_masks() -> Vec<Vec<u8>> {
    vec![ramp(35, 7), ramp(35, 11), ramp(35, 13)]
}

#[test]
fn union_takes_the_largest_sample_of_every_input() {
    let backend = backend();
    let format = mask_format(7, 5);
    let inputs = [StreamFormat::Mask(format); 3];
    let mask = built_mask(&backend, "mask/union", json!({}), &inputs, format);
    let masks = three_masks();

    let selected = backend.build(SelectMasks {
        mask,
        format,
        inputs: masks.clone(),
    });

    let expected: Vec<u8> = (0..35)
        .map(|index| masks.iter().map(|mask| mask[index]).max().expect("three masks"))
        .collect();
    assert_eq!(selected.output, expected);
    assert_eq!(selected.inputs_after, masks);
}

#[test]
fn union_of_one_mask_copies_it() {
    let backend = backend();
    let format = mask_format(7, 5);
    let mask = built_mask(
        &backend,
        "mask/union",
        json!({}),
        &[StreamFormat::Mask(format)],
        format,
    );
    let input = ramp(35, 7);

    let selected = backend.build(SelectMasks {
        mask,
        format,
        inputs: vec![input.clone()],
    });

    assert_eq!(selected.output, input);
}

#[test]
fn vote_selects_where_enough_inputs_agree() {
    let backend = backend();
    let format = mask_format(7, 5);
    let inputs = [StreamFormat::Mask(format); 3];
    let mask = built_mask(
        &backend,
        "mask/vote",
        json!({ "votes": 2, "threshold": 100 }),
        &inputs,
        format,
    );
    let masks = three_masks();

    let selected = backend.build(SelectMasks {
        mask,
        format,
        inputs: masks.clone(),
    });

    let expected: Vec<u8> = (0..35)
        .map(|index| {
            let votes = masks.iter().filter(|mask| mask[index] >= 100).count();
            if votes >= 2 { 255 } else { 0 }
        })
        .collect();
    assert_eq!(selected.output, expected);
}

fn chroma_mask_format() -> MaskFormat {
    let chroma = MaskChroma {
        subsampling: Subsampling::Yuv420,
        location: ChromaLocation::Left,
    };

    let full = mask_format(8, 6);

    MaskFormat {
        chroma: Some(chroma),
        ..full
    }
}

/// Runs `identifier` over `inputs`, each held in a mask that carries a chroma plane.
fn select_chroma_masks(identifier: &str, args: Value, inputs: Vec<Vec<u8>>) -> Selected {
    let backend = backend();
    let input_format = chroma_mask_format();
    let output_format = mask_format(8, 6);
    let offered = vec![StreamFormat::Mask(input_format); inputs.len()];
    let mask = built_mask(&backend, identifier, args, &offered, output_format);

    backend.build(SelectMasks {
        mask,
        format: input_format,
        inputs,
    })
}

fn assert_full_plane_only(selected: &Selected) {
    assert_eq!(selected.output_format, mask_format(8, 6));
    assert!(
        !selected.output_has_chroma_plane,
        "the output carries a chroma plane"
    );
}

#[test]
fn threshold_of_a_mask_with_chroma_produces_only_a_full_plane() {
    let input = ramp(48, 7);

    let selected = select_chroma_masks("mask/threshold", json!({ "cutoff": 100 }), vec![input.clone()]);

    let expected: Vec<u8> = input
        .iter()
        .map(|sample| if *sample >= 100 { 255 } else { 0 })
        .collect();
    assert_eq!(selected.output, expected);
    assert_full_plane_only(&selected);
}

#[test]
fn union_of_masks_with_chroma_produces_only_a_full_plane() {
    let first = ramp(48, 7);
    let second = ramp(48, 11);

    let selected = select_chroma_masks("mask/union", json!({}), vec![first.clone(), second.clone()]);

    let expected: Vec<u8> = first
        .iter()
        .zip(&second)
        .map(|(left, right)| *left.max(right))
        .collect();
    assert_eq!(selected.output, expected);
    assert_full_plane_only(&selected);
}

#[test]
fn vote_of_masks_with_chroma_produces_only_a_full_plane() {
    let first = ramp(48, 7);
    let second = ramp(48, 11);
    let args = json!({ "votes": 2, "threshold": 100 });

    let selected = select_chroma_masks("mask/vote", args, vec![first.clone(), second.clone()]);

    let expected: Vec<u8> = first
        .iter()
        .zip(&second)
        .map(|(left, right)| if *left >= 100 && *right >= 100 { 255 } else { 0 })
        .collect();
    assert_eq!(selected.output, expected);
    assert_full_plane_only(&selected);
}
