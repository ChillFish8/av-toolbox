use av_toolbox_gpu::StreamFormat;
use serde_json::json;

use crate::support::{SelectMasks, backend, built_mask, mask_format};

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
