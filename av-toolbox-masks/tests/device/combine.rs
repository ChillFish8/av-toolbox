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
