use av_toolbox_gpu::{StreamFormat, StreamTemplate};

use super::capabilities;
use crate::{Crossing, Parallelism, PreparedStage, Residency, StageKind};

#[test]
fn a_crossing_passes_its_one_input_through_and_runs_frame_by_frame() {
    let crossing = PreparedStage::Crossing(Crossing::Unload);
    let format = StreamFormat::Mask(av_toolbox_gpu::MaskFormat {
        width: 8,
        height: 8,
        sample: av_toolbox_gpu::SampleFormat::U8,
        chroma: None,
    });
    let capabilities = capabilities();

    let produced = crossing
        .produces(&capabilities, &[Some(format)])
        .expect("passes through");

    assert_eq!(produced.template, StreamTemplate::from_format(&format));
    assert_eq!(crossing.parallelism(), Parallelism::Frame);
    assert_eq!(crossing.chunk_authority(), Some(0));
    assert_eq!(crossing.residency(), Residency::Host);
    assert_eq!(crossing.kind(), StageKind::Transform);
    assert_eq!(crossing.demands(), Vec::new());
    assert_eq!(crossing.suggested_workers(), None);
}
