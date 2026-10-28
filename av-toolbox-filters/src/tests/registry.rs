use av_toolbox_core::{PipelineRegistry, StageKind};

use crate::register;

fn registered(identifier: &str) -> bool {
    let mut registry = PipelineRegistry::new();
    register(&mut registry).expect("the filters register");

    registry
        .stages()
        .any(|info| info.kind() == StageKind::Filter && info.full_identifier() == identifier)
}

#[test]
fn every_filter_is_registered() {
    assert!(registered("av-denoise/nlmeans"));
    assert!(registered("av-denoise/nlmeans-hq"));
    assert!(registered("av-denoise/nl4d"));
    assert!(registered("aa/based"));
    assert!(registered("aa/eedi2"));
    assert!(registered("aa/eedi3"));
    assert!(registered("aa/sangnom"));
    assert!(registered("blend/masked"));
    assert!(registered("mask/to_frames"));
    assert!(registered("crop/auto"));
    assert!(registered("crop/manual"));
    assert!(registered("deband/placebo"));
    assert!(registered("deblock/dpir"));
    assert!(registered("deblock/h264"));
    assert!(registered("deblock/qed"));
    assert!(registered("deblock/pp7"));
    assert!(registered("dehalo/alpha"));
    assert!(registered("dehalo/edge_clean"));
    assert!(registered("dehalo/fine"));
    assert!(registered("dehalo/yahr"));
    assert!(registered("dering/hq"));
    assert!(registered("deinterlace/bwdif"));
    assert!(registered("deinterlace/qtgmc"));
    assert!(registered("ivtc/match"));
    assert!(registered("ivtc/decimate"));
    assert!(registered("lineart/darken"));
    assert!(registered("lineart/thin"));
    assert!(registered("lineart/toon"));
    assert!(registered("lineart/vinverse"));
    assert!(registered("resize/point"));
    assert!(registered("resize/bilinear"));
    assert!(registered("resize/bicubic"));
    assert!(registered("resize/lanczos"));
    assert!(registered("resize/spline16"));
    assert!(registered("resize/spline36"));
    assert!(registered("resize/spline64"));
    assert!(registered("resample/depth"));
    assert!(!registered("resample/min_depth"));
    assert!(registered("resample/chroma"));
    assert!(registered("resample/matrix"));
    assert!(registered("planes/extract"));
    assert!(registered("planes/merge"));
    assert!(registered("warp/awarpsharp"));
    assert!(registered("upscale/artcnn"));
}

#[cfg(feature = "vship")]
#[test]
fn every_vship_filter_is_registered() {
    assert!(registered("vship/ssimulacra2"));
    assert!(registered("vship/butteraugli"));
    assert!(registered("vship/cvvdp"));
}

#[cfg(feature = "vship")]
#[test]
fn every_vship_filter_is_tagged_evaluation() {
    let mut registry = PipelineRegistry::new();
    register(&mut registry).expect("the filters register");

    let vship_stages: Vec<_> = registry
        .stages()
        .filter(|info| info.full_identifier().starts_with("vship/"))
        .collect();

    assert_eq!(vship_stages.len(), 3);
    for info in vship_stages {
        let expected = [av_toolbox_core::Tag::Evaluation];
        assert_eq!(info.tags(), &expected, "{}", info.full_identifier());
    }
}
