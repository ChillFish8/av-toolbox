use av_toolbox_gpu::{FrameMeta, LayerId};

use crate::testing::{FixtureChunk, FixtureCuts};
use crate::{
    DemandKind,
    InputArgument,
    InputParam,
    InputSignature,
    Parallelism,
    ProvisionError,
    StageHandle,
    StreamType,
    bind_inputs,
};

const CLIP: &InputSignature = &InputSignature::new(&[InputParam::required("clip")]);

fn clip_with(demands: &[(DemandKind, LayerId)]) -> InputArgument {
    let mut argument = InputArgument::positional(StageHandle::new(1), StreamType::Frames);
    for (kind, layer) in demands {
        argument = argument.with_demand(*kind, *layer);
    }

    argument
}

#[test]
fn provision_reads_the_value_its_demand_was_given() {
    let kind = DemandKind::of::<FixtureCuts>();
    let argument = clip_with(&[(kind, LayerId(2))]);
    let bound = bind_inputs(CLIP, &[argument]).expect("should bind");
    let meta = FrameMeta::at(0).with_provided(LayerId(2), FixtureChunk(5));

    let provision = bound.provision::<FixtureCuts>("clip").expect("demanded");

    assert_eq!(provision.get(&meta), Some(&FixtureChunk(5)));
    assert_eq!(provision.get(&FrameMeta::at(0)), None);
}

#[test]
fn provision_of_an_undemanded_kind_is_refused() {
    let bound = bind_inputs(CLIP, &[clip_with(&[])]).expect("should bind");

    let error = bound
        .provision::<FixtureCuts>("clip")
        .expect_err("nothing demanded");

    assert!(matches!(error, ProvisionError::NotDemanded { .. }));
}

#[test]
fn provision_refuses_two_demands_of_one_kind_on_an_input() {
    let kind = DemandKind::of::<FixtureCuts>();
    let argument = clip_with(&[(kind, LayerId(0)), (kind, LayerId(1))]);
    let bound = bind_inputs(CLIP, &[argument]).expect("should bind");

    let error = bound.provision::<FixtureCuts>("clip").expect_err("ambiguous");

    assert!(matches!(error, ProvisionError::Ambiguous { .. }));
}

#[test]
fn a_chunk_parallelism_keys_on_its_outputs_chunk() {
    let parallelism = Parallelism::chunk::<FixtureCuts>(());

    let key = parallelism.chunk_key().expect("a chunk parallelism has a key");

    assert_eq!(key(&FixtureChunk(9)), 9);
}
