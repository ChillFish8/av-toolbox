#[macro_use]
mod macros;
mod artefacts;
mod counted;
mod endpoints;
mod filters;
mod host;
mod masks;
mod planes;
mod probing;
mod providers;
mod rated;
mod shared;
mod specs;

use av_toolbox_core::testing::{FixtureCuts, HostStamping, Stamping};
use av_toolbox_core::{Parallelism, PipelineRegistry};

use self::artefacts::{Files, Grain};
pub(crate) use self::counted::COUNTED_OPENS;
use self::counted::{Counted, Unopenable};
use self::endpoints::{Deep, File, FloatFirst, Stdout, Wide, WideningSource};
use self::filters::{Blur, Deband, Either, Failing, Narrow, Precise, Prefers16, Recording, Sparse, Temporal};
#[cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
pub(crate) use self::filters::{RECORDED_INPUT_WIDTH, TEMPORAL_BUILD_CALLS};
pub(crate) use self::filters::{
    SPARSE_ASKED_SLOT,
    SPARSE_CHOSEN_SLOTS,
    SPARSE_FILLED_SLOTS,
    SPARSE_OFFERED_SLOTS,
};
use self::host::{HostBlur, HostFile, HostJoin, HostStdout, Log};
use self::masks::{Anchored, Gather, Matched, Merge, Mislabelled, TemporalMerge};
use self::planes::{Blend, Combine, HostMasker, LayoutFile, PlainBlend};
pub(crate) use self::probing::{
    FixtureProbe,
    MarkRequest,
    PROBE_CALLS,
    PlainProbe,
    SEEKABLE_OPENS,
    SEEKER_READS,
};
use self::probing::{Pair, Probing, Seekable};
use self::providers::{Breaking, Demanding, DemandingPair, Marking, Relaying, TagReading, Tagging};
pub(crate) use self::providers::{BreaksRequest, FixtureBreaks, FixtureMarks, MarksRequest};
use self::rated::{Rated, Retime};
pub(crate) use self::specs::{named, positional, spec};

/// One chunk at a time, chunked by the fixture demand kind.
pub(crate) fn chunked() -> Parallelism {
    Parallelism::chunk::<FixtureCuts>(())
}

pub(crate) fn registry() -> PipelineRegistry {
    let mut registry = registry_without_probe_kinds();

    registry
        .register_probe::<FixtureProbe>()
        .expect("should register");
    registry.register_probe::<PlainProbe>().expect("should register");

    registry
}

/// Every fixture stage, with no probe kind registered.
pub(crate) fn registry_without_probe_kinds() -> PipelineRegistry {
    let mut registry = PipelineRegistry::new();

    registry.register_source::<Counted>().expect("should register");
    registry.register_source::<Unopenable>().expect("should register");
    registry.register_source::<File>().expect("should register");
    registry.register_filter::<Blur>().expect("should register");
    registry.register_filter::<Failing>().expect("should register");
    registry.register_mask::<Merge>().expect("should register");
    registry.register_sink::<Stdout>().expect("should register");
    registry.register_filter::<Deband>().expect("should register");
    registry.register_filter::<Recording>().expect("should register");
    registry.register_source::<Wide>().expect("should register");
    registry.register_source::<Deep>().expect("should register");
    registry.register_filter::<Precise>().expect("should register");
    registry.register_filter::<Either>().expect("should register");
    registry.register_filter::<Narrow>().expect("should register");
    registry.register_filter::<Prefers16>().expect("should register");
    registry.register_mask::<Matched>().expect("should register");
    registry.register_mask::<Anchored>().expect("should register");
    registry.register_source::<FloatFirst>().expect("should register");
    registry.register_mask::<Mislabelled>().expect("should register");
    registry.register_filter::<Sparse>().expect("should register");
    registry.register_filter::<Temporal>().expect("should register");
    registry
        .register_mask::<TemporalMerge>()
        .expect("should register");
    registry.register_mask::<Gather>().expect("should register");
    registry
        .register_source::<WideningSource>()
        .expect("should register");
    registry.register_source::<HostFile>().expect("should register");
    registry.register_filter::<HostBlur>().expect("should register");
    registry.register_sink::<HostStdout>().expect("should register");
    registry.register_filter::<HostJoin>().expect("should register");
    registry.register_meta_sink::<Log>().expect("should register");
    registry.register_source::<Rated>().expect("should register");
    registry.register_filter::<Retime>().expect("should register");
    registry.register_source::<LayoutFile>().expect("should register");
    registry.register_mask::<Combine>().expect("should register");
    registry.register_mask::<HostMasker>().expect("should register");
    registry.register_filter::<Blend>().expect("should register");
    registry.register_filter::<PlainBlend>().expect("should register");
    registry.register_filter::<Demanding>().expect("should register");
    registry
        .register_filter::<DemandingPair>()
        .expect("should register");
    registry.register_provider::<Stamping>().expect("should register");
    registry
        .register_provider::<HostStamping>()
        .expect("should register");
    registry.register_provider::<Breaking>().expect("should register");
    registry.register_provider::<Marking>().expect("should register");
    registry.register_provider::<Relaying>().expect("should register");
    registry.register_provider::<Tagging>().expect("should register");
    registry
        .register_meta_sink::<TagReading>()
        .expect("should register");
    registry.register_filter::<Grain>().expect("should register");
    registry
        .register_artefact_sink::<Files>()
        .expect("should register");
    registry.register_source::<Seekable>().expect("should register");
    registry.register_filter::<Probing>().expect("should register");
    registry.register_filter::<Pair>().expect("should register");

    registry
}

#[cfg(test)]
mod tests {
    use av_toolbox_gpu::{FrameMeta, testing_context};

    use super::shared::frame_format;
    use super::*;

    #[test]
    fn the_test_registry_holds_every_fixture_stage() {
        let registry = registry();
        let identifiers: Vec<&str> = registry.stages().map(|stage| stage.full_identifier()).collect();

        assert_eq!(identifiers.len(), 50);
        assert!(identifiers.contains(&"test/host-stamping"));
        assert!(identifiers.contains(&"test/counted"));
        assert!(identifiers.contains(&"test/unopenable"));
        assert!(identifiers.contains(&"test/blur"));
        assert!(identifiers.contains(&"test/failing"));
        assert!(identifiers.contains(&"test/recording"));
        assert!(identifiers.contains(&"test/temporal"));
        assert!(identifiers.contains(&"test/temporal-merge"));
        assert!(identifiers.contains(&"test/gather"));
        assert!(identifiers.contains(&"test/widening-source"));
        assert!(identifiers.contains(&"test/host-file"));
        assert!(identifiers.contains(&"test/host-blur"));
        assert!(identifiers.contains(&"test/host-stdout"));
        assert!(identifiers.contains(&"test/host-join"));
        assert!(identifiers.contains(&"test/log"));
        assert!(identifiers.contains(&"test/demanding"));
        assert!(identifiers.contains(&"test/demanding-pair"));
        assert!(identifiers.contains(&"test/stamping"));
        assert!(identifiers.contains(&"test/breaking"));
        assert!(identifiers.contains(&"test/marking"));
        assert!(identifiers.contains(&"test/relaying"));
        assert!(identifiers.contains(&"test/tagging"));
        assert!(identifiers.contains(&"test/tag-reading"));
        assert!(identifiers.contains(&"test/grain"));
        assert!(identifiers.contains(&"test/files"));
        assert!(identifiers.contains(&"test/seekable"));
        assert!(identifiers.contains(&"test/probing"));
        assert!(identifiers.contains(&"test/pair"));
    }

    #[test]
    fn testing_context_allocates_a_frame_carrying_its_metadata() {
        let context = testing_context();

        let frame = context.alloc_frame(&frame_format(), FrameMeta::at(4));

        assert_eq!(frame.meta().index, 4);
    }
}
