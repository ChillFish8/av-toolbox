#[macro_use]
mod macros;
mod counted;
mod endpoints;
mod filters;
mod host;
mod masks;
mod rated;
mod shared;
mod specs;

use av_toolbox_core::PipelineRegistry;

pub(crate) use self::counted::COUNTED_OPENS;
use self::counted::{Counted, Unopenable};
use self::endpoints::{Deep, File, FloatFirst, Stdout, Wide, WideningSource};
use self::filters::{
    Blur,
    Deband,
    Either,
    Failing,
    Narrow,
    Precise,
    Prefers16,
    Recording,
    Sparse,
    Split,
    Temporal,
};
#[cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
pub(crate) use self::filters::{RECORDED_INPUT_WIDTH, TEMPORAL_BUILD_CALLS};
pub(crate) use self::filters::{
    SPARSE_ASKED_SLOT,
    SPARSE_CHOSEN_SLOTS,
    SPARSE_FILLED_SLOTS,
    SPARSE_OFFERED_SLOTS,
};
use self::host::{HostBlur, HostFile, HostStdout, Log, Restamp};
use self::masks::{Anchored, Gather, Matched, Merge, Mislabelled, TemporalMerge};
use self::rated::{Rated, Retime};
pub(crate) use self::specs::{named, positional, spec};

pub(crate) fn registry() -> PipelineRegistry {
    let mut registry = PipelineRegistry::new();

    registry.register_source::<Counted>().expect("should register");
    registry.register_source::<Unopenable>().expect("should register");
    registry.register_source::<File>().expect("should register");
    registry.register_filter::<Blur>().expect("should register");
    registry.register_filter::<Failing>().expect("should register");
    registry.register_mask::<Merge>().expect("should register");
    registry.register_transform::<Split>().expect("should register");
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
    registry
        .register_meta_transform::<Restamp>()
        .expect("should register");
    registry.register_meta_sink::<Log>().expect("should register");
    registry.register_source::<Rated>().expect("should register");
    registry.register_filter::<Retime>().expect("should register");

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

        assert_eq!(identifiers.len(), 32);
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
        assert!(identifiers.contains(&"test/restamp"));
        assert!(identifiers.contains(&"test/log"));
    }

    #[test]
    fn testing_context_allocates_a_frame_carrying_its_metadata() {
        let context = testing_context();

        let frame = context.alloc_frame(&frame_format(), FrameMeta::at(4));

        assert_eq!(frame.meta().index, 4);
    }
}
