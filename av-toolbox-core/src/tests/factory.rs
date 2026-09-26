use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    ColourInfo,
    DeviceSide,
    FormatSet,
    Frame,
    FrameContext,
    FrameFormat,
    FrameLayout,
    HostFrame,
    HostSide,
    Offered,
    OutputFormats,
    ResolvedFormats,
    Runtime,
    SampleFormat,
    StreamFormat,
    StreamTemplate,
    Subsampling,
};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::factory::Definition;
use crate::{
    ArgsError,
    BoundInputs,
    BuildStageError,
    Crossing,
    Filter,
    FilterDefinition,
    InputArgument,
    InputParam,
    InputSignature,
    Inputs,
    Mask,
    MaskDefinition,
    OpenSlot,
    Parallelism,
    PipelineRegistry,
    PreparedFilter,
    PreparedSource,
    PreparedStage,
    RegistryError,
    Residency,
    Source,
    SourceDefinition,
    StageDefinition,
    StageError,
    StageFactory,
    StageHandle,
    StageInstance,
    StageKind,
    StreamType,
    Tag,
};

#[derive(Debug, Deserialize, JsonSchema)]
struct ExampleArgs {
    sigma_scale: f32,
}

struct ExampleFilter;

impl StageInstance for ExampleFilter {}

impl Filter for ExampleFilter {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Frame, StageError> {
        inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))
    }
}

static RECORDED_SIGMA_SCALE: AtomicU32 = AtomicU32::new(0);

struct ExampleDefinition;

impl StageDefinition for ExampleDefinition {
    type Args = ExampleArgs;

    fn package() -> &'static str {
        "av-denoise"
    }

    fn identifier() -> &'static str {
        "nl4d"
    }

    fn tags() -> &'static [Tag] {
        &[Tag::Denoise]
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[InputParam::required("reference")]);
        SIGNATURE
    }
}

impl FilterDefinition for ExampleDefinition {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        Ok(FormatSet::of([offered.sample()]))
    }

    fn produces(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let format = chosen.iter().flatten().next().copied().expect("an input");
        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([format.sample()]),
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        if args.sigma_scale < 0.0 {
            return Err(StageError::msg("sigma_scale must not be negative"));
        }

        RECORDED_SIGMA_SCALE.store(args.sigma_scale.to_bits(), Ordering::SeqCst);

        Ok(Box::new(ExampleFilter))
    }
}

fn definition() -> Definition<ExampleDefinition> {
    Definition::new(StageKind::Filter).expect("should build the definition")
}

fn reference_input() -> Vec<InputArgument> {
    vec![InputArgument::positional(StageHandle::new(1), StreamType::Frames)]
}

#[derive(Debug, Deserialize, JsonSchema)]
struct TemporalArgs {
    radius: u32,
}

struct TemporalExample;

impl StageDefinition for TemporalExample {
    type Args = TemporalArgs;

    fn package() -> &'static str {
        "av-denoise"
    }

    fn identifier() -> &'static str {
        "temporal"
    }

    fn parallelism(args: &Self::Args) -> Parallelism {
        if args.radius == 0 {
            Parallelism::Frame
        } else {
            Parallelism::Chunk
        }
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[InputParam::required("reference")]);
        SIGNATURE
    }
}

impl FilterDefinition for TemporalExample {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        Ok(FormatSet::of([offered.sample()]))
    }

    fn produces(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let format = chosen.iter().flatten().next().copied().expect("an input");
        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([format.sample()]),
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        Ok(Box::new(ExampleFilter))
    }
}

#[test]
fn a_prepared_stage_reports_the_granularity_its_arguments_asked_for() {
    let definition = Definition::<TemporalExample>::new(StageKind::Filter).expect("should build");

    let wide = StageFactory::<dyn PreparedFilter>::prepare(
        &definition,
        &serde_json::json!({ "radius": 0 }),
        &reference_input(),
        &OpenSlot::new(),
    )
    .expect("should prepare");

    let narrow = StageFactory::<dyn PreparedFilter>::prepare(
        &definition,
        &serde_json::json!({ "radius": 4 }),
        &reference_input(),
        &OpenSlot::new(),
    )
    .expect("should prepare");

    assert_eq!(wide.parallelism(), Parallelism::Frame);
    assert_eq!(narrow.parallelism(), Parallelism::Chunk);

    let erased = PreparedStage::Filter(narrow);
    assert_eq!(erased.parallelism(), Parallelism::Chunk);
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ChunkAuthorityArgs {}

struct ChunkAuthorityExample;

impl StageDefinition for ChunkAuthorityExample {
    type Args = ChunkAuthorityArgs;

    fn package() -> &'static str {
        "av-denoise"
    }

    fn identifier() -> &'static str {
        "chunk-authority"
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[
            InputParam::required("primary"),
            InputParam::required("secondary").chunk_authority(),
        ]);
        SIGNATURE
    }
}

impl FilterDefinition for ChunkAuthorityExample {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        Ok(FormatSet::of([offered.sample()]))
    }

    fn produces(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let format = chosen.iter().flatten().next().copied().expect("an input");
        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([format.sample()]),
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        Ok(Box::new(ExampleFilter))
    }
}

fn two_inputs() -> Vec<InputArgument> {
    vec![
        InputArgument::positional(StageHandle::new(1), StreamType::Frames),
        InputArgument::positional(StageHandle::new(2), StreamType::Frames),
    ]
}

#[test]
fn chunk_authority_survives_type_erasure_when_a_later_input_is_marked() {
    let definition = Definition::<ChunkAuthorityExample>::new(StageKind::Filter).expect("should build");

    let prepared: Box<dyn PreparedFilter> = StageFactory::<dyn PreparedFilter>::prepare(
        &definition,
        &serde_json::json!({}),
        &two_inputs(),
        &OpenSlot::new(),
    )
    .expect("should prepare");

    assert_eq!(prepared.chunk_authority(), Some(1));
}

#[test]
fn info_reports_the_generated_schema_and_signature() {
    let definition = definition();
    let info = StageFactory::<dyn PreparedFilter>::info(&definition);

    assert_eq!(info.kind(), StageKind::Filter);
    assert_eq!(info.full_identifier(), "av-denoise/nl4d");
    assert_eq!(info.tags(), &[Tag::Denoise]);
    assert_eq!(info.schema()["type"], "object");
    assert_eq!(info.inputs().params().len(), 1);
}

#[test]
fn arguments_are_rejected_without_a_backend() {
    let definition = Definition::<ExampleDefinition>::new(StageKind::Filter).expect("should build");
    let factory: &dyn StageFactory<dyn PreparedFilter> = &definition;

    let result = factory.prepare(
        &serde_json::json!({ "sigma_scale": "wide" }),
        &reference_input(),
        &OpenSlot::new(),
    );
    let Err(error) = result else {
        panic!("the stage should not have been prepared");
    };

    assert!(matches!(error, BuildStageError::InvalidArgs { .. }));
}

#[test]
fn arguments_that_break_the_schema_are_rejected_before_deserialization() {
    let definition = definition();
    let args = serde_json::json!({ "sigma_scale": "not a number" });

    let result =
        StageFactory::<dyn PreparedFilter>::prepare(&definition, &args, &reference_input(), &OpenSlot::new());
    let Err(error) = result else {
        panic!("the stage should not have been prepared");
    };

    assert!(error.to_string().contains("av-denoise/nl4d"));

    let BuildStageError::InvalidArgs { source, .. } = error else {
        panic!("expected invalid arguments, got {error:?}");
    };

    let ArgsError::SchemaViolations(violations) = source else {
        panic!("expected schema violations");
    };

    assert_eq!(violations[0].path(), "/sigma_scale");
}

#[test]
fn inputs_that_break_the_signature_are_rejected() {
    let definition = definition();
    let args = serde_json::json!({ "sigma_scale": 0.8 });

    let result = StageFactory::<dyn PreparedFilter>::prepare(&definition, &args, &[], &OpenSlot::new());
    let Err(error) = result else {
        panic!("the stage should not have been prepared");
    };

    assert!(error.to_string().contains("av-denoise/nl4d"));
    assert!(matches!(error, BuildStageError::InvalidInputs { .. }));
}

#[cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
mod device_tests {
    use av_toolbox_gpu::{Backend, Device};

    use super::*;

    fn backend() -> Backend {
        Backend::open_default(Device::Default).expect("should open")
    }

    fn formats() -> ResolvedFormats {
        ResolvedFormats::new(vec![], None)
    }

    #[test]
    fn valid_arguments_build_a_boxed_filter() {
        let definition = definition();
        let args = serde_json::json!({ "sigma_scale": 0.8 });
        let backend = backend();
        let formats = formats();

        let prepared = StageFactory::<dyn PreparedFilter>::prepare(
            &definition,
            &args,
            &reference_input(),
            &OpenSlot::new(),
        )
        .expect("should prepare");
        let filter = prepared.build(&backend, &formats).expect("should build");
        let filter = filter.into_device().expect("a device filter");
        drop(filter);

        let recorded_sigma_scale = f32::from_bits(RECORDED_SIGMA_SCALE.load(Ordering::SeqCst));
        assert_eq!(recorded_sigma_scale, 0.8);
    }

    #[test]
    fn a_stage_failure_is_wrapped_with_its_identifier() {
        let definition = definition();
        let args = serde_json::json!({ "sigma_scale": -1.0 });
        let backend = backend();
        let formats = formats();

        let prepared = StageFactory::<dyn PreparedFilter>::prepare(
            &definition,
            &args,
            &reference_input(),
            &OpenSlot::new(),
        )
        .expect("should prepare");
        let result = prepared.build(&backend, &formats);
        let Err(error) = result else {
            panic!("the stage should not have been built");
        };

        assert!(matches!(error, BuildStageError::Build { .. }));
        assert!(error.to_string().contains("av-denoise/nl4d"));

        let source = std::error::Error::source(&error).expect("should carry the stage's error");
        assert!(source.to_string().contains("sigma_scale must not be negative"));
    }
}

struct HostExample;

impl StageInstance for HostExample {}

impl Filter<HostSide> for HostExample {
    fn process(
        &mut self,
        _context: &mut FrameContext,
        mut inputs: Inputs<HostSide>,
    ) -> Result<HostFrame, StageError> {
        inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))
    }
}

struct HostDefinition;

impl StageDefinition for HostDefinition {
    type Args = ExampleArgs;

    fn package() -> &'static str {
        "av-denoise"
    }

    fn identifier() -> &'static str {
        "host"
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[InputParam::required("reference")]);
        SIGNATURE
    }
}

impl FilterDefinition for HostDefinition {
    type Side = HostSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        Ok(FormatSet::of([offered.sample()]))
    }

    fn produces(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let format = chosen.iter().flatten().next().copied().expect("an input");
        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([format.sample()]),
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter<HostSide>>, StageError> {
        Ok(Box::new(HostExample))
    }
}

#[test]
fn a_host_definition_prepares_as_a_host_stage() {
    let definition = Definition::<HostDefinition>::new(StageKind::Filter).expect("should build");
    let args = serde_json::json!({ "sigma_scale": 1.0 });

    let prepared =
        StageFactory::<dyn PreparedFilter>::prepare(&definition, &args, &reference_input(), &OpenSlot::new())
            .expect("should prepare");

    assert_eq!(prepared.residency(), Residency::Host);
    assert_eq!(PreparedStage::Filter(prepared).residency(), Residency::Host);
}

#[test]
fn a_definition_on_the_device_side_prepares_as_a_device_stage() {
    let args = serde_json::json!({ "sigma_scale": 1.0 });

    let definition = definition();
    let prepared =
        StageFactory::<dyn PreparedFilter>::prepare(&definition, &args, &reference_input(), &OpenSlot::new())
            .expect("should prepare");

    assert_eq!(prepared.residency(), Residency::Device);
}

#[test]
fn an_identifier_segment_containing_a_slash_is_rejected() {
    struct BadDefinition;

    impl StageDefinition for BadDefinition {
        type Args = ExampleArgs;

        fn package() -> &'static str {
            "av/denoise"
        }

        fn identifier() -> &'static str {
            "nl4d"
        }

        fn inputs() -> &'static InputSignature {
            &InputSignature::EMPTY
        }
    }

    let result = Definition::<BadDefinition>::new(StageKind::Filter);
    let Err(error) = result else {
        panic!("the definition should not have been built");
    };

    assert!(matches!(
        error,
        RegistryError::InvalidIdentifier {
            segment: "package",
            ..
        },
    ));
}

#[derive(Debug, Deserialize, JsonSchema)]
struct PlanesOfArgs {}

struct PlanesOfMask;

impl StageInstance for PlanesOfMask {}

impl Mask for PlanesOfMask {
    fn select(
        &mut self,
        _context: &mut FrameContext,
        _inputs: Inputs,
    ) -> Result<av_toolbox_gpu::Mask, StageError> {
        Err(StageError::msg("not built in this test"))
    }
}

fn planes_of_mask_accepts_input(
    _capabilities: &Capabilities,
    _args: &PlanesOfArgs,
    _index: usize,
    offered: StreamFormat,
    _inputs: &[Offered],
) -> Result<FormatSet, StageError> {
    Ok(FormatSet::of([offered.sample()]))
}

fn planes_of_mask_produces(
    _capabilities: &Capabilities,
    _args: &PlanesOfArgs,
    chosen: &[Option<StreamFormat>],
) -> Result<OutputFormats, StageError> {
    let format = chosen.iter().flatten().next().copied().expect("an input");
    Ok(OutputFormats {
        template: StreamTemplate::from_format(&format),
        samples: FormatSet::of([format.sample()]),
    })
}

fn planes_of_mask_build<R: Runtime>(
    _context: &BuildContext<R>,
    _args: PlanesOfArgs,
    _inputs: BoundInputs,
    _formats: &ResolvedFormats,
) -> Result<Box<dyn Mask>, StageError> {
    Ok(Box::new(PlanesOfMask))
}

struct ValidPlanesOf;

impl StageDefinition for ValidPlanesOf {
    type Args = PlanesOfArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "valid-planes-of"
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[
            InputParam::required("base"),
            InputParam::required("mask")
                .accepting(StreamType::Mask)
                .planes_of("base"),
        ]);
        SIGNATURE
    }
}

impl MaskDefinition for ValidPlanesOf {
    type Side = DeviceSide;

    fn accepts_input(
        capabilities: &Capabilities,
        args: &Self::Args,
        index: usize,
        offered: StreamFormat,
        inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        planes_of_mask_accepts_input(capabilities, args, index, offered, inputs)
    }

    fn produces(
        capabilities: &Capabilities,
        args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        planes_of_mask_produces(capabilities, args, chosen)
    }

    fn build<R: Runtime>(
        context: &BuildContext<R>,
        args: Self::Args,
        inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Mask>, StageError> {
        planes_of_mask_build(context, args, inputs, formats)
    }
}

struct MaskNamingAMissingInput;

impl StageDefinition for MaskNamingAMissingInput {
    type Args = PlanesOfArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "missing-planes-of"
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[InputParam::required("mask")
            .accepting(StreamType::Mask)
            .planes_of("missing")]);
        SIGNATURE
    }
}

impl MaskDefinition for MaskNamingAMissingInput {
    type Side = DeviceSide;

    fn accepts_input(
        capabilities: &Capabilities,
        args: &Self::Args,
        index: usize,
        offered: StreamFormat,
        inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        planes_of_mask_accepts_input(capabilities, args, index, offered, inputs)
    }

    fn produces(
        capabilities: &Capabilities,
        args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        planes_of_mask_produces(capabilities, args, chosen)
    }

    fn build<R: Runtime>(
        context: &BuildContext<R>,
        args: Self::Args,
        inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Mask>, StageError> {
        planes_of_mask_build(context, args, inputs, formats)
    }
}

struct PlanesOfOnAFrameInput;

impl StageDefinition for PlanesOfOnAFrameInput {
    type Args = PlanesOfArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "planes-of-on-a-frame-input"
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[
            InputParam::required("base"),
            InputParam::required("overlay").planes_of("base"),
        ]);
        SIGNATURE
    }
}

impl MaskDefinition for PlanesOfOnAFrameInput {
    type Side = DeviceSide;

    fn accepts_input(
        capabilities: &Capabilities,
        args: &Self::Args,
        index: usize,
        offered: StreamFormat,
        inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        planes_of_mask_accepts_input(capabilities, args, index, offered, inputs)
    }

    fn produces(
        capabilities: &Capabilities,
        args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        planes_of_mask_produces(capabilities, args, chosen)
    }

    fn build<R: Runtime>(
        context: &BuildContext<R>,
        args: Self::Args,
        inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Mask>, StageError> {
        planes_of_mask_build(context, args, inputs, formats)
    }
}

struct FilterNamingAMissingInput;

impl StageDefinition for FilterNamingAMissingInput {
    type Args = PlanesOfArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "filter-missing-planes-of"
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[InputParam::required("mask")
            .accepting(StreamType::Mask)
            .planes_of("missing")]);
        SIGNATURE
    }
}

impl FilterDefinition for FilterNamingAMissingInput {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        Ok(FormatSet::of([offered.sample()]))
    }

    fn produces(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let format = chosen.iter().flatten().next().copied().expect("an input");
        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([format.sample()]),
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        Ok(Box::new(ExampleFilter))
    }
}

#[test]
fn a_mask_input_with_a_valid_planes_of_registers() {
    let mut registry = PipelineRegistry::new();

    let result = registry.register_mask::<ValidPlanesOf>();

    assert!(result.is_ok());
}

#[test]
fn a_planes_of_naming_a_missing_input_is_rejected() {
    let mut registry = PipelineRegistry::new();

    let error = registry
        .register_mask::<MaskNamingAMissingInput>()
        .expect_err("should not register");

    assert!(matches!(
        error,
        RegistryError::InvalidPlanesOf {
            param: "mask",
            reason: "it names no input of this stage",
            ..
        }
    ));
}

#[test]
fn a_planes_of_on_an_input_that_does_not_accept_masks_is_rejected() {
    let mut registry = PipelineRegistry::new();

    let error = registry
        .register_mask::<PlanesOfOnAFrameInput>()
        .expect_err("should not register");

    assert!(matches!(
        error,
        RegistryError::InvalidPlanesOf {
            param: "overlay",
            reason: "it is only allowed on an input that takes masks",
            ..
        }
    ));
}

#[test]
fn the_planes_of_check_applies_to_every_kind() {
    let mut registry = PipelineRegistry::new();

    let error = registry
        .register_filter::<FilterNamingAMissingInput>()
        .expect_err("should not register");

    assert!(matches!(
        error,
        RegistryError::InvalidPlanesOf {
            param: "mask",
            reason: "it names no input of this stage",
            ..
        }
    ));
}

fn capabilities() -> Capabilities {
    let every = [
        av_toolbox_gpu::ElementType::F32,
        av_toolbox_gpu::ElementType::F16,
        av_toolbox_gpu::ElementType::BF16,
        av_toolbox_gpu::ElementType::U8,
        av_toolbox_gpu::ElementType::U16,
        av_toolbox_gpu::ElementType::U32,
        av_toolbox_gpu::ElementType::I8,
        av_toolbox_gpu::ElementType::I16,
        av_toolbox_gpu::ElementType::I32,
    ];

    let accelerator = av_toolbox_gpu::default_accelerators()[0];

    Capabilities::builder(accelerator)
        .storable(every)
        .computable(every)
        .plane_size(32, 32)
        .max_units_per_cube(1024)
        .max_cube_count(65535, 65535, 65535)
        .build()
}

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
}

fn frame_format() -> FrameFormat {
    FrameFormat {
        width: 64,
        height: 64,
        layout: FrameLayout::Yuv {
            subsampling: Subsampling::Yuv420,
        },
        sample: SampleFormat::U8,
        colour: ColourInfo::default(),
        frame_rate: None,
    }
}

/// How many times each counter has been opened, one per test so tests running at once never share.
static COUNTED_OPENS: [AtomicUsize; 3] = [const { AtomicUsize::new(0) }; 3];

const OPENED_ONCE: usize = 0;

const OPENED_AGAIN: usize = 1;

const FAILED_OPEN: usize = 2;

#[derive(Debug, Deserialize, JsonSchema)]
struct CountedArgs {
    fail: bool,
    counter: usize,
}

struct CountedStage;

impl StageInstance for CountedStage {}

impl Source for CountedStage {
    fn next(&mut self, _context: &mut FrameContext) -> Result<Option<Frame>, StageError> {
        Ok(None)
    }
}

struct Counted;

impl StageDefinition for Counted {
    type Args = CountedArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "counted"
    }

    fn inputs() -> &'static InputSignature {
        &InputSignature::EMPTY
    }
}

impl SourceDefinition for Counted {
    type Side = DeviceSide;
    type Opened = StreamFormat;

    fn open(args: &Self::Args) -> Result<Self::Opened, StageError> {
        COUNTED_OPENS[args.counter].fetch_add(1, Ordering::SeqCst);

        if args.fail {
            return Err(StageError::msg("the input could not be opened"));
        }

        Ok(StreamFormat::Frames(frame_format()))
    }

    fn produces(_capabilities: &Capabilities, opened: &Self::Opened) -> Result<OutputFormats, StageError> {
        Ok(OutputFormats {
            template: StreamTemplate::from_format(opened),
            samples: FormatSet::of([opened.sample()]),
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _opened: Self::Opened,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Source>, StageError> {
        Ok(Box::new(CountedStage))
    }
}

fn counted(slot: &OpenSlot, fail: bool, counter: usize) -> Box<dyn PreparedSource> {
    let definition = Definition::<Counted>::new(StageKind::Source).expect("should build");
    let args = serde_json::json!({ "fail": fail, "counter": counter });

    StageFactory::<dyn PreparedSource>::prepare(&definition, &args, &[], slot).expect("should prepare")
}

#[test]
fn a_source_prepared_twice_on_one_slot_opens_once() {
    let slot = OpenSlot::new();
    let capabilities = capabilities();

    counted(&slot, false, OPENED_ONCE)
        .produces(&capabilities)
        .expect("produces");
    counted(&slot, false, OPENED_ONCE)
        .produces(&capabilities)
        .expect("produces again");

    assert_eq!(COUNTED_OPENS[OPENED_ONCE].load(Ordering::SeqCst), 1);
}

#[test]
fn a_source_on_a_fresh_slot_opens_again() {
    let capabilities = capabilities();

    counted(&OpenSlot::new(), false, OPENED_AGAIN)
        .produces(&capabilities)
        .expect("produces");
    counted(&OpenSlot::new(), false, OPENED_AGAIN)
        .produces(&capabilities)
        .expect("produces");

    assert_eq!(COUNTED_OPENS[OPENED_AGAIN].load(Ordering::SeqCst), 2);
}

#[test]
fn a_failed_open_is_the_error_produces_returns() {
    let error = counted(&OpenSlot::new(), true, FAILED_OPEN)
        .produces(&capabilities())
        .expect_err("the open fails");

    assert!(error.to_string().contains("could not be opened"));
}
