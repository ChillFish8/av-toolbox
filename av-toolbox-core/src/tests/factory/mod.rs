mod crossing;
mod demands;
mod granularity;
mod opens;
mod planes_of;
mod prepare;
mod probes;

use std::sync::atomic::{AtomicU32, Ordering};

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
    BoundInputs,
    Filter,
    FilterDefinition,
    InputArgument,
    InputParam,
    InputSignature,
    Inputs,
    PipelineRegistry,
    StageDefinition,
    StageError,
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
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let frame = inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;

        Ok(vec![frame])
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

    fn description() -> &'static str {
        "A stage used by tests."
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

fn registry_with_example() -> PipelineRegistry {
    let mut registry = PipelineRegistry::new();
    registry
        .register_filter::<ExampleDefinition>()
        .expect("registers");

    registry
}

fn reference_input() -> Vec<InputArgument> {
    vec![InputArgument::positional(StageHandle::new(1), StreamType::Frames)]
}

pub(super) fn capabilities() -> Capabilities {
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

pub(crate) fn frame_format() -> FrameFormat {
    FrameFormat {
        width: 64,
        height: 64,
        layout: FrameLayout::Yuv {
            subsampling: Subsampling::Yuv420,
        },
        sample: SampleFormat::U8,
        colour: ColourInfo::default(),
        frame_rate: None,
        field_order: None,
    }
}
