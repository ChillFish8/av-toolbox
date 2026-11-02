use std::sync::Mutex;

use av_toolbox_core::{
    AnyDemand,
    BoundInputs,
    Filter,
    FilterDefinition,
    InputDemand,
    InputParam,
    InputSignature,
    Inputs,
    MetaInputs,
    MetaSink,
    MetaSinkDefinition,
    Parallelism,
    StageDefinition,
    StageError,
    StageInstance,
    StreamType,
};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    DeviceSide,
    FormatSet,
    Frame,
    FrameContext,
    MetaRecord,
    MetaValue,
    Offered,
    OutputFormats,
    ResolvedFormats,
    StreamFormat,
    StreamTemplate,
};
use schemars::JsonSchema;
use serde::Deserialize;

use super::chunked;
use super::tags::{FixtureTags, TagsRequest};
use crate::engine::tests::support::StageLife;

/// Every flow `test/flow-recording` saw, as `frame <index>` or `<kind> <path> <value>`, in order.
pub(crate) static FLOWS: Mutex<Vec<String>> = Mutex::new(Vec::new());

pub(crate) fn clear_flows() {
    FLOWS.lock().expect("not poisoned").clear();
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub(crate) struct MarkingArgs {
    /// Declares the stage per chunk rather than for the whole stream.
    #[serde(default)]
    per_chunk: bool,
    /// Declares the stage frame by frame when it is not per chunk.
    #[serde(default)]
    per_frame: bool,
}

pub(crate) struct MarkingInstance {
    _life: StageLife,
    frames: i64,
}

impl MarkingInstance {
    pub(crate) fn new() -> Self {
        Self {
            _life: StageLife::new(),
            frames: 0,
        }
    }
}

impl StageInstance for MarkingInstance {}

impl Filter for MarkingInstance {
    fn process(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let frame = inputs.take_frame(0).ok_or_else(missing_frame)?;
        self.frames += 1;

        let index = frame.meta().index as i64;
        context.record("mark").attach("index", MetaValue::Int(index));

        Ok(vec![frame])
    }

    fn end_of_stream(&mut self, context: &mut FrameContext) -> Result<Vec<Frame>, StageError> {
        let frames = std::mem::take(&mut self.frames);
        context.record("chunk").attach("frames", MetaValue::Int(frames));

        Ok(Vec::new())
    }
}

fn missing_frame() -> StageError {
    StageError::msg("expected a frame in slot 0")
}

/// A pass through filter recording a `mark` for every frame and a `chunk` at the end of every stream.
pub(crate) struct Marking;

impl StageDefinition for Marking {
    type Args = MarkingArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "marking"
    }

    fn description() -> &'static str {
        "A stage used by tests."
    }

    fn parallelism(args: &Self::Args) -> Parallelism {
        if args.per_chunk {
            return chunked();
        }

        match args.per_frame {
            true => Parallelism::Frame,
            false => Parallelism::Stream,
        }
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl FilterDefinition for Marking {
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

    fn build(
        _context: &BuildContext,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        unreachable!("the engine tests build fixtures directly")
    }
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub(crate) struct FlowRecordingArgs {
    /// Declares the sink frame by frame rather than for the whole stream.
    #[serde(default)]
    per_frame: bool,
    /// Demands tags, served by a metadata provider placed right before the sink.
    #[serde(default)]
    tags: bool,
}

pub(crate) struct FlowRecordingInstance {
    _life: StageLife,
}

impl FlowRecordingInstance {
    pub(crate) fn new() -> Self {
        Self {
            _life: StageLife::new(),
        }
    }
}

impl StageInstance for FlowRecordingInstance {}

impl MetaSink for FlowRecordingInstance {
    fn consume(&mut self, _context: &mut FrameContext, mut inputs: MetaInputs) -> Result<(), StageError> {
        let item = inputs.take(0).expect("an item");
        let entry = format!("frame {}", item.meta().index);

        FLOWS.lock().expect("not poisoned").push(entry);

        Ok(())
    }

    fn consume_record(&mut self, _context: &mut FrameContext, record: &MetaRecord) -> Result<(), StageError> {
        let entry = described_record(record);

        FLOWS.lock().expect("not poisoned").push(entry);

        Ok(())
    }
}

/// A record's kind, followed by its first stamped path and that path's first value, or `-` for each missing.
fn described_record(record: &MetaRecord) -> String {
    let first_path = record.attachments().stages().next();
    let Some((path, values)) = first_path else {
        return format!("{} - -", record.kind());
    };

    let value = match values.first() {
        Some((_, value)) => described_value(value),
        None => String::from("-"),
    };

    format!("{} {path} {value}", record.kind())
}

fn described_value(value: &MetaValue) -> String {
    match value {
        MetaValue::Bool(flag) => flag.to_string(),
        MetaValue::Int(number) => number.to_string(),
        MetaValue::Float(number) => number.to_string(),
        MetaValue::String(text) => text.clone(),
    }
}

/// A metadata-only sink writing every frame and record it is handed into [FLOWS].
pub(crate) struct FlowRecording;

impl StageDefinition for FlowRecording {
    type Args = FlowRecordingArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "flow-recording"
    }

    fn description() -> &'static str {
        "A stage used by tests."
    }

    fn demands(args: &Self::Args) -> Vec<InputDemand> {
        if !args.tags {
            return Vec::new();
        }

        let request = TagsRequest {
            host_cuts: false,
            before_demander: true,
        };

        vec![InputDemand {
            input: "input",
            demand: AnyDemand::new::<FixtureTags>(request),
        }]
    }

    fn parallelism(args: &Self::Args) -> Parallelism {
        match args.per_frame {
            true => Parallelism::Frame,
            false => Parallelism::Stream,
        }
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[InputParam::required("input")]);
        SIGNATURE
    }
}

impl MetaSinkDefinition for FlowRecording {
    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        Ok(FormatSet::of([offered.sample()]))
    }

    fn build(
        _context: &BuildContext,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn MetaSink>, StageError> {
        Ok(Box::new(FlowRecordingInstance::new()))
    }
}
