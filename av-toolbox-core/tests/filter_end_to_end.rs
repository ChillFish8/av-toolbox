// A throwaway filter driven through the whole stage surface against a real device. It is here
// rather than in a shipped crate because nothing else in the workspace calls `process` yet, and a
// surface nobody calls is a surface nobody has checked.
#![cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]

use av_toolbox_core::{
    BoundInputs,
    Filter,
    FilterDefinition,
    InputArgument,
    InputParam,
    InputSignature,
    Inputs,
    PipelineRegistry,
    PreparedStage,
    StageDefinition,
    StageError,
    StageHandle,
    StageInstance,
    StageKind,
    StreamType,
};
use av_toolbox_gpu::{
    Backend,
    BuildContext,
    BuildTask,
    Capabilities,
    Channel,
    ColourInfo,
    Device,
    DeviceSide,
    FormatSet,
    Frame,
    FrameContext,
    FrameFormat,
    FrameLayout,
    FrameMeta,
    Offered,
    OutputFormats,
    Plane,
    ResolvedFormats,
    SampleFormat,
    StreamFormat,
    StreamTemplate,
    Subsampling,
};
// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;
use schemars::JsonSchema;
use serde::Deserialize;

/// What the input frame's luma holds before the filter runs.
const LUMA_IN: f32 = 10.0;

/// What the input frame's chroma holds before the filter runs.
const CHROMA_IN: f32 = 20.0;

/// What a frame left in the pool holds, so a plane the filter forgets to write is recognisable.
const STALE: f32 = 99.0;

const AMOUNT: f32 = 0.25;

#[cube(launch_unchecked)]
fn brighten_in_place(plane: &mut Array<f32>, amount: f32) {
    if ABSOLUTE_POS >= plane.len() {
        terminate!();
    }

    plane[ABSOLUTE_POS] += amount;
}

#[cube(launch_unchecked)]
fn brighten_into(input: &Array<f32>, output: &mut Array<f32>, amount: f32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    output[ABSOLUTE_POS] = input[ABSOLUTE_POS] + amount;
}

#[cube(launch_unchecked)]
fn fill(plane: &mut Array<f32>, value: f32) {
    if ABSOLUTE_POS >= plane.len() {
        terminate!();
    }

    plane[ABSOLUTE_POS] = value;
}

fn frame_format() -> FrameFormat {
    FrameFormat {
        width: 64,
        height: 64,
        layout: FrameLayout::Yuv {
            subsampling: Subsampling::Yuv420,
        },
        sample: SampleFormat::F32,
        colour: ColourInfo::default(),
        frame_rate: None,
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct BrightenArgs {
    amount: f32,
}

/// Adds a constant to a frame's luma, leaving its chroma as it found it.
struct BrightenFilter<R: Runtime> {
    client: ComputeClient<R>,
    format: FrameFormat,
    amount: f32,
}

impl<R: Runtime> BrightenFilter<R> {
    fn brighten_plane(&self, context: &FrameContext, plane: &Plane, amount: f32) {
        let cubes = context.cube_count(plane.samples());
        let shape = context.cube_dim();
        let binding = plane.array::<f32, R>();

        // Safety: the binding names exactly as many elements as the plane holds, and the kernel
        // writes nothing past the length it is given.
        unsafe {
            brighten_in_place::launch_unchecked::<R>(&self.client, cubes, shape, binding, amount);
        }
    }

    fn brighten_into_plane(&self, context: &FrameContext, source: &Plane, target: &Plane, amount: f32) {
        let cubes = context.cube_count(target.samples());
        let shape = context.cube_dim();
        let read = source.array::<f32, R>();
        let write = target.array::<f32, R>();

        // Safety: both bindings name exactly as many elements as their plane holds, and the kernel
        // writes nothing past the length it is given.
        unsafe {
            brighten_into::launch_unchecked::<R>(&self.client, cubes, shape, read, write, amount);
        }
    }
}

impl<R: Runtime> StageInstance for BrightenFilter<R> {}

impl<R: Runtime> Filter for BrightenFilter<R> {
    fn process(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<Frame, StageError> {
        let frame = inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;

        let shared = match frame.try_into_mut() {
            Ok(writable) => {
                let luma = writable
                    .plane(Channel::Y)
                    .ok_or_else(|| StageError::msg("the frame carries no luma"))?;
                self.brighten_plane(context, luma, self.amount);

                return Ok(writable.freeze());
            },
            Err(frame) => frame,
        };

        let output = context.alloc_frame(&self.format, *shared.meta());
        for channel in self.format.channels() {
            let source = shared
                .plane(*channel)
                .ok_or_else(|| StageError::msg("the input is missing a channel"))?;
            let target = output
                .plane(*channel)
                .ok_or_else(|| StageError::msg("the output is missing a channel"))?;

            // Output planes come from the pool holding whatever was last written there, so the
            // channels this filter does not brighten still have to be copied.
            let amount = if *channel == Channel::Y { self.amount } else { 0.0 };
            self.brighten_into_plane(context, source, target, amount);
        }

        Ok(output)
    }
}

struct Brighten;

impl StageDefinition for Brighten {
    type Args = BrightenArgs;

    fn package() -> &'static str {
        "av-toolbox-test"
    }

    fn identifier() -> &'static str {
        "brighten"
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("source").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl FilterDefinition for Brighten {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        offered
            .as_frames()
            .ok_or_else(|| StageError::msg("brighten takes frames, not a mask"))?;

        Ok(FormatSet::of([SampleFormat::F32]))
    }

    fn produces(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let format = chosen.iter().flatten().next().copied();
        let format = format.ok_or_else(|| StageError::msg("no input was bound"))?;

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([SampleFormat::F32]),
        })
    }

    fn build<R: Runtime>(
        context: &BuildContext<R>,
        args: Self::Args,
        _inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        let output = formats
            .output()
            .ok_or_else(|| StageError::msg("a filter resolves to an output"))?;
        let format = output
            .as_frames()
            .ok_or_else(|| StageError::msg("brighten produces frames, not a mask"))?;

        Ok(Box::new(BrightenFilter {
            client: context.client().clone(),
            format: *format,
            amount: args.amount,
        }))
    }
}

fn resolved_formats() -> ResolvedFormats {
    let format = StreamFormat::Frames(frame_format());

    ResolvedFormats::new(vec![Some(format)], Some(format))
}

/// Builds the filter the way a pipeline would, through the registry and a real backend.
fn built_filter(backend: &Backend) -> Box<dyn Filter> {
    let mut registry = PipelineRegistry::new();
    registry.register_filter::<Brighten>().expect("should register");

    let args = serde_json::json!({ "amount": AMOUNT });
    let arguments = vec![InputArgument::positional(StageHandle::new(1), StreamType::Frames)];

    let prepared = registry
        .prepare(StageKind::Filter, "av-toolbox-test/brighten", &args, &arguments)
        .expect("should prepare");
    let PreparedStage::Filter(prepared) = prepared else {
        panic!("expected a filter");
    };

    prepared
        .build(backend, &resolved_formats())
        .expect("should build")
        .into_device()
        .expect("a device filter")
}

/// A frame whose luma and chroma each hold one known value.
fn filled_frame<R: Runtime>(
    build: &BuildContext<R>,
    context: &FrameContext,
    luma: f32,
    chroma: f32,
) -> Frame {
    let frame = context.alloc_frame(&frame_format(), FrameMeta::at(0));

    for channel in frame.format().channels() {
        let plane = frame.plane(*channel).expect("the frame carries this channel");
        let value = if *channel == Channel::Y { luma } else { chroma };

        let cubes = context.cube_count(plane.samples());
        let shape = context.cube_dim();
        let binding = plane.array::<f32, R>();

        // Safety: the binding names exactly as many elements as the plane holds, and the kernel
        // writes nothing past the length it is given.
        unsafe {
            fill::launch_unchecked::<R>(build.client(), cubes, shape, binding, value);
        }
    }

    frame
}

fn read_plane<R: Runtime>(build: &BuildContext<R>, frame: &Frame, channel: Channel) -> Vec<f32> {
    let plane = frame.plane(channel).expect("the frame carries this channel");
    let bytes = build
        .client()
        .read_one(plane.buffer().handle().clone())
        .expect("the device should hand back what it was asked for");

    f32::from_bytes(&bytes).to_vec()
}

/// The samples of one run's output.
struct Processed {
    luma: Vec<f32>,
    chroma: Vec<f32>,
}

/// The same, plus what the shared input held once the filter was done with it.
struct ProcessedShared {
    output: Processed,
    input_luma: Vec<f32>,
}

/// Drives the filter over a frame nobody else holds, so it writes in place.
struct RunExclusive {
    filter: Box<dyn Filter>,
}

impl BuildTask for RunExclusive {
    type Output = Processed;

    fn run<R: Runtime>(mut self, build: &BuildContext<R>) -> Self::Output {
        let mut context = build.frame_context();
        let input = filled_frame(build, &context, LUMA_IN, CHROMA_IN);

        self.filter
            .start_of_stream(&mut context)
            .expect("should start the stream");

        let inputs = Inputs::new(vec![Some(input.into())]);
        let output = self.filter.process(&mut context, inputs).expect("should process");

        Processed {
            luma: read_plane(build, &output, Channel::Y),
            chroma: read_plane(build, &output, Channel::U),
        }
    }
}

/// Drives the filter over a frame somebody else still holds, so it allocates its own output.
struct RunShared {
    filter: Box<dyn Filter>,
}

impl BuildTask for RunShared {
    type Output = ProcessedShared;

    fn run<R: Runtime>(mut self, build: &BuildContext<R>) -> Self::Output {
        let mut context = build.frame_context();
        let input = filled_frame(build, &context, LUMA_IN, CHROMA_IN);

        // A frame left in the pool's free lists, so the output the filter allocates comes back
        // holding it rather than anything blank.
        drop(filled_frame(build, &context, STALE, STALE));

        self.filter
            .start_of_stream(&mut context)
            .expect("should start the stream");

        let held = input.clone();
        let inputs = Inputs::new(vec![Some(input.into())]);
        let output = self.filter.process(&mut context, inputs).expect("should process");

        ProcessedShared {
            output: Processed {
                luma: read_plane(build, &output, Channel::Y),
                chroma: read_plane(build, &output, Channel::U),
            },
            input_luma: read_plane(build, &held, Channel::Y),
        }
    }
}

fn backend() -> Backend {
    Backend::open_default(Device::Default).expect("a backend should open")
}

#[test]
fn a_filter_writes_a_frame_nobody_else_holds_in_place() {
    let backend = backend();
    let filter = built_filter(&backend);

    let processed = backend.build(RunExclusive { filter });

    assert_eq!(processed.luma.len(), 64 * 64);
    assert!(
        processed.luma.iter().all(|sample| *sample == LUMA_IN + AMOUNT),
        "the luma was not brightened",
    );
    assert!(
        processed.chroma.iter().all(|sample| *sample == CHROMA_IN),
        "the chroma did not survive an in place write",
    );
}

/// The case pooled memory makes dangerous. A filter that allocates its own output has to write
/// every plane, because the ones it skips arrive holding an unrelated frame.
#[test]
fn a_filter_copies_the_planes_it_does_not_brighten_into_a_fresh_output() {
    let backend = backend();
    let filter = built_filter(&backend);

    let processed = backend.build(RunShared { filter });

    assert!(
        processed
            .output
            .luma
            .iter()
            .all(|sample| *sample == LUMA_IN + AMOUNT),
        "the luma was not brightened",
    );
    assert!(
        processed.output.chroma.iter().all(|sample| *sample == CHROMA_IN),
        "the output chroma came from the pool rather than the input",
    );
    assert!(
        processed.input_luma.iter().all(|sample| *sample == LUMA_IN),
        "a shared input was written through",
    );
}
