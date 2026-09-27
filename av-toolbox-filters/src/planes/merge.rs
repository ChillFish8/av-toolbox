use av_toolbox_core::{
    BoundInputs,
    Filter,
    FilterDefinition,
    InputParam,
    InputSignature,
    Inputs,
    Parallelism,
    StageDefinition,
    StageError,
    StageInstance,
    StreamType,
};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    Channel,
    DeviceSide,
    FormatSet,
    Frame,
    FrameContext,
    FrameError,
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

use super::ChannelName;

const STAGE: &str = "planes/merge";

/// Where one output plane comes from.
#[derive(Debug, Copy, Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct Pick {
    /// Which input to take the plane from, counting from 0.
    pub(crate) input: usize,
    /// Which of that input's planes to take.
    ///
    /// Unset takes the plane of the same name, or a Grey input's only plane.
    #[serde(default)]
    pub(crate) plane: Option<ChannelName>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct MergeArgs {
    #[serde(default)]
    pub(crate) y: Option<Pick>,
    #[serde(default)]
    pub(crate) u: Option<Pick>,
    #[serde(default)]
    pub(crate) v: Option<Pick>,
    #[serde(default)]
    pub(crate) r: Option<Pick>,
    #[serde(default)]
    pub(crate) g: Option<Pick>,
    #[serde(default)]
    pub(crate) b: Option<Pick>,
}

/// One output plane, and which plane of which input fills it.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) struct Source {
    pub(crate) input: usize,
    pub(crate) plane: Channel,
}

/// The output format, and the source of each of its planes in plane order.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Merged {
    pub(crate) format: FrameFormat,
    pub(crate) sources: Vec<Source>,
}

/// Which layout `planes/merge` is being asked to build.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum Kind {
    Grey,
    Yuv,
    Rgb,
}

fn needs_a_group() -> StageError {
    let message = format!("{STAGE} needs y, y with u and v, or r, g and b");

    StageError::msg(message)
}

/// Which layout `args` picks, from which keys carry a [Pick].
fn kind_of(args: &MergeArgs) -> Result<Kind, StageError> {
    let yuv = (args.y.is_some(), args.u.is_some(), args.v.is_some());
    let rgb = (args.r.is_some(), args.g.is_some(), args.b.is_some());

    match (yuv, rgb) {
        ((true, false, false), (false, false, false)) => Ok(Kind::Grey),
        ((true, true, true), (false, false, false)) => Ok(Kind::Yuv),
        ((false, false, false), (true, true, true)) => Ok(Kind::Rgb),
        _ => Err(needs_a_group()),
    }
}

/// The channels `kind` builds, in plane order.
fn output_channels(kind: Kind) -> &'static [Channel] {
    match kind {
        Kind::Grey => &[Channel::Y],
        Kind::Yuv => &[Channel::Y, Channel::U, Channel::V],
        Kind::Rgb => &[Channel::R, Channel::G, Channel::B],
    }
}

/// The pick `args` holds for `channel`.
fn pick_for(args: &MergeArgs, channel: Channel) -> Option<Pick> {
    match channel {
        Channel::Y => args.y,
        Channel::U => args.u,
        Channel::V => args.v,
        Channel::R => args.r,
        Channel::G => args.g,
        Channel::B => args.b,
    }
}

/// The argument key naming `channel`, for error messages.
fn key_for(channel: Channel) -> &'static str {
    match channel {
        Channel::Y => "y",
        Channel::U => "u",
        Channel::V => "v",
        Channel::R => "r",
        Channel::G => "g",
        Channel::B => "b",
    }
}

fn input_out_of_range(count: usize, key: &str, input: usize) -> StageError {
    let message = format!("{STAGE} has {count} inputs, but {key} picks input {input}");

    StageError::msg(message)
}

/// The plane `pick` takes from an input of `layout`, when it names an output channel of `output`.
fn source_channel_for(pick: Pick, layout: FrameLayout, output: Channel) -> Channel {
    if let Some(name) = pick.plane {
        return name.channel();
    }

    if let FrameLayout::Grey = layout {
        return Channel::Y;
    }

    output
}

fn missing_channel(channel: Channel, input: usize, layout: FrameLayout) -> StageError {
    let message = format!("{STAGE} cannot take {channel} from input {input}, a {layout} clip");

    StageError::msg(message)
}

/// One entry per output channel: which input and plane fills it, and that plane's dimensions.
struct Picked {
    source: Source,
    dimensions: (u32, u32),
}

/// Works out where every output channel of `kind` comes from, and checks each input carries it.
fn picked_sources(kind: Kind, args: &MergeArgs, inputs: &[FrameFormat]) -> Result<Vec<Picked>, StageError> {
    let mut picked = Vec::with_capacity(output_channels(kind).len());

    for channel in output_channels(kind) {
        let key = key_for(*channel);
        let pick = pick_for(args, *channel).expect("kind_of confirmed this channel is picked");

        if pick.input >= inputs.len() {
            return Err(input_out_of_range(inputs.len(), key, pick.input));
        }

        let input_format = inputs[pick.input];
        let source_channel = source_channel_for(pick, input_format.layout, *channel);

        if !input_format.layout.carries(source_channel) {
            return Err(missing_channel(source_channel, pick.input, input_format.layout));
        }

        let dimensions = input_format
            .channel_dimensions(source_channel)
            .expect("carries was just checked");

        picked.push(Picked {
            source: Source {
                input: pick.input,
                plane: source_channel,
            },
            dimensions,
        });
    }

    Ok(picked)
}

fn unused_input(index: usize) -> StageError {
    let message = format!("{STAGE} is given input {index} but never picks from it");

    StageError::msg(message)
}

/// Refuses any input index that no source names.
fn checked_every_input_used(sources: &[Source], inputs: usize) -> Result<(), StageError> {
    for index in 0..inputs {
        let used = sources.iter().any(|source| source.input == index);
        if !used {
            return Err(unused_input(index));
        }
    }

    Ok(())
}

fn different_samples(first: SampleFormat, second: SampleFormat) -> StageError {
    let message = format!("{STAGE} needs every input the same sample, got {first} and {second}");

    StageError::msg(message)
}

/// The sample every input shares, refusing inputs that differ.
fn checked_same_sample(inputs: &[FrameFormat]) -> Result<SampleFormat, StageError> {
    let reference = inputs[0].sample;

    for input in inputs {
        if input.sample != reference {
            return Err(different_samples(reference, input.sample));
        }
    }

    Ok(reference)
}

fn axis_factor(luma: u32, chroma: u32) -> Option<u32> {
    if chroma == luma {
        return Some(1);
    }

    if luma > 1 && chroma == luma.div_ceil(2) {
        return Some(2);
    }

    None
}

fn bad_chroma_size(luma: (u32, u32), chroma: (u32, u32)) -> StageError {
    let message = format!(
        "{STAGE} cannot fit a {}x{} chroma plane onto a {}x{} luma plane",
        chroma.0, chroma.1, luma.0, luma.1,
    );

    StageError::msg(message)
}

/// The subsampling a chroma plane of `chroma` implies against a luma plane of `luma`.
fn subsampling_for(luma: (u32, u32), chroma: (u32, u32)) -> Result<Subsampling, StageError> {
    let factor_x = axis_factor(luma.0, chroma.0);
    let factor_y = axis_factor(luma.1, chroma.1);

    match (factor_x, factor_y) {
        (Some(1), Some(1)) => Ok(Subsampling::Yuv444),
        (Some(2), Some(1)) => Ok(Subsampling::Yuv422),
        (Some(2), Some(2)) => Ok(Subsampling::Yuv420),
        _ => Err(bad_chroma_size(luma, chroma)),
    }
}

fn rgb_size_mismatch(r: (u32, u32), g: (u32, u32), b: (u32, u32)) -> StageError {
    let message = format!(
        "{STAGE} needs r, g and b the same size, got {}x{}, {}x{} and {}x{}",
        r.0, r.1, g.0, g.1, b.0, b.1,
    );

    StageError::msg(message)
}

fn uv_size_mismatch(u: (u32, u32), v: (u32, u32)) -> StageError {
    let message = format!(
        "{STAGE} needs u and v the same size, got {}x{} and {}x{}",
        u.0, u.1, v.0, v.1,
    );

    StageError::msg(message)
}

/// The width, height and layout `kind` builds, from the dimensions of each picked plane.
fn shaped(kind: Kind, dimensions: &[(u32, u32)]) -> Result<(u32, u32, FrameLayout), StageError> {
    match kind {
        Kind::Grey => {
            let (width, height) = dimensions[0];

            Ok((width, height, FrameLayout::Grey))
        },
        Kind::Rgb => {
            let (r, g, b) = (dimensions[0], dimensions[1], dimensions[2]);
            if r != g || r != b {
                return Err(rgb_size_mismatch(r, g, b));
            }

            Ok((r.0, r.1, FrameLayout::Rgb))
        },
        Kind::Yuv => {
            let (luma, u, v) = (dimensions[0], dimensions[1], dimensions[2]);
            if u != v {
                return Err(uv_size_mismatch(u, v));
            }

            let subsampling = subsampling_for(luma, u)?;

            Ok((luma.0, luma.1, FrameLayout::Yuv { subsampling }))
        },
    }
}

/// Works out the output format and the source of each of its planes.
///
/// `args` picks which of `inputs` fills each output plane, checking every pick is bound, carried
/// and the right shape and sample along the way.
pub(crate) fn merged(args: &MergeArgs, inputs: &[FrameFormat]) -> Result<Merged, StageError> {
    let kind = kind_of(args)?;
    let picked = picked_sources(kind, args, inputs)?;

    let sources: Vec<Source> = picked.iter().map(|entry| entry.source).collect();
    let dimensions: Vec<(u32, u32)> = picked.iter().map(|entry| entry.dimensions).collect();

    checked_every_input_used(&sources, inputs.len())?;
    let sample = checked_same_sample(inputs)?;
    let (width, height, layout) = shaped(kind, &dimensions)?;

    let mut colour = inputs[0].colour;
    if let Kind::Yuv = kind {
        let u_input = sources[1].input;
        colour.chroma_location = inputs[u_input].colour.chroma_location;
    }

    let format = FrameFormat {
        width,
        height,
        layout,
        sample,
        colour,
        frame_rate: inputs[0].frame_rate,
    };

    Ok(Merged { format, sources })
}

fn not_frames() -> StageError {
    let message = format!("{STAGE} merges frames, not masks");

    StageError::msg(message)
}

/// What `index` may accept, given the samples every other input has settled or proposed.
fn accepts_clip(offered: StreamFormat, index: usize, inputs: &[Offered]) -> Result<FormatSet, StageError> {
    let Some(format) = offered.as_frames() else {
        return Err(not_frames());
    };

    let mismatched = inputs
        .iter()
        .enumerate()
        .filter(|(other_index, _)| *other_index != index)
        .filter_map(|(_, other)| other.format())
        .any(|other_format| other_format.sample() != format.sample);

    if mismatched {
        return Ok(FormatSet::new());
    }

    Ok(FormatSet::of([format.sample]))
}

fn missing_input() -> StageError {
    let message = format!("{STAGE} needs every input bound");

    StageError::msg(message)
}

/// Every chosen input as frames, once none is unbound or a mask.
fn checked_inputs(chosen: &[Option<StreamFormat>]) -> Result<Vec<FrameFormat>, StageError> {
    let mut formats = Vec::with_capacity(chosen.len());

    for slot in chosen {
        let Some(stream) = slot else {
            return Err(missing_input());
        };

        let Some(format) = stream.as_frames() else {
            return Err(not_frames());
        };

        formats.push(*format);
    }

    Ok(formats)
}

/// Builds a clip by picking each plane from one of several input clips.
pub(crate) struct Merge;

impl StageDefinition for Merge {
    type Args = MergeArgs;

    fn package() -> &'static str {
        "planes"
    }

    fn identifier() -> &'static str {
        "merge"
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Frame
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::with_variadic_of(
            &[InputParam::required("first")
                .accepting(StreamType::Frames)
                .chunk_authority()],
            StreamType::Frames,
        );
        SIGNATURE
    }
}

impl FilterDefinition for Merge {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        index: usize,
        offered: StreamFormat,
        inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        accepts_clip(offered, index, inputs)
    }

    fn produces(
        _capabilities: &Capabilities,
        args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let inputs = checked_inputs(chosen)?;
        let merge = merged(args, &inputs)?;
        let format = StreamFormat::Frames(merge.format);

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([merge.format.sample]),
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        args: Self::Args,
        _inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        let chosen = formats.inputs();
        let inputs = checked_inputs(chosen)?;
        let merge = merged(&args, &inputs)?;
        let merger = Merger::new(merge);

        Ok(Box::new(merger))
    }
}

fn missing_frame(index: usize) -> StageError {
    let message = format!("{STAGE} needs input {index} bound");

    StageError::msg(message)
}

fn bad_frame(error: FrameError) -> StageError {
    let message = format!("{STAGE} built a bad frame: {error}");

    StageError::msg(message)
}

/// Builds a frame by sharing planes out of several other frames, rather than copying them.
pub(crate) struct Merger {
    merged: Merged,
}

impl Merger {
    pub(crate) fn new(merged: Merged) -> Self {
        Self { merged }
    }
}

impl StageInstance for Merger {}

impl Filter for Merger {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Frame, StageError> {
        let mut frames = Vec::with_capacity(inputs.len());

        for index in 0..inputs.len() {
            let Some(frame) = inputs.take_frame(index) else {
                return Err(missing_frame(index));
            };

            frames.push(frame);
        }

        let meta = frames[0].meta().clone();
        let mut planes = Vec::with_capacity(self.merged.sources.len());

        for source in &self.merged.sources {
            let frame = &frames[source.input];
            let plane = frame
                .plane(source.plane)
                .expect("merged validated every source plane exists");

            planes.push(plane.clone());
        }

        let output = Frame::from_planes(self.merged.format, planes, meta).map_err(bad_frame)?;

        Ok(output)
    }
}
