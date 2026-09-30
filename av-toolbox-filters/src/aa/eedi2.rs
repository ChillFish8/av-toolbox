use av_toolbox_core::{
    BoundInputs,
    Filter,
    FilterDefinition,
    InputSignature,
    Inputs,
    Parallelism,
    StageDefinition,
    StageError,
    StageInstance,
};
use av_toolbox_cubek::DevicePlane;
use av_toolbox_cubek::aa::{Eedi2Arguments, Eedi2Scratch, Eedi2Settings, eedi2_field, eedi2_scratch};
use av_toolbox_cubek::depth::Element;
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    Channel,
    ComputeClient,
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
    StreamFormat,
    StreamTemplate,
};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::aa::direction::{Direction, FieldPass, checked_dimensions, missing_plane, run_direction};
use crate::bits::bits_of;
use crate::checks::between_u32;
use crate::masked::{
    INPUT,
    MASK,
    MASKED_INPUTS,
    MaskedFilter,
    accepts_masked_input,
    checked_input,
    checked_mask,
    missing_input,
};
use crate::planes::ChannelName;
use crate::planewise::{PlaneFilter, Planewise};
use crate::selection::{PlaneDefault, selected_planes};
use crate::sizing::Sizing;

const STAGE: &str = "aa/eedi2";

const MIN_MAXD: u32 = 1;

const MAX_MAXD: u32 = 29;

const MAX_PP: u32 = 1;

/// The fewest samples in a row EEDI2 rebuilds.
const MIN_ROW_WIDTH: u32 = 8;

/// The fewest rows a plane needs in the direction EEDI2 runs, two for each of a field's 7 rows.
const MIN_PLANE_ROWS: u32 = 14;

/// The most samples in a row EEDI2 rebuilds, the widest its gap fill rounds exactly.
const MAX_ROW_WIDTH: u32 = 65535;

fn default_mthresh() -> i32 {
    10
}

fn default_lthresh() -> i32 {
    20
}

fn default_vthresh() -> i32 {
    20
}

fn default_estr() -> i32 {
    2
}

fn default_dstr() -> i32 {
    4
}

fn default_maxd() -> u32 {
    24
}

fn default_nt() -> i32 {
    50
}

fn default_pp() -> u32 {
    1
}

/// Arguments for `aa/eedi2`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct Eedi2Args {
    /// The edge magnitude threshold, at least 0.
    #[serde(default = "default_mthresh")]
    pub(crate) mthresh: i32,
    /// The Laplacian threshold, at least 0.
    #[serde(default = "default_lthresh")]
    pub(crate) lthresh: i32,
    /// The variance threshold, at least 0.
    #[serde(default = "default_vthresh")]
    pub(crate) vthresh: i32,
    /// How strongly the edge mask erodes, at least 0.
    #[serde(default = "default_estr")]
    pub(crate) estr: i32,
    /// How strongly the edge mask dilates, at least 0.
    #[serde(default = "default_dstr")]
    pub(crate) dstr: i32,
    /// The furthest direction searched, between 1 and 29.
    #[serde(default = "default_maxd")]
    pub(crate) maxd: u32,
    /// The noise threshold, at least 0.
    #[serde(default = "default_nt")]
    pub(crate) nt: i32,
    /// Whether the output is post-processed, 0 or 1.
    #[serde(default = "default_pp")]
    pub(crate) pp: u32,
    /// The planes to anti-alias, the first plane when not given.
    #[serde(default)]
    pub(crate) planes: Option<Vec<ChannelName>>,
    /// The axes to anti-alias along, both when not given.
    #[serde(default)]
    pub(crate) direction: Direction,
}

fn negative(name: &str, value: i32) -> StageError {
    let message = format!("{STAGE} needs {name} of at least 0, got {value}");

    StageError::msg(message)
}

fn too_small(channel: Channel, direction: Direction, size: (u32, u32), smallest: (u32, u32)) -> StageError {
    let (width, height) = size;
    let (smallest_width, smallest_height) = smallest;
    let message = format!(
        "{STAGE} needs plane {channel} to be at least {smallest_width}x{smallest_height} with direction \
         {direction}, got {width}x{height}"
    );

    StageError::msg(message)
}

fn too_large(channel: Channel, direction: Direction, dimension: &str, size: u32) -> StageError {
    let message = format!(
        "{STAGE} needs a {dimension} of at most {MAX_ROW_WIDTH} on plane {channel} with direction \
         {direction}, got {size}"
    );

    StageError::msg(message)
}

/// Refuses a negative `value` for the argument `name`, handing it back unsigned otherwise.
fn non_negative(name: &str, value: i32) -> Result<u32, StageError> {
    match u32::try_from(value) {
        Ok(unsigned) => Ok(unsigned),
        Err(_) => Err(negative(name, value)),
    }
}

/// The arguments the device pass takes, once `args` are checked.
fn checked_arguments(args: &Eedi2Args) -> Result<Eedi2Arguments, StageError> {
    let mthresh = non_negative("mthresh", args.mthresh)?;
    let lthresh = non_negative("lthresh", args.lthresh)?;
    let vthresh = non_negative("vthresh", args.vthresh)?;
    let estr = non_negative("estr", args.estr)?;
    let dstr = non_negative("dstr", args.dstr)?;
    let nt = non_negative("nt", args.nt)?;
    between_u32(STAGE, "maxd", args.maxd, MIN_MAXD, MAX_MAXD)?;
    between_u32(STAGE, "pp", args.pp, 0, MAX_PP)?;

    Ok(Eedi2Arguments {
        mthresh,
        lthresh,
        vthresh,
        estr,
        dstr,
        maxd: args.maxd,
        nt,
        pp: args.pp,
    })
}

/// The smallest plane, as width and height, that EEDI2 runs on along `direction`.
///
/// A pass rebuilds rows at least 8 wide from a plane at least 14 rows tall. A horizontal pass swaps the
/// two.
fn smallest_plane(direction: Direction) -> (u32, u32) {
    match direction {
        Direction::Vertical => (MIN_ROW_WIDTH, MIN_PLANE_ROWS),
        Direction::Horizontal => (MIN_PLANE_ROWS, MIN_ROW_WIDTH),
        Direction::Both => (MIN_PLANE_ROWS, MIN_PLANE_ROWS),
    }
}

/// Refuses a `width` by `height` plane `channel` too small or too wide for EEDI2 along `direction`.
///
/// A vertical pass rebuilds rows as wide as the plane, and a horizontal pass rows as long as the plane
/// is tall.
fn checked_size(channel: Channel, direction: Direction, width: u32, height: u32) -> Result<(), StageError> {
    let smallest = smallest_plane(direction);
    let (smallest_width, smallest_height) = smallest;
    if width < smallest_width || height < smallest_height {
        return Err(too_small(channel, direction, (width, height), smallest));
    }

    let runs_vertically = matches!(direction, Direction::Vertical | Direction::Both);
    if runs_vertically && width > MAX_ROW_WIDTH {
        return Err(too_large(channel, direction, "width", width));
    }

    let runs_horizontally = matches!(direction, Direction::Horizontal | Direction::Both);
    if runs_horizontally && height > MAX_ROW_WIDTH {
        return Err(too_large(channel, direction, "height", height));
    }

    Ok(())
}

/// Whether each plane of `input` is anti-aliased, once `args` are checked against it.
fn checked_planes(args: &Eedi2Args, input: &FrameFormat) -> Result<Vec<bool>, StageError> {
    let requested = args.planes.as_deref();
    let selected = selected_planes(STAGE, requested, input.layout, PlaneDefault::First)?;
    checked_dimensions(STAGE, input, &selected, args.direction)?;

    let channels = input.channels();
    let chosen = channels
        .iter()
        .zip(&selected)
        .filter(|(_, is_selected)| **is_selected);
    for (channel, _) in chosen {
        let Some((width, height)) = input.channel_dimensions(*channel) else {
            return Err(missing_plane(STAGE, *channel));
        };

        checked_size(*channel, args.direction, width, height)?;
    }

    Ok(selected)
}

/// The log2 subsampling of plane `plane_index` of `layout`, across and down.
///
/// Only the chroma planes of a YUV layout are subsampled. A `Chroma` layout's planes are the frame's own
/// size.
fn subsampling_shifts(layout: FrameLayout, plane_index: usize) -> (u32, u32) {
    let channels = layout.channels();
    let Some(channel) = channels.get(plane_index) else {
        return (0, 0);
    };

    match (layout, channel) {
        (FrameLayout::Yuv { subsampling }, Channel::U | Channel::V) => {
            let (across, down) = subsampling.factors();

            (across.ilog2(), down.ilog2())
        },
        _ => (0, 0),
    }
}

/// Anti-aliases by re-interpolating each field along a smoothed map of edge directions, the way EEDI2
/// does.
pub(crate) struct Eedi2;

impl StageDefinition for Eedi2 {
    type Args = Eedi2Args;

    fn package() -> &'static str {
        "aa"
    }

    fn identifier() -> &'static str {
        "eedi2"
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Frame
    }

    fn inputs() -> &'static InputSignature {
        MASKED_INPUTS
    }
}

impl FilterDefinition for Eedi2 {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        accepts_masked_input(STAGE, index, offered)
    }

    fn produces(
        _capabilities: &Capabilities,
        args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let input = checked_input(STAGE, chosen)?;
        let mask = chosen.get(MASK).copied().flatten();
        checked_arguments(args)?;
        checked_planes(args, &input)?;
        checked_mask(STAGE, &input, mask)?;

        let format = StreamFormat::Frames(input);

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([input.sample]),
        })
    }

    fn build<R: Runtime>(
        context: &BuildContext<R>,
        args: Self::Args,
        _inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        let inputs = formats.inputs();
        let input = checked_input(STAGE, inputs)?;
        let mask = inputs.get(MASK).copied().flatten();
        let arguments = checked_arguments(&args)?;
        let selected = checked_planes(&args, &input)?;
        checked_mask(STAGE, &input, mask)?;

        let bits = bits_of(input.sample);
        let element = if bits == 8 { Element::U8 } else { Element::U16 };
        let peak = ((1u32 << bits) - 1) as f32;

        let planewise = Planewise {
            stage: STAGE,
            format: input,
            selected,
            element,
            peak,
        };
        let antialiaser = Antialiaser {
            client: context.client().clone(),
            planewise,
            direction: args.direction,
            arguments,
            bits,
        };

        Ok(Box::new(antialiaser))
    }
}

/// One EEDI2 field pass at the thresholds of one plane and orientation.
struct Eedi2Pass<'a, R: Runtime> {
    client: &'a ComputeClient<R>,
    settings: Eedi2Settings,
}

impl<R: Runtime> FieldPass for Eedi2Pass<'_, R> {
    fn run(&self, context: &FrameContext, input: DevicePlane<'_>, output: DevicePlane<'_>, offset: u32) {
        let sizes = eedi2_scratch(input.width(), input.height());
        let edges = context.scratch(sizes.edges);
        let working = context.scratch(sizes.working);
        let directions = context.scratch(sizes.directions);
        let spare = context.scratch(sizes.spare);
        let candidates = context.scratch(sizes.candidates);
        let rebuilt = context.scratch(sizes.rebuilt);
        let scratch = Eedi2Scratch {
            edges: edges.handle(),
            working: working.handle(),
            directions: directions.handle(),
            spare: spare.handle(),
            candidates: candidates.handle(),
            rebuilt: rebuilt.handle(),
        };

        let sizing = Sizing(context);
        eedi2_field(
            self.client,
            &sizing,
            input,
            output,
            scratch,
            offset,
            &self.settings,
        );
    }
}

/// Runs `aa/eedi2` on every selected plane of a frame.
struct Antialiaser<R: Runtime> {
    client: ComputeClient<R>,
    planewise: Planewise,
    direction: Direction,
    arguments: Eedi2Arguments,
    bits: u32,
}

impl<R: Runtime> Antialiaser<R> {
    /// A field pass over rows whose samples each span `2^ssw` samples of the full plane.
    fn pass(&self, ssw: u32) -> Eedi2Pass<'_, R> {
        let settings = Eedi2Settings::new(&self.arguments, self.bits, ssw);

        Eedi2Pass {
            client: &self.client,
            settings,
        }
    }
}

impl<R: Runtime> PlaneFilter for Antialiaser<R> {
    fn filter(
        &self,
        context: &FrameContext,
        plane_index: usize,
        input: DevicePlane<'_>,
        output: DevicePlane<'_>,
    ) -> Result<(), StageError> {
        let layout = self.planewise.format.layout;
        let (across, down) = subsampling_shifts(layout, plane_index);
        let vertical = self.pass(across);
        let horizontal = self.pass(down);

        run_direction(
            &self.client,
            context,
            &vertical,
            &horizontal,
            input,
            output,
            self.direction,
        );

        Ok(())
    }
}

impl<R: Runtime> StageInstance for Antialiaser<R> {}

impl<R: Runtime> Filter for Antialiaser<R> {
    fn process(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<Frame, StageError> {
        let Some(input) = inputs.take_frame(INPUT) else {
            return Err(missing_input(STAGE));
        };

        let mask = inputs.take_mask(MASK);
        let masked = MaskedFilter {
            client: &self.client,
            stage: STAGE,
            format: &self.planewise.format,
            mask: mask.as_ref(),
            peak: self.planewise.peak,
            inner: &*self,
        };

        self.planewise.run(&self.client, context, input, &masked)
    }
}
