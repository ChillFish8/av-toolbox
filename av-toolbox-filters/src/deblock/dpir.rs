use av_toolbox_core::StageError;
use av_toolbox_cubek::dpir::{PACK, Variant};
use av_toolbox_gpu::{FormatSet, FrameFormat, FrameLayout, SampleFormat, StreamFormat};
use av_toolbox_models::{Located, MissingModel, Store};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::checks::between_f32;
use crate::cnn::PrecisionName;
use crate::planes::ChannelName;
use crate::selection::{PlaneDefault, selected_planes};

pub(crate) const STAGE: &str = "deblock/dpir";

const DEFAULT_STRENGTH: f32 = 10.0;

/// How strongly to deblock, on an 8-bit scale, once for every plane or once per plane.
#[derive(Debug, Clone, PartialEq, Deserialize, JsonSchema)]
#[serde(untagged)]
pub(crate) enum StrengthArg {
    One(f32),
    PerPlane(Vec<f32>),
}

impl Default for StrengthArg {
    fn default() -> Self {
        Self::One(DEFAULT_STRENGTH)
    }
}

/// Which DPIR network runs, as the stage's arguments name it.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ModelChoice {
    /// The colour network on RGB, the gray network otherwise.
    #[default]
    Auto,
    Gray,
    Colour,
}

/// Arguments for `deblock/dpir`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct DpirArgs {
    /// How strongly to deblock, between 0 and 255. A list gives one per selected plane, gray model only.
    #[serde(default)]
    pub(crate) strength: StrengthArg,
    /// `colour` runs the RGB planes together, `gray` runs each plane on its own, `auto` picks by layout.
    #[serde(default)]
    pub(crate) model: ModelChoice,
    /// The planes the gray model runs on, every plane by default.
    pub(crate) planes: Option<Vec<ChannelName>>,
    /// The precision of the feature maps. `f32` is slower and more exact than `f16`.
    #[serde(default)]
    pub(crate) precision: PrecisionName,
}

/// What each frame runs through the network as.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Runs {
    /// The three RGB planes together through the colour network.
    Colour { strength: f32 },
    /// Each plane on its own through the gray network, one entry per plane in channel order, `None` for a
    /// plane left alone.
    Gray { strengths: Vec<Option<f32>> },
}

impl Runs {
    pub(crate) fn variant(&self) -> Variant {
        match self {
            Self::Colour { .. } => Variant::Colour,
            Self::Gray { .. } => Variant::Gray,
        }
    }
}

fn not_frames() -> StageError {
    let message = format!("{STAGE} deblocks frames, not masks");

    StageError::msg(message)
}

fn missing_input() -> StageError {
    let message = format!("{STAGE} needs its input bound");

    StageError::msg(message)
}

fn colour_needs_rgb(layout: FrameLayout) -> StageError {
    let message = format!(
        "{STAGE}'s colour model runs on rgb frames, not {layout} frames. Convert with resample/chroma to \
         4:4:4 and resample/matrix to rgb first, or set model to \"gray\""
    );

    StageError::msg(message)
}

fn planes_with_colour() -> StageError {
    let message = format!(
        "{STAGE}'s colour model runs on all three rgb planes together, so it takes no planes. Set model to \
         \"gray\" to choose planes"
    );

    StageError::msg(message)
}

fn strength_list_with_colour() -> StageError {
    let message =
        format!("{STAGE}'s colour model takes one strength, not a list. Set model to \"gray\" for a list");

    StageError::msg(message)
}

fn strength_count(given: usize, planes: usize) -> StageError {
    let message = format!("{STAGE} needs one strength per selected plane, got {given} for {planes} planes");

    StageError::msg(message)
}

fn model_missing(variant: Variant, missing: MissingModel) -> StageError {
    let name = model_argument(variant);
    let message = format!("{STAGE} cannot find the weights for the {name} model. {missing}");

    StageError::msg(message)
}

fn model_argument(variant: Variant) -> &'static str {
    match variant {
        Variant::Gray => "gray",
        Variant::Colour => "colour",
    }
}

fn checked_strength(strength: f32) -> Result<f32, StageError> {
    between_f32(STAGE, "strength", strength, 0.0, 255.0)?;

    Ok(strength)
}

/// The input format, once it is known to be frames.
pub(crate) fn checked_input(chosen: &[Option<StreamFormat>]) -> Result<FrameFormat, StageError> {
    let Some(input) = chosen.first().copied().flatten() else {
        return Err(missing_input());
    };

    let Some(format) = input.as_frames() else {
        return Err(not_frames());
    };

    Ok(*format)
}

/// U8, U16 and F32 frames, refusing masks.
pub(crate) fn accepted_samples(offered: StreamFormat) -> Result<FormatSet, StageError> {
    let Some(format) = offered.as_frames() else {
        return Err(not_frames());
    };

    if !matches!(
        format.sample,
        SampleFormat::U8 | SampleFormat::U16 { .. } | SampleFormat::F32
    ) {
        return Ok(FormatSet::new());
    }

    Ok(FormatSet::of([format.sample]))
}

fn colour_runs(args: &DpirArgs, layout: FrameLayout) -> Result<Runs, StageError> {
    if layout != FrameLayout::Rgb {
        return Err(colour_needs_rgb(layout));
    }

    if args.planes.is_some() {
        return Err(planes_with_colour());
    }

    let StrengthArg::One(strength) = args.strength else {
        return Err(strength_list_with_colour());
    };

    let strength = checked_strength(strength)?;

    Ok(Runs::Colour { strength })
}

/// The plane indices to run, in the order `planes` lists them, or in channel order without it.
fn ordered_planes(args: &DpirArgs, layout: FrameLayout) -> Result<Vec<usize>, StageError> {
    let requested = args.planes.as_deref();
    selected_planes(STAGE, requested, layout, PlaneDefault::Every)?;
    let channels = layout.channels();

    let Some(names) = requested else {
        let every = (0..channels.len()).collect();

        return Ok(every);
    };

    let ordered = names
        .iter()
        .filter_map(|name| channels.iter().position(|channel| *channel == name.channel()))
        .collect();

    Ok(ordered)
}

fn gray_runs(args: &DpirArgs, layout: FrameLayout) -> Result<Runs, StageError> {
    let ordered = ordered_planes(args, layout)?;
    let mut strengths = vec![None; layout.channels().len()];

    let given = match &args.strength {
        StrengthArg::One(strength) => vec![*strength; ordered.len()],
        StrengthArg::PerPlane(list) => list.clone(),
    };

    if given.len() != ordered.len() {
        return Err(strength_count(given.len(), ordered.len()));
    }

    for (index, strength) in ordered.into_iter().zip(given) {
        let checked = checked_strength(strength)?;
        strengths[index] = Some(checked);
    }

    Ok(Runs::Gray { strengths })
}

/// Checks the arguments against the input, giving what each frame runs through the network as.
pub(crate) fn checked_runs(args: &DpirArgs, input: &FrameFormat) -> Result<Runs, StageError> {
    let model = match (args.model, input.layout) {
        (ModelChoice::Auto, FrameLayout::Rgb) => ModelChoice::Colour,
        (ModelChoice::Auto, _) => ModelChoice::Gray,
        (chosen, _) => chosen,
    };

    match model {
        ModelChoice::Colour => colour_runs(args, input.layout),
        _ => gray_runs(args, input.layout),
    }
}

/// Finds `variant`'s weight file in `store`.
pub(crate) fn locate(store: &Store, variant: Variant) -> Result<Located, StageError> {
    let found = store.find(PACK, variant.file_name());

    match found {
        Ok(located) => Ok(located),
        Err(missing) => Err(model_missing(variant, missing)),
    }
}
