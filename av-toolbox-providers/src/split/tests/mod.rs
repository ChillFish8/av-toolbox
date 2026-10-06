mod layers;
mod provider;

use av_toolbox_core::{
    AnyDemand,
    DemandKind,
    InputArgument,
    Inputs,
    LayerDemand,
    Provider,
    Provision,
    StageDefinition,
    StageHandle,
    StreamType,
    bind_inputs,
    end_meta_provider,
    run_meta_provider,
};
use av_toolbox_gpu::{AnyItem, FrameContext, FrameFormat, HostFrame, HostSide, HostStreamItem, LayerId};

use crate::scene::score::ServeScores;
use crate::scene::{SceneBreaks, SceneChunk, SceneRequest, SceneScores, Scenes};
use crate::split::provider::{SceneSplitProvider, ServeSplits};

/// The layer the split reads its scores from, apart from every scene layer a test numbers.
const SCORE_LAYER: LayerId = LayerId(99);

/// One layer per request, numbered in order from 0.
fn demands(requests: &[SceneRequest]) -> Vec<LayerDemand> {
    requests
        .iter()
        .enumerate()
        .map(|(position, request)| LayerDemand {
            layer: LayerId(position as u32),
            demand: AnyDemand::new::<Scenes>(request.clone()),
        })
        .collect()
}

/// The score layer the split's own demand asks for, served on [SCORE_LAYER].
fn score_demands(args: &[LayerDemand]) -> Vec<LayerDemand> {
    let args = args.to_vec();

    SceneSplitProvider::demands(&args)
        .into_iter()
        .map(|demanded| LayerDemand {
            layer: SCORE_LAYER,
            demand: demanded.demand,
        })
        .collect()
}

/// A handle reading the scores written on [SCORE_LAYER].
fn scores_provision() -> Provision<SceneScores> {
    let kind = DemandKind::of::<SceneScores>();
    let argument =
        InputArgument::named("input", StageHandle::new(0), StreamType::Frames).with_demand(kind, SCORE_LAYER);
    let bound = bind_inputs(SceneSplitProvider::inputs(), &[argument]).expect("binds");

    bound.provision::<SceneScores>("input").expect("demanded")
}

fn limited(min_frames: u32, max_frames: Option<u32>) -> SceneRequest {
    SceneRequest {
        breaks: SceneBreaks::Limited {
            min_frames,
            max_frames,
            max_seconds: None,
        },
        ..SceneRequest::natural()
    }
}

/// A score provider feeding a split provider, the way the graph stacks them.
struct Chain {
    score: ServeScores,
    split: ServeSplits,
}

impl Chain {
    fn new(requests: &[SceneRequest], format: &FrameFormat) -> Self {
        let args = demands(requests);
        let score_args = score_demands(&args);

        let score = ServeScores::new(&score_args, format).expect("builds");
        let split = ServeSplits::new(&args, format.frame_rate, scores_provision()).expect("builds");

        Self { score, split }
    }

    /// Hands every scored item to the split, returning the frames it emits.
    fn split_all(&mut self, context: &mut FrameContext, scored: Vec<HostStreamItem>) -> Vec<HostFrame> {
        let mut emitted = Vec::new();
        for item in scored {
            let item = AnyItem::Host(item);
            let out = run_meta_provider(&mut self.split, context, item).expect("splits");
            emitted.extend(frames_of(out));
        }

        emitted
    }
}

/// Passes one frame through `chain`, returning the frames it emits.
fn process(chain: &mut Chain, context: &mut FrameContext, frame: HostFrame) -> Vec<HostFrame> {
    let slot = HostStreamItem::Frame(frame);
    let inputs = Inputs::<HostSide>::from_slots(vec![Some(slot)]);

    let scored = chain.score.process(context, inputs).expect("processes");
    chain.split_all(context, scored)
}

fn finish(chain: &mut Chain, context: &mut FrameContext) -> Vec<HostFrame> {
    let scored = chain.score.end_of_stream(context).expect("finishes");
    let mut emitted = chain.split_all(context, scored);

    let rest = end_meta_provider(&mut chain.split, context).expect("finishes");
    emitted.extend(frames_of(rest));

    emitted
}

fn frames_of(items: Vec<AnyItem>) -> Vec<HostFrame> {
    items
        .into_iter()
        .map(|item| match item {
            AnyItem::Host(HostStreamItem::Frame(frame)) => frame,
            other => panic!("the scene providers only emit host frames here, not {other:?}"),
        })
        .collect()
}

/// Runs `frames` through providers serving `requests`, returning everything they emit.
fn serve(
    context: &mut FrameContext,
    format: &FrameFormat,
    requests: &[SceneRequest],
    frames: Vec<HostFrame>,
) -> Vec<HostFrame> {
    let mut chain = Chain::new(requests, format);

    let mut emitted = Vec::new();
    for frame in frames {
        let out = process(&mut chain, context, frame);
        emitted.extend(out);
    }

    let rest = finish(&mut chain, context);
    emitted.extend(rest);

    emitted
}

fn chunks_in(frames: &[HostFrame], layer: u32) -> Vec<u64> {
    frames
        .iter()
        .map(|frame| {
            frame
                .meta()
                .provided
                .get::<SceneChunk>(LayerId(layer))
                .map_or(0, |scene| scene.0)
        })
        .collect()
}

/// The frames at which `chunks` moves to a new chunk, the first frame included.
fn breaks_of(chunks: &[u64]) -> Vec<usize> {
    let mut breaks = Vec::new();
    for (position, chunk) in chunks.iter().enumerate() {
        let starts = position == 0 || chunks[position - 1] != *chunk;
        if starts {
            breaks.push(position);
        }
    }

    breaks
}
