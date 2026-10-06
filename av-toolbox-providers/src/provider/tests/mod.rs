mod layers;
mod provider;

use av_toolbox_core::{AnyDemand, Inputs, LayerDemand, Provider};
use av_toolbox_gpu::{FrameContext, FrameFormat, HostFrame, HostSide, HostStreamItem, LayerId};

use crate::provider::{SceneBreaks, SceneChunk, SceneRequest, Scenes, ServeScenes};

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

/// Passes one frame through `provider`, returning the frames it emits.
fn process(provider: &mut ServeScenes, context: &mut FrameContext, frame: HostFrame) -> Vec<HostFrame> {
    let slot = HostStreamItem::Frame(frame);
    let inputs = Inputs::<HostSide>::from_slots(vec![Some(slot)]);

    let emitted = provider.process(context, inputs).expect("processes");
    frames_of(emitted)
}

fn finish(provider: &mut ServeScenes, context: &mut FrameContext) -> Vec<HostFrame> {
    let emitted = provider.end_of_stream(context).expect("finishes");

    frames_of(emitted)
}

fn frames_of(items: Vec<HostStreamItem>) -> Vec<HostFrame> {
    items
        .into_iter()
        .map(|item| match item {
            HostStreamItem::Frame(frame) => frame,
            HostStreamItem::Mask(_) => panic!("the scene provider only emits frames"),
        })
        .collect()
}

/// Runs `frames` through a provider serving `requests`, returning everything it emits.
fn serve(
    context: &mut FrameContext,
    format: &FrameFormat,
    requests: &[SceneRequest],
    frames: Vec<HostFrame>,
) -> Vec<HostFrame> {
    let args = demands(requests);
    let mut provider = ServeScenes::new(&args, format).expect("builds");

    let mut emitted = Vec::new();
    for frame in frames {
        let out = process(&mut provider, context, frame);
        emitted.extend(out);
    }

    let rest = finish(&mut provider, context);
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
