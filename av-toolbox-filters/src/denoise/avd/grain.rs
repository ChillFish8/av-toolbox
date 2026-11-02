use std::fmt::Display;

use av_denoise_core::{GrainChunk, SceneGrain, build_table};
use av_toolbox_core::{Artefact, ArtefactData, ArtefactDecl, ArtefactPiece, StageError};
use serde::{Deserialize, Serialize};

use super::args::{AvdArgs, Nl4dArgs};

/// The name `av-denoise/nl4d` hands its grain table over under.
pub(crate) const GRAIN_TABLE: &str = "grain_table";

/// An AV1 film grain table of the grain NL4D removed from luma.
const GRAIN_TABLE_DECL: ArtefactDecl = ArtefactDecl::opaque(
    GRAIN_TABLE,
    "tbl",
    "An AV1 film grain table of the grain removed from luma, for an encoder to add back. Its entries follow the \
     frame numbers and frame rate this stage sees, so a later stage that drops frames or changes the rate shifts \
     the table off the encoded video.",
);

/// The artefacts `av-denoise/nl4d` declares.
pub(crate) const ARTEFACTS: &[ArtefactDecl] = &[GRAIN_TABLE_DECL];

/// A grain table with no entries, which encoders read as no grain.
const EMPTY_TABLE: &str = "filmgrn1\n";

const STAGE: &str = Nl4dArgs::STAGE;

fn unreadable_piece(error: impl Display) -> StageError {
    let message = format!("{STAGE} could not read a piece of {GRAIN_TABLE}: {error}");

    StageError::msg(message)
}

fn unwritable_piece(error: serde_json::Error) -> StageError {
    let message = format!("{STAGE} could not write a piece of {GRAIN_TABLE}: {error}");

    StageError::msg(message)
}

fn mixed_rates(first: (u64, u64), second: (u64, u64)) -> StageError {
    let message = format!(
        "{STAGE} was handed {GRAIN_TABLE} pieces at {}/{} and {}/{} frames per second",
        first.0, first.1, second.0, second.1
    );

    StageError::msg(message)
}

fn non_finite_statistics() -> StageError {
    let message = format!("{STAGE} measured grain statistics that are not finite numbers");

    StageError::msg(message)
}

fn no_pieces() -> StageError {
    let message = format!("{STAGE} was handed no pieces of {GRAIN_TABLE} to merge");

    StageError::msg(message)
}

/// The grain statistics of one chunk, as a piece holds them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct ChunkPiece {
    frames: u32,
    source_hist: Vec<u32>,
    kept_hist: Vec<u32>,
    autocov: Vec<f64>,
    pixels: Vec<f64>,
}

impl From<&GrainChunk> for ChunkPiece {
    fn from(chunk: &GrainChunk) -> Self {
        Self {
            frames: chunk.frames,
            source_hist: chunk.source_hist.clone(),
            kept_hist: chunk.kept_hist.clone(),
            autocov: chunk.autocov.clone(),
            pixels: chunk.pixels.clone(),
        }
    }
}

impl From<ChunkPiece> for GrainChunk {
    fn from(piece: ChunkPiece) -> Self {
        Self {
            frames: piece.frames,
            source_hist: piece.source_hist,
            kept_hist: piece.kept_hist,
            autocov: piece.autocov,
            pixels: piece.pixels,
        }
    }
}

/// One scene's chunks, as a piece holds them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct ScenePiece {
    first_frame: u64,
    chunks: Vec<ChunkPiece>,
}

impl From<&SceneGrain> for ScenePiece {
    fn from(scene: &SceneGrain) -> Self {
        let chunks = scene.chunks.iter().map(ChunkPiece::from).collect();

        Self {
            first_frame: scene.first_frame,
            chunks,
        }
    }
}

impl From<ScenePiece> for SceneGrain {
    fn from(piece: ScenePiece) -> Self {
        let chunks = piece.chunks.into_iter().map(GrainChunk::from).collect();

        Self {
            first_frame: piece.first_frame,
            chunks,
        }
    }
}

impl ChunkPiece {
    fn is_finite(&self) -> bool {
        let autocov_finite = self.autocov.iter().all(|value| value.is_finite());
        let pixels_finite = self.pixels.iter().all(|value| value.is_finite());

        autocov_finite && pixels_finite
    }
}

/// What one NL4D stream hands over towards the grain table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct GrainPiece {
    /// Frames per second as `(numerator, denominator)`.
    frame_rate: (u64, u64),
    scenes: Vec<ScenePiece>,
}

impl GrainPiece {
    pub(crate) fn new(frame_rate: (u64, u64), scenes: &[SceneGrain]) -> Self {
        let scenes = scenes.iter().map(ScenePiece::from).collect();

        Self { frame_rate, scenes }
    }

    /// Reads a piece handed over as structured data.
    pub(crate) fn from_piece(piece: ArtefactPiece) -> Result<Self, StageError> {
        let ArtefactData::Structured(value) = piece.data else {
            return Err(unreadable_piece("it holds opaque data"));
        };

        serde_json::from_value(value).map_err(unreadable_piece)
    }

    fn is_finite(&self) -> bool {
        self.scenes
            .iter()
            .flat_map(|scene| &scene.chunks)
            .all(ChunkPiece::is_finite)
    }

    pub(crate) fn frame_rate(&self) -> (u64, u64) {
        self.frame_rate
    }

    pub(crate) fn into_scenes(self) -> Vec<SceneGrain> {
        self.scenes.into_iter().map(SceneGrain::from).collect()
    }

    pub(crate) fn to_artefact(&self) -> Result<Artefact, StageError> {
        if !self.is_finite() {
            return Err(non_finite_statistics());
        }

        let value = serde_json::to_value(self).map_err(unwritable_piece)?;

        Ok(Artefact {
            name: GRAIN_TABLE,
            data: ArtefactData::Structured(value),
        })
    }
}

/// Every piece's scenes and the frame rate they share.
pub(crate) fn joined_scenes(pieces: Vec<ArtefactPiece>) -> Result<((u64, u64), Vec<SceneGrain>), StageError> {
    let mut frame_rate = None;
    let mut scenes = Vec::new();

    for piece in pieces {
        let decoded = GrainPiece::from_piece(piece)?;
        let piece_rate = decoded.frame_rate();

        match frame_rate {
            Some(rate) if rate != piece_rate => return Err(mixed_rates(rate, piece_rate)),
            _ => frame_rate = Some(piece_rate),
        }

        scenes.extend(decoded.into_scenes());
    }

    let Some(frame_rate) = frame_rate else {
        return Err(no_pieces());
    };

    Ok((frame_rate, scenes))
}

/// Fits every scene in `pieces` into one grain table.
pub(crate) fn merge_grain_table(pieces: Vec<ArtefactPiece>) -> Result<ArtefactData, StageError> {
    let (frame_rate, scenes) = joined_scenes(pieces)?;
    let table = build_table(&scenes, frame_rate);

    if table == EMPTY_TABLE {
        tracing::warn!(
            stage = STAGE,
            "no grain could be measured, so {GRAIN_TABLE} has no entries"
        );
    }

    Ok(ArtefactData::Opaque(table.into_bytes()))
}

/// The grain one NL4D instance measured since it last handed a piece over.
pub(crate) struct GrainLog {
    /// Frames per second as `(numerator, denominator)`.
    frame_rate: (u64, u64),
    /// The index of the stream's first frame, once it has one.
    first_frame: Option<u64>,
    scenes: Vec<SceneGrain>,
}

impl GrainLog {
    pub(crate) fn new(frame_rate: (u64, u64)) -> Self {
        Self {
            frame_rate,
            first_frame: None,
            scenes: Vec::new(),
        }
    }

    /// Forgets the last stream's first frame, ready for the next stream.
    pub(crate) fn restart(&mut self) {
        self.first_frame = None;
    }

    /// Notes the frame at `index` arriving in the stream under way.
    pub(crate) fn saw(&mut self, index: u64) {
        self.first_frame.get_or_insert(index);
    }

    /// Keeps the chunks measured over the stream that just ended as one scene, unless there are none.
    pub(crate) fn keep(&mut self, chunks: Vec<GrainChunk>) {
        let Some(first_frame) = self.first_frame else {
            return;
        };

        if chunks.is_empty() {
            return;
        }

        self.scenes.push(SceneGrain { first_frame, chunks });
    }

    /// Hands over every scene kept since the last call as one piece.
    pub(crate) fn take_piece(&mut self) -> Result<Artefact, StageError> {
        let scenes = std::mem::take(&mut self.scenes);
        let piece = GrainPiece::new(self.frame_rate, &scenes);

        piece.to_artefact()
    }
}
