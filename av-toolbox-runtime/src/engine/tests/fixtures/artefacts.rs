use std::sync::Mutex;

use av_toolbox_core::{
    Artefact,
    ArtefactData,
    ArtefactDecl,
    ArtefactPiece,
    ArtefactSink,
    ArtefactSinkDefinition,
    BoundInputs,
    Filter,
    FilterDefinition,
    InputParam,
    InputSignature,
    Inputs,
    KeyFilter,
    MergedArtefact,
    Parallelism,
    Pattern,
    StageDefinition,
    StageError,
    StageInstance,
    StreamType,
    WantedArtefacts,
    merge_single,
};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    DeviceSide,
    FormatSet,
    Frame,
    FrameContext,
    Offered,
    OutputFormats,
    ResolvedFormats,
    Runtime,
    StreamFormat,
    StreamTemplate,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use super::chunked;

/// Whether each `test/artefacting` instance was told its `frames` artefact is wanted, one entry per instance.
pub(crate) static WANTED: Mutex<Vec<bool>> = Mutex::new(Vec::new());

/// Every artefact `test/collecting` was handed, as its key and its data read as JSON.
pub(crate) static COLLECTED: Mutex<Vec<(String, serde_json::Value)>> = Mutex::new(Vec::new());

/// What a `test/artefacting` instance hands over at the end of each stream.
#[derive(Debug, Default, Copy, Clone, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ArtefactingMode {
    /// A `frames` piece counting the frames it saw.
    #[default]
    Counting,
    /// An artefact it never declared.
    Undeclared,
    /// `frames` as opaque bytes, though it is declared structured.
    WrongKind,
    /// A `broken` piece, whose merge always fails.
    Broken,
    /// A `bytes` piece of opaque data.
    Opaque,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub(crate) struct ArtefactingArgs {
    #[serde(default)]
    mode: ArtefactingMode,
    /// Declares the stage per chunk rather than for the whole stream.
    #[serde(default)]
    per_chunk: bool,
}

pub(crate) struct ArtefactingInstance {
    mode: ArtefactingMode,
    frames: u64,
}

impl ArtefactingInstance {
    pub(crate) fn from_args(args: ArtefactingArgs) -> Self {
        Self {
            mode: args.mode,
            frames: 0,
        }
    }
}

impl StageInstance for ArtefactingInstance {
    fn wants_artefacts(&mut self, wanted: &WantedArtefacts) {
        let mut seen = WANTED.lock().expect("no test panics holding WANTED");
        seen.push(wanted.contains("frames"));
    }

    fn take_artefacts(&mut self, _context: &mut FrameContext) -> Result<Vec<Artefact>, StageError> {
        let frames = std::mem::take(&mut self.frames);

        let artefact = match self.mode {
            ArtefactingMode::Counting => Artefact {
                name: "frames",
                data: ArtefactData::Structured(json!(frames)),
            },
            ArtefactingMode::Undeclared => Artefact {
                name: "surprise",
                data: ArtefactData::Structured(json!(frames)),
            },
            ArtefactingMode::WrongKind => Artefact {
                name: "frames",
                data: ArtefactData::Opaque(vec![1, 2, 3]),
            },
            ArtefactingMode::Broken => Artefact {
                name: "broken",
                data: ArtefactData::Structured(json!(frames)),
            },
            ArtefactingMode::Opaque => Artefact {
                name: "bytes",
                data: ArtefactData::Opaque(vec![1, 2, 3]),
            },
        };

        Ok(vec![artefact])
    }
}

impl Filter for ArtefactingInstance {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let frame = inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;
        self.frames += 1;

        Ok(vec![frame])
    }
}

/// A pass through filter handing over how many frames it saw at the end of every stream.
pub(crate) struct Artefacting;

impl StageDefinition for Artefacting {
    type Args = ArtefactingArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "artefacting"
    }

    fn parallelism(args: &Self::Args) -> Parallelism {
        match args.per_chunk {
            true => chunked(),
            false => Parallelism::Stream,
        }
    }

    fn artefacts() -> &'static [ArtefactDecl] {
        const DECLARED: &[ArtefactDecl] = &[
            ArtefactDecl::structured("frames"),
            ArtefactDecl::structured("broken"),
            ArtefactDecl::opaque("bytes", "bin"),
        ];
        DECLARED
    }

    /// Lists each chunk's count of `frames` as `[chunk, frames]`, fails for `broken`, and keeps a lone `bytes`.
    fn merge_artefact(name: &'static str, pieces: Vec<ArtefactPiece>) -> Result<ArtefactData, StageError> {
        if name == "broken" {
            return Err(StageError::msg("the broken artefact never merges"));
        }

        if name == "bytes" {
            return merge_single(name, pieces);
        }

        let listed: Vec<serde_json::Value> = pieces.into_iter().map(listed_piece).collect();

        Ok(ArtefactData::Structured(serde_json::Value::Array(listed)))
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

fn listed_piece(piece: ArtefactPiece) -> serde_json::Value {
    let ArtefactData::Structured(frames) = piece.data else {
        return serde_json::Value::Null;
    };

    json!([piece.chunk, frames])
}

impl FilterDefinition for Artefacting {
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
        unreachable!("the engine tests build fixtures directly")
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct CollectingArgs {
    include: Option<Vec<Pattern>>,
}

struct CollectingSink;

impl ArtefactSink for CollectingSink {
    fn write(&mut self, artefacts: &[&MergedArtefact]) -> Result<usize, StageError> {
        let mut collected = COLLECTED.lock().expect("no test panics holding COLLECTED");

        for artefact in artefacts {
            let ArtefactData::Structured(value) = &artefact.data else {
                return Err(StageError::msg(
                    "the collecting sink only takes structured artefacts",
                ));
            };

            collected.push((artefact.key(), value.clone()));
        }

        Ok(artefacts.len())
    }
}

/// An artefact sink recording everything it was handed into [COLLECTED].
pub(crate) struct Collecting;

impl StageDefinition for Collecting {
    type Args = CollectingArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "collecting"
    }

    fn inputs() -> &'static InputSignature {
        &InputSignature::EMPTY
    }
}

impl ArtefactSinkDefinition for Collecting {
    fn selection(args: &Self::Args) -> KeyFilter {
        let include = args.include.clone();

        KeyFilter::new(include, None)
    }

    fn build(_args: Self::Args) -> Result<Box<dyn ArtefactSink>, StageError> {
        Ok(Box::new(CollectingSink))
    }
}
