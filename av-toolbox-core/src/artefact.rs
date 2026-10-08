use std::fmt;

use crate::{StageDefinition, StageError};

/// What an artefact holds.
#[derive(Debug, Clone, PartialEq)]
pub enum ArtefactData {
    /// Records written as JSON.
    Structured(serde_json::Value),
    /// Bytes written as they are, such as an image or a film-grain table.
    Opaque(Vec<u8>),
}

impl ArtefactData {
    /// Whether this data has the shape `kind` declares.
    pub fn is_kind(&self, kind: ArtefactKind) -> bool {
        matches!(
            (self, kind),
            (Self::Structured(_), ArtefactKind::Structured) | (Self::Opaque(_), ArtefactKind::Opaque { .. })
        )
    }
}

/// One output a stage hands over at the end of a stream.
#[derive(Debug, Clone, PartialEq)]
pub struct Artefact {
    pub name: &'static str,
    pub data: ArtefactData,
}

/// The shape of data an artefact holds.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ArtefactKind {
    Structured,
    Opaque { extension: &'static str },
}

impl ArtefactKind {
    /// The file extension an artefact of this kind is written with.
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Structured => "json",
            Self::Opaque { extension } => extension,
        }
    }
}

/// An artefact a stage can produce.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct ArtefactDecl {
    pub name: &'static str,
    pub kind: ArtefactKind,
}

impl ArtefactDecl {
    pub const fn structured(name: &'static str) -> Self {
        Self {
            name,
            kind: ArtefactKind::Structured,
        }
    }

    pub const fn opaque(name: &'static str, extension: &'static str) -> Self {
        Self {
            name,
            kind: ArtefactKind::Opaque { extension },
        }
    }
}

/// The part of an artefact one instance produced for one chunk.
///
/// `chunk` is `None` for a stage that is not given one chunk at a time in a region routed by frame.
#[derive(Debug, Clone, PartialEq)]
pub struct ArtefactPiece {
    pub chunk: Option<u64>,
    pub data: ArtefactData,
}

/// A stage's artefact after its pieces were merged.
#[derive(Debug, Clone, PartialEq)]
pub struct MergedArtefact {
    /// The stage's path, such as `filter.grain`.
    pub stage: String,
    pub name: &'static str,
    pub kind: ArtefactKind,
    pub data: ArtefactData,
}

impl MergedArtefact {
    /// The stage path and the name joined by a `.`, such as `filter.grain.table`.
    pub fn key(&self) -> String {
        format!("{}.{}", self.stage, self.name)
    }
}

/// Merges the pieces of one artefact, as [StageDefinition::merge_artefact](crate::StageDefinition::merge_artefact) does.
pub type MergeArtefactFn = fn(&'static str, Vec<ArtefactPiece>) -> Result<ArtefactData, StageError>;

/// The artefacts a stage declares and how its pieces of one are merged.
#[derive(Copy, Clone)]
pub struct StageArtefacts {
    declared: &'static [ArtefactDecl],
    merge: MergeArtefactFn,
}

impl StageArtefacts {
    /// What a stage that declares no artefacts carries, such as a crossing.
    pub const NONE: Self = Self {
        declared: &[],
        merge: merge_single,
    };

    pub fn of<D: StageDefinition>() -> Self {
        Self {
            declared: D::artefacts(),
            merge: D::merge_artefact,
        }
    }

    pub fn declared(&self) -> &'static [ArtefactDecl] {
        self.declared
    }

    pub fn find(&self, name: &str) -> Option<&'static ArtefactDecl> {
        self.declared.iter().find(|decl| decl.name == name)
    }

    pub fn merge(&self, name: &'static str, pieces: Vec<ArtefactPiece>) -> Result<ArtefactData, StageError> {
        (self.merge)(name, pieces)
    }
}

impl fmt::Debug for StageArtefacts {
    /// Shows only the declarations, since a function pointer says nothing useful.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StageArtefacts")
            .field("declared", &self.declared)
            .finish()
    }
}

/// Returns a lone piece unchanged, refusing more than one.
pub fn merge_single(name: &'static str, pieces: Vec<ArtefactPiece>) -> Result<ArtefactData, StageError> {
    let count = pieces.len();
    let mut remaining = pieces.into_iter();
    let first = remaining.next();
    let second = remaining.next();

    let (Some(piece), None) = (first, second) else {
        let message =
            format!("artefact {name} came in {count} pieces, and its stage has no merge for more than one");
        return Err(StageError::msg(message));
    };

    Ok(piece.data)
}

/// The artefacts of one stage that at least one sink will write.
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct WantedArtefacts {
    names: Vec<&'static str>,
}

impl WantedArtefacts {
    pub fn new(names: Vec<&'static str>) -> Self {
        Self { names }
    }

    pub fn contains(&self, name: &str) -> bool {
        self.names.contains(&name)
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

/// A sink handed every artefact it selected once the run has ended.
pub trait ArtefactSink: Send + 'static {
    /// Writes `artefacts`, answering with how many files were written.
    fn write(&mut self, artefacts: &[&MergedArtefact]) -> Result<usize, StageError>;
}
