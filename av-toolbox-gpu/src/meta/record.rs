use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use super::Attachments;

/// The next id to hand a record, shared by every thread so no two records ever share one.
static NEXT_RECORD: AtomicU64 = AtomicU64::new(0);

/// What tells two records apart, unique within a process.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct RecordId(u64);

/// A metadata-only item a stage emits into the stream.
///
/// It holds no samples, only a kind its stage names and the values its stage attached. A clone keeps its id, so
/// the same record reaching a stage by two routes can be recognised.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct MetaRecord {
    kind: Arc<str>,
    id: RecordId,
    attachments: Attachments,
}

impl MetaRecord {
    pub fn new(kind: &str) -> Self {
        let next = NEXT_RECORD.fetch_add(1, Ordering::Relaxed);

        Self {
            kind: Arc::from(kind),
            id: RecordId(next),
            attachments: Attachments::new(),
        }
    }

    pub fn kind(&self) -> &str {
        &self.kind
    }

    pub fn id(&self) -> RecordId {
        self.id
    }

    pub fn attachments(&self) -> &Attachments {
        &self.attachments
    }

    pub fn attachments_mut(&mut self) -> &mut Attachments {
        &mut self.attachments
    }

    /// Checks a kind is non-empty, uses only lowercase letters, digits and underscores, and is not `frame`.
    pub fn check_kind(kind: &str) -> Result<(), String> {
        if kind == "frame" {
            return Err(String::from("the kind frame is reserved for frames"));
        }

        let well_formed = !kind.is_empty() && kind.chars().all(is_kind_character);
        if well_formed {
            return Ok(());
        }

        let message = format!("a record's kind {kind:?} must be non-empty and use only a-z, 0-9 and _");

        Err(message)
    }
}

fn is_kind_character(character: char) -> bool {
    character.is_ascii_lowercase() || character.is_ascii_digit() || character == '_'
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MetaValue;

    #[test]
    fn a_record_files_its_values_under_a_path() {
        let mut record = MetaRecord::new("chunk");
        record.attachments_mut().attach("frames", MetaValue::Int(24));

        record.attachments_mut().stamp("filter.stats");
        let stages: Vec<&str> = record.attachments().stages().map(|(path, _)| path).collect();

        assert_eq!(record.kind(), "chunk");
        assert_eq!(stages, vec!["filter.stats"]);
    }

    #[test]
    fn two_records_get_different_ids() {
        let first = MetaRecord::new("chunk");
        let second = MetaRecord::new("chunk");

        assert_ne!(first.id(), second.id());
    }

    #[test]
    fn a_clone_keeps_its_id() {
        let record = MetaRecord::new("gop");

        let cloned = record.clone();

        assert_eq!(cloned.id(), record.id());
    }

    #[test]
    fn a_kind_of_lowercase_letters_digits_and_underscores_is_accepted() {
        assert!(MetaRecord::check_kind("chunk_2").is_ok());
    }

    #[test]
    fn the_frame_kind_is_reserved() {
        let refused = MetaRecord::check_kind("frame").expect_err("frame is reserved");

        assert!(refused.contains("frame"));
    }

    #[test]
    fn an_empty_kind_is_refused() {
        assert!(MetaRecord::check_kind("").is_err());
    }

    #[test]
    fn a_malformed_kind_is_refused_as_a_record_kind() {
        let refused = MetaRecord::check_kind("Chunk").expect_err("malformed");

        assert!(refused.starts_with("a record's kind \"Chunk\""), "{refused}");
    }

    #[test]
    fn a_kind_with_other_characters_is_refused() {
        assert!(MetaRecord::check_kind("Chunk").is_err());
        assert!(MetaRecord::check_kind("chunk.end").is_err());
        assert!(MetaRecord::check_kind("chunk-end").is_err());
    }
}
