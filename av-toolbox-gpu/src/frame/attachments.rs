use std::hash::{Hash, Hasher};
use std::mem;
use std::sync::Arc;

/// A value a stage attaches to a frame.
///
/// A float compares and hashes by its bits, so a NaN equals itself and metadata holding one can
/// still be compared.
#[derive(Debug, Clone)]
pub enum MetaValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
}

impl PartialEq for MetaValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Bool(left), Self::Bool(right)) => left == right,
            (Self::Int(left), Self::Int(right)) => left == right,
            (Self::Float(left), Self::Float(right)) => left.to_bits() == right.to_bits(),
            (Self::String(left), Self::String(right)) => left == right,
            _ => false,
        }
    }
}

impl Eq for MetaValue {}

impl Hash for MetaValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        mem::discriminant(self).hash(state);

        match self {
            Self::Bool(value) => value.hash(state),
            Self::Int(value) => value.hash(state),
            Self::Float(value) => value.to_bits().hash(state),
            Self::String(value) => value.hash(state),
        }
    }
}

/// One stage's key-value pairs, in the order they were attached.
type StageValues = Vec<(String, MetaValue)>;

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
struct Stored {
    stages: Vec<(Arc<str>, StageValues)>,
    pending: StageValues,
}

/// Values stages attached to a frame, kept apart per stage.
///
/// A stage attaches to a pending set, and the runtime files that set under the stage's path once
/// the stage returns. A clone shares the values until either side writes.
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct Attachments(Option<Arc<Stored>>);

impl Attachments {
    pub const fn new() -> Self {
        Self(None)
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_none()
    }

    /// Attaches `value` under `key`, replacing a pending value with the same key.
    pub fn attach(&mut self, key: impl Into<String>, value: MetaValue) {
        let key = key.into();
        debug_assert!(!key.is_empty(), "an attached value needs a key");

        let shared = self.0.get_or_insert_with(Arc::default);
        let stored = Arc::make_mut(shared);
        upsert(&mut stored.pending, key, value);
    }

    /// Files every pending value under `path`.
    ///
    /// This is for the runtime, which knows the path of the stage that just ran.
    pub fn stamp(&mut self, path: &str) {
        let Some(shared) = self.0.as_mut() else {
            return;
        };

        if shared.pending.is_empty() {
            return;
        }

        let stored = Arc::make_mut(shared);
        let pending = mem::take(&mut stored.pending);
        let existing = stored.stages.iter_mut().find(|(stage, _)| &**stage == path);

        match existing {
            Some((_, values)) => {
                for (key, value) in pending {
                    upsert(values, key, value);
                }
            },
            None => stored.stages.push((Arc::from(path), pending)),
        }
    }

    /// Each stage's values, in the order the stages were stamped.
    pub fn stages(&self) -> impl Iterator<Item = (&str, &[(String, MetaValue)])> {
        let stages = self.0.iter().flat_map(|stored| stored.stages.iter());

        stages.map(|(path, values)| (&**path, values.as_slice()))
    }
}

fn upsert(values: &mut Vec<(String, MetaValue)>, key: String, value: MetaValue) {
    let existing = values.iter_mut().find(|(existing_key, _)| *existing_key == key);

    match existing {
        Some((_, slot)) => *slot = value,
        None => values.push((key, value)),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    use super::*;

    fn hash_of(value: &MetaValue) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);

        hasher.finish()
    }

    fn collected(attachments: &Attachments) -> Vec<(String, Vec<(String, MetaValue)>)> {
        attachments
            .stages()
            .map(|(path, values)| (path.to_string(), values.to_vec()))
            .collect()
    }

    #[test]
    fn new_attachments_are_empty() {
        let attachments = Attachments::new();

        assert!(attachments.is_empty());
        assert_eq!(attachments.stages().count(), 0);
    }

    #[test]
    fn stamping_files_pending_values_under_the_path_in_order() {
        let mut attachments = Attachments::new();
        attachments.attach("first", MetaValue::Int(1));
        attachments.attach("second", MetaValue::Bool(true));

        attachments.stamp("filter.convert_colour");

        let expected = vec![(
            "filter.convert_colour".to_string(),
            vec![
                ("first".to_string(), MetaValue::Int(1)),
                ("second".to_string(), MetaValue::Bool(true)),
            ],
        )];
        assert_eq!(collected(&attachments), expected);
    }

    #[test]
    fn attaching_a_key_twice_keeps_the_last_value() {
        let mut attachments = Attachments::new();
        attachments.attach("score", MetaValue::Int(1));
        attachments.attach("score", MetaValue::Int(2));

        attachments.stamp("filter.a");

        let expected = vec![(
            "filter.a".to_string(),
            vec![("score".to_string(), MetaValue::Int(2))],
        )];
        assert_eq!(collected(&attachments), expected);
    }

    #[test]
    fn stamping_with_nothing_pending_changes_nothing() {
        let mut attachments = Attachments::new();
        attachments.attach("score", MetaValue::Int(1));
        attachments.stamp("filter.a");
        let before = attachments.clone();

        attachments.stamp("transform.b");

        assert_eq!(attachments, before);
        assert_eq!(attachments.stages().count(), 1);
    }

    #[test]
    fn stamping_the_same_path_twice_extends_its_entry() {
        let mut attachments = Attachments::new();
        attachments.attach("first", MetaValue::Int(1));
        attachments.stamp("filter.a");
        attachments.attach("second", MetaValue::Int(2));
        attachments.attach("first", MetaValue::Int(3));

        attachments.stamp("filter.a");

        let expected = vec![(
            "filter.a".to_string(),
            vec![
                ("first".to_string(), MetaValue::Int(3)),
                ("second".to_string(), MetaValue::Int(2)),
            ],
        )];
        assert_eq!(collected(&attachments), expected);
    }

    #[test]
    fn a_clone_is_unchanged_by_writes_to_the_original() {
        let mut original = Attachments::new();
        original.attach("shared", MetaValue::Int(1));
        original.stamp("filter.a");
        let copy = original.clone();

        original.attach("later", MetaValue::Int(2));
        original.stamp("filter.b");

        assert_eq!(copy.stages().count(), 1);
        assert_eq!(original.stages().count(), 2);
    }

    #[test]
    fn floats_compare_and_hash_by_their_bits() {
        let nan = MetaValue::Float(f64::NAN);
        let positive_zero = MetaValue::Float(0.0);
        let negative_zero = MetaValue::Float(-0.0);

        assert_eq!(nan, nan.clone());
        assert_eq!(hash_of(&nan), hash_of(&nan.clone()));
        assert_ne!(positive_zero, negative_zero);
    }

    #[test]
    fn values_of_different_kinds_are_not_equal() {
        assert_ne!(MetaValue::Int(1), MetaValue::Float(1.0));
        assert_ne!(MetaValue::Bool(true), MetaValue::Int(1));
    }
}
