use std::collections::HashSet;

use av_toolbox_gpu::{AnyItem, MetaRecord, RecordId};

/// One thing moving through the stream, either an item with samples or a metadata-only record.
#[derive(Debug, Clone)]
pub enum Flow {
    Item(AnyItem),
    Record(MetaRecord),
}

impl Flow {
    pub fn into_item(self) -> Option<AnyItem> {
        match self {
            Self::Item(item) => Some(item),
            Self::Record(_) => None,
        }
    }

    pub fn item(&self) -> Option<&AnyItem> {
        match self {
            Self::Item(item) => Some(item),
            Self::Record(_) => None,
        }
    }

    pub fn is_record(&self) -> bool {
        matches!(self, Self::Record(_))
    }
}

/// The records already seen at a place a record can reach by more than one route.
///
/// It only remembers records when there are several routes in, since one route never carries a record twice.
pub(crate) struct Sightings {
    seen: Option<HashSet<RecordId>>,
}

impl Sightings {
    pub(crate) fn for_routes(routes: usize) -> Self {
        let seen = (routes > 1).then(HashSet::new);

        Self { seen }
    }

    /// Whether `record` is seen here for the first time, which is always so with one route in.
    pub(crate) fn first(&mut self, record: &MetaRecord) -> bool {
        match &mut self.seen {
            Some(seen) => seen.insert(record.id()),
            None => true,
        }
    }

    pub(crate) fn forget(&mut self) {
        if let Some(seen) = &mut self.seen {
            seen.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use av_toolbox_gpu::MetaRecord;

    use super::*;

    #[test]
    fn several_routes_drop_a_repeated_record() {
        let mut sightings = Sightings::for_routes(2);
        let record = MetaRecord::new("mark");

        let first = sightings.first(&record);
        let again = sightings.first(&record);

        assert!(first);
        assert!(!again);
    }

    #[test]
    fn one_route_remembers_no_record() {
        let mut sightings = Sightings::for_routes(1);
        let record = MetaRecord::new("mark");

        sightings.first(&record);
        let again = sightings.first(&record);

        assert!(again);
    }

    #[test]
    fn a_forgotten_record_is_a_first_sighting_again() {
        let mut sightings = Sightings::for_routes(2);
        let record = MetaRecord::new("mark");

        sightings.first(&record);
        sightings.forget();
        let again = sightings.first(&record);

        assert!(again);
    }
}
