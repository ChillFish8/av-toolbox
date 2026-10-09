use av_toolbox_gpu::{AnyItem, MetaRecord};

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
