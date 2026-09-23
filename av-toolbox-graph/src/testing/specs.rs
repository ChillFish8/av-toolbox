use av_toolbox_core::StageKind;
use serde_json::json;

use crate::{InputRef, Span, Spanned, StageSpec};

/// Builds a spec with no inputs, empty arguments and unknown spans.
///
/// Every field is public, so a test that cares about a span or an argument sets it directly.
pub(crate) fn spec(kind: StageKind, name: &str, stage_type: &str) -> StageSpec {
    StageSpec {
        name: Spanned::unknown(name.to_string()),
        kind,
        stage_type: Spanned::unknown(stage_type.to_string()),
        inputs: Vec::new(),
        args: Spanned::unknown(json!({})),
        workers: None,
        span: Span::unknown(),
    }
}

pub(crate) fn positional(target: &str) -> Spanned<InputRef> {
    Spanned::unknown(InputRef {
        name: None,
        target: target.to_string(),
    })
}

pub(crate) fn named(name: &str, target: &str) -> Spanned<InputRef> {
    Spanned::unknown(InputRef {
        name: Some(name.to_string()),
        target: target.to_string(),
    })
}
