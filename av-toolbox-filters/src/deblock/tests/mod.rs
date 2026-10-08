mod dpir;
mod h264;
mod pp7;
mod qed;

use av_toolbox_core::{InputArgument, PipelineRegistry, StageHandle, StageKind, StreamType};

/// Every message in the chain the pipeline refuses `stage` and `value` with, one per line.
fn prepared_refusal(stage: &str, value: serde_json::Value) -> String {
    let mut registry = PipelineRegistry::new();
    crate::register(&mut registry).expect("the filters register");

    let handle = StageHandle::new(1);
    let arguments = [InputArgument::positional(handle, StreamType::Frames)];
    let prepared = registry.prepare(StageKind::Filter, stage, &value, &arguments);
    let Err(error) = prepared else {
        panic!("{stage} should refuse {value}");
    };

    let mut lines = vec![error.to_string()];
    let mut cause = std::error::Error::source(&error);
    while let Some(inner) = cause {
        lines.push(inner.to_string());
        cause = inner.source();
    }

    lines.join("\n")
}
