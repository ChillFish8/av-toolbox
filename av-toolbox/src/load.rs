use av_toolbox_core::PipelineRegistry;
use codespan_reporting::term::termcolor::WriteColor;

use crate::report;

/// Every built-in stage, or `None` after saying why they could not be registered.
pub(crate) fn registry(stderr: &mut dyn WriteColor) -> Option<PipelineRegistry> {
    let registered = av_toolbox_pipeline::registry();

    match registered {
        Ok(registry) => Some(registry),
        Err(error) => {
            let message = format!("the built-in stages could not be registered: {error}");
            report::error_line(stderr, &message);

            None
        },
    }
}
