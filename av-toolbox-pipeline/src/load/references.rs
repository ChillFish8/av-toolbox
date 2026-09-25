use std::collections::HashMap;

use av_toolbox_graph::{Diagnostic, DiagnosticCode, Span, suggest_name};

use super::expand::Loader;
use crate::toml::RawFile;

impl Loader<'_> {
    /// Reports each call input and output that reads a name this file does not define.
    ///
    /// Names reading a call's output, such as `denoise.clean`, are skipped.
    pub(super) fn check_targets(&mut self, raw: &RawFile, outputs_of: &HashMap<String, Option<Vec<String>>>) {
        let stages = raw.stages.iter().map(|stage| stage.name.value.clone());
        let inputs = raw.inputs.iter().map(|input| input.name.value.clone());
        let calls = raw.calls.iter().map(|call| call.name.value.clone());
        let call_outputs = outputs_of.iter().flat_map(|(call_name, outputs)| {
            let outputs = outputs.iter().flatten();
            outputs.map(move |output| format!("{call_name}.{output}"))
        });
        let mut known: Vec<String> = stages.chain(inputs).chain(calls).chain(call_outputs).collect();
        known.sort();

        for call in &raw.calls {
            let path = format!("call.{}.inputs", call.name.value);
            for input in &call.inputs {
                self.check_target(&input.value.target, input.span, &path, &known, outputs_of);
            }
        }

        for output in &raw.outputs {
            let path = format!("output.{}.inputs", output.name.value);
            self.check_target(
                &output.source.value,
                output.source.span,
                &path,
                &known,
                outputs_of,
            );
        }
    }

    fn check_target(
        &mut self,
        target: &str,
        span: Span,
        path: &str,
        known: &[String],
        outputs_of: &HashMap<String, Option<Vec<String>>>,
    ) {
        if known.iter().any(|name| name == target) {
            return;
        }

        let call_name = target.split_once('.').map(|(call_name, _)| call_name);
        if call_name.is_some_and(|call_name| outputs_of.contains_key(call_name)) {
            return;
        }

        let message = format!("no stage named {target} is defined in this file");
        let mut diagnostic = Diagnostic::error(DiagnosticCode::UnknownStageReference, path, span, message);

        let names = known.iter().map(String::as_str);
        let closest = suggest_name(target, names);
        if let Some(closest) = closest {
            let note = format!("did you mean `{closest}`?");
            diagnostic = diagnostic.with_note(note, Span::unknown());
        }

        self.diagnostics.push(diagnostic);
    }
}
