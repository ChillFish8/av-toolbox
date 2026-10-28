use std::collections::{HashMap, HashSet};

use av_toolbox_graph::{Diagnostic, DiagnosticCode, Diagnostics, Expr, Piece, StageSpec};

use super::aliases::follow;

/// Points every stage the deferred args of `spec` read at the stage its name stands for.
///
/// Names reading a call's input or output are followed like stage inputs are, and each output
/// passed on the way is added to `read_outputs`. A name that loops is reported.
pub(super) fn follow_references(
    spec: &mut StageSpec,
    aliases: &HashMap<String, String>,
    outputs: &HashSet<String>,
    read_outputs: &mut HashSet<String>,
    diagnostics: &mut Diagnostics,
) {
    for deferred in &mut spec.deferred {
        for piece in deferred.template.pieces_mut() {
            let Piece::Expr(found) = piece else {
                continue;
            };

            let mut visit = |expr: &mut Expr| follow_field(expr, aliases, outputs, read_outputs);
            let followed = found.expr.rewrite(&mut visit);
            if let Err(message) = followed {
                let diagnostic = Diagnostic::error(DiagnosticCode::Cycle, "", deferred.span, message);
                diagnostics.push(diagnostic);
            }
        }
    }
}

fn follow_field(
    expr: &mut Expr,
    aliases: &HashMap<String, String>,
    outputs: &HashSet<String>,
    read_outputs: &mut HashSet<String>,
) -> Result<(), String> {
    let Expr::Field { stage, .. } = expr else {
        return Ok(());
    };

    let followed = follow(aliases, outputs, stage, read_outputs);
    let Some(target) = followed else {
        let message = format!("{stage} loops back to itself through calls");
        return Err(message);
    };

    *stage = target;

    Ok(())
}
