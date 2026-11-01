mod supplied;
mod values;

use av_toolbox_graph::{Span, Spanned};
use serde_json::Value;

use super::{Declaration, VariableType};

fn declaration(variable_type: VariableType) -> Declaration {
    let name = String::from("setting");
    let header = Span::new(0, 10);

    Declaration {
        name: Spanned::new(name, header),
        variable_type,
        description: None,
        default: None,
        min: None,
        max: None,
        choices: None,
        optional: false,
    }
}

fn spanned(value: Value) -> Option<Spanned<Value>> {
    let span = Span::new(1, 2);
    Some(Spanned::new(value, span))
}
