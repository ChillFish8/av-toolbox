use serde_json::{Value, json};

use crate::{PathNode, PathView};

/// A path through `stages`, in a graph that also holds a stage named `elsewhere`.
fn path_of<'g>(stages: &[(&'g str, &'g str)], args: &'g Value) -> PathView<'g> {
    let nodes = stages
        .iter()
        .map(|(name, identifier)| PathNode {
            name,
            identifier,
            args,
        })
        .collect();
    let mut names: Vec<&str> = stages.iter().map(|(name, _)| *name).collect();
    names.push("elsewhere");

    PathView::new(nodes, names)
}

#[test]
fn after_named_finds_a_node_on_the_path() {
    let args = json!({});
    let stages = [
        ("input", "y4m/stdin"),
        ("soften", "test/blur"),
        ("deband", "test/deband"),
    ];
    let path = path_of(&stages, &args);

    let edge = path.after_named("soften").expect("soften is on the path");

    assert_eq!(edge, path.after(1));
    assert_eq!(edge.node(), 1);
}

#[test]
fn after_named_refuses_a_node_off_the_path() {
    let args = json!({});
    let stages = [("input", "y4m/stdin"), ("soften", "test/blur")];
    let path = path_of(&stages, &args);

    let Err(error) = path.after_named("elsewhere") else {
        panic!("a stage off the path should be refused");
    };

    assert_eq!(
        error.to_string(),
        "`elsewhere` is not a stage on the path this demand travels, so the provider cannot go after it",
    );
}

#[test]
fn after_named_refuses_a_name_no_stage_has() {
    let args = json!({});
    let stages = [("input", "y4m/stdin"), ("soften", "test/blur")];
    let path = path_of(&stages, &args);

    let Err(error) = path.after_named("sofen") else {
        panic!("a name no stage has should be refused");
    };

    assert_eq!(error.to_string(), "there is no stage named `sofen`");
}

#[test]
fn after_last_of_picks_the_last_match() {
    let args = json!({});
    let stages = [
        ("input", "y4m/stdin"),
        ("first", "test/blur"),
        ("between", "test/deband"),
        ("second", "test/blur"),
        ("last", "test/deband"),
    ];
    let path = path_of(&stages, &args);

    let edge = path.after_last_of(&["test/blur", "test/missing"]);

    assert_eq!(edge, Some(path.after(3)));
    assert_eq!(path.after_last_of(&["test/missing"]), None);
}

#[test]
fn before_demander_is_after_the_last_node() {
    let args = json!({});
    let stages = [
        ("input", "y4m/stdin"),
        ("soften", "test/blur"),
        ("deband", "test/deband"),
    ];
    let path = path_of(&stages, &args);

    assert_eq!(path.before_demander(), path.after(2));
    assert_eq!(path.after_source(), path.after(0));
    assert_eq!(path.nodes().len(), 3);
}

#[test]
#[should_panic]
fn after_past_the_end_panics() {
    let args = json!({});
    let stages = [("input", "y4m/stdin")];
    let path = path_of(&stages, &args);

    path.after(1);
}
