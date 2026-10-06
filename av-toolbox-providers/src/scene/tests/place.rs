use av_toolbox_core::{PathNode, PathView};
use serde_json::{Value, json};

use crate::scene::place;

/// The path of a demand made by `log`, which reads `denoise`, in a graph that also holds `elsewhere`.
fn path_of(args: &Value) -> PathView<'_> {
    let input = PathNode {
        name: "input",
        identifier: "y4m/stdin",
        args,
    };
    let denoise = PathNode {
        name: "denoise",
        identifier: "av-denoise/nl4d",
        args,
    };

    let stages = vec!["input", "denoise", "log", "elsewhere"];

    PathView::new(vec![input, denoise], stages)
}

#[test]
fn a_scene_request_goes_after_the_source() {
    let args = json!({});
    let path = path_of(&args);

    let edge = place(None, &path).expect("should place");

    assert_eq!(edge, path.after_source());
}

#[test]
fn a_scene_request_with_after_goes_after_that_stage() {
    let args = json!({});
    let path = path_of(&args);

    let edge = place(Some("denoise"), &path).expect("should place");

    assert_eq!(edge, path.after(1));
}

#[test]
fn after_naming_the_demanding_stage_is_refused() {
    let args = json!({});
    let path = path_of(&args);

    let Err(error) = place(Some("log"), &path) else {
        panic!("the demanding stage is not on its own path");
    };

    assert!(error.to_string().contains("`log`"), "{error}");
}

#[test]
fn after_naming_a_stage_off_the_path_is_refused() {
    let args = json!({});
    let path = path_of(&args);

    let Err(error) = place(Some("elsewhere"), &path) else {
        panic!("a stage off the path should be refused");
    };

    assert_eq!(
        error.to_string(),
        "`elsewhere` is not a stage on the path this demand travels, so the provider cannot go after it",
    );
}

#[test]
fn after_naming_no_stage_at_all_is_refused() {
    let args = json!({});
    let path = path_of(&args);

    let Err(error) = place(Some("denoize"), &path) else {
        panic!("a name no stage has should be refused");
    };

    assert_eq!(error.to_string(), "there is no stage named `denoize`");
}
