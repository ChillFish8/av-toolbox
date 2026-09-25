use crate::variables::{Supplied, SuppliedValue, environment_name};

fn environment(entries: &[(&str, &str)]) -> Supplied {
    let owned = entries
        .iter()
        .map(|(key, value)| (key.to_string(), value.to_string()));

    Supplied::from_environment(owned)
}

#[test]
fn only_prefixed_environment_variables_are_read() {
    let supplied = environment(&[("AVTB_VAR_SIGMA", "1.2"), ("SIGMA", "9")]);

    assert_eq!(supplied.get("sigma"), Some(SuppliedValue::Environment("1.2")));
    assert_eq!(supplied.get("other"), None);
}

#[test]
fn a_mixed_case_suffix_reaches_the_lower_case_name() {
    let supplied = environment(&[("AVTB_VAR_Speed", "fast")]);

    assert_eq!(supplied.get("speed"), Some(SuppliedValue::Environment("fast")));
}

#[test]
fn a_flag_wins_over_the_environment() {
    let mut supplied = environment(&[("AVTB_VAR_SIGMA", "1.2")]);
    supplied.set_flag("sigma", "2.0");

    assert_eq!(supplied.get("sigma"), Some(SuppliedValue::Flag("2.0")));
}

#[test]
fn the_last_flag_for_a_name_wins() {
    let mut supplied = Supplied::new();
    supplied.set_flag("sigma", "1");
    supplied.set_flag("sigma", "2");

    assert_eq!(supplied.get("sigma"), Some(SuppliedValue::Flag("2")));
}

#[test]
fn flag_names_lists_every_flag() {
    let mut supplied = Supplied::new();
    supplied.set_flag("a", "1");
    supplied.set_flag("b", "2");

    let names: Vec<&str> = supplied.flag_names().collect();

    assert_eq!(names, ["a", "b"]);
}

#[test]
fn the_environment_name_is_prefixed_and_upper_case() {
    assert_eq!(environment_name("clip_input"), "AVTB_VAR_CLIP_INPUT");
}
