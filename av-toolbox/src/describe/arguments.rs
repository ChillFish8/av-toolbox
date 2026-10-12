use serde_json::Value;

/// One argument a stage takes, read from its schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Argument {
    pub(super) name: String,
    pub(super) shape: String,
    pub(super) required: bool,
    pub(super) default: Option<String>,
    pub(super) description: Option<String>,
}

/// The prefix of a reference to one of the schema's own definitions.
const DEFS_PREFIX: &str = "#/$defs/";

/// Every argument in `schema`, required ones first and each group sorted by name.
pub(super) fn read(schema: &Value) -> Vec<Argument> {
    let Some(properties) = schema.get("properties").and_then(Value::as_object) else {
        return Vec::new();
    };

    let required = required_names(schema);
    let mut arguments: Vec<Argument> = properties
        .iter()
        .map(|(name, property)| read_argument(schema, name, property, &required))
        .collect();

    arguments.sort_by(|left, right| {
        let by_need = right.required.cmp(&left.required);
        by_need.then_with(|| left.name.cmp(&right.name))
    });

    arguments
}

fn required_names(schema: &Value) -> Vec<&str> {
    let Some(listed) = schema.get("required").and_then(Value::as_array) else {
        return Vec::new();
    };

    listed.iter().filter_map(Value::as_str).collect()
}

fn read_argument(root: &Value, name: &str, property: &Value, required: &[&str]) -> Argument {
    let shape = shape_of(root, property);
    let description = description_of(root, property);
    let default = property.get("default").and_then(default_text);

    Argument {
        name: name.to_string(),
        shape,
        required: required.contains(&name),
        default,
        description,
    }
}

/// The field's own description, or the one on whatever it refers to.
fn description_of(root: &Value, property: &Value) -> Option<String> {
    let own = property.get("description").and_then(Value::as_str);
    if let Some(text) = own {
        return Some(text.to_string());
    }

    let inner = nullable_branch(property).unwrap_or(property);
    let resolved = resolve(root, inner)?;
    let referenced = resolved.get("description").and_then(Value::as_str);
    referenced.map(str::to_string)
}

fn default_text(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::String(text) => Some(text.clone()),
        other => Some(other.to_string()),
    }
}

/// How an argument's value is written, such as `float, optional` or `fast | slow`.
fn shape_of(root: &Value, schema: &Value) -> String {
    if let Some(inner) = nullable_branch(schema) {
        let shape = shape_of(root, inner);
        return format!("{shape}, optional");
    }

    let Some(resolved) = resolve(root, schema) else {
        return "value".to_string();
    };

    if let Some(values) = enum_values(resolved) {
        return values.join(" | ");
    }

    let declared = resolved.get("type");
    if let Some(Value::Array(types)) = declared {
        return typed_list_shape(root, resolved, types);
    }

    if let Some(Value::String(name)) = declared {
        return type_shape(root, resolved, name);
    }

    if let Some(branches) = union_branches(resolved) {
        let shapes: Vec<String> = branches.iter().map(|branch| shape_of(root, branch)).collect();
        let joined = shapes.join(" | ");
        return format!("one of {joined}");
    }

    "value".to_string()
}

/// The shape of a schema whose `type` is a list such as `["number", "null"]`.
fn typed_list_shape(root: &Value, schema: &Value, types: &[Value]) -> String {
    let named: Vec<&str> = types
        .iter()
        .filter_map(Value::as_str)
        .filter(|name| *name != "null")
        .collect();
    let nullable = named.len() < types.len();

    let base = match named.as_slice() {
        [single] => type_shape(root, schema, single),
        _ => "value".to_string(),
    };

    if nullable {
        return format!("{base}, optional");
    }

    base
}

fn type_shape(root: &Value, schema: &Value, name: &str) -> String {
    match name {
        "integer" => "integer".to_string(),
        "number" => "float".to_string(),
        "boolean" => "bool".to_string(),
        "string" => "string".to_string(),
        "object" => "object".to_string(),
        "array" => list_shape(root, schema),
        _ => "value".to_string(),
    }
}

fn list_shape(root: &Value, schema: &Value) -> String {
    let item_shape = match schema.get("items") {
        Some(items) => shape_of(root, items),
        None => "value".to_string(),
    };

    let shortest = schema.get("minItems").and_then(Value::as_u64);
    let longest = schema.get("maxItems").and_then(Value::as_u64);
    match (shortest, longest) {
        (Some(length), Some(same)) if length == same => format!("list of {length} {item_shape}"),
        _ => format!("list of {item_shape}"),
    }
}

/// The other branch of an `anyOf` or `oneOf` of two where one branch is `null`.
fn nullable_branch(schema: &Value) -> Option<&Value> {
    let branches = union_branches(schema)?;
    let [first, second] = branches else {
        return None;
    };

    if is_null(second) {
        return Some(first);
    }

    if is_null(first) {
        return Some(second);
    }

    None
}

fn is_null(schema: &Value) -> bool {
    schema.get("type").and_then(Value::as_str) == Some("null")
}

fn union_branches(schema: &Value) -> Option<&[Value]> {
    let branches = schema.get("anyOf").or_else(|| schema.get("oneOf"))?;
    let listed = branches.as_array()?;
    Some(listed.as_slice())
}

/// The definition `schema` refers to, or `schema` itself when it holds no reference.
///
/// A reference to a definition the schema does not hold resolves to nothing.
fn resolve<'a>(root: &'a Value, schema: &'a Value) -> Option<&'a Value> {
    let Some(reference) = schema.get("$ref").and_then(Value::as_str) else {
        return Some(schema);
    };

    let name = reference.strip_prefix(DEFS_PREFIX)?;
    let definitions = root.get("$defs")?;
    definitions.get(name)
}

/// The values a string enum allows, written as `enum` or as a union of `const` branches.
fn enum_values(schema: &Value) -> Option<Vec<String>> {
    if let Some(listed) = schema.get("enum").and_then(Value::as_array) {
        return string_values(listed);
    }

    let branches = union_branches(schema)?;
    let mut values = Vec::new();
    for branch in branches {
        let value = single_value(branch)?;
        values.push(value);
    }

    Some(values)
}

fn single_value(branch: &Value) -> Option<String> {
    if let Some(constant) = branch.get("const").and_then(Value::as_str) {
        return Some(constant.to_string());
    }

    let listed = branch.get("enum").and_then(Value::as_array)?;
    let [only] = listed.as_slice() else {
        return None;
    };

    only.as_str().map(str::to_string)
}

fn string_values(listed: &[Value]) -> Option<Vec<String>> {
    listed
        .iter()
        .map(|value| value.as_str().map(str::to_string))
        .collect()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn only(schema: &Value) -> Argument {
        let mut arguments = read(schema);
        assert_eq!(arguments.len(), 1, "{arguments:?}");
        arguments.remove(0)
    }

    #[test]
    fn plain_types_read_as_simple_names() {
        let schema = json!({
            "properties": {
                "count": { "type": "integer", "format": "uint32" },
                "on": { "type": "boolean" },
                "path": { "type": "string" },
                "sigma": { "type": "number", "format": "float" },
            },
        });

        let shapes: Vec<String> = read(&schema).into_iter().map(|argument| argument.shape).collect();

        assert_eq!(shapes, ["integer", "bool", "string", "float"]);
    }

    #[test]
    fn a_ref_to_an_enum_reads_as_its_values() {
        let schema = json!({
            "$defs": { "Preset": { "enum": ["fast", "slow"], "type": "string", "description": "A preset." } },
            "properties": { "preset": { "$ref": "#/$defs/Preset" } },
        });

        let argument = only(&schema);

        assert_eq!(argument.shape, "fast | slow");
        assert_eq!(argument.description.as_deref(), Some("A preset."));
    }

    #[test]
    fn a_field_description_overrides_the_referenced_one() {
        let schema = json!({
            "$defs": { "Preset": { "enum": ["fast"], "type": "string", "description": "A preset." } },
            "properties": { "preset": { "$ref": "#/$defs/Preset", "description": "The preset used." } },
        });

        let argument = only(&schema);

        assert_eq!(argument.description.as_deref(), Some("The preset used."));
    }

    #[test]
    fn a_nullable_number_reads_as_an_optional_float() {
        let schema = json!({
            "properties": { "lambda": { "type": ["number", "null"], "default": null } },
        });

        let argument = only(&schema);

        assert_eq!(argument.shape, "float, optional");
        assert_eq!(argument.default, None);
    }

    #[test]
    fn a_nullable_ref_reads_as_optional() {
        let schema = json!({
            "$defs": { "FrameRate": { "type": "string", "pattern": "^[0-9]+$" } },
            "properties": {
                "frame_rate": { "anyOf": [{ "$ref": "#/$defs/FrameRate" }, { "type": "null" }] },
            },
        });

        let argument = only(&schema);

        assert_eq!(argument.shape, "string, optional");
    }

    #[test]
    fn a_list_of_a_referenced_enum_reads_as_a_list_of_its_values() {
        let schema = json!({
            "$defs": { "ChannelName": { "enum": ["y", "u", "v"], "type": "string" } },
            "properties": {
                "planes": { "items": { "$ref": "#/$defs/ChannelName" }, "type": ["array", "null"] },
            },
        });

        let argument = only(&schema);

        assert_eq!(argument.shape, "list of y | u | v, optional");
    }

    #[test]
    fn a_fixed_length_list_says_how_many() {
        let schema = json!({
            "properties": {
                "thresholds": {
                    "items": { "type": "number" },
                    "minItems": 3,
                    "maxItems": 3,
                    "type": "array",
                },
            },
        });

        let argument = only(&schema);

        assert_eq!(argument.shape, "list of 3 float");
    }

    #[test]
    fn a_one_of_documented_consts_reads_as_an_enum() {
        let schema = json!({
            "$defs": {
                "Output": {
                    "oneOf": [
                        { "const": "clip", "type": "string", "description": "The clip." },
                        { "enum": ["map"], "type": "string", "description": "The map." },
                    ],
                },
            },
            "properties": { "output": { "$ref": "#/$defs/Output" } },
        });

        let argument = only(&schema);

        assert_eq!(argument.shape, "clip | map");
    }

    #[test]
    fn a_mixed_union_reads_each_branch() {
        let schema = json!({
            "$defs": {
                "Colour": {
                    "anyOf": [
                        { "type": "string" },
                        { "items": { "type": "integer" }, "type": "array" },
                    ],
                },
            },
            "properties": { "colour": { "$ref": "#/$defs/Colour" } },
        });

        let argument = only(&schema);

        assert_eq!(argument.shape, "one of string | list of integer");
    }

    #[test]
    fn a_nested_object_reads_as_object() {
        let schema = json!({
            "properties": {
                "tuning": { "properties": { "alpha": { "type": "number" } }, "type": "object" },
            },
        });

        let argument = only(&schema);

        assert_eq!(argument.shape, "object");
    }

    #[test]
    fn a_dangling_ref_reads_as_a_value() {
        let schema = json!({
            "properties": { "mystery": { "$ref": "#/$defs/Missing" } },
        });

        let argument = only(&schema);

        assert_eq!(argument.shape, "value");
    }

    #[test]
    fn required_arguments_come_first_and_each_group_is_sorted() {
        let schema = json!({
            "properties": {
                "zeta": { "type": "integer" },
                "alpha": { "type": "integer" },
                "width": { "type": "integer" },
                "height": { "type": "integer" },
            },
            "required": ["width", "height"],
        });

        let names: Vec<String> = read(&schema).into_iter().map(|argument| argument.name).collect();

        assert_eq!(names, ["height", "width", "alpha", "zeta"]);
    }

    #[test]
    fn defaults_print_as_compact_json_with_strings_unquoted() {
        let schema = json!({
            "properties": {
                "display": { "type": "string", "default": "standard_fhd" },
                "sigmas": { "items": { "type": "number" }, "type": "array", "default": [50.0, 200.0] },
            },
        });

        let defaults: Vec<Option<String>> = read(&schema)
            .into_iter()
            .map(|argument| argument.default)
            .collect();

        assert_eq!(
            defaults,
            [Some("standard_fhd".to_string()), Some("[50.0,200.0]".to_string())]
        );
    }

    #[test]
    fn a_schema_without_properties_has_no_arguments() {
        let schema = json!({ "items": { "type": "object" }, "type": "array" });

        assert!(read(&schema).is_empty());
    }
}
