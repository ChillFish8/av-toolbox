use std::collections::{HashMap, HashSet};

use av_toolbox_core::StageKind;
use av_toolbox_graph::{FileId, StageSpec};

/// Follows aliases from `name` to a stage name, or `None` if they loop.
///
/// Every output passed on the way is added to `read_outputs`.
pub(super) fn follow(
    aliases: &HashMap<String, String>,
    outputs: &HashSet<String>,
    name: &str,
    read_outputs: &mut HashSet<String>,
) -> Option<String> {
    let mut current = name.to_string();
    let mut seen: HashSet<String> = HashSet::new();

    while let Some(next) = aliases.get(&current) {
        if !seen.insert(current.clone()) {
            return None;
        }

        if outputs.contains(&current) {
            read_outputs.insert(current.clone());
        }

        current = next.clone();
    }

    Some(current)
}

/// Removes stages from called files whose frames only reach outputs nothing reads.
pub(super) fn drop_unread(
    specs: &mut Vec<StageSpec>,
    aliases: &HashMap<String, String>,
    outputs: &HashSet<String>,
    read_outputs: &HashSet<String>,
) {
    let mut ignored = HashSet::new();
    let mut read_targets = Vec::new();
    let mut unread_targets = Vec::new();

    for output in outputs {
        let target = follow(aliases, outputs, output, &mut ignored);
        let Some(target) = target else {
            continue;
        };

        if read_outputs.contains(output) {
            read_targets.push(target);
        } else {
            unread_targets.push(target);
        }
    }

    let unread = upstream_of(specs, unread_targets);

    let sinks = specs
        .iter()
        .filter(|spec| spec.kind == StageKind::Sink)
        .map(|spec| spec.name.value.clone());
    let outside_unread = specs
        .iter()
        .filter(|spec| !unread.contains(&spec.name.value))
        .map(|spec| spec.name.value.clone());
    let kept_roots: Vec<String> = sinks.chain(read_targets).chain(outside_unread).collect();

    let kept = upstream_of(specs, kept_roots);

    specs.retain(|spec| {
        let name = &spec.name.value;
        let from_call = spec.span.file != FileId::ROOT;
        let only_unread = unread.contains(name) && !kept.contains(name);

        !(from_call && only_unread)
    });
}

/// Every stage named in `roots` and every stage they read from, directly or not.
fn upstream_of(specs: &[StageSpec], roots: Vec<String>) -> HashSet<String> {
    let by_name: HashMap<&str, &StageSpec> = specs
        .iter()
        .map(|spec| (spec.name.value.as_str(), spec))
        .collect();
    let mut found: HashSet<String> = HashSet::new();
    let mut pending = roots;

    while let Some(name) = pending.pop() {
        if !found.insert(name.clone()) {
            continue;
        }

        let Some(spec) = by_name.get(name.as_str()) else {
            continue;
        };

        for input in &spec.inputs {
            pending.push(input.value.target.clone());
        }
    }

    found
}
