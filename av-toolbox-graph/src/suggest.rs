/// The closest of `names` to `target`, when one is near enough to be worth suggesting.
pub fn suggest_name<'a>(target: &str, names: impl IntoIterator<Item = &'a str>) -> Option<&'a str> {
    let mut best: Option<(usize, &'a str)> = None;

    for name in names {
        let distance = edit_distance(target, name);
        let is_better = best.is_none_or(|(current, _)| distance < current);

        if distance <= 2 && is_better {
            best = Some((distance, name));
        }
    }

    best.map(|(_, name)| name)
}

/// The number of single character edits between two names.
fn edit_distance(left: &str, right: &str) -> usize {
    let right_chars: Vec<char> = right.chars().collect();
    let mut previous: Vec<usize> = (0..=right_chars.len()).collect();
    let mut current = vec![0; right_chars.len() + 1];

    for (row, left_char) in left.chars().enumerate() {
        current[0] = row + 1;

        for (column, right_char) in right_chars.iter().enumerate() {
            let substitution = previous[column] + usize::from(left_char != *right_char);
            let deletion = previous[column + 1] + 1;
            let insertion = current[column] + 1;

            current[column + 1] = substitution.min(deletion).min(insertion);
        }

        std::mem::swap(&mut previous, &mut current);
    }

    previous[right_chars.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_close_name_is_suggested() {
        let names = ["clip_input", "sigma"];

        assert_eq!(suggest_name("clip_inptu", names), Some("clip_input"));
    }

    #[test]
    fn a_distant_name_is_not_suggested() {
        let names = ["clip_input"];

        assert_eq!(suggest_name("sigma", names), None);
    }
}
