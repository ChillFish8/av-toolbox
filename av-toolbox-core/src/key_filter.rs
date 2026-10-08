use schemars::JsonSchema;
use serde::de::Error;
use serde::{Deserialize, Deserializer};

const EMPTY_PATTERN: &str = "a pattern cannot be empty, expected a key such as \"index\" or \"filter.*\"";

/// A pattern matching a key, where `*` matches any run of characters.
#[derive(Debug, Clone, PartialEq, Eq, JsonSchema)]
pub struct Pattern(String);

impl Pattern {
    pub fn matches(&self, key: &str) -> bool {
        glob(self.0.as_bytes(), key.as_bytes())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for Pattern {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        if text.is_empty() {
            return Err(D::Error::custom(EMPTY_PATTERN));
        }

        Ok(Self(text))
    }
}

/// Which keys to keep.
///
/// With no `include`, every key starts kept. `exclude` then drops what it matches, so it wins
/// over `include`.
#[derive(Debug, Default, Clone)]
pub struct KeyFilter {
    include: Option<Vec<Pattern>>,
    exclude: Vec<Pattern>,
}

impl KeyFilter {
    pub fn new(include: Option<Vec<Pattern>>, exclude: Option<Vec<Pattern>>) -> Self {
        Self {
            include,
            exclude: exclude.unwrap_or_default(),
        }
    }

    pub fn keeps(&self, key: &str) -> bool {
        let included = match &self.include {
            Some(patterns) => patterns.iter().any(|pattern| pattern.matches(key)),
            None => true,
        };

        if !included {
            return false;
        }

        let excluded = self.exclude.iter().any(|pattern| pattern.matches(key));

        !excluded
    }

    /// Every include pattern, then every exclude pattern.
    pub fn patterns(&self) -> impl Iterator<Item = &Pattern> {
        let included = self.include.iter().flatten();

        included.chain(self.exclude.iter())
    }
}

/// Whether `text` matches `pattern`, where `*` matches any run of bytes.
///
/// On a mismatch this goes back to the last `*` and lets it take one more byte.
fn glob(pattern: &[u8], text: &[u8]) -> bool {
    let mut pattern_index = 0;
    let mut text_index = 0;
    let mut last_star: Option<(usize, usize)> = None;

    while text_index < text.len() {
        let current = pattern.get(pattern_index);

        if current == Some(&b'*') {
            last_star = Some((pattern_index, text_index));
            pattern_index += 1;
            continue;
        }

        if current == Some(&text[text_index]) {
            pattern_index += 1;
            text_index += 1;
            continue;
        }

        let Some((star_pattern, star_text)) = last_star else {
            return false;
        };

        pattern_index = star_pattern + 1;
        text_index = star_text + 1;
        last_star = Some((star_pattern, star_text + 1));
    }

    let remaining = &pattern[pattern_index..];

    remaining.iter().all(|byte| *byte == b'*')
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn pattern(text: &str) -> Pattern {
        serde_json::from_value(json!(text)).expect("a pattern")
    }

    fn patterns(texts: &[&str]) -> Vec<Pattern> {
        texts.iter().map(|text| pattern(text)).collect()
    }

    #[test]
    fn a_pattern_without_a_star_matches_only_itself() {
        let exact = pattern("index");

        assert!(exact.matches("index"));
        assert!(!exact.matches("indexes"));
        assert!(!exact.matches("chunk"));
    }

    #[test]
    fn a_star_matches_any_run_including_dots() {
        let filters = pattern("filter.*");

        assert!(filters.matches("filter.convert_colour"));
        assert!(filters.matches("filter.scenes.detect"));
        assert!(filters.matches("filter."));
        assert!(!filters.matches("sink.detect"));
    }

    #[test]
    fn stars_can_sit_anywhere_in_a_pattern() {
        let debug = pattern("*.debug_*");

        assert!(debug.matches("filter.debug_levels"));
        assert!(debug.matches("mask.scenes.debug_cut"));
        assert!(!debug.matches("filter.levels"));
    }

    #[test]
    fn an_empty_pattern_fails_to_deserialize() {
        let result = serde_json::from_value::<Pattern>(json!(""));

        let error = result.expect_err("an empty pattern");
        assert!(error.to_string().contains("filter.*"), "{error}");
    }

    #[test]
    fn with_neither_list_every_key_is_kept() {
        let filter = KeyFilter::default();

        assert!(filter.keeps("index"));
        assert!(filter.keeps("filter.convert_colour"));
    }

    #[test]
    fn include_keeps_only_the_keys_it_matches() {
        let filter = KeyFilter::new(Some(patterns(&["index", "filter.*"])), None);

        assert!(filter.keeps("index"));
        assert!(filter.keeps("filter.convert_colour"));
        assert!(!filter.keeps("chunk"));
    }

    #[test]
    fn exclude_wins_over_include() {
        let filter = KeyFilter::new(Some(patterns(&["filter.*"])), Some(patterns(&["filter.debug_*"])));

        assert!(filter.keeps("filter.convert_colour"));
        assert!(!filter.keeps("filter.debug_levels"));
    }

    #[test]
    fn an_empty_include_keeps_nothing() {
        let filter = KeyFilter::new(Some(Vec::new()), None);

        assert!(!filter.keeps("index"));
    }

    #[test]
    fn patterns_lists_includes_then_excludes() {
        let filter = KeyFilter::new(Some(patterns(&["filter.*"])), Some(patterns(&["*.debug"])));

        let listed: Vec<&str> = filter.patterns().map(Pattern::as_str).collect();

        assert_eq!(listed, vec!["filter.*", "*.debug"]);
    }
}
