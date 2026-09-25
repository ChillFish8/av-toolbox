const PREFIX: &str = "AVTB_VAR_";

/// Variable values given from outside a pipeline, by flag or environment variable.
#[derive(Debug, Clone, Default)]
pub struct Supplied {
    flags: Vec<(String, String)>,
    environment: Vec<(String, String)>,
}

/// Where a supplied value came from, and its text.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum SuppliedValue<'a> {
    Flag(&'a str),
    Environment(&'a str),
}

impl Supplied {
    pub fn new() -> Self {
        Self::default()
    }

    /// Every `AVTB_VAR_*` variable in this process's environment.
    ///
    /// Variables whose name or value is not valid UTF-8 are skipped.
    pub fn from_env() -> Self {
        let variables = std::env::vars_os().filter_map(|(key, value)| {
            let key = key.into_string().ok()?;
            let value = value.into_string().ok()?;
            Some((key, value))
        });

        Self::from_environment(variables)
    }

    pub fn from_environment(variables: impl IntoIterator<Item = (String, String)>) -> Self {
        let mut environment = Vec::new();

        for (key, value) in variables {
            let Some(suffix) = key.strip_prefix(PREFIX) else {
                continue;
            };

            if suffix.is_empty() {
                continue;
            }

            let name = suffix.to_ascii_lowercase();
            environment.push((name, value));
        }

        Self {
            flags: Vec::new(),
            environment,
        }
    }

    pub fn set_flag(&mut self, name: impl Into<String>, value: impl Into<String>) {
        let entry = (name.into(), value.into());
        self.flags.push(entry);
    }

    pub(crate) fn get(&self, name: &str) -> Option<SuppliedValue<'_>> {
        let flag = self.flags.iter().rev().find(|(flag_name, _)| flag_name == name);
        if let Some((_, value)) = flag {
            return Some(SuppliedValue::Flag(value));
        }

        let from_environment = self
            .environment
            .iter()
            .rev()
            .find(|(variable_name, _)| variable_name == name);
        from_environment.map(|(_, value)| SuppliedValue::Environment(value))
    }

    pub(crate) fn flag_names(&self) -> impl Iterator<Item = &str> {
        self.flags.iter().map(|(name, _)| name.as_str())
    }
}

pub(crate) fn environment_name(name: &str) -> String {
    let upper = name.to_ascii_uppercase();
    format!("{PREFIX}{upper}")
}
