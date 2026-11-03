use strum_macros::{Display, EnumString};

/// How the av-denoise stages pick between their candidate kernels.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Hash, EnumString, Display)]
#[strum(serialize_all = "snake_case")]
pub enum Tuning {
    /// Uses stored results and times the candidates that have none.
    Auto,
    /// Uses stored results only. A kernel without one runs its baseline.
    #[default]
    Cached,
    /// Always runs the baseline.
    Off,
}

/// Whether the stages that can tune run a fresh tuning pass, ignoring stored results.
///
/// A fresh pass saves its results over the stored ones and runs whatever the [Tuning].
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Hash, Display)]
#[strum(serialize_all = "snake_case")]
pub enum Retune {
    /// Keeps stored results.
    #[default]
    No,
    /// Times one candidate from each kernel family, then only the families that look worth it.
    Families,
    /// Times every candidate, which takes far longer.
    Exhaustive,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_tuning_mode_renders_and_parses_back() {
        let modes = [
            (Tuning::Auto, "auto"),
            (Tuning::Cached, "cached"),
            (Tuning::Off, "off"),
        ];

        for (mode, spelling) in modes {
            assert_eq!(mode.to_string(), spelling);

            let parsed = spelling.parse::<Tuning>().expect("should parse");
            assert_eq!(parsed, mode, "{spelling}");
        }

        assert_eq!(Tuning::default(), Tuning::Cached);
    }

    #[test]
    fn every_retune_renders_its_name() {
        let retunes = [
            (Retune::No, "no"),
            (Retune::Families, "families"),
            (Retune::Exhaustive, "exhaustive"),
        ];

        for (retune, spelling) in retunes {
            assert_eq!(retune.to_string(), spelling);
        }

        assert_eq!(Retune::default(), Retune::No);
    }
}
