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
}
