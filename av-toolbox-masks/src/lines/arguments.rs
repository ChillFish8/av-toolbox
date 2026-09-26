use av_toolbox_core::StageError;
use av_toolbox_cubek::blur::MAX_SIGMA;

/// Checks `value` is a finite number above 0, naming `stage` and `argument` when it is not.
pub(crate) fn finite_positive(stage: &str, argument: &str, value: f32) -> Result<f32, StageError> {
    if !value.is_finite() {
        let message = format!("{stage} needs a finite {argument}, got {value}");
        return Err(StageError::msg(message));
    }

    if value <= 0.0 {
        let message = format!("{stage} needs {argument} above 0, got {value}");
        return Err(StageError::msg(message));
    }

    Ok(value)
}

/// Checks `sigma` is a finite number above 0 and no wider than the widest blur, naming `stage` and
/// `argument` when it is not.
pub(crate) fn reachable_sigma(stage: &str, argument: &str, sigma: f32) -> Result<f32, StageError> {
    let checked = finite_positive(stage, argument, sigma)?;
    if checked > MAX_SIGMA {
        let message = format!("{stage} needs {argument} of at most {MAX_SIGMA}, got {sigma}");
        return Err(StageError::msg(message));
    }

    Ok(checked)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_finite_positive_value_is_kept() {
        let kept = finite_positive("stage", "value", 2.5).expect("2.5 is finite and positive");

        assert_eq!(kept, 2.5);
    }

    #[test]
    fn the_widest_blur_sigma_is_kept() {
        let kept = reachable_sigma("stage", "sigma", MAX_SIGMA).expect("the widest sigma is allowed");

        assert_eq!(kept, MAX_SIGMA);
    }

    #[test]
    fn a_negative_infinity_is_refused_as_not_finite() {
        let error = finite_positive("stage", "value", f32::NEG_INFINITY).expect_err("not finite");

        assert!(error.to_string().contains("stage needs a finite value, got -inf"));
    }
}
