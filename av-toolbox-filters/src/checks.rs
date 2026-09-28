use av_toolbox_core::StageError;

fn below_minimum(stage: &str, name: &str, value: f32, minimum: f32) -> StageError {
    let message = format!("{stage} needs a finite {name} of at least {minimum}, got {value}");

    StageError::msg(message)
}

fn out_of_range_u32(stage: &str, name: &str, value: u32, minimum: u32, maximum: u32) -> StageError {
    let message = format!("{stage} needs {name} between {minimum} and {maximum}, got {value}");

    StageError::msg(message)
}

fn out_of_range_f32(stage: &str, name: &str, value: f32, minimum: f32, maximum: f32) -> StageError {
    let message = format!("{stage} needs {name} between {minimum} and {maximum}, got {value}");

    StageError::msg(message)
}

/// Refuses `value` unless it is finite and at least `minimum`.
pub(crate) fn at_least(stage: &str, name: &str, value: f32, minimum: f32) -> Result<(), StageError> {
    if value.is_finite() && value >= minimum {
        return Ok(());
    }

    Err(below_minimum(stage, name, value, minimum))
}

/// Refuses `value` unless it falls between `minimum` and `maximum`, inclusive.
pub(crate) fn between_u32(
    stage: &str,
    name: &str,
    value: u32,
    minimum: u32,
    maximum: u32,
) -> Result<(), StageError> {
    if (minimum..=maximum).contains(&value) {
        return Ok(());
    }

    Err(out_of_range_u32(stage, name, value, minimum, maximum))
}

/// Refuses `value` unless it falls between `minimum` and `maximum`, inclusive.
pub(crate) fn between_f32(
    stage: &str,
    name: &str,
    value: f32,
    minimum: f32,
    maximum: f32,
) -> Result<(), StageError> {
    if (minimum..=maximum).contains(&value) {
        return Ok(());
    }

    Err(out_of_range_f32(stage, name, value, minimum, maximum))
}

#[cfg(test)]
mod tests {
    use super::*;

    const STAGE: &str = "test/stage";

    #[test]
    fn at_least_accepts_the_minimum_itself() {
        assert!(at_least(STAGE, "value", 1.0, 1.0).is_ok());
    }

    #[test]
    fn at_least_refuses_nan_and_values_below_the_minimum() {
        let below = at_least(STAGE, "value", 0.5, 1.0).expect_err("below minimum");
        let nan = at_least(STAGE, "value", f32::NAN, 1.0).expect_err("not finite");

        assert!(below.to_string().contains(STAGE) && below.to_string().contains("value"));
        assert!(nan.to_string().contains(STAGE) && nan.to_string().contains("value"));
    }

    #[test]
    fn between_u32_accepts_the_bounds_and_refuses_outside_them() {
        assert!(between_u32(STAGE, "value", 0, 0, 16).is_ok());
        assert!(between_u32(STAGE, "value", 16, 0, 16).is_ok());

        let error = between_u32(STAGE, "value", 17, 0, 16).expect_err("above maximum");

        assert!(error.to_string().contains(STAGE) && error.to_string().contains("17"));
    }

    #[test]
    fn between_f32_refuses_nan_and_out_of_range_values() {
        let above = between_f32(STAGE, "value", 256.0, 0.0, 255.0).expect_err("above maximum");
        let nan = between_f32(STAGE, "value", f32::NAN, 0.0, 255.0).expect_err("not finite");

        assert!(above.to_string().contains("256"));
        assert!(nan.to_string().contains(STAGE));
    }
}
