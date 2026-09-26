use av_toolbox_core::StageError;

/// The output size and the part of the input it is made from, in input samples.
///
/// A window left unsized covers the input from its start to the input's own size, so a start on its
/// own shifts the picture.
#[derive(Debug, Copy, Clone, PartialEq)]
pub(crate) struct Window {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) src_left: f32,
    pub(crate) src_top: f32,
    pub(crate) src_width: Option<f32>,
    pub(crate) src_height: Option<f32>,
}

fn zero_output(stage: &str) -> StageError {
    let message = format!("{stage} needs a width and height above 0");

    StageError::msg(message)
}

fn not_finite(stage: &str, field: &str) -> StageError {
    let message = format!("{stage} needs {field} to be a finite number");

    StageError::msg(message)
}

fn not_positive(stage: &str, field: &str) -> StageError {
    let message = format!("{stage} needs {field} to be above 0");

    StageError::msg(message)
}

impl Window {
    pub(crate) fn check(&self, stage: &str) -> Result<(), StageError> {
        if self.width == 0 || self.height == 0 {
            return Err(zero_output(stage));
        }

        let starts = [("src_left", self.src_left), ("src_top", self.src_top)];
        for (field, value) in starts {
            if !value.is_finite() {
                return Err(not_finite(stage, field));
            }
        }

        let sizes = [("src_width", self.src_width), ("src_height", self.src_height)];
        for (field, value) in sizes {
            let Some(value) = value else {
                continue;
            };

            if !value.is_finite() {
                return Err(not_finite(stage, field));
            }

            if value <= 0.0 {
                return Err(not_positive(stage, field));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window() -> Window {
        Window {
            width: 8,
            height: 6,
            src_left: 0.0,
            src_top: 0.0,
            src_width: None,
            src_height: None,
        }
    }

    #[test]
    fn a_whole_frame_window_passes() {
        let checked = window().check("resize/point");

        assert!(checked.is_ok());
    }

    #[test]
    fn a_zero_output_size_is_refused() {
        let zero_width = Window { width: 0, ..window() };
        let zero_height = Window {
            height: 0,
            ..window()
        };

        assert!(zero_width.check("resize/point").is_err());
        assert!(zero_height.check("resize/point").is_err());
    }

    #[test]
    fn a_window_of_no_size_is_refused() {
        let empty = Window {
            src_width: Some(0.0),
            ..window()
        };
        let negative = Window {
            src_height: Some(-4.0),
            ..window()
        };

        let error = empty.check("resize/point").expect_err("an empty window");

        assert!(error.to_string().contains("src_width"));
        assert!(negative.check("resize/point").is_err());
    }

    #[test]
    fn a_window_that_is_not_finite_is_refused() {
        let left = Window {
            src_left: f32::INFINITY,
            ..window()
        };
        let width = Window {
            src_width: Some(f32::NAN),
            ..window()
        };

        assert!(left.check("resize/point").is_err());
        assert!(width.check("resize/point").is_err());
    }

    #[test]
    fn a_window_starting_outside_the_frame_passes() {
        let shifted = Window {
            src_left: -2.0,
            src_top: -1.5,
            ..window()
        };

        assert!(shifted.check("resize/point").is_ok());
    }
}
