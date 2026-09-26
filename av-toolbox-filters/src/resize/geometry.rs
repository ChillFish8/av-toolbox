use av_toolbox_core::StageError;
use av_toolbox_gpu::{FrameFormat, FrameLayout, Subsampling};

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

/// Output sample `i` of one axis reads the input at `scale * i + offset`, in the plane's own samples.
#[derive(Debug, Copy, Clone, PartialEq)]
pub(crate) struct Axis {
    pub(crate) scale: f32,
    pub(crate) offset: f32,
}

/// The output size of one plane and where each of its axes reads from.
#[derive(Debug, Copy, Clone, PartialEq)]
pub(crate) struct PlaneShape {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) across: Axis,
    pub(crate) down: Axis,
}

/// The shapes of every plane of a resize.
///
/// Chroma only has its own shape when it is subsampled. Every other plane is read like luma.
#[derive(Debug, Copy, Clone, PartialEq)]
pub(crate) struct Geometry {
    pub(crate) luma: PlaneShape,
    pub(crate) chroma: Option<PlaneShape>,
}

/// One axis of a window in luma samples.
struct Span {
    start: f32,
    length: f32,
    output: u32,
}

/// Maps one axis of a plane subsampled by `factor`, whose samples sit `siting` luma samples past
/// `factor * i`.
///
/// Output luma sample `i` reads input luma position `start + (i + 0.5) * scale - 0.5`. A chroma
/// sample is moved into luma space by its siting, mapped the same way, and moved back.
fn axis(span: &Span, factor: u32, siting: f32) -> Axis {
    let scale = span.length / span.output as f32;
    let factor = factor as f32;
    let luma_position = span.start + (siting + 0.5) * scale - 0.5;
    let offset = (luma_position - siting) / factor;

    Axis { scale, offset }
}

fn spans(input: &FrameFormat, window: &Window) -> (Span, Span) {
    let across = Span {
        start: window.src_left,
        length: window.src_width.unwrap_or(input.width as f32),
        output: window.width,
    };
    let down = Span {
        start: window.src_top,
        length: window.src_height.unwrap_or(input.height as f32),
        output: window.height,
    };

    (across, down)
}

pub(crate) fn geometry(input: &FrameFormat, window: &Window) -> Geometry {
    let (across, down) = spans(input, window);

    let luma_across = axis(&across, 1, 0.0);
    let luma_down = axis(&down, 1, 0.0);
    let luma = PlaneShape {
        width: window.width,
        height: window.height,
        across: luma_across,
        down: luma_down,
    };

    let FrameLayout::Yuv { subsampling } = input.layout else {
        return Geometry { luma, chroma: None };
    };

    if subsampling == Subsampling::Yuv444 {
        return Geometry { luma, chroma: None };
    }

    let (factor_x, factor_y) = subsampling.factors();
    let (siting_x, siting_y) = input.colour.chroma_location.offsets(factor_x, factor_y);
    let (chroma_width, chroma_height) = subsampling.chroma_dimensions(window.width, window.height);
    let chroma_across = axis(&across, factor_x, siting_x);
    let chroma_down = axis(&down, factor_y, siting_y);
    let chroma = PlaneShape {
        width: chroma_width,
        height: chroma_height,
        across: chroma_across,
        down: chroma_down,
    };

    Geometry {
        luma,
        chroma: Some(chroma),
    }
}

fn not_divisible(stage: &str, field: &str, value: u32, factor: u32) -> StageError {
    let message = format!("{stage} needs a {field} divisible by {factor} for this subsampling, got {value}");

    StageError::msg(message)
}

/// Refuses an output size the input's subsampling cannot split evenly.
pub(crate) fn check_output(input: &FrameFormat, window: &Window, stage: &str) -> Result<(), StageError> {
    let FrameLayout::Yuv { subsampling } = input.layout else {
        return Ok(());
    };

    let (factor_x, factor_y) = subsampling.factors();
    if !window.width.is_multiple_of(factor_x) {
        return Err(not_divisible(stage, "width", window.width, factor_x));
    }

    if !window.height.is_multiple_of(factor_y) {
        return Err(not_divisible(stage, "height", window.height, factor_y));
    }

    Ok(())
}

fn source_too_large(stage: &str, field: &str, limit: u32) -> StageError {
    let message = format!("{stage} needs {field} of at most {limit}, twice the input");

    StageError::msg(message)
}

/// Refuses a source window more than twice the input's size.
pub(crate) fn check_source(input: &FrameFormat, window: &Window, stage: &str) -> Result<(), StageError> {
    let width_limit = input.width.saturating_mul(2);
    let height_limit = input.height.saturating_mul(2);
    let sizes = [
        ("src_width", window.src_width, width_limit),
        ("src_height", window.src_height, height_limit),
    ];

    for (field, value, limit) in sizes {
        let Some(value) = value else {
            continue;
        };

        if value > limit as f32 {
            return Err(source_too_large(stage, field, limit));
        }
    }

    Ok(())
}

/// Whether the resize would hand back exactly what it was given.
pub(crate) fn is_identity(input: &FrameFormat, window: &Window) -> bool {
    let (across, down) = spans(input, window);
    let same_size = window.width == input.width && window.height == input.height;
    let whole_across = across.start == 0.0 && across.length == input.width as f32;
    let whole_down = down.start == 0.0 && down.length == input.height as f32;

    same_size && whole_across && whole_down
}

#[cfg(test)]
mod tests {
    use av_toolbox_gpu::{ChromaLocation, ColourInfo, FrameLayout, SampleFormat, Subsampling};

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

    fn yuv(width: u32, height: u32, subsampling: Subsampling, location: ChromaLocation) -> FrameFormat {
        let colour = ColourInfo {
            chroma_location: location,
            ..ColourInfo::default()
        };

        FrameFormat {
            width,
            height,
            layout: FrameLayout::Yuv { subsampling },
            sample: SampleFormat::U8,
            colour,
            frame_rate: None,
        }
    }

    fn sized(width: u32, height: u32) -> Window {
        Window {
            width,
            height,
            ..window()
        }
    }

    fn assert_axis(found: Axis, scale: f32, offset: f32) {
        assert!(
            (found.scale - scale).abs() < 1e-6,
            "scale {} not {scale}",
            found.scale
        );
        assert!(
            (found.offset - offset).abs() < 1e-6,
            "offset {} not {offset}",
            found.offset
        );
    }

    #[test]
    fn luma_is_centre_aligned() {
        let input = yuv(4, 4, Subsampling::Yuv444, ChromaLocation::Left);
        let doubled = sized(8, 8);

        let found = geometry(&input, &doubled);

        assert_axis(found.luma.across, 0.5, -0.25);
        assert_axis(found.luma.down, 0.5, -0.25);
        assert_eq!(found.chroma, None);
    }

    #[test]
    fn left_sited_chroma_doubles_where_it_sits() {
        let input = yuv(8, 8, Subsampling::Yuv420, ChromaLocation::Left);
        let doubled = sized(16, 16);

        let found = geometry(&input, &doubled);
        let chroma = found.chroma.expect("4:2:0 has its own chroma shape");

        assert_eq!((chroma.width, chroma.height), (8, 8));
        assert_axis(chroma.across, 0.5, -0.125);
        assert_axis(chroma.down, 0.5, -0.25);
    }

    #[test]
    fn centre_sited_chroma_doubles_where_it_sits() {
        let input = yuv(8, 8, Subsampling::Yuv420, ChromaLocation::Center);
        let doubled = sized(16, 16);

        let found = geometry(&input, &doubled);
        let chroma = found.chroma.expect("4:2:0 has its own chroma shape");

        assert_axis(chroma.across, 0.5, -0.25);
        assert_axis(chroma.down, 0.5, -0.25);
    }

    #[test]
    fn left_sited_chroma_halves_where_it_sits() {
        let input = yuv(16, 16, Subsampling::Yuv420, ChromaLocation::Left);
        let halved = sized(8, 8);

        let found = geometry(&input, &halved);
        let chroma = found.chroma.expect("4:2:0 has its own chroma shape");

        assert_eq!((chroma.width, chroma.height), (4, 4));
        assert_axis(chroma.across, 2.0, 0.25);
        assert_axis(chroma.down, 2.0, 0.5);
    }

    #[test]
    fn centre_sited_chroma_halves_where_it_sits() {
        let input = yuv(16, 16, Subsampling::Yuv420, ChromaLocation::Center);
        let halved = sized(8, 8);

        let found = geometry(&input, &halved);
        let chroma = found.chroma.expect("4:2:0 has its own chroma shape");

        assert_axis(chroma.across, 2.0, 0.5);
        assert_axis(chroma.down, 2.0, 0.5);
    }

    #[test]
    fn a_window_moves_left_sited_chroma() {
        let input = yuv(16, 16, Subsampling::Yuv420, ChromaLocation::Left);
        let cropped = Window {
            width: 8,
            height: 8,
            src_left: 2.0,
            src_top: 2.0,
            src_width: Some(8.0),
            src_height: Some(8.0),
        };

        let found = geometry(&input, &cropped);
        let chroma = found.chroma.expect("4:2:0 has its own chroma shape");

        assert_eq!((chroma.width, chroma.height), (4, 4));
        assert_axis(chroma.across, 1.0, 1.0);
        assert_axis(chroma.down, 1.0, 1.0);
    }

    #[test]
    fn a_window_moves_and_scales_luma() {
        let input = yuv(16, 16, Subsampling::Yuv444, ChromaLocation::Left);
        let cropped = Window {
            width: 4,
            height: 2,
            src_left: 2.0,
            src_top: 4.0,
            src_width: Some(4.0),
            src_height: Some(2.0),
        };

        let found = geometry(&input, &cropped);

        assert_axis(found.luma.across, 1.0, 2.0);
        assert_axis(found.luma.down, 1.0, 4.0);
    }

    #[test]
    fn a_422_frame_only_subsamples_across() {
        let input = yuv(8, 6, Subsampling::Yuv422, ChromaLocation::Left);
        let doubled = sized(16, 12);

        let found = geometry(&input, &doubled);
        let chroma = found.chroma.expect("4:2:2 has its own chroma shape");

        assert_eq!((chroma.width, chroma.height), (8, 12));
        assert_axis(chroma.down, 0.5, -0.25);
    }

    #[test]
    fn an_odd_width_is_refused_at_420() {
        let input = yuv(8, 8, Subsampling::Yuv420, ChromaLocation::Left);
        let odd = sized(7, 8);

        let error = check_output(&input, &odd, "resize/point").expect_err("7 is not even");

        assert!(error.to_string().contains("width"));
        assert!(error.to_string().contains('2'));
    }

    #[test]
    fn an_odd_height_is_allowed_at_422() {
        let input = yuv(8, 8, Subsampling::Yuv422, ChromaLocation::Left);
        let odd = sized(8, 7);

        let checked = check_output(&input, &odd, "resize/point");

        assert!(checked.is_ok());
    }

    #[test]
    fn a_resize_to_the_same_size_over_the_whole_frame_is_identity() {
        let input = yuv(8, 8, Subsampling::Yuv420, ChromaLocation::Left);
        let same = sized(8, 8);
        let explicit = Window {
            src_width: Some(8.0),
            src_height: Some(8.0),
            ..same
        };
        let shifted = Window {
            src_left: 0.5,
            ..same
        };

        assert!(is_identity(&input, &same));
        assert!(is_identity(&input, &explicit));
        assert!(!is_identity(&input, &shifted));
    }
}
