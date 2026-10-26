use crate::matrix::{Coefficients, MatrixSamples, SampleRange};

/// Smoothed map strength below this fraction of the full scale paints nothing.
pub const FLOOR: f32 = 0.25;

/// The colouring a heatmap is painted with.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Style {
    /// A greyscale picture shaded from white to deep blue.
    Blue,
    /// A lifted greyscale picture shaded from cyan through yellow to red.
    Heat,
}

/// How the planes of a clip hold the picture being painted.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum PaintLayout {
    /// Red, green and blue planes.
    Rgb,
    /// Luma and two chroma planes, each chroma plane shrunk by the shifts.
    Yuv { width_shift: u32, height_shift: u32 },
}

/// Everything a painting depends on besides the planes themselves.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Painting {
    pub style: Style,
    /// The smoothed map value that paints at full strength.
    pub scale: f32,
    pub layout: PaintLayout,
    pub samples: MatrixSamples,
    pub range: SampleRange,
    pub coefficients: Coefficients,
}

/// The standard deviation of the smoothing applied to a map of `height` rows.
pub fn sigma_for(height: u32) -> f32 {
    height as f32 / 90.0
}

/// The colours at strength 0, 0.5 and 1, as red, green and blue between 0 and 1.
pub fn ramp_stops(style: Style) -> [[f32; 3]; 3] {
    let stops = match style {
        Style::Blue => [[255.0, 255.0, 255.0], [110.0, 160.0, 255.0], [20.0, 40.0, 200.0]],
        Style::Heat => [[60.0, 220.0, 230.0], [240.0, 220.0, 40.0], [230.0, 30.0, 30.0]],
    };

    stops.map(|stop| stop.map(|channel| channel / 255.0))
}

/// The picture's brightness as `(base, gain)`, so a grey of `grey` shows as `base + gain * grey`.
pub fn lift(style: Style) -> (f32, f32) {
    match style {
        Style::Blue => (0.0, 1.0),
        Style::Heat => (FLOOR, 1.0 - FLOOR),
    }
}
