#[cfg(test)]
mod tests;

use crate::depth::Element;

/// Which blur smooths an edge mask.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum BlurKind {
    /// Reaches 6 samples either side.
    R6,
    /// Reaches 2 samples either side.
    R2,
}

/// Everything one aWarpSharp2 warp needs.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct WarpParams {
    pub element: Element,
    /// The bits one sample holds, between 8 and 16.
    pub bits: u32,
    /// The most an edge can weigh in the mask, on the 8-bit scale.
    pub thresh: u32,
    pub blur: BlurKind,
    /// How many times the blur runs.
    pub passes: u32,
    /// How far samples move toward edges, between -128 and 127. A negative depth moves them away.
    pub depth: i32,
}

impl WarpParams {
    /// The largest sample value.
    pub fn peak(&self) -> i32 {
        let levels = 1i32 << self.bits;

        levels - 1
    }

    /// `thresh` at the sample's own depth.
    pub fn scaled_thresh(&self) -> i32 {
        let shift = self.bits - 8;
        let scaled = self.thresh << shift;

        scaled as i32
    }
}
