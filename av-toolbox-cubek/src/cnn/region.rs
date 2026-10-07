/// A rectangle of a plane, in samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Region {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl Region {
    pub fn samples(&self) -> u32 {
        self.width * self.height
    }

    pub(crate) fn contains(&self, other: &Region) -> bool {
        other.x >= self.x
            && other.y >= self.y
            && other.x + other.width <= self.x + self.width
            && other.y + other.height <= self.y + self.height
    }
}

/// Splits a plane into tiles, giving each tile's region and core.
///
/// The cores are at most `tile` on a side and cover the plane exactly once. Each region is its core widened
/// by `halo` on every side, clipped to the plane.
pub fn tiles(width: u32, height: u32, tile: u32, halo: u32) -> Vec<(Region, Region)> {
    assert!(tile >= 1, "a tile must be at least one sample on a side");

    let mut list = Vec::new();

    for core_y in (0..height).step_by(tile as usize) {
        for core_x in (0..width).step_by(tile as usize) {
            let core_width = tile.min(width - core_x);
            let core_height = tile.min(height - core_y);
            let region_x = core_x.saturating_sub(halo);
            let region_y = core_y.saturating_sub(halo);
            let region_right = (core_x + core_width + halo).min(width);
            let region_bottom = (core_y + core_height + halo).min(height);

            let core = Region {
                x: core_x,
                y: core_y,
                width: core_width,
                height: core_height,
            };
            let region = Region {
                x: region_x,
                y: region_y,
                width: region_right - region_x,
                height: region_bottom - region_y,
            };
            list.push((region, core));
        }
    }

    list
}
