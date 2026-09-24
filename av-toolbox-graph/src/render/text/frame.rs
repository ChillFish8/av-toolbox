use super::draw::{outline, widest};
use super::grid::Grid;
use super::place::Rect;

/// Border and padding either side of a frame's contents.
pub(super) const FRAME_INSET: usize = 2;

/// The size a frame needs around contents of the given size, wide enough for its heading.
pub(super) fn frame_size(inner: (usize, usize), heading: &str) -> (usize, usize) {
    let heading_width = heading.chars().count() + 4;
    let width = (inner.0 + FRAME_INSET * 2).max(heading_width);

    (width, inner.1 + FRAME_INSET * 2)
}

/// Draws a frame with its heading set into the top border.
pub(super) fn draw_frame(grid: &mut Grid, rect: Rect, heading: &str) {
    outline(grid, rect, false);

    let title = format!(" {heading} ");
    grid.text(rect.x + 1, rect.y, &title);
}

/// The size an exchange marker needs, with a cell either side for the lines it joins.
pub(super) fn marker_size(lines: &[String]) -> (usize, usize) {
    (widest(lines) + 4, lines.len())
}

/// Draws an exchange marker, running its first line on to the cell it leaves by.
pub(super) fn draw_marker(grid: &mut Grid, rect: Rect, lines: &[String]) {
    for (offset, line) in lines.iter().enumerate() {
        grid.text(rect.x + 2, rect.y + offset, line);
    }

    let first = lines.first().map_or(0, |line| line.chars().count());
    for x in (rect.x + 3 + first)..rect.right() {
        grid.draw(x, rect.y, '─');
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_frame_is_wide_enough_for_its_heading() {
        assert_eq!(frame_size((1, 0), "r 1"), (7, 4));
        assert_eq!(frame_size((20, 3), "r 1"), (24, 7));
    }

    #[test]
    fn a_frame_carries_its_heading_in_the_top_border() {
        let mut grid = Grid::new(12, 3);
        let rect = Rect {
            x: 0,
            y: 0,
            width: 12,
            height: 3,
        };

        draw_frame(&mut grid, rect, "r 1");

        assert_eq!(
            grid.render(),
            concat!("┌ r 1 ─────┐\n", "│          │\n", "└──────────┘\n")
        );
    }

    #[test]
    fn a_marker_runs_its_first_line_out_to_its_right_edge() {
        let mut grid = Grid::new(12, 2);
        let labels = vec![String::from("◁ 1:4"), String::from("disperse")];
        let size = marker_size(&labels);
        let rect = Rect {
            x: 0,
            y: 0,
            width: size.0,
            height: size.1,
        };

        draw_marker(&mut grid, rect, &labels);

        assert_eq!(grid.render(), concat!("  ◁ 1:4 ───\n", "  disperse\n"));
    }
}
