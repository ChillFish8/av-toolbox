use super::draw::outline;
use super::grid::Grid;
use super::tracks::Rect;

/// The line printed under a planned diagram.
pub(super) const LEGEND: &str = concat!(
    "◁ disperse, unbounded   ▷ aggregate, bounded   ┄ host and gpu crossing   ",
    "┄ implied provider   ×N workers",
);

/// One piece of a frame's text, and the cell it starts in.
#[derive(Debug, Clone, Eq, PartialEq)]
pub(super) struct FrameText {
    pub(super) x: usize,
    pub(super) y: usize,
    pub(super) text: String,
}

/// Draws a frame's border.
pub(super) fn draw_frame(grid: &mut Grid, rect: Rect) {
    outline(grid, rect, false);
}

/// Where a frame's heading and detail sit, the heading in the top border and the detail just inside.
///
/// Text longer than the frame is cut short rather than widening it. The heading keeps a space
/// either side so it stands clear of the border.
pub(super) fn frame_text(rect: Rect, heading: &str, detail: Option<&str>) -> Vec<FrameText> {
    let heading_room = rect.width.saturating_sub(5);
    let shown_heading = shorten(heading, heading_room);
    let title = FrameText {
        x: rect.x + 1,
        y: rect.y,
        text: format!(" {shown_heading} "),
    };

    let mut pieces = vec![title];

    if let Some(detail) = detail {
        let detail_room = rect.width.saturating_sub(4);
        let shown_detail = shorten(detail, detail_room);

        pieces.push(FrameText {
            x: rect.x + 2,
            y: rect.y + 1,
            text: shown_detail,
        });
    }

    pieces
}

/// Writes a frame's heading and detail over whatever is there.
pub(super) fn draw_frame_text(grid: &mut Grid, rect: Rect, heading: &str, detail: Option<&str>) {
    for piece in frame_text(rect, heading, detail) {
        grid.text(piece.x, piece.y, &piece.text);
    }
}

/// Cuts text down to `room` characters, ending in `…` when anything was cut.
pub(super) fn shorten(text: &str, room: usize) -> String {
    let length = text.chars().count();

    if length <= room {
        return text.to_string();
    }

    if room == 0 {
        return String::new();
    }

    let kept: String = text.chars().take(room - 1).collect();
    format!("{kept}…")
}

/// The size a marker needs, one row per line entering it.
pub(super) fn marker_size(ratio: &str, arrivals: usize) -> (usize, usize) {
    let width = ratio.chars().count() + 4;
    (width, arrivals.max(1))
}

/// Draws a marker's ratio, running a line on to the cell it leaves by.
pub(super) fn draw_marker(grid: &mut Grid, rect: Rect, ratio: &str) {
    grid.text(rect.x + 2, rect.y, ratio);

    let length = ratio.chars().count();
    for x in (rect.x + 3 + length)..rect.right() {
        grid.draw(x, rect.y, '─');
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_that_fits_is_left_alone() {
        assert_eq!(shorten("region 1", 8), "region 1");
    }

    #[test]
    fn text_that_does_not_fit_ends_in_an_ellipsis() {
        assert_eq!(shorten("region 2 · stream", 10), "region 2 …");
    }

    #[test]
    fn a_long_heading_is_cut_inside_the_frame() {
        let mut grid = Grid::new(12, 3);
        let rect = Rect {
            x: 0,
            y: 0,
            width: 12,
            height: 3,
        };

        draw_frame(&mut grid, rect);
        draw_frame_text(&mut grid, rect, "region 2 · stream", None);

        assert_eq!(
            grid.render(),
            concat!("┌ region… ─┐\n", "│          │\n", "└──────────┘\n")
        );
    }

    #[test]
    fn a_marker_runs_its_line_out_to_its_right_edge() {
        let mut grid = Grid::new(10, 1);
        let rect = Rect {
            x: 0,
            y: 0,
            width: 10,
            height: 1,
        };

        draw_marker(&mut grid, rect, "◁ 1:4");

        assert_eq!(grid.render(), "  ◁ 1:4 ─\n");
    }

    #[test]
    fn the_detail_sits_just_inside_the_top_border() {
        let rect = Rect {
            x: 3,
            y: 4,
            width: 14,
            height: 4,
        };

        let pieces = frame_text(rect, "region 1", Some("by chunk"));
        let detail = FrameText {
            x: 5,
            y: 5,
            text: String::from("by chunk"),
        };

        assert_eq!(pieces[0].text, " region 1 ");
        assert_eq!(pieces[1], detail);
    }
}
