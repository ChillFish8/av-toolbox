use super::grid::Grid;
use super::place::Rect;

/// Two spaces of padding inside a box, and a border either side of that.
const BOX_PADDING: usize = 4;

/// Where a line starts, and whether it leaves a box border as a tee.
#[derive(Debug, Copy, Clone)]
pub(super) struct Exit {
    pub(super) x: usize,
    pub(super) y: usize,
    pub(super) tee: bool,
}

/// Where a line ends, and whether it ends in an arrow.
#[derive(Debug, Copy, Clone)]
pub(super) struct Entry {
    pub(super) x: usize,
    pub(super) y: usize,
    pub(super) arrow: bool,
}

pub(super) fn widest(lines: &[String]) -> usize {
    lines.iter().map(|line| line.chars().count()).max().unwrap_or(0)
}

/// The size a box needs, including the shadow behind it when it stands for several workers.
pub(super) fn box_size(lines: &[String], stacked: bool) -> (usize, usize) {
    let shadow = usize::from(stacked);
    let width = widest(lines) + BOX_PADDING + shadow;
    let height = lines.len() + 2 + shadow;

    (width, height)
}

/// The row a line leaves a box by.
pub(super) fn box_port(rect: Rect, stacked: bool) -> usize {
    let height = rect.height - usize::from(stacked);
    rect.y + height / 2
}

/// The row the line at `index` of the `count` lines entering a box arrives on.
///
/// Each line gets its own row on the left side, and the rows sit together around the middle.
/// A box is sized with at least as many rows inside its border as lines entering it.
pub(super) fn box_entry(rect: Rect, stacked: bool, count: usize, index: usize) -> usize {
    let inside = rect.height - usize::from(stacked) - 2;
    let first = rect.y + 1 + (inside - count).div_ceil(2);

    first + index
}

/// Draws a box and its lines, dashed for a crossing, with a shadow copy behind it when stacked.
///
/// The shadow sits one cell right and one cell down. The box is drawn over it and its inside
/// cleared, so only the shadow's right and bottom edges show.
pub(super) fn draw_box(grid: &mut Grid, rect: Rect, lines: &[String], dashed: bool, stacked: bool) {
    let shadow = usize::from(stacked);
    let front = Rect {
        width: rect.width - shadow,
        height: rect.height - shadow,
        ..rect
    };

    if stacked {
        let behind = front.shifted(1, 1);
        outline(grid, behind, dashed);
    }

    outline(grid, front, dashed);
    clear_inside(grid, front);

    let inner = front.width - 2;
    for (offset, line) in lines.iter().enumerate() {
        let padding = (inner - line.chars().count()) / 2;
        grid.text(front.x + 1 + padding, front.y + 1 + offset, line);
    }
}

/// Draws a rectangle's border over whatever is there.
pub(super) fn outline(grid: &mut Grid, rect: Rect, dashed: bool) {
    let (horizontal, vertical) = match dashed {
        true => ('┄', '┆'),
        false => ('─', '│'),
    };
    let right = rect.right();
    let bottom = rect.bottom();

    for x in (rect.x + 1)..right {
        grid.set(x, rect.y, horizontal);
        grid.set(x, bottom, horizontal);
    }

    for y in (rect.y + 1)..bottom {
        grid.set(rect.x, y, vertical);
        grid.set(right, y, vertical);
    }

    grid.set(rect.x, rect.y, '┌');
    grid.set(right, rect.y, '┐');
    grid.set(rect.x, bottom, '└');
    grid.set(right, bottom, '┘');
}

fn clear_inside(grid: &mut Grid, rect: Rect) {
    for y in (rect.y + 1)..rect.bottom() {
        for x in (rect.x + 1)..rect.right() {
            grid.set(x, y, ' ');
        }
    }
}

/// Runs a line straight across a virtual slot, so an edge crossing its column has something to follow.
pub(super) fn draw_lane(grid: &mut Grid, rect: Rect) {
    for x in rect.x..=rect.right() {
        grid.draw(x, rect.y, '─');
    }
}

/// Runs a line right from an exit, turning once at `turn` to reach the entry's row.
///
/// Every run is horizontal or vertical, and each character joins any line already there.
pub(super) fn route(grid: &mut Grid, exit: Exit, turn: usize, entry: Entry) {
    let leaving = match exit.tee {
        true => '├',
        false => '─',
    };
    grid.draw(exit.x, exit.y, leaving);

    if exit.y == entry.y {
        for x in (exit.x + 1)..entry.x {
            grid.draw(x, exit.y, '─');
        }
    } else {
        let going_down = entry.y > exit.y;
        let (first_corner, second_corner) = match going_down {
            true => ('┐', '└'),
            false => ('┘', '┌'),
        };

        for x in (exit.x + 1)..turn {
            grid.draw(x, exit.y, '─');
        }

        grid.draw(turn, exit.y, first_corner);

        let upper = exit.y.min(entry.y);
        let lower = exit.y.max(entry.y);
        for y in (upper + 1)..lower {
            grid.draw(turn, y, '│');
        }

        grid.draw(turn, entry.y, second_corner);

        for x in (turn + 1)..entry.x {
            grid.draw(x, entry.y, '─');
        }
    }

    if entry.arrow {
        grid.set(entry.x, entry.y, '▶');
        return;
    }

    grid.draw(entry.x, entry.y, '─');
}

/// Writes a label so it ends in the cell just before `end`.
pub(super) fn draw_label(grid: &mut Grid, end: usize, y: usize, label: &str) {
    let length = label.chars().count();
    grid.text(end - length, y, label);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    #[test]
    fn a_stacked_box_shows_its_shadow_to_the_right_and_below() {
        let mut grid = Grid::new(8, 5);
        let labels = lines(&["ab", "x4"]);
        let size = box_size(&labels, true);
        let rect = Rect {
            x: 0,
            y: 0,
            width: size.0,
            height: size.1,
        };

        draw_box(&mut grid, rect, &labels, false, true);

        assert_eq!(
            grid.render(),
            concat!("┌────┐\n", "│ ab │┐\n", "│ x4 ││\n", "└────┘│\n", " └────┘\n")
        );
    }

    #[test]
    fn lines_entering_one_box_arrive_on_neighbouring_rows_around_its_middle() {
        let rect = Rect {
            x: 0,
            y: 0,
            width: 6,
            height: 5,
        };

        let single = box_entry(rect, false, 1, 0);
        let pair = [box_entry(rect, false, 2, 0), box_entry(rect, false, 2, 1)];

        assert_eq!(single, box_port(rect, false));
        assert_eq!(pair, [2, 3]);
    }

    #[test]
    fn a_route_between_rows_turns_once_in_the_gap() {
        let mut grid = Grid::new(6, 3);
        let exit = Exit {
            x: 0,
            y: 0,
            tee: false,
        };
        let entry = Entry {
            x: 5,
            y: 2,
            arrow: true,
        };

        route(&mut grid, exit, 2, entry);

        assert_eq!(grid.render(), concat!("──┐\n", "  │\n", "  └──▶\n"));
    }

    #[test]
    fn a_label_ends_just_before_the_cell_given() {
        let mut grid = Grid::new(8, 1);

        draw_label(&mut grid, 7, 0, "ref");

        assert_eq!(grid.render(), "    ref\n");
    }
}
