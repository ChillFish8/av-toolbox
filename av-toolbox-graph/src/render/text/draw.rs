use super::grid::{Grid, Sides};
use super::tracks::Rect;

/// Two spaces of padding inside a box, and a border either side of that.
const BOX_PADDING: usize = 4;

fn widest(lines: &[String]) -> usize {
    lines.iter().map(|line| line.chars().count()).max().unwrap_or(0)
}

/// The size a box needs, with its shadow when stacked and a row inside for every entering line.
///
/// Entering lines take rows from the middle downwards, so the inside is at least twice as tall as
/// the number of lines entering, less one.
pub(super) fn box_size(lines: &[String], stacked: bool, arrivals: usize) -> (usize, usize) {
    let shadow = usize::from(stacked);
    let entry_rows = (arrivals * 2).saturating_sub(1);
    let inside = lines.len().max(entry_rows).max(1);
    let width = widest(lines) + BOX_PADDING + shadow;

    (width, inside + 2 + shadow)
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

/// Writes a label so it ends in the cell just before `end`.
pub(super) fn draw_label(grid: &mut Grid, end: usize, y: usize, label: &str) {
    let length = label.chars().count();
    grid.text(end - length, y, label);
}

/// Draws a route through its corner points, as a tee or plain run where it starts and an arrow where it ends.
pub(super) fn draw_route(grid: &mut Grid, points: &[(usize, usize)], tee: bool) {
    let cells = route_cells(points);
    let last = cells.len().saturating_sub(1);

    for (index, (x, y, sides)) in cells.into_iter().enumerate() {
        if index == 0 {
            let leaving = match tee {
                true => '├',
                false => '─',
            };
            grid.draw(x, y, leaving);
            continue;
        }

        if index == last {
            grid.set(x, y, '▶');
            continue;
        }

        grid.draw_sides(x, y, sides);
    }
}

/// Every cell a route passes, with the sides the route touches in each.
pub(super) fn route_cells(points: &[(usize, usize)]) -> Vec<(usize, usize, Sides)> {
    let Some(&(start_x, start_y)) = points.first() else {
        return Vec::new();
    };

    let mut cells = vec![(start_x, start_y, Sides::default())];

    for pair in points.windows(2) {
        let (target_x, target_y) = pair[1];

        loop {
            let (x, y, _) = *cells.last().expect("a route has a start");
            if (x, y) == (target_x, target_y) {
                break;
            }

            let (next_x, next_y, leaving, arriving) = step(x, y, target_x, target_y);
            if let Some(current) = cells.last_mut() {
                current.2 = current.2.merged(leaving);
            }

            cells.push((next_x, next_y, arriving));
        }
    }

    cells
}

/// One cell's step towards a point, with the side it leaves by and the side it arrives by.
fn step(x: usize, y: usize, target_x: usize, target_y: usize) -> (usize, usize, Sides, Sides) {
    if target_x > x {
        return (x + 1, y, Sides::RIGHT, Sides::LEFT);
    }

    if target_x < x {
        return (x - 1, y, Sides::LEFT, Sides::RIGHT);
    }

    if target_y > y {
        return (x, y + 1, Sides::DOWN, Sides::UP);
    }

    (x, y - 1, Sides::UP, Sides::DOWN)
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
        let size = box_size(&labels, true, 0);
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
    fn a_route_between_rows_turns_once_in_the_gap() {
        let mut grid = Grid::new(6, 3);
        let points = [(0, 0), (2, 0), (2, 2), (5, 2)];

        draw_route(&mut grid, &points, false);

        assert_eq!(grid.render(), concat!("──┐\n", "  │\n", "  └──▶\n"));
    }

    #[test]
    fn a_label_ends_just_before_the_cell_given() {
        let mut grid = Grid::new(8, 1);

        draw_label(&mut grid, 7, 0, "ref");

        assert_eq!(grid.render(), "    ref\n");
    }
}
