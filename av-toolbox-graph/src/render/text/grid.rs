/// Which sides a box drawing character connects to.
#[derive(Debug, Copy, Clone, Default, Eq, PartialEq)]
pub(crate) struct Sides {
    pub(crate) up: bool,
    pub(crate) down: bool,
    pub(crate) left: bool,
    pub(crate) right: bool,
}

impl Sides {
    pub(crate) fn merged(self, other: Self) -> Self {
        Self {
            up: self.up || other.up,
            down: self.down || other.down,
            left: self.left || other.left,
            right: self.right || other.right,
        }
    }

    pub(crate) const UP: Self = Self {
        up: true,
        down: false,
        left: false,
        right: false,
    };
    pub(crate) const DOWN: Self = Self {
        up: false,
        down: true,
        left: false,
        right: false,
    };
    pub(crate) const LEFT: Self = Self {
        up: false,
        down: false,
        left: true,
        right: false,
    };
    pub(crate) const RIGHT: Self = Self {
        up: false,
        down: false,
        left: false,
        right: true,
    };
}

/// A character canvas that joins lines where they meet.
pub(crate) struct Grid {
    cells: Vec<Vec<char>>,
}

impl Grid {
    pub(crate) fn new(width: usize, height: usize) -> Self {
        Self {
            cells: vec![vec![' '; width]; height],
        }
    }

    /// Writes a character over whatever is there.
    pub(crate) fn set(&mut self, x: usize, y: usize, value: char) {
        let Some(row) = self.cells.get_mut(y) else {
            return;
        };

        if let Some(cell) = row.get_mut(x) {
            *cell = value;
        }
    }

    /// Writes a line character, joining it to any line already there.
    pub(crate) fn draw(&mut self, x: usize, y: usize, value: char) {
        let existing = self.cells.get(y).and_then(|row| row.get(x)).copied();
        let Some(existing) = existing else {
            return;
        };

        let (Some(incoming), Some(current)) = (sides_of(value), sides_of(existing)) else {
            self.set(x, y, value);
            return;
        };

        self.set(x, y, character_for(incoming.merged(current)));
    }

    /// Draws a line cell reaching out on the given sides, joining it to any line already there.
    pub(crate) fn draw_sides(&mut self, x: usize, y: usize, sides: Sides) {
        let value = character_for(sides);
        self.draw(x, y, value);
    }

    pub(crate) fn text(&mut self, x: usize, y: usize, value: &str) {
        for (offset, character) in value.chars().enumerate() {
            self.set(x + offset, y, character);
        }
    }

    pub(crate) fn render(&self) -> String {
        let mut out = String::new();

        for row in &self.cells {
            let line: String = row.iter().collect();
            out.push_str(line.trim_end());
            out.push('\n');
        }

        out
    }
}

/// The sides a character connects, or `None` when it is not a line character.
fn sides_of(value: char) -> Option<Sides> {
    let sides = match value {
        ' ' => Sides::default(),
        '│' => Sides {
            up: true,
            down: true,
            ..Sides::default()
        },
        '─' => Sides {
            left: true,
            right: true,
            ..Sides::default()
        },
        '┌' => Sides {
            down: true,
            right: true,
            ..Sides::default()
        },
        '┐' => Sides {
            down: true,
            left: true,
            ..Sides::default()
        },
        '└' => Sides {
            up: true,
            right: true,
            ..Sides::default()
        },
        '┘' => Sides {
            up: true,
            left: true,
            ..Sides::default()
        },
        '├' => Sides {
            up: true,
            down: true,
            right: true,
            ..Sides::default()
        },
        '┤' => Sides {
            up: true,
            down: true,
            left: true,
            ..Sides::default()
        },
        '┬' => Sides {
            down: true,
            left: true,
            right: true,
            ..Sides::default()
        },
        '┴' => Sides {
            up: true,
            left: true,
            right: true,
            ..Sides::default()
        },
        '┼' => Sides {
            up: true,
            down: true,
            left: true,
            right: true,
        },
        _ => return None,
    };

    Some(sides)
}

fn character_for(sides: Sides) -> char {
    match (sides.up, sides.down, sides.left, sides.right) {
        (true, true, true, true) => '┼',
        (true, true, true, false) => '┤',
        (true, true, false, true) => '├',
        (true, true, false, false) => '│',
        (true, false, true, true) => '┴',
        (true, false, true, false) => '┘',
        (true, false, false, true) => '└',
        (true, false, false, false) => '│',
        (false, true, true, true) => '┬',
        (false, true, true, false) => '┐',
        (false, true, false, true) => '┌',
        (false, true, false, false) => '│',
        (false, false, true, true) => '─',
        (false, false, true, false) => '─',
        (false, false, false, true) => '─',
        (false, false, false, false) => ' ',
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_horizontal_line_crossing_a_vertical_one_becomes_a_junction() {
        let mut grid = Grid::new(3, 2);

        grid.draw(1, 0, '│');
        grid.draw(1, 1, '│');
        grid.draw(0, 1, '─');
        grid.draw(1, 1, '─');
        grid.draw(2, 1, '─');

        assert_eq!(grid.render(), " │\n─┼─\n");
    }

    #[test]
    fn a_line_meeting_a_box_border_becomes_a_tee() {
        let mut grid = Grid::new(3, 1);

        grid.draw(0, 0, '─');
        grid.draw(1, 0, '─');
        grid.draw(2, 0, '─');
        grid.draw(1, 0, '│');

        assert_eq!(grid.render(), "─┼─\n");
    }

    #[test]
    fn text_overwrites_whatever_is_underneath_it() {
        let mut grid = Grid::new(5, 1);

        grid.draw(0, 0, '─');
        grid.text(0, 0, "name");

        assert_eq!(grid.render(), "name\n");
    }

    #[test]
    fn an_arrow_replaces_the_border_it_lands_on() {
        let mut grid = Grid::new(3, 1);

        grid.draw(1, 0, '─');
        grid.set(1, 0, '▼');

        assert_eq!(grid.render(), " ▼\n");
    }

    #[test]
    fn trailing_spaces_are_trimmed_from_every_line() {
        let mut grid = Grid::new(6, 2);

        grid.text(0, 0, "ab");

        assert_eq!(grid.render(), "ab\n\n");
    }
}
