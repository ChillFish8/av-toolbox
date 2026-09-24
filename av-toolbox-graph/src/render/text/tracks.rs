use std::collections::{BTreeSet, HashMap};

use super::cells::Cells;

/// Cells either side of a frame border that sits in a gutter or channel.
const FRAME_RESERVE: usize = 2;

/// A rectangle of cells on the grid.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq)]
pub(super) struct Rect {
    pub(super) x: usize,
    pub(super) y: usize,
    pub(super) width: usize,
    pub(super) height: usize,
}

impl Rect {
    pub(super) fn right(&self) -> usize {
        self.x + self.width - 1
    }

    pub(super) fn bottom(&self) -> usize {
        self.y + self.height - 1
    }

    pub(super) fn shifted(&self, x: usize, y: usize) -> Self {
        Self {
            x: self.x + x,
            y: self.y + y,
            ..*self
        }
    }
}

/// The size of one item, and the row inside it that lines leave from and first enter on.
#[derive(Debug, Copy, Clone)]
pub(super) struct Size {
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) port: usize,
    /// Whether the item is a box with a shadow, whose last column holds only that shadow.
    pub(super) stacked: bool,
}

/// One link to route, as item indices, with its entry index at the target and its label's width.
///
/// `region` names the frame both ends sit inside, when they share one, and is `None` otherwise.
#[derive(Debug, Copy, Clone)]
pub(super) struct Wire {
    pub(super) from: usize,
    pub(super) to: usize,
    pub(super) entry: usize,
    pub(super) label: usize,
    pub(super) region: Option<usize>,
}

/// The cells one region covers, and whether its frame carries a detail line.
#[derive(Debug, Copy, Clone)]
pub(super) struct FrameCells {
    pub(super) first_column: usize,
    pub(super) last_column: usize,
    pub(super) top_row: usize,
    pub(super) bottom_row: usize,
    pub(super) detail: bool,
}

/// Where everything lands on the character grid.
#[derive(Debug)]
pub(super) struct Geometry {
    pub(super) items: Vec<Rect>,
    pub(super) frames: Vec<Rect>,
    /// Each wire's corner points, from its exit to its entry.
    pub(super) routes: Vec<Vec<(usize, usize)>>,
    /// For each wire with a label, the cell its label ends just before and the row it sits on.
    pub(super) labels: Vec<Option<(usize, usize)>>,
    pub(super) width: usize,
    pub(super) height: usize,
}

/// A horizontal run a line sits on inside one gutter.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
enum Line {
    Port(usize),
    Entry(usize, usize),
    /// A track in the open middle of a channel.
    Channel(usize, usize),
    /// A track inside a frame's bottom edge, in the same channel.
    InsideChannel(usize, usize),
}

/// One wire's turn inside one gutter.
#[derive(Debug, Copy, Clone)]
struct Pass {
    key: usize,
    leave: Line,
    arrive: Line,
}

/// How a wire gets from its source to its target.
#[derive(Debug, Copy, Clone)]
enum Path {
    Straight,
    Turn {
        gutter: usize,
    },
    Channel {
        channel: usize,
        first: usize,
        last: usize,
    },
}

/// One wire's run along a channel.
#[derive(Debug, Copy, Clone)]
struct Run {
    key: usize,
    first: usize,
    last: usize,
    down: bool,
}

/// Turns cells and item sizes into character positions, frames and routes.
///
/// An empty `frames` lays out an unplanned diagram, with no margins and no frame reserves.
pub(super) fn lay_out(cells: &Cells, sizes: &[Size], wires: &[Wire], frames: &[FrameCells]) -> Geometry {
    let planned = !frames.is_empty();
    let widths = column_widths(cells, sizes);
    let (ports, heights) = row_lines(cells, sizes);
    let paths: Vec<Path> = wires.iter().map(|wire| path_of(cells, wire)).collect();

    let channel_tracks = channel_tracks(cells, wires, &paths, frames);
    let passes = gutter_passes(cells, wires, &paths, frames, &channel_tracks);
    let gutter_tracks: Vec<HashMap<usize, usize>> =
        passes.iter().map(|passes| gutter_order(passes)).collect();

    let columns = Columns::new(cells, wires, frames, &widths, &gutter_tracks, planned);
    let rows = Rows::new(cells, frames, &heights, &channel_tracks, planned);

    let items: Vec<Rect> = cells
        .cells
        .iter()
        .zip(sizes)
        .map(|(cell, size)| {
            let x = columns.x[cell.column];
            let y = rows.top[cell.row] + ports[cell.row] - size.port;

            Rect {
                x,
                y,
                width: widths[cell.column],
                height: size.height,
            }
        })
        .collect();

    let mut routes = Vec::with_capacity(wires.len());
    let mut labels = Vec::with_capacity(wires.len());

    for (wire, path) in wires.iter().zip(&paths) {
        let source = items[wire.from];
        let target = items[wire.to];
        let exit = (source.right(), source.y + sizes[wire.from].port);
        let entry = (target.x, target.y + sizes[wire.to].port + wire.entry);

        let route = match *path {
            Path::Straight => vec![exit, entry],
            Path::Turn { gutter } => {
                let turn = columns.track_x(gutter, gutter_tracks[gutter][&wire.from]);
                vec![exit, (turn, exit.1), (turn, entry.1), entry]
            },
            Path::Channel { channel, first, last } => {
                let inside = inside_channel(wire, channel, frames);
                let track = match inside {
                    true => channel_tracks.inside[channel][&wire.from],
                    false => channel_tracks.outside[channel][&wire.from],
                };
                let run_y = match inside {
                    true => rows.inside_track_y(channel, track),
                    false => rows.track_y(channel, track),
                };
                let out_turn = columns.track_x(first, gutter_tracks[first][&wire.from]);
                let in_turn = columns.track_x(last, gutter_tracks[last][&wire.from]);

                vec![
                    exit,
                    (out_turn, exit.1),
                    (out_turn, run_y),
                    (in_turn, run_y),
                    (in_turn, entry.1),
                    entry,
                ]
            },
        };

        let target_column = cells.cells[wire.to].column;
        let label = match wire.label {
            0 => None,
            _ => {
                let end = columns.label_end(target_column - 1);
                Some((end, entry.1))
            },
        };

        routes.push(route);
        labels.push(label);
    }

    let frame_rects = frames
        .iter()
        .map(|frame| {
            let left = columns.x[frame.first_column] - FRAME_RESERVE;
            let right = columns.x[frame.last_column] + widths[frame.last_column] + 1;
            let top = rows.top[frame.top_row] - rows.top_reserve[frame.top_row];
            let bottom = rows.bottom_border(frame.bottom_row + 1);

            Rect {
                x: left,
                y: top,
                width: right - left + 1,
                height: bottom - top + 1,
            }
        })
        .collect();

    Geometry {
        items,
        frames: frame_rects,
        routes,
        labels,
        width: columns.width,
        height: rows.height,
    }
}

/// Each column's width, as wide as its widest item.
///
/// A column holding a stacked box gives every other item one more cell, since a plain box there is
/// drawn one cell narrower to line up with the stacked box's front.
fn column_widths(cells: &Cells, sizes: &[Size]) -> Vec<usize> {
    let mut holds_stacked = vec![false; cells.columns];

    for (cell, size) in cells.cells.iter().zip(sizes) {
        holds_stacked[cell.column] |= size.stacked;
    }

    let mut widths = vec![1; cells.columns];

    for (cell, size) in cells.cells.iter().zip(sizes) {
        let narrowed = holds_stacked[cell.column] && !size.stacked;
        let needed = size.width + usize::from(narrowed);
        widths[cell.column] = widths[cell.column].max(needed);
    }

    widths
}

/// Each row's port line, and how tall the row is once every item sits on it.
fn row_lines(cells: &Cells, sizes: &[Size]) -> (Vec<usize>, Vec<usize>) {
    let mut ports = vec![0; cells.rows];

    for (cell, size) in cells.cells.iter().zip(sizes) {
        ports[cell.row] = ports[cell.row].max(size.port);
    }

    let mut heights = vec![0; cells.rows];

    for (cell, size) in cells.cells.iter().zip(sizes) {
        let reach = ports[cell.row] - size.port + size.height;
        heights[cell.row] = heights[cell.row].max(reach);
    }

    (ports, heights)
}

fn path_of(cells: &Cells, wire: &Wire) -> Path {
    let from = cells.cells[wire.from];
    let to = cells.cells[wire.to];

    if to.column == from.column + 1 {
        if from.row == to.row && wire.entry == 0 {
            return Path::Straight;
        }

        return Path::Turn { gutter: from.column };
    }

    let channel = match to.row < from.row {
        true => from.row,
        false => from.row + 1,
    };

    Path::Channel {
        channel,
        first: from.column,
        last: to.column - 1,
    }
}

fn entry_line(cells: &Cells, wire: &Wire) -> Line {
    let row = cells.cells[wire.to].row;

    match wire.entry {
        0 => Line::Port(row),
        entry => Line::Entry(row, entry),
    }
}

/// Whether a wire's channel run sits inside its region's frame, along that frame's bottom edge.
fn inside_channel(wire: &Wire, channel: usize, frames: &[FrameCells]) -> bool {
    wire.region
        .is_some_and(|region| frames[region].bottom_row + 1 == channel)
}

/// Every channel's tracks, split into those inside a frame's bottom edge and those in the open
/// middle of the channel.
struct ChannelTracks {
    inside: Vec<HashMap<usize, usize>>,
    outside: Vec<HashMap<usize, usize>>,
}

/// Every channel's tracks, keyed by the exit that owns them.
///
/// Lines running down from the row above come first, shortest first, so longer ones sit further
/// from that row. Lines running up from the row below follow, longest first, for the same reason.
/// A line inside a frame's bottom edge always runs down from the row above.
fn channel_tracks(cells: &Cells, wires: &[Wire], paths: &[Path], frames: &[FrameCells]) -> ChannelTracks {
    let mut inside_runs: Vec<Vec<Run>> = vec![Vec::new(); cells.rows + 1];
    let mut outside_runs: Vec<Vec<Run>> = vec![Vec::new(); cells.rows + 1];

    for (wire, path) in wires.iter().zip(paths) {
        let Path::Channel { channel, first, last } = *path else {
            continue;
        };

        let down = cells.cells[wire.to].row >= cells.cells[wire.from].row;
        let runs = match inside_channel(wire, channel, frames) {
            true => &mut inside_runs[channel],
            false => &mut outside_runs[channel],
        };
        let existing = runs.iter_mut().find(|run| run.key == wire.from);

        match existing {
            Some(run) => {
                run.first = run.first.min(first);
                run.last = run.last.max(last);
            },
            None => runs.push(Run {
                key: wire.from,
                first,
                last,
                down,
            }),
        }
    }

    ChannelTracks {
        inside: order_runs(inside_runs),
        outside: order_runs(outside_runs),
    }
}

/// Orders one channel's runs into tracks, keyed by the exit that owns them.
fn order_runs(runs: Vec<Vec<Run>>) -> Vec<HashMap<usize, usize>> {
    runs.into_iter()
        .map(|mut channel_runs| {
            channel_runs.sort_by_key(|run| {
                let span = run.last - run.first;
                let signed = match run.down {
                    true => span as isize,
                    false => -(span as isize),
                };

                (!run.down, signed, run.key)
            });

            channel_runs
                .iter()
                .enumerate()
                .map(|(track, run)| (run.key, track))
                .collect()
        })
        .collect()
}

fn gutter_passes(
    cells: &Cells,
    wires: &[Wire],
    paths: &[Path],
    frames: &[FrameCells],
    channel_tracks: &ChannelTracks,
) -> Vec<Vec<Pass>> {
    let gutters = cells.columns.saturating_sub(1);
    let mut passes: Vec<Vec<Pass>> = vec![Vec::new(); gutters];

    for (wire, path) in wires.iter().zip(paths) {
        let key = wire.from;
        let leaving = Line::Port(cells.cells[wire.from].row);
        let arriving = entry_line(cells, wire);

        match *path {
            Path::Straight => {},
            Path::Turn { gutter } => passes[gutter].push(Pass {
                key,
                leave: leaving,
                arrive: arriving,
            }),
            Path::Channel { channel, first, last } => {
                let inside = inside_channel(wire, channel, frames);
                let run = match inside {
                    true => Line::InsideChannel(channel, channel_tracks.inside[channel][&key]),
                    false => Line::Channel(channel, channel_tracks.outside[channel][&key]),
                };

                passes[first].push(Pass {
                    key,
                    leave: leaving,
                    arrive: run,
                });
                passes[last].push(Pass {
                    key,
                    leave: run,
                    arrive: arriving,
                });
            },
        }
    }

    passes
}

/// Orders one gutter's tracks, keyed by the exit that owns them.
///
/// A line arriving on another line's leaving row goes after it, so neither runs along the other.
/// When no line is free to go next, the earliest declared exit goes.
fn gutter_order(passes: &[Pass]) -> HashMap<usize, usize> {
    let keys: BTreeSet<usize> = passes.iter().map(|pass| pass.key).collect();
    let mut after: HashMap<usize, BTreeSet<usize>> = HashMap::new();

    for arriving in passes {
        for leaving in passes {
            if arriving.key != leaving.key && arriving.arrive == leaving.leave {
                after.entry(arriving.key).or_default().insert(leaving.key);
            }
        }
    }

    let mut remaining = keys;
    let mut order = HashMap::new();

    while !remaining.is_empty() {
        let ready = remaining
            .iter()
            .copied()
            .find(|key| after.get(key).is_none_or(|before| before.is_disjoint(&remaining)));
        let first_left = *remaining.first().expect("remaining is not empty");
        let next = ready.unwrap_or(first_left);

        let track = order.len();
        order.insert(next, track);
        remaining.remove(&next);
    }

    order
}

/// Every column's position, and every gutter's layout.
struct Columns {
    x: Vec<usize>,
    widths: Vec<usize>,
    right_reserve: Vec<usize>,
    left_reserve: Vec<usize>,
    width: usize,
}

impl Columns {
    /// Lays the columns out left to right.
    ///
    /// A gutter holds its right frame reserve, a run, its tracks, a run, its longest label and its
    /// left frame reserve, in that order.
    fn new(
        cells: &Cells,
        wires: &[Wire],
        frames: &[FrameCells],
        widths: &[usize],
        gutter_tracks: &[HashMap<usize, usize>],
        planned: bool,
    ) -> Self {
        let margin = match planned {
            true => FRAME_RESERVE,
            false => 0,
        };
        let gutters = cells.columns.saturating_sub(1);

        let mut right_reserve = vec![0; gutters];
        let mut left_reserve = vec![0; gutters];
        let mut labels = vec![0; gutters];

        for gutter in 0..gutters {
            let right_edge = frames.iter().any(|frame| frame.last_column == gutter);
            let left_edge = frames.iter().any(|frame| frame.first_column == gutter + 1);

            right_reserve[gutter] = usize::from(right_edge) * FRAME_RESERVE;
            left_reserve[gutter] = usize::from(left_edge) * FRAME_RESERVE;
        }

        for wire in wires {
            let target_column = cells.cells[wire.to].column;

            if target_column > 0 {
                let gutter = target_column - 1;
                labels[gutter] = labels[gutter].max(wire.label);
            }
        }

        let mut x = Vec::with_capacity(cells.columns);
        let mut next = margin;

        for (column, width) in widths.iter().enumerate() {
            x.push(next);
            next += width;

            if column < gutters {
                let tracks = gutter_tracks[column].len();
                next += right_reserve[column] + 1 + tracks + 1 + labels[column] + left_reserve[column];
            }
        }

        Self {
            x,
            widths: widths.to_vec(),
            right_reserve,
            left_reserve,
            width: next + margin,
        }
    }

    fn track_x(&self, gutter: usize, track: usize) -> usize {
        let start = self.x[gutter] + self.widths[gutter];
        start + self.right_reserve[gutter] + 1 + track
    }

    /// The cell a label ending in this gutter stops just before.
    fn label_end(&self, gutter: usize) -> usize {
        self.x[gutter + 1] - self.left_reserve[gutter]
    }
}

/// Every row's position, and every channel's layout.
struct Rows {
    top: Vec<usize>,
    channel_start: Vec<usize>,
    inside_area: Vec<usize>,
    bottom_reserve: Vec<usize>,
    top_reserve: Vec<usize>,
    height: usize,
}

impl Rows {
    /// Lays the rows out top to bottom.
    ///
    /// A channel holds its inside area, its bottom frame reserve, its tracks with a row of padding
    /// either side, and its top frame reserve, in that order. The inside area holds the tracks running
    /// inside a frame's bottom edge, each on their own row starting one row below the row above, and
    /// ends with one padding row before the bottom reserve. It is empty when the channel holds none.
    /// A channel between two rows holding no tracks keeps one empty row so boxes never touch.
    fn new(
        cells: &Cells,
        frames: &[FrameCells],
        heights: &[usize],
        channel_tracks: &ChannelTracks,
        planned: bool,
    ) -> Self {
        let channels = cells.rows + 1;
        let mut inside_area = vec![0; channels];
        let mut bottom_reserve = vec![0; channels];
        let mut top_reserve = vec![0; channels];
        let mut channel_heights = vec![0; channels];

        for channel in 0..channels {
            let bottom_edge = planned && frames.iter().any(|frame| frame.bottom_row + 1 == channel);
            let top_frames: Vec<&FrameCells> =
                frames.iter().filter(|frame| frame.top_row == channel).collect();
            let detail = top_frames.iter().any(|frame| frame.detail);

            let inside_tracks = channel_tracks.inside[channel].len();
            inside_area[channel] = match inside_tracks {
                0 => 0,
                count => count + 1,
            };

            bottom_reserve[channel] = usize::from(bottom_edge) * FRAME_RESERVE;
            top_reserve[channel] = match planned && !top_frames.is_empty() {
                true => FRAME_RESERVE + usize::from(detail),
                false => 0,
            };

            let outside_tracks = channel_tracks.outside[channel].len();
            let between_rows = channel > 0 && channel < cells.rows;
            let middle = match (outside_tracks, between_rows) {
                (0, true) => 1,
                (0, false) => 0,
                (count, _) => count + 2,
            };

            channel_heights[channel] =
                inside_area[channel] + bottom_reserve[channel] + middle + top_reserve[channel];
        }

        let mut top = Vec::with_capacity(cells.rows);
        let mut channel_start = Vec::with_capacity(channels);
        let mut next = 0;

        for (row, height) in heights.iter().enumerate() {
            channel_start.push(next);
            next += channel_heights[row];
            top.push(next);
            next += height;
        }

        channel_start.push(next);
        next += channel_heights[cells.rows];

        Self {
            top,
            channel_start,
            inside_area,
            bottom_reserve,
            top_reserve,
            height: next,
        }
    }

    /// A track in the open middle of a channel.
    fn track_y(&self, channel: usize, track: usize) -> usize {
        self.channel_start[channel] + self.inside_area[channel] + self.bottom_reserve[channel] + 1 + track
    }

    /// A track inside a frame's bottom edge, in the same channel.
    fn inside_track_y(&self, channel: usize, track: usize) -> usize {
        self.channel_start[channel] + 1 + track
    }

    /// The row a frame's bottom border sits on, when its bottom edge falls in this channel.
    fn bottom_border(&self, channel: usize) -> usize {
        self.channel_start[channel] + self.inside_area[channel] + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::text::cells::Cell;

    fn cells(placed: &[(usize, usize)]) -> Cells {
        let cells: Vec<Cell> = placed
            .iter()
            .map(|(column, row)| Cell {
                column: *column,
                row: *row,
            })
            .collect();
        let columns = placed.iter().map(|(column, _)| column + 1).max().unwrap_or(0);
        let rows = placed.iter().map(|(_, row)| row + 1).max().unwrap_or(0);

        Cells { cells, columns, rows }
    }

    fn size(width: usize, height: usize, port: usize) -> Size {
        Size {
            width,
            height,
            port,
            stacked: false,
        }
    }

    fn wire(from: usize, to: usize, entry: usize, label: usize) -> Wire {
        Wire {
            from,
            to,
            entry,
            label,
            region: None,
        }
    }

    fn inside(point: (usize, usize), rect: Rect) -> bool {
        let (x, y) = point;
        x >= rect.x && x <= rect.right() && y >= rect.y && y <= rect.bottom()
    }

    fn strictly_inside(point: (usize, usize), rect: Rect) -> bool {
        let (x, y) = point;
        x > rect.x && x < rect.right() && y > rect.y && y < rect.bottom()
    }

    #[test]
    fn a_link_along_one_row_is_a_straight_run_across_a_two_cell_gutter() {
        let placed = cells(&[(0, 0), (1, 0)]);
        let sizes = [size(9, 3, 1), size(7, 3, 1)];
        let wires = [wire(0, 1, 0, 0)];
        let geometry = lay_out(&placed, &sizes, &wires, &[]);

        assert_eq!(geometry.items[1].x, 11);
        assert_eq!(geometry.routes[0], vec![(8, 1), (11, 1)]);
    }

    #[test]
    fn a_column_is_as_wide_as_its_widest_item_and_every_item_fills_it() {
        let placed = cells(&[(0, 0), (0, 1)]);
        let sizes = [size(9, 3, 1), size(5, 3, 1)];
        let geometry = lay_out(&placed, &sizes, &[], &[]);

        assert_eq!(geometry.items[0].width, 9);
        assert_eq!(geometry.items[1].width, 9);
    }

    #[test]
    fn lines_from_different_exits_never_share_a_track() {
        let placed = cells(&[(0, 0), (0, 2), (1, 1)]);
        let sizes = [size(5, 3, 1), size(5, 3, 1), size(5, 5, 2)];
        let wires = [wire(0, 2, 0, 0), wire(1, 2, 1, 0)];
        let geometry = lay_out(&placed, &sizes, &wires, &[]);

        let first_turn = geometry.routes[0][1].0;
        let second_turn = geometry.routes[1][1].0;
        assert_ne!(first_turn, second_turn);
    }

    #[test]
    fn a_fan_out_from_one_exit_shares_its_turn() {
        let placed = cells(&[(0, 1), (1, 0), (1, 2)]);
        let sizes = [size(5, 3, 1), size(5, 3, 1), size(5, 3, 1)];
        let wires = [wire(0, 1, 0, 0), wire(0, 2, 0, 0)];
        let geometry = lay_out(&placed, &sizes, &wires, &[]);

        assert_eq!(geometry.routes[0][1].0, geometry.routes[1][1].0);
    }

    #[test]
    fn a_line_never_runs_along_another_lines_leaving_row() {
        let placed = cells(&[(0, 0), (0, 1), (1, 1), (1, 2)]);
        let sizes = [size(5, 3, 1); 4];
        let arriving_on_row_one = wire(0, 2, 0, 0);
        let leaving_row_one = wire(1, 3, 0, 0);
        let geometry = lay_out(&placed, &sizes, &[arriving_on_row_one, leaving_row_one], &[]);

        let arriving_turn = geometry.routes[0][1].0;
        let leaving_turn = geometry.routes[1][1].0;
        assert!(leaving_turn < arriving_turn);
    }

    #[test]
    fn a_gutter_widens_with_its_tracks_and_labels() {
        let placed = cells(&[(0, 0), (1, 1)]);
        let sizes = [size(5, 3, 1), size(5, 3, 1)];
        let wires = [wire(0, 1, 0, 5)];
        let geometry = lay_out(&placed, &sizes, &wires, &[]);

        assert_eq!(geometry.items[1].x, 5 + 1 + 1 + 1 + 5);
        assert_eq!(
            geometry.labels[0],
            Some((geometry.items[1].x, geometry.routes[0][3].1))
        );
    }

    #[test]
    fn a_link_spanning_columns_runs_in_a_channel_through_no_cell() {
        let placed = cells(&[(0, 0), (1, 0), (2, 0)]);
        let sizes = [size(5, 3, 1), size(5, 3, 1), size(5, 5, 2)];
        let wires = [wire(0, 1, 0, 0), wire(1, 2, 0, 0), wire(0, 2, 1, 0)];
        let geometry = lay_out(&placed, &sizes, &wires, &[]);

        let route = &geometry.routes[2];
        let cells_passed = crate::render::text::draw::route_cells(route);
        let middle = geometry.items[1];

        assert_eq!(route.len(), 6);
        assert!(cells_passed.iter().all(|(x, y, _)| !inside((*x, *y), middle)));
        assert!(route[2].1 > middle.bottom());
    }

    #[test]
    fn every_entry_sits_inside_its_target() {
        let placed = cells(&[(0, 0), (0, 1), (0, 2), (1, 1)]);
        let sizes = [size(5, 3, 1), size(5, 3, 1), size(5, 3, 1), size(5, 7, 3)];
        let wires = [wire(0, 3, 0, 0), wire(1, 3, 1, 0), wire(2, 3, 2, 0)];
        let geometry = lay_out(&placed, &sizes, &wires, &[]);
        let target = geometry.items[3];

        for route in &geometry.routes {
            let end = *route.last().expect("a route has an end");
            assert_eq!(end.0, target.x);
            assert!(end.1 > target.y && end.1 < target.bottom());
        }
    }

    #[test]
    fn a_frame_sits_in_the_reserved_space_around_its_cells() {
        let placed = cells(&[(0, 0), (1, 0)]);
        let sizes = [size(5, 3, 1), size(5, 3, 1)];
        let wires = [wire(0, 1, 0, 0)];
        let frames = [
            FrameCells {
                first_column: 0,
                last_column: 0,
                top_row: 0,
                bottom_row: 0,
                detail: true,
            },
            FrameCells {
                first_column: 1,
                last_column: 1,
                top_row: 0,
                bottom_row: 0,
                detail: false,
            },
        ];
        let geometry = lay_out(&placed, &sizes, &wires, &frames);

        let first_box = geometry.items[0];
        let first_frame = geometry.frames[0];
        let second_frame = geometry.frames[1];

        assert_eq!(first_frame.x + 2, first_box.x);
        assert_eq!(first_frame.right(), first_box.right() + 2);
        assert_eq!(first_frame.y + 3, first_box.y);
        assert!(first_frame.right() < second_frame.x);
    }

    #[test]
    fn a_skip_link_along_a_regions_bottom_row_stays_inside_its_frame() {
        let placed = cells(&[(0, 0), (1, 0), (2, 0)]);
        let sizes = [size(5, 3, 1), size(5, 3, 1), size(5, 3, 1)];
        let skip = Wire {
            from: 0,
            to: 2,
            entry: 1,
            label: 0,
            region: Some(0),
        };
        let wires = [wire(0, 1, 0, 0), wire(1, 2, 0, 0), skip];
        let frames = [FrameCells {
            first_column: 0,
            last_column: 2,
            top_row: 0,
            bottom_row: 0,
            detail: false,
        }];
        let geometry = lay_out(&placed, &sizes, &wires, &frames);

        let route = &geometry.routes[2];
        let cells_passed = crate::render::text::draw::route_cells(route);
        let frame = geometry.frames[0];

        assert!(
            cells_passed
                .iter()
                .all(|(x, y, _)| strictly_inside((*x, *y), frame))
        );
    }

    #[test]
    fn a_longer_channel_run_sits_further_from_its_row() {
        let placed = cells(&[(0, 0), (1, 0), (3, 1)]);
        let sizes = [size(5, 3, 1), size(5, 3, 1), size(5, 3, 1)];
        let wires = [wire(0, 2, 0, 0), wire(1, 2, 1, 0)];
        let geometry = lay_out(&placed, &sizes, &wires, &[]);

        let longer_run_y = geometry.routes[0][2].1;
        let shorter_run_y = geometry.routes[1][2].1;
        assert!(longer_run_y > shorter_run_y);
    }

    #[test]
    fn an_upward_far_link_uses_the_channel_above() {
        let placed = cells(&[(0, 1), (1, 1), (3, 0)]);
        let sizes = [size(5, 3, 1), size(5, 3, 1), size(5, 3, 1)];
        let wires = [wire(0, 2, 0, 0)];
        let geometry = lay_out(&placed, &sizes, &wires, &[]);

        let run_y = geometry.routes[0][2].1;
        assert!(run_y < geometry.items[0].y);
    }

    #[test]
    fn stacked_frames_share_a_channel_without_touching() {
        let placed = cells(&[(0, 0), (0, 1)]);
        let sizes = [size(5, 3, 1), size(5, 3, 1)];
        let frames = [
            FrameCells {
                first_column: 0,
                last_column: 0,
                top_row: 0,
                bottom_row: 0,
                detail: false,
            },
            FrameCells {
                first_column: 0,
                last_column: 0,
                top_row: 1,
                bottom_row: 1,
                detail: false,
            },
        ];
        let geometry = lay_out(&placed, &sizes, &[], &frames);

        let first_frame = geometry.frames[0];
        let second_frame = geometry.frames[1];

        assert!(first_frame.bottom() < second_frame.y);
    }

    #[test]
    fn a_label_ends_before_a_left_frame_border() {
        let placed = cells(&[(0, 0), (1, 0)]);
        let sizes = [size(5, 3, 1), size(5, 3, 1)];
        let wires = [wire(0, 1, 0, 5)];
        let frames = [FrameCells {
            first_column: 1,
            last_column: 1,
            top_row: 0,
            bottom_row: 0,
            detail: false,
        }];
        let geometry = lay_out(&placed, &sizes, &wires, &frames);

        let label = geometry.labels[0].expect("the link is labelled");
        assert_eq!(label.0, geometry.frames[0].x);
    }

    #[test]
    fn a_turn_track_sits_after_the_right_frame_reserve() {
        let placed = cells(&[(0, 0), (1, 0)]);
        let sizes = [size(5, 3, 1), size(5, 3, 1)];
        let wires = [wire(0, 1, 1, 0)];
        let frames = [FrameCells {
            first_column: 0,
            last_column: 0,
            top_row: 0,
            bottom_row: 0,
            detail: false,
        }];
        let geometry = lay_out(&placed, &sizes, &wires, &frames);

        let column_end = geometry.items[0].x + geometry.items[0].width;
        assert_eq!(geometry.routes[0][1].0, column_end + 2 + 1);
    }
}
