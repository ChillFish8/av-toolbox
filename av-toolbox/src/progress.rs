use std::io::IsTerminal;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use av_toolbox_runtime::Progress;
use console::Term;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};

use crate::cli::ProgressMode;

/// How often the live view redraws.
const LIVE_INTERVAL: Duration = Duration::from_millis(250);

/// How often a progress line is logged when stderr is not a terminal.
const LOGGED_INTERVAL: Duration = Duration::from_secs(10);

/// How many times a second a forced live view may draw.
const FORCED_DRAW_HZ: u8 = 4;

/// How a run's progress is shown.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum View {
    Live,
    Logged,
}

fn view_for(mode: ProgressMode, is_terminal: bool) -> Option<View> {
    match (mode, is_terminal) {
        (ProgressMode::Never, _) => None,
        (ProgressMode::Always, _) => Some(View::Live),
        (ProgressMode::Auto, true) => Some(View::Live),
        (ProgressMode::Auto, false) => Some(View::Logged),
    }
}

/// Formats `elapsed` as `HH:MM:SS`.
pub(crate) fn clock(elapsed: Duration) -> String {
    let seconds = elapsed.as_secs();
    let hours = seconds / 3600;
    let minutes = seconds / 60 % 60;
    let remainder = seconds % 60;

    format!("{hours:02}:{minutes:02}:{remainder:02}")
}

/// Formats `count` with a comma between every group of three digits.
fn grouped(count: u64) -> String {
    let digits = count.to_string();
    let mut output = String::with_capacity(digits.len() + digits.len() / 3);

    for (index, digit) in digits.chars().enumerate() {
        let remaining = digits.len() - index;
        if index > 0 && remaining.is_multiple_of(3) {
            output.push(',');
        }

        output.push(digit);
    }

    output
}

/// Frames per second over `elapsed`, zero before any time has passed.
fn average_fps(frames: u64, elapsed: Duration) -> f64 {
    if elapsed.is_zero() {
        return 0.0;
    }

    frames as f64 / elapsed.as_secs_f64()
}

/// One sink's name and how many frames it has written.
pub(crate) struct SinkRow {
    pub(crate) name: String,
    pub(crate) frames: u64,
}

/// A run's progress at one moment.
pub(crate) struct Snapshot {
    pub(crate) elapsed: Duration,
    pub(crate) read: u64,
    pub(crate) sinks: Vec<SinkRow>,
}

impl Snapshot {
    /// Reads `progress`, naming each sink from `names` in the same order.
    pub(crate) fn take(progress: &Progress, names: &[String], elapsed: Duration) -> Self {
        let sinks = progress
            .sinks()
            .iter()
            .zip(names)
            .map(|(sink, name)| SinkRow {
                name: name.clone(),
                frames: sink.steps,
            })
            .collect();

        Self {
            elapsed,
            read: progress.read(),
            sinks,
        }
    }

    fn header(&self) -> String {
        let elapsed = clock(self.elapsed);
        let read = grouped(self.read);

        format!("{elapsed} │ {read} frames read")
    }

    /// One live row per sink, with names padded so the counts line up.
    fn rows(&self) -> Vec<String> {
        let width = self
            .sinks
            .iter()
            .map(|sink| sink.name.chars().count())
            .max()
            .unwrap_or(0);

        self.sinks
            .iter()
            .map(|sink| {
                let frames = grouped(sink.frames);
                let fps = average_fps(sink.frames, self.elapsed);

                format!("  {:<width$}  {frames} frames │ {fps:.1} fps", sink.name)
            })
            .collect()
    }

    /// One log line per sink.
    pub(crate) fn sink_lines(&self) -> Vec<String> {
        self.sinks
            .iter()
            .map(|sink| {
                let frames = grouped(sink.frames);
                let fps = average_fps(sink.frames, self.elapsed);

                format!("{}: {frames} frames at {fps:.1} fps", sink.name)
            })
            .collect()
    }

    /// The whole snapshot on one log line.
    fn summary(&self) -> String {
        let elapsed = clock(self.elapsed);
        let read = grouped(self.read);
        let mut summary = format!("{elapsed}, {read} frames read");

        for line in self.sink_lines() {
            summary.push_str(" | ");
            summary.push_str(&line);
        }

        summary
    }
}

/// A thread reporting a run's progress until it is stopped.
pub(crate) struct Reporter {
    stop: Sender<()>,
    handle: JoinHandle<()>,
}

impl Reporter {
    /// Starts reporting `progress` the way `mode` asks, or nothing for [ProgressMode::Never].
    ///
    /// `names` names each sink in the order
    /// [Progress::sinks](av_toolbox_runtime::Progress::sinks) lists them.
    pub(crate) fn start(
        mode: ProgressMode,
        bars: &MultiProgress,
        progress: Progress,
        names: Vec<String>,
    ) -> Option<Reporter> {
        let is_terminal = std::io::stderr().is_terminal();
        let view = view_for(mode, is_terminal)?;

        if mode == ProgressMode::Always {
            let term = Box::new(Term::stderr());
            let target = ProgressDrawTarget::term_like_with_hz(term, FORCED_DRAW_HZ);
            bars.set_draw_target(target);
        }

        let (stop, stopped) = mpsc::channel();
        let handle = match view {
            View::Live => {
                let live = LiveView::new(bars, names.len());
                thread::spawn(move || report_live(live, progress, names, stopped))
            },
            View::Logged => thread::spawn(move || report_logged(progress, names, stopped)),
        };

        Some(Reporter { stop, handle })
    }

    /// Stops the thread and waits for it, clearing any live view.
    pub(crate) fn stop(self) {
        let Reporter { stop, handle } = self;
        drop(stop);

        if handle.join().is_err() {
            tracing::warn!("the progress reporter panicked");
        }
    }
}

/// The bars of a live view, one header and one per sink.
struct LiveView {
    header: ProgressBar,
    rows: Vec<ProgressBar>,
}

impl LiveView {
    fn new(bars: &MultiProgress, sinks: usize) -> Self {
        let style = ProgressStyle::with_template("{msg}").expect("a fixed template is valid");
        let header = add_bar(bars, &style);
        let rows = (0..sinks).map(|_| add_bar(bars, &style)).collect();

        Self { header, rows }
    }

    fn show(&self, snapshot: &Snapshot) {
        let header = snapshot.header();
        self.header.set_message(header);

        let rows = snapshot.rows();
        for (bar, row) in self.rows.iter().zip(rows) {
            bar.set_message(row);
        }
    }

    fn clear(&self) {
        self.header.finish_and_clear();

        for bar in &self.rows {
            bar.finish_and_clear();
        }
    }
}

fn add_bar(bars: &MultiProgress, style: &ProgressStyle) -> ProgressBar {
    let bar = ProgressBar::new_spinner().with_style(style.clone());
    bars.add(bar)
}

/// Whether the reporter should keep going after waiting up to `interval` for a stop.
fn keep_going(stopped: &Receiver<()>, interval: Duration) -> bool {
    let waited = stopped.recv_timeout(interval);
    matches!(waited, Err(RecvTimeoutError::Timeout))
}

fn report_live(view: LiveView, progress: Progress, names: Vec<String>, stopped: Receiver<()>) {
    loop {
        let elapsed = progress.elapsed();
        let snapshot = Snapshot::take(&progress, &names, elapsed);
        view.show(&snapshot);

        if !keep_going(&stopped, LIVE_INTERVAL) {
            break;
        }
    }

    view.clear();
}

fn report_logged(progress: Progress, names: Vec<String>, stopped: Receiver<()>) {
    while keep_going(&stopped, LOGGED_INTERVAL) {
        let elapsed = progress.elapsed();
        let snapshot = Snapshot::take(&progress, &names, elapsed);
        let summary = snapshot.summary();

        tracing::info!("{summary}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot() -> Snapshot {
        Snapshot {
            elapsed: Duration::from_secs(133),
            read: 4812,
            sinks: vec![
                SinkRow {
                    name: "out".to_string(),
                    frames: 2406,
                },
                SinkRow {
                    name: "preview".to_string(),
                    frames: 2406,
                },
            ],
        }
    }

    #[test]
    fn clock_pads_hours_minutes_and_seconds() {
        assert_eq!(clock(Duration::from_secs(133)), "00:02:13");
    }

    #[test]
    fn clock_runs_past_an_hour() {
        assert_eq!(clock(Duration::from_secs(3 * 3600 + 61)), "03:01:01");
    }

    #[test]
    fn grouped_separates_thousands() {
        assert_eq!(grouped(0), "0");
        assert_eq!(grouped(999), "999");
        assert_eq!(grouped(4812), "4,812");
        assert_eq!(grouped(1_234_567), "1,234,567");
    }

    #[test]
    fn fps_is_zero_at_zero_elapsed() {
        assert_eq!(average_fps(10, Duration::ZERO), 0.0);
    }

    #[test]
    fn fps_divides_frames_by_seconds() {
        assert_eq!(average_fps(50, Duration::from_secs(2)), 25.0);
    }

    #[test]
    fn the_header_shows_elapsed_and_frames_read() {
        assert_eq!(snapshot().header(), "00:02:13 │ 4,812 frames read");
    }

    #[test]
    fn rows_pad_names_to_the_longest() {
        let rows = snapshot().rows();

        assert_eq!(rows[0], "  out      2,406 frames │ 18.1 fps");
        assert_eq!(rows[1], "  preview  2,406 frames │ 18.1 fps");
    }

    #[test]
    fn the_summary_names_every_sink() {
        let summary = snapshot().summary();

        assert_eq!(
            summary,
            "00:02:13, 4,812 frames read | out: 2,406 frames at 18.1 fps | preview: 2,406 frames at 18.1 fps"
        );
    }

    #[test]
    fn auto_draws_on_a_terminal() {
        assert_eq!(view_for(ProgressMode::Auto, true), Some(View::Live));
    }

    #[test]
    fn auto_logs_when_stderr_is_not_a_terminal() {
        assert_eq!(view_for(ProgressMode::Auto, false), Some(View::Logged));
    }

    #[test]
    fn always_draws_without_a_terminal() {
        assert_eq!(view_for(ProgressMode::Always, false), Some(View::Live));
    }

    #[test]
    fn never_shows_nothing() {
        assert_eq!(view_for(ProgressMode::Never, true), None);
        assert_eq!(view_for(ProgressMode::Never, false), None);
    }
}
