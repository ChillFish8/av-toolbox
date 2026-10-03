use std::fmt::Write;
use std::path::PathBuf;
use std::time::Duration;

/// How many characters a full share bar takes.
const BAR_WIDTH: f64 = 20.0;

/// One stage's measured cost per frame.
pub(crate) struct Row {
    pub(crate) label: String,
    pub(crate) host: Duration,
    pub(crate) device: Duration,
    pub(crate) busy: Duration,
    /// `None` for a node that takes no width of its own.
    pub(crate) workers: Option<WorkersCell>,
}

/// The baseline and suggested worker counts of one stage.
pub(crate) struct WorkersCell {
    /// The width the baseline pass planned, inherited widths included.
    pub(crate) baseline: usize,
    /// The width the pipeline itself declares, if any.
    pub(crate) declared: Option<usize>,
    pub(crate) suggested: usize,
}

/// Everything the breakdown report prints. All durations are per frame.
pub(crate) struct Summary {
    pub(crate) device: String,
    pub(crate) frames: u64,
    pub(crate) rows: Vec<Row>,
    pub(crate) overhead: Duration,
    pub(crate) floor: Duration,
    pub(crate) baseline_fps: f64,
    pub(crate) tuned_fps: f64,
    pub(crate) kept_tuned: bool,
    pub(crate) discarded: Vec<String>,
    pub(crate) written: PathBuf,
}

/// The breakdown table and the frame rates, as printed to stdout.
pub(crate) fn render(summary: &Summary, bars: bool) -> String {
    let mut output = String::new();
    let total_busy: Duration = summary.rows.iter().map(|row| row.busy).sum();

    let _ = writeln!(output, "Profiled {} frames on {}", summary.frames, summary.device);
    let _ = writeln!(output);
    let _ = writeln!(
        output,
        "{:<24} {:>9} {:>9} {:>9} {:>7}   workers",
        "stage", "host ms", "device ms", "busy ms", "share"
    );

    for row in &summary.rows {
        let line = row_line(row, total_busy, bars);
        let _ = writeln!(output, "{line}");
    }

    let overhead = milliseconds(summary.overhead);
    let _ = writeln!(
        output,
        "{:<24} {:>9} {:>9} {:>9.2}",
        "runtime overhead", "", "", overhead
    );
    let _ = writeln!(output);

    let floor = milliseconds(summary.floor);
    let change = percent_change(summary.baseline_fps, summary.tuned_fps);
    let _ = writeln!(output, "device floor: {floor:.2} ms/frame");
    let _ = writeln!(
        output,
        "baseline:  {:.1} fps (declared widths)",
        summary.baseline_fps
    );
    let _ = writeln!(output, "tuned:     {:.1} fps ({change:+.1} %)", summary.tuned_fps);

    if !summary.kept_tuned {
        let _ = writeln!(
            output,
            concat!(
                "the suggested widths did not help here, ",
                "so the baseline's widths were written"
            )
        );
    }

    if !summary.discarded.is_empty() {
        let names = summary.discarded.join(", ");
        let _ = writeln!(output, "sinks replaced by null/discard: {names}");
    }

    let _ = writeln!(output, "wrote {}", summary.written.display());

    output
}

fn row_line(row: &Row, total_busy: Duration, bars: bool) -> String {
    let share = share_of(row.busy, total_busy);
    let workers = workers_text(row.workers.as_ref());
    let host = milliseconds(row.host);
    let device = milliseconds(row.device);
    let busy = milliseconds(row.busy);

    let mut line = format!(
        "{:<24} {host:>9.2} {device:>9.2} {busy:>9.2} {share:>5.1} %   {workers}",
        row.label
    );

    if bars {
        let length = (share / 100.0 * BAR_WIDTH).round() as usize;
        let bar = "#".repeat(length);
        line.push_str("  ");
        line.push_str(&bar);
    }

    line
}

fn workers_text(cell: Option<&WorkersCell>) -> String {
    let Some(cell) = cell else {
        return "-".to_string();
    };

    let differs = cell.declared.is_some_and(|declared| declared != cell.suggested);
    if differs {
        format!("{} -> {} (declared)", cell.baseline, cell.suggested)
    } else {
        format!("{} -> {}", cell.baseline, cell.suggested)
    }
}

fn share_of(busy: Duration, total: Duration) -> f64 {
    if total.is_zero() {
        return 0.0;
    }

    busy.as_secs_f64() / total.as_secs_f64() * 100.0
}

fn percent_change(baseline: f64, tuned: f64) -> f64 {
    if baseline == 0.0 {
        return 0.0;
    }

    (tuned - baseline) / baseline * 100.0
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn milliseconds(value: f64) -> Duration {
        Duration::from_secs_f64(value / 1000.0)
    }

    fn summary() -> Summary {
        let source = Row {
            label: "source.clip".to_string(),
            host: milliseconds(6.2),
            device: Duration::ZERO,
            busy: milliseconds(6.2),
            workers: None,
        };
        let filter = Row {
            label: "filter.based".to_string(),
            host: milliseconds(0.9),
            device: milliseconds(71.2),
            busy: milliseconds(72.1),
            workers: Some(WorkersCell {
                baseline: 2,
                declared: Some(2),
                suggested: 1,
            }),
        };

        Summary {
            device: "vulkan discrete:1".to_string(),
            frames: 300,
            rows: vec![source, filter],
            overhead: milliseconds(1.3),
            floor: milliseconds(71.3),
            baseline_fps: 13.4,
            tuned_fps: 13.9,
            kept_tuned: true,
            discarded: vec!["sink.output".to_string()],
            written: PathBuf::from("clip.profile.toml"),
        }
    }

    #[test]
    fn the_report_names_the_device_and_frames() {
        let rendered = render(&summary(), false);

        assert!(
            rendered.starts_with("Profiled 300 frames on vulkan discrete:1\n"),
            "{rendered}"
        );
    }

    #[test]
    fn each_row_shows_its_share_of_the_busy_time() {
        let rendered = render(&summary(), false);

        let filter_line = rendered.lines().find(|line| line.starts_with("filter.based"));
        let filter_line = filter_line.expect("the filter has a row");
        assert!(filter_line.contains("92.1 %"), "{filter_line}");
        assert!(filter_line.contains("72.10"), "{filter_line}");
    }

    #[test]
    fn a_declared_width_differing_from_the_suggestion_is_marked() {
        let rendered = render(&summary(), false);

        assert!(rendered.contains("2 -> 1 (declared)"), "{rendered}");
    }

    #[test]
    fn an_inherited_baseline_width_is_shown_without_the_declared_mark() {
        let cell = WorkersCell {
            baseline: 3,
            declared: None,
            suggested: 1,
        };

        assert_eq!(workers_text(Some(&cell)), "3 -> 1");
    }

    #[test]
    fn the_frame_rates_and_their_change_are_shown() {
        let rendered = render(&summary(), false);

        assert!(rendered.contains("device floor: 71.30 ms/frame"), "{rendered}");
        assert!(rendered.contains("baseline:  13.4 fps"), "{rendered}");
        assert!(rendered.contains("tuned:     13.9 fps (+3.7 %)"), "{rendered}");
        assert!(rendered.contains("wrote clip.profile.toml"), "{rendered}");
    }

    #[test]
    fn replaced_sinks_are_named() {
        let rendered = render(&summary(), false);

        assert!(
            rendered.contains("sinks replaced by null/discard: sink.output"),
            "{rendered}"
        );
    }

    #[test]
    fn a_tuned_run_that_did_not_help_says_so() {
        let mut summary = summary();
        summary.kept_tuned = false;

        let rendered = render(&summary, false);

        assert!(
            rendered.contains("the suggested widths did not help"),
            "{rendered}"
        );
    }

    #[test]
    fn bars_are_only_drawn_when_asked() {
        let plain = render(&summary(), false);
        let barred = render(&summary(), true);

        assert!(!plain.contains('#'));
        assert!(barred.contains('#'));
    }
}
