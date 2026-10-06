use std::time::Duration;

use av_toolbox_core::Granularity;

/// How much faster than the floor a widened stage aims to keep up.
const HEADROOM: f64 = 1.25;

/// How close to the baseline's frame rate a tuned run must come for its widths to be kept.
const KEEP_WITHIN: f64 = 0.98;

/// One node's measured cost per frame.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NodeCost {
    pub(crate) key: String,
    /// `None` for a node that takes no width of its own.
    pub(crate) granularity: Option<Granularity>,
    pub(crate) host: Duration,
    pub(crate) busy: Duration,
}

impl NodeCost {
    pub(crate) fn device(&self) -> Duration {
        self.busy.saturating_sub(self.host)
    }

    fn widenable(&self) -> bool {
        matches!(self.granularity, Some(Granularity::Frame | Granularity::Chunk))
    }
}

/// How many workers one stage runs.
#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct Width {
    pub(crate) key: String,
    pub(crate) workers: usize,
    pub(crate) host: Duration,
    /// Whether the pipeline declares this stage's workers, so a profile cannot change them.
    pub(crate) declared: bool,
}

/// The time per frame no width can beat.
///
/// The device runs one dispatch at a time, so every node's device time adds up. A stage that
/// cannot widen is a floor of its own.
pub(crate) fn floor(costs: &[NodeCost]) -> Duration {
    let device: Duration = costs.iter().map(NodeCost::device).sum();

    let streamed = costs
        .iter()
        .filter(|cost| cost.granularity == Some(Granularity::Stream))
        .map(|cost| cost.host)
        .max()
        .unwrap_or_default();

    device.max(streamed)
}

/// The fewest workers keeping each widenable stage's host time under the floor.
pub(crate) fn suggest(costs: &[NodeCost], max_workers: usize) -> Vec<Width> {
    let floor = floor(costs);

    costs
        .iter()
        .filter(|cost| cost.widenable())
        .map(|cost| Width {
            key: cost.key.clone(),
            workers: workers_for(cost.host, floor, max_workers),
            host: cost.host,
            declared: false,
        })
        .collect()
}

fn workers_for(host: Duration, floor: Duration, max_workers: usize) -> usize {
    if floor.is_zero() {
        return 1;
    }

    let wanted = host.as_secs_f64() * HEADROOM / floor.as_secs_f64();
    let rounded = wanted.ceil() as usize;
    let ceiling = max_workers.max(1);

    rounded.clamp(1, ceiling)
}

/// Takes one worker from the widest undeclared stage, ties going to the larger host time.
///
/// Returns false once no undeclared stage is wider than one.
pub(crate) fn step_down(widths: &mut [Width]) -> bool {
    let widest = widths
        .iter_mut()
        .filter(|width| width.workers > 1 && !width.declared)
        .max_by(|left, right| {
            let by_workers = left.workers.cmp(&right.workers);
            by_workers.then(left.host.cmp(&right.host))
        });

    let Some(widest) = widest else {
        return false;
    };

    widest.workers -= 1;
    true
}

pub(crate) fn keeps_tuned(baseline_fps: f64, tuned_fps: f64) -> bool {
    tuned_fps >= baseline_fps * KEEP_WITHIN
}

#[cfg(test)]
mod tests {
    use super::*;

    fn milliseconds(value: u64) -> Duration {
        Duration::from_millis(value)
    }

    fn cost(key: &str, granularity: Option<Granularity>, host: u64, busy: u64) -> NodeCost {
        NodeCost {
            key: key.to_string(),
            granularity,
            host: milliseconds(host),
            busy: milliseconds(busy),
        }
    }

    fn widenable(key: &str, host: u64, busy: u64) -> NodeCost {
        cost(key, Some(Granularity::Frame), host, busy)
    }

    fn workers_of(widths: &[Width], key: &str) -> usize {
        let found = widths.iter().find(|width| width.key == key);
        found.expect("the stage is sized").workers
    }

    fn width(key: &str, workers: usize, host: u64, declared: bool) -> Width {
        Width {
            key: key.to_string(),
            workers,
            host: milliseconds(host),
            declared,
        }
    }

    #[test]
    fn the_floor_is_the_device_time_when_that_is_largest() {
        let costs = vec![
            widenable("filter.a", 1, 11),
            cost("source.clip", Some(Granularity::Stream), 4, 4),
        ];

        assert_eq!(floor(&costs), milliseconds(10));
    }

    #[test]
    fn the_floor_is_a_slow_stream_stage_when_that_is_largest() {
        let costs = vec![
            widenable("filter.a", 1, 11),
            cost("source.clip", Some(Granularity::Stream), 30, 30),
        ];

        assert_eq!(floor(&costs), milliseconds(30));
    }

    #[test]
    fn a_host_bound_stage_is_widened_to_keep_up_with_the_floor() {
        let costs = vec![widenable("filter.gpu", 1, 11), widenable("filter.cpu", 25, 25)];

        let widths = suggest(&costs, 16);

        assert_eq!(workers_of(&widths, "filter.cpu"), 4);
        assert_eq!(workers_of(&widths, "filter.gpu"), 1);
    }

    #[test]
    fn widths_are_clamped_to_the_most_workers_allowed() {
        let costs = vec![widenable("filter.gpu", 1, 2), widenable("filter.cpu", 100, 100)];

        let widths = suggest(&costs, 3);

        assert_eq!(workers_of(&widths, "filter.cpu"), 3);
    }

    #[test]
    fn a_zero_floor_leaves_every_stage_at_one() {
        let costs = vec![widenable("filter.a", 0, 0), widenable("filter.b", 0, 0)];

        let widths = suggest(&costs, 8);

        assert_eq!(workers_of(&widths, "filter.a"), 1);
        assert_eq!(workers_of(&widths, "filter.b"), 1);
    }

    #[test]
    fn stream_stages_and_transfers_are_not_sized() {
        let costs = vec![
            widenable("filter.a", 1, 11),
            cost("filter.ordered", Some(Granularity::Stream), 1, 1),
            cost("download -> sink.out", None, 1, 2),
        ];

        let widths = suggest(&costs, 8);

        assert_eq!(widths.len(), 1);
        assert_eq!(widths[0].key, "filter.a");
    }

    #[test]
    fn stepping_down_takes_from_the_widest_then_the_slowest() {
        let mut widths = vec![
            Width {
                key: "filter.a".to_string(),
                workers: 3,
                host: milliseconds(5),
                declared: false,
            },
            Width {
                key: "filter.b".to_string(),
                workers: 3,
                host: milliseconds(9),
                declared: false,
            },
        ];

        assert!(step_down(&mut widths));

        assert_eq!(workers_of(&widths, "filter.a"), 3);
        assert_eq!(workers_of(&widths, "filter.b"), 2);
    }

    #[test]
    fn stepping_down_takes_from_the_widest_even_when_a_narrower_stage_is_slower() {
        let mut widths = vec![
            width("filter.wide", 3, 5, false),
            width("filter.slow", 2, 90, false),
        ];

        assert!(step_down(&mut widths));

        assert_eq!(workers_of(&widths, "filter.wide"), 2);
        assert_eq!(workers_of(&widths, "filter.slow"), 2);
    }

    #[test]
    fn a_declared_wide_stage_is_never_stepped_down() {
        let mut widths = vec![
            width("filter.declared", 4, 50, true),
            width("filter.free", 2, 5, false),
        ];

        assert!(step_down(&mut widths));
        assert_eq!(workers_of(&widths, "filter.declared"), 4);
        assert_eq!(workers_of(&widths, "filter.free"), 1);

        assert!(!step_down(&mut widths));
        assert_eq!(workers_of(&widths, "filter.declared"), 4);
    }

    #[test]
    fn stepping_down_stops_once_every_stage_is_at_one() {
        let mut widths = vec![Width {
            key: "filter.a".to_string(),
            workers: 1,
            host: milliseconds(5),
            declared: false,
        }];

        assert!(!step_down(&mut widths));
    }

    #[test]
    fn the_tuned_widths_are_kept_within_two_percent_of_the_baseline() {
        assert!(keeps_tuned(10.0, 9.85));
        assert!(!keeps_tuned(10.0, 9.7));
    }
}
