use av_toolbox_gpu::{Attachments, MetaValue};

/// Every score a stage gave, for the statistics it writes at the end of the stream.
#[derive(Debug, Default)]
pub(crate) struct Summary {
    scores: Vec<f64>,
}

impl Summary {
    pub(crate) fn push(&mut self, score: f64) {
        self.scores.push(score);
    }

    pub(crate) fn frames(&self) -> usize {
        self.scores.len()
    }

    pub(crate) fn mean(&self) -> Option<f64> {
        if self.scores.is_empty() {
            return None;
        }

        let total: f64 = self.scores.iter().sum();

        Some(total / self.scores.len() as f64)
    }

    pub(crate) fn min(&self) -> Option<f64> {
        self.scores.iter().copied().reduce(f64::min)
    }

    pub(crate) fn max(&self) -> Option<f64> {
        self.scores.iter().copied().reduce(f64::max)
    }

    /// The score at `fraction` of the way through the sorted scores, by nearest rank.
    pub(crate) fn percentile(&self, fraction: f64) -> Option<f64> {
        if self.scores.is_empty() {
            return None;
        }

        let mut sorted = self.scores.clone();
        sorted.sort_by(f64::total_cmp);

        let last = (sorted.len() - 1) as f64;
        let rank = (fraction * last).round() as usize;

        sorted.get(rank).copied()
    }

    /// Attaches `frames`, then `mean`, `min`, `max`, `p5` and `p95` each named with `prefix` in front.
    ///
    /// With no scores, only `frames` is attached.
    pub(crate) fn write(&self, record: &mut Attachments, prefix: &str) {
        let frames = i64::try_from(self.frames()).unwrap_or(i64::MAX);
        record.attach("frames", MetaValue::Int(frames));

        let statistics = [
            ("mean", self.mean()),
            ("min", self.min()),
            ("max", self.max()),
            ("p5", self.percentile(0.05)),
            ("p95", self.percentile(0.95)),
        ];
        for (name, value) in statistics {
            let Some(value) = value else {
                continue;
            };

            record.attach(format!("{prefix}{name}"), MetaValue::Float(value));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summary_of(scores: &[f64]) -> Summary {
        let mut summary = Summary::default();
        for score in scores {
            summary.push(*score);
        }

        summary
    }

    /// The values `summary` writes under `prefix`, in the order it wrote them.
    fn written(summary: &Summary, prefix: &str) -> Vec<(String, MetaValue)> {
        let mut record = Attachments::new();
        summary.write(&mut record, prefix);
        record.stamp("vship/test");

        record
            .stages()
            .flat_map(|(_, values)| values.iter().cloned())
            .collect()
    }

    #[test]
    fn known_scores_give_their_statistics() {
        let scores: Vec<f64> = (1..=21).map(f64::from).collect();
        let summary = summary_of(&scores);

        assert_eq!(summary.frames(), 21);
        assert_eq!(summary.mean(), Some(11.0));
        assert_eq!(summary.min(), Some(1.0));
        assert_eq!(summary.max(), Some(21.0));
        assert_eq!(summary.percentile(0.05), Some(2.0));
        assert_eq!(summary.percentile(0.95), Some(20.0));
    }

    #[test]
    fn unsorted_scores_are_ranked_by_value() {
        let summary = summary_of(&[90.0, 10.0, 50.0]);

        assert_eq!(summary.min(), Some(10.0));
        assert_eq!(summary.max(), Some(90.0));
        assert_eq!(summary.percentile(0.5), Some(50.0));
        assert_eq!(summary.percentile(0.0), Some(10.0));
        assert_eq!(summary.percentile(1.0), Some(90.0));
    }

    #[test]
    fn every_statistic_is_written() {
        let summary = summary_of(&[2.0, 4.0]);

        let values = written(&summary, "");

        let expected = vec![
            ("frames".to_string(), MetaValue::Int(2)),
            ("mean".to_string(), MetaValue::Float(3.0)),
            ("min".to_string(), MetaValue::Float(2.0)),
            ("max".to_string(), MetaValue::Float(4.0)),
            ("p5".to_string(), MetaValue::Float(2.0)),
            ("p95".to_string(), MetaValue::Float(4.0)),
        ];
        assert_eq!(values, expected);
    }

    #[test]
    fn an_empty_summary_writes_only_frames() {
        let summary = Summary::default();

        let values = written(&summary, "");

        assert_eq!(summary.mean(), None);
        assert_eq!(values, vec![("frames".to_string(), MetaValue::Int(0))]);
    }

    #[test]
    fn a_prefix_is_applied_to_every_statistic() {
        let summary = summary_of(&[1.0]);

        let values = written(&summary, "norm_q_");

        let names: Vec<&str> = values.iter().map(|(name, _)| name.as_str()).collect();
        let expected = [
            "frames",
            "norm_q_mean",
            "norm_q_min",
            "norm_q_max",
            "norm_q_p5",
            "norm_q_p95",
        ];
        assert_eq!(names, expected);
    }
}
