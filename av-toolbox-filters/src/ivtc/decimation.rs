use av_toolbox_cubek::ivtc::DecimateMetric;

/// VDecimate's thresholds in difference units.
pub(crate) struct Thresholds {
    pub(crate) duplicate: i64,
    pub(crate) scene: i64,
}

impl Thresholds {
    /// Scales the percentages `dupthresh` and `scthresh` the way VDecimate does.
    ///
    /// `dupthresh` is a share of one block of `block_x` by `block_y` samples, and `scthresh` a share of the
    /// whole luma plane. Each is truncated to a whole number.
    pub(crate) fn new(
        dupthresh: f64,
        scthresh: f64,
        bits: u32,
        width: u32,
        height: u32,
        block_x: u32,
        block_y: u32,
    ) -> Self {
        let peak = (1i64 << bits) - 1;
        let luma = peak * i64::from(width) * i64::from(height);
        let block = peak * i64::from(block_x) * i64::from(block_y);
        let scene = (luma as f64 * scthresh / 100.0) as i64;
        let duplicate = (block as f64 * dupthresh / 100.0) as i64;

        Self { duplicate, scene }
    }
}

/// The index of the frame of a cycle to drop.
///
/// The frame with the lowest block difference is dropped if it is below the duplicate threshold. With no
/// duplicate, the frame at the cycle's last scene change goes, and with no scene change either, the lowest.
/// Ties go to the earliest frame.
pub(crate) fn find_drop(metrics: &[DecimateMetric], thresholds: &Thresholds) -> usize {
    let mut scene_position = None;
    let mut lowest = 0;
    for (index, metric) in metrics.iter().enumerate() {
        if metric.total > thresholds.scene {
            scene_position = Some(index);
        }

        if metric.max_block < metrics[lowest].max_block {
            lowest = index;
        }
    }

    let duplicate = metrics[lowest].max_block < thresholds.duplicate;
    match scene_position {
        Some(position) if !duplicate => position,
        _ => lowest,
    }
}

/// Replaces the first frame's metric, which compares it with itself and so would always be a duplicate.
///
/// It takes the second frame's block difference and counts as a scene change. A cycle of fewer than two
/// frames is left as it is.
pub(crate) fn first_cycle_metrics(metrics: &mut [DecimateMetric], thresholds: &Thresholds) {
    if metrics.len() < 2 {
        return;
    }

    metrics[0].max_block = metrics[1].max_block;
    metrics[0].total = thresholds.scene + 1;
}
