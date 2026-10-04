use av_toolbox_cubek::ivtc::FieldSums;

/// VFM's `p` match, the current frame's matching field woven with the previous frame's.
pub(crate) const P: usize = 0;
/// VFM's `c` match, the current frame itself.
pub(crate) const C: usize = 1;
/// VFM's `n` match, the current frame's matching field woven with the next frame's.
pub(crate) const N: usize = 2;
/// VFM's `b` match, the previous frame's matching field woven with the current frame's.
pub(crate) const B: usize = 3;
/// VFM's `u` match, the next frame's matching field woven with the current frame's.
pub(crate) const U: usize = 4;

const SAME_ORDER: [usize; 5] = [0, 1, 2, 3, 4];
const OPPOSITE_ORDER: [usize; 5] = [2, 1, 0, 4, 3];

fn rounded_sixth(sum: u64) -> i64 {
    // VFM divides in f32 and truncates after adding a half, so the rounding is kept exactly.
    let divided = sum as f32 / 6.0;

    (divided + 0.5) as i64
}

/// Which of `first` and `second` `compareFieldsSlow` keeps for these sums.
pub(crate) fn compare_fields(sums: FieldSums, first: usize, second: usize, shift: u32) -> usize {
    let five_hundred = 500u64 << shift;
    let mut first_motion = sums.first.motion;
    let mut second_motion = sums.second.motion;
    let first_isolated = sums.first.isolated;
    let second_isolated = sums.second.isolated;

    let both_still = first_motion < five_hundred && second_motion < five_hundred;
    let isolated_large = first_isolated >= five_hundred || second_isolated >= five_hundred;
    let isolated_high = u64::max(first_isolated, second_isolated);
    let isolated_low = u64::min(first_isolated, second_isolated);
    if both_still && isolated_large && isolated_high > 3 * isolated_low {
        first_motion = first_isolated;
        second_motion = second_isolated;
    }

    let first_norm = rounded_sixth(sums.first.combing);
    let second_norm = rounded_sixth(sums.second.combing);
    let first_mtn = rounded_sixth(first_motion);
    let second_mtn = rounded_sixth(second_motion);

    let norm_high = i64::max(first_norm, second_norm);
    let norm_low = i64::min(first_norm, second_norm);
    let mtn_high = i64::max(first_mtn, second_mtn);
    let mtn_low = i64::min(first_mtn, second_mtn);
    let c1 = norm_high as f32 / i64::max(norm_low, 1) as f32;
    let c2 = mtn_high as f32 / i64::max(mtn_low, 1) as f32;
    let mr = mtn_high as f32 / i64::max(norm_high, 1) as f32;

    let motion = MotionPair {
        first: first_mtn,
        second: second_mtn,
        shift,
    };
    let clearly_apart = motion.apart(500, 2, 1)
        || motion.apart(1000, 3, 2)
        || motion.apart(2000, 5, 4)
        || (motion.reaches(4000) && c2 > c1);
    if clearly_apart {
        return less_moving(first, second, first_mtn, second_mtn);
    }

    let some_motion = f64::from(mr) > 0.005 && mtn_high > (150 << shift);
    if some_motion && motion.ratio_apart(2, 1) {
        return less_moving(first, second, first_mtn, second_mtn);
    }

    less_moving(first, second, first_norm, second_norm)
}

/// The two matches' motion figures, with the bit depth their thresholds scale by.
struct MotionPair {
    first: i64,
    second: i64,
    shift: u32,
}

impl MotionPair {
    /// Whether either figure reaches `base`, scaled to the bit depth.
    fn reaches(&self, base: i64) -> bool {
        let threshold = base << self.shift;

        self.first >= threshold || self.second >= threshold
    }

    /// Whether `low × numerator < high × denominator` holds one way or the other.
    ///
    /// It is VFM's `mtn1*2 < mtn2*1 || mtn2*2 < mtn1*1` for a numerator of 2 and a denominator of 1.
    fn ratio_apart(&self, numerator: i64, denominator: i64) -> bool {
        let first_under = self.first * numerator < self.second * denominator;
        let second_under = self.second * numerator < self.first * denominator;

        first_under || second_under
    }

    fn apart(&self, base: i64, numerator: i64, denominator: i64) -> bool {
        self.reaches(base) && self.ratio_apart(numerator, denominator)
    }
}

/// `second` when the first match's figure is the larger, `first` otherwise.
fn less_moving(first: usize, second: usize, first_figure: i64, second_figure: i64) -> usize {
    if first_figure > second_figure {
        second
    } else {
        first
    }
}

/// `checkmm`, keeping `first` unless `second` is clearly less combed.
pub(crate) fn check_mm(first: usize, second: usize, mics: &[u32; 5], mi: u32) -> usize {
    let first_mic = i64::from(mics[first]);
    let second_mic = i64::from(mics[second]);
    let limit = i64::from(mi);
    let much_lower = second_mic * 3 < first_mic || (second_mic * 2 < first_mic && first_mic > limit);
    let margin = (second_mic - first_mic).abs() >= 30;

    if much_lower && margin && second_mic < limit {
        second
    } else {
        first
    }
}

/// Whether a frame is a scene change, from its own scene measure and the next frame's.
///
/// `scthresh` is VFM's argument already divided by 100.
pub(crate) fn is_scene_change(current: f64, next: f64, scthresh: f64) -> bool {
    current > scthresh || next > scthresh
}

/// The field compare sums of the three match pairs [Decision::pairs] lists.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PairSums {
    pub(crate) current_previous: FieldSums,
    pub(crate) current_next: FieldSums,
    pub(crate) previous_next: FieldSums,
}

/// What VFM measures of one frame before it decides.
///
/// `mics` holds each match's mic by raw match number. `scene_change` is only ever set under micmatch 1.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FrameMetrics {
    pub(crate) pairs: PairSums,
    pub(crate) mics: [u32; 5],
    pub(crate) scene_change: bool,
}

/// VFM's match decision, `vfmProcess` from its field compares to its mode table.
///
/// `shift` is the bit depth less 8. `opposite` is true when VFM's field is not the clip's first field.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Decision {
    pub(crate) mode: u32,
    pub(crate) micmatch: u32,
    pub(crate) mi: u32,
    pub(crate) shift: u32,
    pub(crate) opposite: bool,
}

impl Decision {
    /// VFM's `fxo`, the raw match number of each of [P], [C], [N], [B] and [U].
    pub(crate) fn order(&self) -> [usize; 5] {
        if self.opposite { OPPOSITE_ORDER } else { SAME_ORDER }
    }

    /// The raw match pairs of [PairSums]'s fields, in order, `match1` first.
    pub(crate) fn pairs(&self) -> [(usize, usize); 3] {
        let order = self.order();

        [(order[C], order[P]), (order[C], order[N]), (order[P], order[N])]
    }

    /// The raw match `vfmProcess` picks for a frame with these metrics.
    pub(crate) fn choose(&self, metrics: &FrameMetrics) -> usize {
        let order = self.order();
        let pairs = &metrics.pairs;
        let mut matched = compare_fields(pairs.current_previous, order[C], order[P], self.shift);
        if self.mode >= 4 {
            let sums = if matched == order[C] {
                pairs.current_next
            } else {
                pairs.previous_next
            };
            matched = compare_fields(sums, matched, order[N], self.shift);
        }

        let checks_mics = self.micmatch == 2 || (self.micmatch == 1 && metrics.scene_change);
        if !checks_mics {
            return matched;
        }

        let rivals = self.mic_rivals(matched);
        rivals.into_iter().fold(matched, |kept, rival| {
            check_mm(kept, rival, &metrics.mics, self.mi)
        })
    }

    /// The matches the mode table checks `matched` against, in turn.
    fn mic_rivals(&self, matched: usize) -> Vec<usize> {
        let order = self.order();
        let discarded = if matched == order[P] { order[C] } else { order[P] };

        match self.mode {
            0 | 4 => vec![discarded],
            1 => vec![order[N]],
            2 => vec![order[U]],
            3 => vec![order[N], order[U], order[B]],
            5 => vec![order[U], order[B]],
            _ => Vec::new(),
        }
    }
}
