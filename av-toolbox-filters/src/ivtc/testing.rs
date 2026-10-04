use av_toolbox_cubek::ivtc::FieldSums;

use crate::ivtc::decision::{Decision, FrameMetrics, PairSums, is_scene_change};

/// VFM's match decision and its settings, for checking `ivtc/match` against a CPU pipeline.
///
/// `shift` is the bit depth less 8, `opposite` is set when VFM's field is not the clip's first field, and
/// `scthresh` is a fraction rather than a percentage.
#[derive(Debug, Clone, Copy)]
pub struct VfmDecision {
    pub mode: u32,
    pub micmatch: u32,
    pub mi: u32,
    pub shift: u32,
    pub opposite: bool,
    pub scthresh: f64,
}

impl VfmDecision {
    fn decision(&self) -> Decision {
        Decision {
            mode: self.mode,
            micmatch: self.micmatch,
            mi: self.mi,
            shift: self.shift,
            opposite: self.opposite,
        }
    }

    /// The raw match pairs of the field compares [VfmDecision::choose] reads, `match1` first.
    pub fn pairs(&self) -> [(usize, usize); 3] {
        self.decision().pairs()
    }

    /// VFM's scene change flag for a frame with these scene measures, only ever set under micmatch 1.
    pub fn scene_change(&self, current_scene: f64, next_scene: f64) -> bool {
        self.micmatch == 1 && is_scene_change(current_scene, next_scene, self.scthresh)
    }

    /// The raw match VFM picks, from the sums of each of [VfmDecision::pairs] and every match's mic.
    pub fn choose(&self, sums: [FieldSums; 3], mics: [u32; 5], scene_change: bool) -> usize {
        let [current_previous, current_next, previous_next] = sums;
        let metrics = FrameMetrics {
            pairs: PairSums {
                current_previous,
                current_next,
                previous_next,
            },
            mics,
            scene_change,
        };

        self.decision().choose(&metrics)
    }
}
