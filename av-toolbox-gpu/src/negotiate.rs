use crate::frame::{FrameFormat, MaskFormat, SampleFormat, StreamFormat};

/// Sample formats a stage will work with, most preferred first.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct FormatSet {
    preferences: Vec<SampleFormat>,
}

impl FormatSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// A set holding each of `samples`, in the order given.
    pub fn of(samples: impl IntoIterator<Item = SampleFormat>) -> Self {
        let mut set = Self::new();
        for sample in samples {
            set.push(sample);
        }

        set
    }

    /// Adds `sample` as the least preferred, unless it is already held.
    pub fn push(&mut self, sample: SampleFormat) {
        if !self.accepts(sample) {
            self.preferences.push(sample);
        }
    }

    /// What this set holds, most preferred first.
    pub fn preferences(&self) -> &[SampleFormat] {
        &self.preferences
    }

    pub fn accepts(&self, sample: SampleFormat) -> bool {
        self.preferences.contains(&sample)
    }

    pub fn is_empty(&self) -> bool {
        self.preferences.is_empty()
    }

    /// The sample of this set that moves the fewest bytes.
    ///
    /// Ties go to whichever came first, so a stage's own preference decides between two formats of
    /// the same width.
    pub fn narrowest(&self) -> Option<SampleFormat> {
        self.preferences
            .iter()
            .min_by_key(|sample| sample.bytes())
            .copied()
    }

    /// The samples both sets hold, in this set's order.
    pub fn intersect(&self, other: &Self) -> Self {
        let kept = self
            .preferences
            .iter()
            .filter(|sample| other.accepts(**sample))
            .copied();

        Self::of(kept)
    }
}

/// What the solver knows about one of a stage's inputs.
///
/// A stage is asked about one input at a time, and the rest may not be settled yet. A proposal is
/// real information to steer by, so a stage that cares which format a sibling carries can say so
/// while the decision is still open. It is not a promise, so it is not something to reject a
/// pipeline over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Offered {
    /// Already decided. It will not change.
    Resolved(StreamFormat),
    /// What its producer would prefer, not yet agreed with anyone.
    Proposed(StreamFormat),
    /// Its producer has not been reached yet.
    Unknown,
}

impl Offered {
    /// The format this carries, whether settled or only proposed.
    pub fn format(self) -> Option<StreamFormat> {
        match self {
            Self::Resolved(format) | Self::Proposed(format) => Some(format),
            Self::Unknown => None,
        }
    }

    pub fn is_resolved(self) -> bool {
        matches!(self, Self::Resolved(_))
    }
}

/// Everything about a stream except the sample format.
///
/// Dimensions, layout and colour are fixed by the stage that produces them. Only the sample is
/// negotiated, so a template plus a chosen sample is a whole format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamTemplate {
    Frames(FrameFormat),
    Mask(MaskFormat),
}

impl StreamTemplate {
    pub const fn from_format(format: &StreamFormat) -> Self {
        match format {
            StreamFormat::Frames(format) => Self::Frames(*format),
            StreamFormat::Mask(format) => Self::Mask(*format),
        }
    }

    /// This template as a whole format, carrying `sample`.
    pub fn with_sample(self, sample: SampleFormat) -> StreamFormat {
        match self {
            Self::Frames(mut format) => {
                format.sample = sample;
                StreamFormat::Frames(format)
            },
            Self::Mask(mut format) => {
                format.sample = sample;
                StreamFormat::Mask(format)
            },
        }
    }
}

/// What a stage will produce.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputFormats {
    /// Everything the stage fixes for itself.
    pub template: StreamTemplate,
    /// The samples it can emit, most preferred first.
    pub samples: FormatSet,
}

/// The formats a stage was actually given and will actually emit.
///
/// The inputs are in the stage's signature order, one entry per parameter followed by one per
/// variadic input, so they line up with the inputs the stage was bound to. An optional parameter
/// nobody bound is a hole rather than a gap that shifts everything after it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedFormats {
    inputs: Vec<Option<StreamFormat>>,
    output: Option<StreamFormat>,
}

impl ResolvedFormats {
    /// A sink resolves to no output, every other kind to one.
    pub fn new(inputs: Vec<Option<StreamFormat>>, output: Option<StreamFormat>) -> Self {
        Self { inputs, output }
    }

    pub fn inputs(&self) -> &[Option<StreamFormat>] {
        &self.inputs
    }

    /// The format in one signature slot.
    ///
    /// `None` covers two cases a caller may want to tell apart. The slot is an optional parameter
    /// nobody bound, or `index` is past the end of the signature.
    pub fn input(&self, index: usize) -> Option<StreamFormat> {
        self.inputs.get(index).copied().flatten()
    }

    pub fn output(&self) -> Option<StreamFormat> {
        self.output
    }

    /// Replaces the format in one signature slot.
    ///
    /// Panics when `index` is past the end of the signature.
    pub fn set_input(&mut self, index: usize, format: StreamFormat) {
        self.inputs[index] = Some(format);
    }

    pub fn set_output(&mut self, format: StreamFormat) {
        self.output = Some(format);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ChromaLocation,
        ColourInfo,
        Depth,
        FrameFormat,
        FrameLayout,
        MaskChroma,
        MaskFormat,
        SampleFormat,
        Subsampling,
    };

    fn frames(sample: SampleFormat) -> StreamFormat {
        StreamFormat::Frames(FrameFormat {
            width: 64,
            height: 64,
            layout: FrameLayout::Yuv {
                subsampling: Subsampling::Yuv420,
            },
            sample,
            colour: ColourInfo::default(),
            frame_rate: None,
            field_order: None,
        })
    }

    #[test]
    fn a_format_set_keeps_the_order_it_was_given() {
        let mut set = FormatSet::new();
        set.push(SampleFormat::F32);
        set.push(SampleFormat::U8);

        assert_eq!(set.preferences(), &[SampleFormat::F32, SampleFormat::U8]);
    }

    #[test]
    fn a_format_set_holds_each_sample_once() {
        let mut set = FormatSet::new();
        set.push(SampleFormat::U8);
        set.push(SampleFormat::U8);

        assert_eq!(set.preferences(), &[SampleFormat::U8]);
    }

    #[test]
    fn a_format_set_reports_what_it_holds() {
        let set = FormatSet::of([SampleFormat::U8, SampleFormat::F32]);

        assert!(set.accepts(SampleFormat::F32));
        assert!(!set.accepts(SampleFormat::F16));
        assert!(!set.is_empty());
        assert!(FormatSet::new().is_empty());
    }

    /// The solver picks the narrowest format both ends accept, so a set has to be able to say which
    /// of its own preferences is narrowest.
    #[test]
    fn a_format_set_finds_its_narrowest_sample() {
        let set = FormatSet::of([SampleFormat::F32, SampleFormat::U8, SampleFormat::F16]);

        assert_eq!(set.narrowest(), Some(SampleFormat::U8));
        assert_eq!(FormatSet::new().narrowest(), None);
    }

    /// F16 and a ten-bit U16 are both two bytes wide, so which one wins is decided by the order
    /// they were pushed rather than by their width.
    #[test]
    fn a_tie_on_width_goes_to_the_more_preferred() {
        let ten_bit = SampleFormat::U16 { depth: Depth::TEN };

        let float_first = FormatSet::of([SampleFormat::F16, ten_bit]);
        assert_eq!(float_first.narrowest(), Some(SampleFormat::F16));

        let integer_first = FormatSet::of([ten_bit, SampleFormat::F16]);
        assert_eq!(integer_first.narrowest(), Some(ten_bit));
    }

    #[test]
    fn two_format_sets_keep_only_what_both_hold() {
        let ours = FormatSet::of([SampleFormat::F32, SampleFormat::U8]);
        let theirs = FormatSet::of([SampleFormat::U8, SampleFormat::F16]);

        assert_eq!(ours.intersect(&theirs).preferences(), &[SampleFormat::U8]);
    }

    /// Intersection keeps the producer's order, because that is whose preference decides a tie.
    #[test]
    fn intersection_keeps_the_left_hand_order() {
        let ours = FormatSet::of([SampleFormat::F32, SampleFormat::U8]);
        let theirs = FormatSet::of([SampleFormat::U8, SampleFormat::F32]);

        assert_eq!(
            ours.intersect(&theirs).preferences(),
            &[SampleFormat::F32, SampleFormat::U8],
        );
    }

    #[test]
    fn a_template_becomes_a_stream_format_once_a_sample_is_chosen() {
        let format = frames(SampleFormat::U8);
        let template = StreamTemplate::from_format(&format);

        let chosen = template.with_sample(SampleFormat::F32);
        assert_eq!(chosen.sample(), SampleFormat::F32);
        assert_eq!(chosen.dimensions(), (64, 64));
    }

    #[test]
    fn the_sample_can_be_swapped_without_losing_chroma() {
        let format = MaskFormat {
            frame_rate: None,
            width: 7,
            height: 5,
            sample: SampleFormat::U8,
            chroma: Some(MaskChroma {
                subsampling: Subsampling::Yuv422,
                location: ChromaLocation::Left,
            }),
        };
        let template = StreamTemplate::Mask(format);

        let swapped = template.with_sample(SampleFormat::F16);
        let swapped_chroma = swapped.as_mask().and_then(|mask| mask.chroma);

        assert_eq!(swapped.sample(), SampleFormat::F16);
        assert_eq!(swapped_chroma, format.chroma);
    }

    #[test]
    fn a_mask_template_stays_a_mask() {
        let mask = StreamFormat::Mask(MaskFormat {
            frame_rate: None,
            width: 32,
            height: 16,
            sample: SampleFormat::U8,
            chroma: None,
        });

        let chosen = StreamTemplate::from_format(&mask).with_sample(SampleFormat::F16);
        assert!(matches!(chosen, StreamFormat::Mask(_)));
        assert_eq!(chosen.dimensions(), (32, 16));
    }

    #[test]
    fn what_a_stage_produces_carries_a_template_and_its_samples() {
        let format = frames(SampleFormat::U8);
        let output = OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([SampleFormat::F32]),
        };

        assert_eq!(output.samples.narrowest(), Some(SampleFormat::F32));
        assert_eq!(
            output.template.with_sample(SampleFormat::F32).dimensions(),
            (64, 64)
        );
    }

    #[test]
    fn resolved_formats_report_what_a_stage_was_given_and_will_emit() {
        let resolved = ResolvedFormats::new(
            vec![Some(frames(SampleFormat::U8))],
            Some(frames(SampleFormat::F32)),
        );

        assert_eq!(
            resolved.input(0).map(|format| format.sample()),
            Some(SampleFormat::U8)
        );
        assert_eq!(resolved.input(1), None);
        assert_eq!(
            resolved.output().map(|format| format.sample()),
            Some(SampleFormat::F32)
        );
    }

    /// An optional parameter nobody bound keeps its slot, so everything after it keeps its index.
    #[test]
    fn an_unbound_slot_is_a_hole_rather_than_a_gap() {
        let resolved = ResolvedFormats::new(vec![None, Some(frames(SampleFormat::U8))], None);

        assert_eq!(resolved.inputs().len(), 2);
        assert_eq!(resolved.input(0), None);
        assert_eq!(
            resolved.input(1).map(|format| format.sample()),
            Some(SampleFormat::U8)
        );
    }

    #[test]
    fn a_sink_resolves_to_no_output() {
        let resolved = ResolvedFormats::new(vec![Some(frames(SampleFormat::U8))], None);

        assert_eq!(resolved.output(), None);
        assert_eq!(resolved.inputs().len(), 1);
    }
}
