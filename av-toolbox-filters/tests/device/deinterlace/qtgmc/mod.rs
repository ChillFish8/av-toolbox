mod conventions;
mod formats;
mod progressive;
mod reference;

use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::ops::Range;
use std::time::Duration;

use av_toolbox_cubek::deinterlace::qtgmc::Parity;
use av_toolbox_cubek::testing::{
    Eedi3Clips,
    Eedi3Parameters,
    Interpolate,
    QtgmcReferenceSettings,
    ReferenceInput,
    ReferenceSearch,
    field_clips,
    reference_bob,
    reference_eedi3_field,
    reference_qtgmc_run,
};
use av_toolbox_filters::testing::{QtgmcMotion, QtgmcRepairFlag, qtgmc_with_motion_log};
use av_toolbox_gpu::{
    BuildContext,
    BuildTask,
    ChromaLocation,
    Depth,
    FieldOrder,
    FrameFormat,
    FrameLayout,
    FrameMeta,
    FrameRate,
    SampleFormat,
    Subsampling,
};
use cubecl::prelude::Runtime;

pub(super) use self::progressive::{mode_args, progressive_format};
use crate::stream::{Input, Output, run_chunk};
use crate::support::{backend, element_of, float_planes, frame, planes_as_codes, ten_bit, to_bytes};

/// How far an output sample may sit from the CPU reference, in code values.
const TOLERANCE: u16 = 1;

/// The share of a field plane's samples that may sit further than [TOLERANCE] from the reference.
///
/// It allows for f32 threshold flips between the device and the CPU.
const OUTLIER_SHARE: f64 = 0.001;

/// [OUTLIER_SHARE] for the chroma of progressive, repair and repair reversed input.
///
/// These modes keep the frame's integer-valued chroma, so compensated values sit on whole or half codes
/// within an ulp. Rep0 and Rep2 then meet their restore limits exactly, and the tie flips between the
/// device and the CPU.
const PROGRESSIVE_CHROMA_OUTLIER_SHARE: f64 = 0.005;

/// How far apart, relative to the reference's SAD, two SADs of a near-tie may be.
const NEAR_TIE_SAD: f32 = 1e-5;

/// The share of the blocks a field's searches cover whose vector may differ from the reference on a near-tie.
const NEAR_TIE_SHARE: f64 = 0.01;

/// The largest difference from the reference any output sample may have, in 8-bit code values.
const LARGEST_8_BIT: u16 = 8;

/// The furthest direction QTGMC's EEDI3 searches.
const QTGMC_MDIS: u32 = 8;

/// An interlaced clip, top field first, as Y, U and V code values.
///
/// A gray clip has empty chroma planes.
#[derive(Clone)]
pub(super) struct Clip {
    pub(super) name: String,
    pub(super) bits: u32,
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) subsampling: Subsampling,
    pub(super) frames: Vec<[Vec<i32>; 3]>,
}

impl Clip {
    /// The field clip called `name`.
    pub(super) fn named(name: &str) -> Self {
        let clips = field_clips();
        let found = clips
            .into_iter()
            .find(|clip| clip.name == name)
            .expect("the clip exists");

        Self {
            name: name.to_string(),
            bits: found.bits,
            width: found.width,
            height: found.height,
            subsampling: Subsampling::Yuv420,
            frames: found.frames,
        }
    }

    /// The clip's frames in `frames` alone.
    fn cut(&self, frames: Range<usize>) -> Self {
        let name = format!("{} frames {frames:?}", self.name);

        Self {
            name,
            frames: self.frames[frames].to_vec(),
            ..self.clone()
        }
    }

    /// The clip's luma alone.
    fn gray(&self) -> Self {
        let frames = self
            .frames
            .iter()
            .map(|planes| [planes[0].clone(), Vec::new(), Vec::new()])
            .collect();
        let name = format!("{} gray", self.name);

        Self {
            name,
            frames,
            ..self.clone()
        }
    }

    fn is_gray(&self) -> bool {
        self.frames[0][1].is_empty()
    }

    pub(super) fn sample(&self) -> SampleFormat {
        match self.bits {
            8 => SampleFormat::U8,
            10 => ten_bit(),
            bits => {
                let depth = Depth::new(bits as u8).expect("the clip has a supported depth");
                SampleFormat::U16 { depth }
            },
        }
    }

    fn field_count(&self) -> usize {
        2 * self.frames.len()
    }
}

/// The 25 frames per second format of `clip`, with its fields in `order`.
pub(super) fn interlaced(clip: &Clip, order: FieldOrder) -> FrameFormat {
    let layout = if clip.is_gray() {
        FrameLayout::Grey
    } else {
        FrameLayout::Yuv {
            subsampling: clip.subsampling,
        }
    };
    let base = frame(
        clip.width as u32,
        clip.height as u32,
        layout,
        clip.sample(),
        ChromaLocation::Left,
    );

    FrameFormat {
        frame_rate: FrameRate::new(25, 1),
        field_order: Some(order),
        ..base
    }
}

/// The format `deinterlace/qtgmc` emits for `input`, at twice its rate when `double` is set.
pub(super) fn deinterlaced(input: FrameFormat, double: bool) -> FrameFormat {
    let frame_rate = match (double, input.frame_rate) {
        (true, Some(rate)) => FrameRate::new(rate.numerator() * 2, rate.denominator()),
        (_, unchanged) => unchanged,
    };

    FrameFormat {
        frame_rate,
        field_order: Some(FieldOrder::Progressive),
        sample: SampleFormat::F32,
        ..input
    }
}

/// The planes of `codes`, one frame of `format`, as bytes.
///
/// An F32 frame holds `bits` deep codes.
fn frame_bytes(format: &FrameFormat, codes: &[Vec<u16>], bits: u32) -> Vec<Vec<u8>> {
    if format.sample == SampleFormat::F32 {
        return float_planes(format, codes, bits);
    }

    let element = element_of(format.sample);

    codes.iter().map(|plane| to_bytes(plane, element)).collect()
}

/// The frames of `clip` as inputs of `format`, numbered from `first` and 40 ms apart.
///
/// An F32 input holds the clip's codes at its depth.
pub(super) fn inputs(clip: &Clip, format: &FrameFormat, first: u64) -> Vec<Input> {
    let plane_count = format.channels().len();

    clip.frames
        .iter()
        .enumerate()
        .map(|(offset, planes)| {
            let index = first + offset as u64;
            let timestamp = Duration::from_millis(40 * index);
            let meta = FrameMeta {
                timestamp: Some(timestamp),
                ..FrameMeta::at(index)
            };
            let codes: Vec<Vec<u16>> = planes[..plane_count]
                .iter()
                .map(|plane| plane.iter().map(|sample| *sample as u16).collect())
                .collect();
            let bytes = frame_bytes(format, &codes, clip.bits);

            Input { meta, planes: bytes }
        })
        .collect()
}

/// Every plane of `output`, an F32 output of `clip`, as unrounded codes of the clip's depth.
pub(super) fn output_codes(clip: &Clip, output: &Output) -> Vec<Vec<f32>> {
    let input = interlaced(clip, FieldOrder::TopFirst);
    let float = FrameFormat {
        sample: SampleFormat::F32,
        ..input
    };

    planes_as_codes(&float, &output.planes, clip.bits)
}

/// Every plane of `output`, an F32 output of `clip`, as the codes an integer output of the clip's depth
/// would hold.
pub(super) fn narrowed_codes(clip: &Clip, output: &Output) -> Vec<Vec<u16>> {
    let peak = f32::from(peak_code(clip.sample()));
    let planes = output_codes(clip, output);

    planes
        .iter()
        .map(|plane| {
            plane
                .iter()
                .map(|code| code.round().clamp(0.0, peak) as u16)
                .collect()
        })
        .collect()
}

/// What the stage gave for one chunk, its outputs, every motion search it ran and every repair mask flag.
pub(super) struct ChunkRun {
    pub(super) outputs: Vec<Output>,
    motion: Vec<QtgmcMotion>,
    repairs: Vec<QtgmcRepairFlag>,
}

/// Runs each chunk through one stage built with a motion log, starting and ending the stream around each.
struct LoggedStream {
    args: serde_json::Value,
    input: FrameFormat,
    output: FrameFormat,
    chunks: Vec<Vec<Input>>,
}

impl BuildTask for LoggedStream {
    type Output = Vec<ChunkRun>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let built = qtgmc_with_motion_log(build, self.args, self.input);
        let (mut filter, log) = built.expect("the stage should build");
        let mut context = build.frame_context();

        let mut runs = Vec::with_capacity(self.chunks.len());
        for chunk in self.chunks {
            let formats = (&self.input, &self.output);
            let outputs = run_chunk(build, &mut context, filter.as_mut(), formats, chunk);
            let motion = log.take();
            let repairs = log.take_repairs();
            runs.push(ChunkRun {
                outputs,
                motion,
                repairs,
            });
        }

        runs
    }
}

/// Runs `chunks` of `input` through the stage built with `args`, giving each chunk's run.
pub(super) fn run_chunks(
    args: serde_json::Value,
    input: FrameFormat,
    output: FrameFormat,
    chunks: Vec<Vec<Input>>,
) -> Vec<ChunkRun> {
    let backend = backend();
    let stream = LoggedStream {
        args,
        input,
        output,
        chunks,
    };

    backend.build(stream)
}

/// Runs `clip` at double rate, top field first, as one chunk.
pub(super) fn run_clip(args: serde_json::Value, clip: &Clip) -> ChunkRun {
    let input = interlaced(clip, FieldOrder::TopFirst);
    let output = deinterlaced(input, true);
    let chunk = inputs(clip, &input, 0);

    let mut runs = run_chunks(args, input, output, vec![chunk]);
    runs.pop().expect("one chunk was run")
}

/// The reference's settings at QTGMC's defaults.
fn reference_defaults() -> QtgmcReferenceSettings {
    QtgmcReferenceSettings {
        tr0: 2,
        tr1: 2,
        tr2: 1,
        sharpness: 1.0,
        thsad1: 640.0,
        thsad2: 256.0,
        input: ReferenceInput::Interlaced,
        first_frame: 0,
    }
}

fn reference_frames(clip: &Clip) -> Vec<[Vec<f32>; 3]> {
    clip.frames
        .iter()
        .map(|planes| {
            planes
                .each_ref()
                .map(|plane| plane.iter().map(|sample| *sample as f32).collect())
        })
        .collect()
}

/// The `spatial: cubic` interpolation, QTGMC's bob.
fn cubic(plane: &[f32], width: usize, height: usize, parity: Parity) -> Vec<f32> {
    reference_bob(plane, (width, height), parity)
}

/// QTGMC's EEDI3 on the CPU, remembering every plane it has interpolated.
struct Eedi3Reference {
    parameters: Eedi3Parameters,
    remembered: RefCell<HashMap<(u64, bool), Vec<f32>>>,
}

impl Eedi3Reference {
    fn new(bits: u32) -> Self {
        let parameters = Eedi3Parameters {
            mdis: QTGMC_MDIS,
            bits,
            ..Eedi3Parameters::default()
        };

        Self {
            parameters,
            remembered: RefCell::new(HashMap::new()),
        }
    }

    fn interpolate(&self, plane: &[f32], width: usize, height: usize, parity: Parity) -> Vec<f32> {
        let mut hasher = DefaultHasher::new();
        for sample in plane {
            sample.to_bits().hash(&mut hasher);
        }
        width.hash(&mut hasher);
        let key = (hasher.finish(), parity == Parity::Top);

        let mut remembered = self.remembered.borrow_mut();
        let found = remembered.entry(key).or_insert_with(|| {
            let kept = parity.row_offset();
            let clips = Eedi3Clips::default();
            let pass =
                reference_eedi3_field(plane, width as u32, height as u32, kept, &self.parameters, clips);
            pass.output
        });

        found.clone()
    }
}

/// What the CPU reference gives for a clip, every output field and every field's motion.
struct Expected {
    fields: Vec<[Vec<f32>; 3]>,
    motion: Vec<Vec<ReferenceSearch>>,
    input: ReferenceInput,
}

fn expected(clip: &Clip, settings: &QtgmcReferenceSettings, interpolate: Interpolate<'_>) -> Expected {
    let frames = reference_frames(clip);
    let luma = (clip.width, clip.height);
    let run = reference_qtgmc_run(&frames, luma, clip.bits, settings, interpolate);

    Expected {
        fields: run.output,
        motion: run.motion,
        input: settings.input,
    }
}

/// Whether two SADs a search chose between are equal to within f32 rounding.
fn is_near_tie(found: f32, wanted: f32) -> bool {
    let allowed = NEAR_TIE_SAD * wanted.max(1.0);

    (found - wanted).abs() <= allowed
}

/// Asserts the searches the stage ran for its `field` match the reference's `wanted`, giving how many
/// blocks differ on a near-tie.
fn field_near_ties(found: &[QtgmcMotion], field: usize, wanted: &[ReferenceSearch], label: &str) -> usize {
    let searches: Vec<&QtgmcMotion> = found.iter().filter(|search| search.field == field).collect();
    let found_deltas: Vec<i32> = searches.iter().map(|search| search.delta).collect();
    let wanted_deltas: Vec<i32> = wanted.iter().map(|motion| motion.delta).collect();
    assert_eq!(
        found_deltas, wanted_deltas,
        "{label}: field {field} searched other neighbours"
    );

    let mut near_ties = 0;
    for (search, motion) in searches.iter().zip(wanted) {
        let delta = motion.delta;
        assert_eq!(
            search.flagged, motion.flagged,
            "{label}: field {field} delta {delta} differs on its scene-change flag"
        );
        assert_eq!(
            search.vectors.len(),
            motion.field.vectors.len(),
            "{label}: field {field} delta {delta} has another block count"
        );

        let blocks = search.vectors.chunks(2).zip(motion.field.vectors.chunks(2));
        for (block, (got, want)) in blocks.enumerate() {
            if got == want {
                continue;
            }

            let found_sad = search.sads[block];
            let wanted_sad = motion.field.sads[block];
            let location = format!("{label}: field {field} delta {delta} block {block}");
            assert!(
                is_near_tie(found_sad, wanted_sad),
                "{location} has vector {got:?} with SAD {found_sad}, expected {want:?} with SAD {wanted_sad}"
            );

            println!(
                "{location} is a near-tie: {got:?} SAD {found_sad}, reference {want:?} SAD {wanted_sad}"
            );
            near_ties += 1;
        }
    }

    let blocks: usize = wanted.iter().map(|motion| motion.field.sads.len()).sum();
    let allowed = (blocks as f64 * NEAR_TIE_SHARE) as usize;
    assert!(
        near_ties <= allowed,
        "{label}: field {field} has {near_ties} near-tie blocks, more than {allowed}"
    );

    near_ties
}

/// Asserts each of `fields`, a stage field and the reference's searches for it, matches the reference.
///
/// A vector may differ only on a near-tie, where both SADs agree to within [NEAR_TIE_SAD], in at most
/// [NEAR_TIE_SHARE] of the blocks a field's searches cover. TR0's f32 sum order on the device leaves
/// last-bit differences in the search clip that can flip a near-tie between two candidate vectors. Flags
/// must always be equal.
fn assert_motion_fields<'a>(
    found: &[QtgmcMotion],
    fields: impl Iterator<Item = (usize, &'a [ReferenceSearch])>,
    label: &str,
) {
    let mut worst = (0, 0);
    for (field, wanted) in fields {
        let near_ties = field_near_ties(found, field, wanted, label);
        if near_ties > worst.1 {
            worst = (field, near_ties);
        }
    }

    println!(
        "{label}: the worst field, {}, has {} near-tie blocks",
        worst.0, worst.1
    );
}

/// Asserts every search the stage ran matches the reference's, field by field.
fn assert_motion(found: &[QtgmcMotion], expected: &Expected, label: &str) {
    let fields = expected
        .motion
        .iter()
        .enumerate()
        .map(|(field, wanted)| (field, wanted.as_slice()));

    assert_motion_fields(found, fields, label);
}

/// How far one output plane sits from the reference.
#[derive(Debug, Clone, Copy, Default)]
struct Closeness {
    /// The share of samples further than [TOLERANCE] from the reference.
    share: f64,
    largest: u16,
    field: usize,
    plane: usize,
}

/// The bit depth of `sample`, an integer format.
pub(super) fn depth_of(sample: SampleFormat) -> u32 {
    match sample {
        SampleFormat::U8 => 8,
        SampleFormat::U16 { depth } => u32::from(depth.bits()),
        other => panic!("{other:?} is not an integer format"),
    }
}

/// The largest code value of `sample`, an integer format.
pub(super) fn peak_code(sample: SampleFormat) -> u16 {
    let depth = depth_of(sample);

    ((1u32 << depth) - 1) as u16
}

/// The largest difference from the reference a sample of `sample` may have, [LARGEST_8_BIT] scaled to its
/// depth.
fn largest_allowed(sample: SampleFormat) -> u16 {
    let depth = depth_of(sample);

    LARGEST_8_BIT << (depth - 8)
}

/// How far `codes`, plane `plane` of output `field`, sits from `wanted`, narrowed to `sample`.
fn closeness(codes: &[u16], wanted: &[f32], sample: SampleFormat, field: usize, plane: usize) -> Closeness {
    let peak = f32::from(peak_code(sample));
    assert_eq!(
        codes.len(),
        wanted.len(),
        "field {field} plane {plane} has another size"
    );

    let differences: Vec<u16> = codes
        .iter()
        .zip(wanted)
        .map(|(got, want)| {
            let narrowed = want.round().clamp(0.0, peak) as u16;
            got.abs_diff(narrowed)
        })
        .collect();
    let outliers = differences
        .iter()
        .filter(|difference| **difference > TOLERANCE)
        .count();
    let largest = differences.iter().copied().max().unwrap_or(0);

    Closeness {
        share: outliers as f64 / wanted.len() as f64,
        largest,
        field,
        plane,
    }
}

/// The share of plane `plane` that may sit further than [TOLERANCE] from the reference for `input`.
fn allowed_share(input: ReferenceInput, plane: usize) -> f64 {
    let is_chroma = plane > 0;
    let is_progressive = input != ReferenceInput::Interlaced;
    if is_chroma && is_progressive {
        return PROGRESSIVE_CHROMA_OUTLIER_SHARE;
    }

    OUTLIER_SHARE
}

/// Asserts each of `found` lies within the tolerance of the reference field it is paired with, giving the
/// worst plane.
///
/// Each pair is an output and the reference's planes for it, from a run with `input`.
fn assert_paired_outputs<'a>(
    pairs: impl Iterator<Item = (usize, &'a Output, &'a [Vec<f32>; 3])>,
    clip: &Clip,
    input: ReferenceInput,
    label: &str,
) -> Closeness {
    let sample = clip.sample();
    let ceiling = largest_allowed(sample);

    let mut worst = Closeness::default();
    let mut worst_planes: Vec<Closeness> = Vec::new();
    for (field, output, wanted) in pairs {
        let planes = narrowed_codes(clip, output);
        for (plane, found) in planes.iter().enumerate() {
            let found_closeness = closeness(found, &wanted[plane], sample, field, plane);
            let allowed = allowed_share(input, plane);
            assert!(
                found_closeness.share <= allowed,
                "{label}: {:.3} % of field {field} plane {plane} is over {TOLERANCE} code, the largest {}",
                found_closeness.share * 100.0,
                found_closeness.largest
            );
            assert!(
                found_closeness.largest <= ceiling,
                "{label}: field {field} plane {plane} has a sample {} codes off, over {ceiling}",
                found_closeness.largest
            );

            if found_closeness.share > worst.share || found_closeness.largest > worst.largest {
                worst = found_closeness;
            }

            if worst_planes.len() <= plane {
                worst_planes.push(Closeness::default());
            }

            let plane_worst = &mut worst_planes[plane];
            if found_closeness.share > plane_worst.share || found_closeness.largest > plane_worst.largest {
                *plane_worst = found_closeness;
            }
        }
    }

    println!(
        "{label}: the worst plane is plane {} of field {}",
        worst.plane, worst.field
    );
    for (plane, plane_worst) in worst_planes.iter().enumerate() {
        println!(
            "{label}: plane {plane} worst share over {TOLERANCE} code is {:.3} % (field {}), largest \
             difference {}",
            plane_worst.share * 100.0,
            plane_worst.field,
            plane_worst.largest
        );
    }

    worst
}

/// Asserts each of `outputs` lies within the tolerance of the reference field of the same number.
fn assert_outputs(outputs: &[Output], expected: &Expected, clip: &Clip, label: &str) -> Closeness {
    assert_eq!(
        outputs.len(),
        expected.fields.len(),
        "{label}: one output per field"
    );

    let pairs = outputs
        .iter()
        .zip(&expected.fields)
        .enumerate()
        .map(|(field, (output, wanted))| (field, output, wanted));

    assert_paired_outputs(pairs, clip, expected.input, label)
}

/// Asserts `run` matches the reference, its motion first and then its outputs.
fn assert_matches(run: &ChunkRun, expected: &Expected, clip: &Clip, label: &str) -> Closeness {
    assert_motion(&run.motion, expected, label);

    assert_outputs(&run.outputs, expected, clip, label)
}
