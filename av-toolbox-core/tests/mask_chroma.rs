// A mask stage that knows nothing of chroma, built through the registry and run on a real device,
// to check the chroma plane its readers ask for is added and downscaled for it.
#![cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]

use av_toolbox_core::{
    BoundInputs,
    BuildStageError,
    BuiltMask,
    InputSignature,
    Inputs,
    Mask,
    MaskDefinition,
    PipelineRegistry,
    PreparedStage,
    StageDefinition,
    StageError,
    StageInstance,
    StageKind,
};
use av_toolbox_cubek::resize::{BILINEAR_SUPPORT, Taps, bilinear, taps};
use av_toolbox_gpu::{
    Backend,
    BuildContext,
    BuildTask,
    Capabilities,
    ChromaLocation,
    Device,
    DeviceSide,
    FormatSet,
    FrameContext,
    FrameMeta,
    HostMask,
    HostSide,
    MaskChroma,
    MaskFormat,
    Offered,
    OutputFormats,
    ResolvedFormats,
    SampleFormat,
    StreamFormat,
    StreamTemplate,
    Subsampling,
};
// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;
use schemars::JsonSchema;
use serde::Deserialize;

const WIDTH: u32 = 7;
const HEIGHT: u32 = 5;
const CHROMA_WIDTH: u32 = 4;
const CHROMA_HEIGHT: u32 = 3;

#[cube(launch_unchecked)]
fn copy(input: &Array<u8>, output: &mut Array<u8>) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    output[ABSOLUTE_POS] = input[ABSOLUTE_POS];
}

/// The samples the fixture writes, varied enough that every tap position matters.
fn pattern() -> Vec<u8> {
    let samples = WIDTH * HEIGHT;

    (0..samples)
        .map(|index| ((index * 53 + 17) % 256) as u8)
        .collect()
}

fn mask_format(chroma: Option<MaskChroma>) -> MaskFormat {
    MaskFormat {
        width: WIDTH,
        height: HEIGHT,
        sample: SampleFormat::U8,
        chroma,
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct PatternArgs {}

/// Writes [pattern] into a fresh mask with no chroma plane.
struct PatternMask<R: Runtime> {
    client: ComputeClient<R>,
}

impl<R: Runtime> StageInstance for PatternMask<R> {}

impl<R: Runtime> Mask for PatternMask<R> {
    fn select(
        &mut self,
        context: &mut FrameContext,
        _inputs: Inputs,
    ) -> Result<av_toolbox_gpu::Mask, StageError> {
        let format = mask_format(None);
        let meta = FrameMeta::at(0);
        let mask = context.alloc_mask(&format, meta);
        let samples = pattern();
        let uploaded = self.client.create_from_slice(&samples);
        let plane = mask.plane();
        let plane_samples = plane.samples();

        let cubes = context.cube_count(plane_samples);
        let shape = context.cube_dim();
        let read = bytes_array::<R>(&uploaded, plane_samples);
        let write = plane.array::<u8, R>();

        // Safety: both bindings name exactly `WIDTH * HEIGHT` bytes, and the kernel writes nothing
        // past the length it is given.
        unsafe {
            copy::launch_unchecked::<R>(&self.client, cubes, shape, read, write);
        }

        Ok(mask)
    }
}

fn bytes_array<R: Runtime>(handle: &cubecl::server::Handle, len: usize) -> ArrayArg<R> {
    let owned = handle.clone();

    // Safety: every caller passes a buffer made from exactly `len` bytes.
    unsafe { ArrayArg::from_raw_parts(owned, len) }
}

fn mask_accepts_input(offered: StreamFormat) -> Result<FormatSet, StageError> {
    let sample = offered.sample();

    Ok(FormatSet::of([sample]))
}

fn mask_produces() -> Result<OutputFormats, StageError> {
    let format = mask_format(None);
    let stream = StreamFormat::Mask(format);
    let template = StreamTemplate::from_format(&stream);
    let samples = FormatSet::of([SampleFormat::U8]);

    Ok(OutputFormats { template, samples })
}

struct Pattern;

impl StageDefinition for Pattern {
    type Args = PatternArgs;

    fn package() -> &'static str {
        "av-toolbox-test"
    }

    fn identifier() -> &'static str {
        "pattern"
    }

    fn inputs() -> &'static InputSignature {
        &InputSignature::EMPTY
    }
}

impl MaskDefinition for Pattern {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        mask_accepts_input(offered)
    }

    fn produces(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        mask_produces()
    }

    fn build<R: Runtime>(
        context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Mask>, StageError> {
        let client = context.client().clone();

        Ok(Box::new(PatternMask { client }))
    }
}

struct HostPatternMask;

impl StageInstance for HostPatternMask {}

impl Mask<HostSide> for HostPatternMask {
    fn select(
        &mut self,
        _context: &mut FrameContext,
        _inputs: Inputs<HostSide>,
    ) -> Result<HostMask, StageError> {
        Err(StageError::msg("not run in this test"))
    }
}

struct HostPattern;

impl StageDefinition for HostPattern {
    type Args = PatternArgs;

    fn package() -> &'static str {
        "av-toolbox-test"
    }

    fn identifier() -> &'static str {
        "host-pattern"
    }

    fn inputs() -> &'static InputSignature {
        &InputSignature::EMPTY
    }
}

impl MaskDefinition for HostPattern {
    type Side = HostSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        mask_accepts_input(offered)
    }

    fn produces(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        mask_produces()
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Mask<HostSide>>, StageError> {
        Ok(Box::new(HostPatternMask))
    }
}

fn backend() -> Backend {
    Backend::open_default(Device::Default).expect("a backend should open")
}

/// Builds `identifier` the way a pipeline would, with its output resolved to carry `chroma`.
fn build_mask(
    backend: &Backend,
    identifier: &str,
    chroma: Option<MaskChroma>,
) -> Result<BuiltMask, BuildStageError> {
    let mut registry = PipelineRegistry::new();
    registry.register_mask::<Pattern>().expect("should register");
    registry.register_mask::<HostPattern>().expect("should register");

    let args = serde_json::json!({});
    let prepared = registry
        .prepare(StageKind::Mask, identifier, &args, &[])
        .expect("should prepare");
    let PreparedStage::Mask(prepared) = prepared else {
        panic!("expected a mask");
    };

    let format = mask_format(chroma);
    let output = StreamFormat::Mask(format);
    let formats = ResolvedFormats::new(vec![], Some(output));

    prepared.build(backend, &formats)
}

fn device_mask(backend: &Backend, chroma: Option<MaskChroma>) -> Box<dyn Mask> {
    let built = build_mask(backend, "av-toolbox-test/pattern", chroma).expect("should build");

    built.into_device().expect("a device mask")
}

/// The planes one run produced.
struct Selected {
    full: Vec<u8>,
    chroma: Option<Vec<u8>>,
}

struct RunMask {
    mask: Box<dyn Mask>,
}

impl BuildTask for RunMask {
    type Output = Selected;

    fn run<R: Runtime>(mut self, build: &BuildContext<R>) -> Self::Output {
        let mut context = build.frame_context();

        self.mask
            .start_of_stream(&mut context)
            .expect("should start the stream");

        let inputs = Inputs::new(vec![]);
        let selected = self.mask.select(&mut context, inputs).expect("should select");

        let full_plane = selected.plane();
        let full = read_plane(build, full_plane);
        let chroma = selected
            .chroma_plane()
            .map(|chroma_plane| read_plane(build, chroma_plane));

        Selected { full, chroma }
    }
}

fn read_plane<R: Runtime>(build: &BuildContext<R>, plane: &av_toolbox_gpu::Plane) -> Vec<u8> {
    let handle = plane.buffer().handle().clone();
    let bytes = build
        .client()
        .read_one(handle)
        .expect("the device should hand back what it was asked for");

    let samples = plane.samples();

    bytes[..samples].to_vec()
}

/// Mirrors [av_toolbox_cubek::reflect] on the CPU.
fn reflect(index: i32, length: i32) -> i32 {
    if length <= 1 {
        return 0;
    }

    let period = 2 * length - 2;
    let folded = index.abs() % period;
    if folded >= length { period - folded } else { folded }
}

/// Applies `across` then `down` on the CPU, rounding to U8 at the end.
fn reference_resize(input: &[u8], across: &Taps, down: &Taps) -> Vec<u8> {
    let out_width = across.starts.len() as u32;
    let out_height = down.starts.len() as u32;

    let mut rows = vec![0.0f32; (out_width * HEIGHT) as usize];
    for row in 0..HEIGHT {
        for column in 0..out_width {
            let mut total = 0.0f32;
            for step in 0..across.count {
                let tap = across.starts[column as usize] + step as i32;
                let source = reflect(tap, WIDTH as i32) as u32;
                let weight = across.weights[(column * across.count + step) as usize];
                let sample = input[(row * WIDTH + source) as usize];
                total += f32::from(sample) * weight;
            }

            rows[(row * out_width + column) as usize] = total;
        }
    }

    let mut output = Vec::with_capacity((out_width * out_height) as usize);
    for row in 0..out_height {
        for column in 0..out_width {
            let mut total = 0.0f32;
            for step in 0..down.count {
                let tap = down.starts[row as usize] + step as i32;
                let source = reflect(tap, HEIGHT as i32) as u32;
                let weight = down.weights[(row * down.count + step) as usize];
                total += rows[(source * out_width + column) as usize] * weight;
            }

            let rounded = total.round().clamp(0.0, 255.0) as u8;
            output.push(rounded);
        }
    }

    output
}

/// The 4:2:0 chroma plane of [pattern], with each chroma sample `offset_x` and `offset_y` luma
/// samples past the first it covers.
fn reference_chroma(offset_x: f32, offset_y: f32) -> Vec<u8> {
    let input = pattern();
    let across = taps(CHROMA_WIDTH, 2.0, offset_x, bilinear, BILINEAR_SUPPORT);
    let down = taps(CHROMA_HEIGHT, 2.0, offset_y, bilinear, BILINEAR_SUPPORT);

    reference_resize(&input, &across, &down)
}

fn largest_difference(found: &[u8], expected: &[u8]) -> u8 {
    found
        .iter()
        .zip(expected)
        .map(|(got, want)| got.abs_diff(*want))
        .max()
        .unwrap_or(0)
}

fn run_with_chroma(location: ChromaLocation) -> Selected {
    let backend = backend();
    let chroma = MaskChroma {
        subsampling: Subsampling::Yuv420,
        location,
    };
    let mask = device_mask(&backend, Some(chroma));

    backend.build(RunMask { mask })
}

fn assert_chroma_matches(selected: &Selected, expected: &[u8]) {
    let chroma = selected
        .chroma
        .as_ref()
        .expect("the mask should carry a chroma plane");
    let largest = largest_difference(chroma, expected);
    let written = pattern();

    assert_eq!(selected.full, written, "the full plane was changed");
    assert_eq!(chroma.len(), (CHROMA_WIDTH * CHROMA_HEIGHT) as usize);
    assert!(largest <= 1, "the chroma plane differed by {largest}");
}

#[test]
fn a_left_sited_420_mask_gets_a_downscaled_chroma_plane() {
    let selected = run_with_chroma(ChromaLocation::Left);
    let expected = reference_chroma(0.0, 0.5);

    assert_chroma_matches(&selected, &expected);
}

#[test]
fn a_centred_420_mask_gets_a_chroma_plane_sited_between_samples() {
    let left = reference_chroma(0.0, 0.5);
    let expected = reference_chroma(0.5, 0.5);
    let apart = largest_difference(&left, &expected);
    assert!(apart > 1, "the pattern should tell the two locations apart");

    let selected = run_with_chroma(ChromaLocation::Center);

    assert_chroma_matches(&selected, &expected);
}

#[test]
fn a_mask_without_chroma_in_its_output_is_left_alone() {
    let backend = backend();
    let mask = device_mask(&backend, None);

    let selected = backend.build(RunMask { mask });
    let written = pattern();

    assert_eq!(selected.full, written);
    assert!(selected.chroma.is_none());
}

#[test]
fn a_host_mask_with_chroma_in_its_output_is_refused() {
    let backend = backend();
    let chroma = MaskChroma {
        subsampling: Subsampling::Yuv420,
        location: ChromaLocation::Left,
    };

    let result = build_mask(&backend, "av-toolbox-test/host-pattern", Some(chroma));
    let Err(error) = result else {
        panic!("the host mask should not have been built");
    };

    let source = std::error::Error::source(&error).expect("should carry the stage's error");
    assert!(
        source
            .to_string()
            .contains("host masks cannot carry chroma planes")
    );
}
