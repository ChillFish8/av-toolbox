use av_toolbox_core::{
    BuildStageError,
    Filter,
    InputArgument,
    Inputs,
    PipelineRegistry,
    PreparedStage,
    StageError,
    StageHandle,
    StageKind,
    StreamType,
};
use av_toolbox_cubek::depth::{Conversion, convert_to_float_sample};
use av_toolbox_cubek::matrix::SampleRange;
use av_toolbox_gpu::{
    Attachments,
    Backend,
    Channel,
    ChromaLocation,
    ColourInfo,
    ColourRange,
    FrameFormat,
    FrameLayout,
    FrameMeta,
    HostFrame,
    HostPool,
    HostSide,
    HostStreamItem,
    Matrix,
    MetaRecord,
    MetaValue,
    Primaries,
    ResolvedFormats,
    SampleFormat,
    StreamFormat,
    Subsampling,
    Transfer,
    test_device,
    testing_context,
};

/// The width and height of every test clip.
pub(crate) const SIZE: u32 = 256;

/// What a run of a stage handed back.
pub(crate) struct Run {
    pub(crate) frames: Vec<HostFrame>,
    pub(crate) records: Vec<MetaRecord>,
}

/// A BT.709 limited range 4:2:0 clip of [SIZE] by [SIZE] in `sample`.
pub(crate) fn yuv420(sample: SampleFormat) -> FrameFormat {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let colour = ColourInfo {
        range: ColourRange::Limited,
        matrix: Matrix::Bt709,
        primaries: Primaries::Bt709,
        transfer: Transfer::Bt709,
        chroma_location: ChromaLocation::Left,
    };

    FrameFormat {
        width: SIZE,
        height: SIZE,
        layout,
        sample,
        colour,
        frame_rate: None,
        field_order: None,
    }
}

/// Builds `stage` from `args` over a reference and a distorted clip, the way a pipeline would.
///
/// The output format is the first one the stage's `produces` offers.
pub(crate) fn built(
    stage: &str,
    args: serde_json::Value,
    reference: FrameFormat,
    distorted: FrameFormat,
) -> Result<Box<dyn Filter<HostSide>>, BuildStageError> {
    let mut registry = PipelineRegistry::new();
    crate::register(&mut registry).expect("the filters register");

    let reference_handle = StageHandle::new(1);
    let distorted_handle = StageHandle::new(2);
    let arguments = [
        InputArgument::positional(reference_handle, StreamType::Frames),
        InputArgument::positional(distorted_handle, StreamType::Frames),
    ];
    let prepared = registry
        .prepare(StageKind::Filter, stage, &args, &arguments)
        .expect("should prepare");
    let PreparedStage::Filter(prepared) = prepared else {
        panic!("expected a filter");
    };

    let inputs = vec![
        Some(StreamFormat::Frames(reference)),
        Some(StreamFormat::Frames(distorted)),
    ];
    let backend = Backend::open_default(test_device()).expect("a backend should open");
    let produced = match prepared.produces(backend.capabilities(), &inputs) {
        Ok(produced) => produced,
        Err(source) => return Err(produce_error(stage, source)),
    };

    let sample = produced.samples.preferences()[0];
    let output = produced.template.with_sample(sample);
    let formats = ResolvedFormats::new(inputs, Some(output));

    let filter = prepared.build(&backend, &formats)?;
    let host = filter.into_host().expect("a host filter");

    Ok(host)
}

fn produce_error(stage: &str, source: StageError) -> BuildStageError {
    BuildStageError::Build {
        identifier: stage.to_string(),
        source,
    }
}

/// Runs every pair of frames through `filter` as one stream.
pub(crate) fn run(filter: &mut dyn Filter<HostSide>, pairs: Vec<(HostFrame, HostFrame)>) -> Run {
    let mut context = testing_context();
    let mut frames = Vec::new();

    filter
        .start_of_stream(&mut context)
        .expect("should start the stream");

    for (reference, distorted) in pairs {
        let slots = vec![
            Some(HostStreamItem::Frame(reference)),
            Some(HostStreamItem::Frame(distorted)),
        ];
        let inputs = Inputs::from_slots(slots);

        let emitted = filter.process(&mut context, inputs).expect("should process");
        frames.extend(emitted);
    }

    let flushed = filter.end_of_stream(&mut context).expect("should end the stream");
    frames.extend(flushed);

    let records = context.take_records();

    Run { frames, records }
}

/// The stage's own reason for refusing to build.
pub(crate) fn build_error(
    stage: &str,
    args: serde_json::Value,
    reference: FrameFormat,
    distorted: FrameFormat,
) -> String {
    let refused = built(stage, args, reference, distorted);
    let Err(error) = refused else {
        panic!("the stage should not build");
    };

    match error {
        BuildStageError::Build { source, .. } => source.to_string(),
        other => panic!("expected a build failure, got {other}"),
    }
}

pub(crate) fn float_named(attachments: &Attachments, name: &str) -> f64 {
    let found = attachments
        .stages()
        .flat_map(|(_, values)| values.iter())
        .find(|(key, _)| key == name);

    match found {
        Some((_, MetaValue::Float(value))) => *value,
        other => panic!("expected a float named {name}, found {other:?}"),
    }
}

/// The value `stage` attached to `frame` under `name`.
pub(crate) fn frame_float(frame: &HostFrame, stage: &str, name: &str) -> f64 {
    let mut attachments = frame.meta().attachments.clone();
    attachments.stamp(stage);

    float_named(&attachments, name)
}

/// The names and values of the run's one `summary` record, in the order they were written.
pub(crate) fn summary_values(run: &Run, stage: &str) -> Vec<(String, MetaValue)> {
    assert_eq!(run.records.len(), 1);
    let record = &run.records[0];
    assert_eq!(record.kind(), "summary");

    let mut attachments = record.attachments().clone();
    attachments.stamp(stage);

    attachments
        .stages()
        .flat_map(|(_, values)| values.iter().cloned())
        .collect()
}

/// The `Y` plane of an `F32` frame, read as little-endian floats.
pub(crate) fn f32_plane(frame: &HostFrame) -> Vec<f32> {
    let plane = frame.plane(Channel::Y).expect("the frame has a Y plane");

    let (values, _) = plane.bytes().as_chunks::<4>();

    values.iter().map(|bytes| f32::from_le_bytes(*bytes)).collect()
}

/// A frame of `format` at `index`, with luma from `luma` and neutral chroma.
///
/// `luma` gives an 8-bit value for a row and column, which is scaled up to the clip's depth.
pub(crate) fn frame(format: &FrameFormat, index: u64, luma: fn(u32, u32) -> u32) -> HostFrame {
    let host = HostPool::new();
    let mut frame = host.alloc_frame(format, FrameMeta::at(index));

    for (channel, plane) in frame.planes_mut() {
        let (width, height) = format
            .channel_dimensions(channel)
            .expect("the format carries this channel");

        for row in 0..height {
            for column in 0..width {
                let value = match channel {
                    Channel::Y => luma(row, column),
                    _ => 128,
                };
                let position = (row * width + column) as usize;
                write_sample(plane, format.sample, position, value);
            }
        }
    }

    frame.freeze()
}

/// A smooth gradient with a fine checkerboard on top, so a metric has structure to compare.
pub(crate) fn detailed_luma(row: u32, column: u32) -> u32 {
    let gradient = 16 + (row + column) * 200 / (2 * SIZE);
    let detail = if (row / 4 + column / 4).is_multiple_of(2) {
        10
    } else {
        0
    };

    gradient + detail
}

/// [detailed_luma] with its checkerboard blurred away.
pub(crate) fn blurred_luma(row: u32, column: u32) -> u32 {
    let gradient = 16 + (row + column) * 200 / (2 * SIZE);

    gradient + 5
}

/// Writes the 8-bit `value`, scaled to `sample`'s depth, as sample `position` of `plane`.
fn write_sample(plane: &mut [u8], sample: SampleFormat, position: usize, value: u32) {
    match sample {
        SampleFormat::U8 => plane[position] = value as u8,
        SampleFormat::U16 { depth } => {
            let shift = depth.bits() - 8;
            let scaled = (value << shift) as u16;
            let start = position * 2;
            plane[start..start + 2].copy_from_slice(&scaled.to_le_bytes());
        },
        _ => panic!("the test clips are integer"),
    }
}

/// `source` as an F32 clip, converted the way `resample/depth` turns limited 8-bit codes into floats.
pub(crate) fn float_copy(source: &FrameFormat, frame: &HostFrame, index: u64) -> (FrameFormat, HostFrame) {
    let float_format = FrameFormat {
        sample: SampleFormat::F32,
        ..*source
    };
    let luma_conversion = Conversion::to_float(8, SampleRange::Limited, false);
    let chroma_conversion = Conversion::to_float(8, SampleRange::Limited, true);

    let host = HostPool::new();
    let mut converted = host.alloc_frame(&float_format, FrameMeta::at(index));

    for (channel, plane) in converted.planes_mut() {
        let conversion = match channel {
            Channel::Y => &luma_conversion,
            _ => &chroma_conversion,
        };
        let codes = frame.plane(channel).expect("the source has this plane").bytes();
        let (samples, _) = plane.as_chunks_mut::<4>();

        for (sample, code) in samples.iter_mut().zip(codes) {
            let value = convert_to_float_sample(f32::from(*code), conversion);
            *sample = value.to_le_bytes();
        }
    }

    (float_format, converted.freeze())
}
