use av_toolbox_core::{
    BuildStageError,
    Filter,
    InputArgument,
    Inputs,
    PipelineRegistry,
    PreparedStage,
    StageHandle,
    StageKind,
    StreamType,
};
use av_toolbox_gpu::{
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
    let output = Some(StreamFormat::Frames(distorted));
    let formats = ResolvedFormats::new(inputs, output);
    let backend = Backend::open_default(test_device()).expect("a backend should open");

    let filter = prepared.build(&backend, &formats)?;
    let host = filter.into_host().expect("a host filter");

    Ok(host)
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
