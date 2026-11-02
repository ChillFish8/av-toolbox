use av_toolbox_core::{Filter, Inputs};
use av_toolbox_gpu::{
    BuildContext,
    BuildTask,
    Frame,
    FrameContext,
    FrameFormat,
    FrameMeta,
    HostPool,
    HostStreamItem,
    StreamItem,
};

/// One frame handed to the filter, its metadata and every plane's bytes.
#[derive(Clone)]
pub(crate) struct Input {
    pub(crate) meta: FrameMeta,
    pub(crate) planes: Vec<Vec<u8>>,
}

/// One frame the filter emitted, its metadata and every plane's bytes.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Output {
    pub(crate) meta: FrameMeta,
    pub(crate) planes: Vec<Vec<u8>>,
}

/// Runs every chunk of `chunks` through one filter instance in turn, starting and ending the stream around
/// each.
pub(crate) struct Stream {
    pub(crate) filter: Box<dyn Filter>,
    pub(crate) input: FrameFormat,
    pub(crate) output: FrameFormat,
    pub(crate) chunks: Vec<Vec<Input>>,
}

fn read_back(build: &BuildContext, format: &FrameFormat, emitted: Frame) -> Output {
    assert_eq!(emitted.format(), format);

    let mut planes = Vec::new();
    for channel in format.channels() {
        let plane = emitted.plane(*channel).expect("the output carries this channel");
        let handle = plane.buffer().handle().clone();
        let bytes = build
            .client()
            .read_one(handle)
            .expect("the device should hand back the plane");
        planes.push(bytes.to_vec());
    }

    Output {
        meta: emitted.meta().clone(),
        planes,
    }
}

/// Runs one chunk through `filter`, starting and ending the stream around it.
pub(crate) fn run_chunk(
    build: &BuildContext,
    context: &mut FrameContext,
    filter: &mut dyn Filter,
    formats: (&FrameFormat, &FrameFormat),
    chunk: Vec<Input>,
) -> Vec<Output> {
    let (input_format, output_format) = formats;
    let host = HostPool::new();
    let mut outputs = Vec::new();

    filter.start_of_stream(context).expect("should start the stream");

    for input in chunk {
        let mut frame = host.alloc_frame(input_format, input.meta);
        for (index, channel) in input_format.channels().iter().enumerate() {
            let plane = frame
                .plane_mut(*channel)
                .expect("the format carries this channel");
            plane.copy_from_slice(&input.planes[index]);
        }

        let frozen = frame.freeze();
        let item = HostStreamItem::Frame(frozen);
        let loaded: StreamItem = context.load_host(&item).expect("the item should load");
        let inputs = Inputs::new(vec![Some(loaded)]);

        let emitted = filter.process(context, inputs).expect("should process");
        for frame in emitted {
            let output = read_back(build, output_format, frame);
            outputs.push(output);
        }
    }

    let flushed = filter.end_of_stream(context).expect("should end the stream");
    for frame in flushed {
        let output = read_back(build, output_format, frame);
        outputs.push(output);
    }

    outputs
}

impl BuildTask for Stream {
    type Output = Vec<Output>;

    fn run(mut self, build: &BuildContext) -> Self::Output {
        let mut context = build.frame_context();
        let mut outputs = Vec::new();
        for chunk in self.chunks {
            let formats = (&self.input, &self.output);
            let emitted = run_chunk(build, &mut context, self.filter.as_mut(), formats, chunk);
            outputs.extend(emitted);
        }

        outputs
    }
}
