use std::io;
use std::io::Write;
use std::sync::{Arc, Mutex};

use av_toolbox_core::{Inputs, Sink};
use av_toolbox_gpu::{
    ChromaLocation,
    ColourInfo,
    FrameFormat,
    FrameLayout,
    FrameMeta,
    FrameRate,
    HostFrame,
    HostSide,
    HostStreamItem,
    ResolvedFormats,
    SampleFormat,
    StreamFormat,
    Subsampling,
    testing_context,
};

/// A writer whose bytes can still be read after it has been moved into a sink.
#[derive(Debug, Clone, Default)]
pub(crate) struct SharedBuffer(Arc<Mutex<Vec<u8>>>);

impl SharedBuffer {
    pub(crate) fn contents(&self) -> Vec<u8> {
        self.0.lock().expect("not poisoned").clone()
    }
}

impl Write for SharedBuffer {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.0.lock().expect("not poisoned").extend_from_slice(buffer);

        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub(crate) fn yuv420(width: u32, height: u32, sample: SampleFormat, rate: Option<FrameRate>) -> FrameFormat {
    FrameFormat {
        width,
        height,
        layout: FrameLayout::Yuv {
            subsampling: Subsampling::Yuv420,
        },
        sample,
        colour: ColourInfo {
            chroma_location: ChromaLocation::Center,
            ..ColourInfo::default()
        },
        frame_rate: rate,
        field_order: None,
    }
}

pub(crate) fn resolved(format: FrameFormat) -> ResolvedFormats {
    let input = StreamFormat::Frames(format);

    ResolvedFormats::new(vec![Some(input)], None)
}

/// A frame whose every byte is `value`.
pub(crate) fn filled_frame(format: &FrameFormat, index: u64, value: u8) -> HostFrame {
    let context = testing_context();
    let mut frame = context.alloc_host_frame(format, FrameMeta::at(index));

    for channel in format.channels() {
        let plane = frame.plane_mut(*channel).expect("the layout carries it");
        plane.fill(value);
    }

    frame.freeze()
}

pub(crate) fn consume<S: Sink<HostSide>>(sink: &mut S, frame: HostFrame) {
    let mut context = testing_context();
    let item = HostStreamItem::from(frame);
    let inputs = Inputs::<HostSide>::from_slots(vec![Some(item)]);

    sink.consume(&mut context, inputs).expect("consumes");
}
