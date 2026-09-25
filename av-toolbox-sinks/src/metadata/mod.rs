mod filter;
mod line;

use std::io;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use av_toolbox_core::{
    BoundInputs,
    InputSignature,
    MetaInputs,
    MetaSink,
    MetaSinkDefinition,
    StageDefinition,
    StageError,
    StageInstance,
};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    FormatSet,
    FrameContext,
    Offered,
    ResolvedFormats,
    Runtime,
    StreamFormat,
};
use schemars::JsonSchema;
use serde::Deserialize;

use self::filter::{KeyFilter, Pattern};
use self::line::write_line;
use crate::shared::{INPUTS, open_target};

const STAGE: &str = "metadata/ndjson";

/// Arguments for `metadata/ndjson`, which takes exactly one of `fd` or `path`.
#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct NdjsonArgs {
    /// An inherited file descriptor, Unix only.
    pub(crate) fd: Option<u32>,
    /// A FIFO, or a file to create or truncate.
    pub(crate) path: Option<PathBuf>,
    /// Keys to keep, where `*` matches any run of characters. Every key is kept when unset.
    pub(crate) include: Option<Vec<Pattern>>,
    /// Keys to drop, which wins over `include`.
    pub(crate) exclude: Option<Vec<Pattern>>,
}

pub(crate) struct Ndjson;

impl StageDefinition for Ndjson {
    type Args = NdjsonArgs;

    fn package() -> &'static str {
        "metadata"
    }

    fn identifier() -> &'static str {
        "ndjson"
    }

    fn inputs() -> &'static InputSignature {
        INPUTS
    }
}

impl MetaSinkDefinition for Ndjson {
    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        accepts(offered)
    }

    /// Opens the descriptor or path.
    ///
    /// Opening a FIFO waits until something opens it to read.
    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn MetaSink>, StageError> {
        let path = args.path.as_deref();
        let file = open_target(STAGE, args.fd, path)?;
        let filter = KeyFilter::new(args.include, args.exclude);
        let sink = NdjsonSink::new(file, filter);

        Ok(Box::new(sink))
    }
}

/// The samples `metadata/ndjson` will take for `offered`, which has to be a frame stream.
fn accepts(offered: StreamFormat) -> Result<FormatSet, StageError> {
    let StreamFormat::Frames(_) = offered else {
        return Err(StageError::msg("metadata/ndjson logs frames, not a mask"));
    };

    Ok(FormatSet::of([offered.sample()]))
}

/// A sink writing each frame's metadata as one line of JSON.
pub(crate) struct NdjsonSink<W: Write> {
    output: BufWriter<W>,
    filter: KeyFilter,
}

impl<W: Write> NdjsonSink<W> {
    pub(crate) fn new(output: W, filter: KeyFilter) -> Self {
        Self {
            output: BufWriter::new(output),
            filter,
        }
    }
}

impl<W: Write + Send + 'static> StageInstance for NdjsonSink<W> {}

impl<W: Write + Send + 'static> MetaSink for NdjsonSink<W> {
    fn consume(&mut self, _context: &mut FrameContext, mut inputs: MetaInputs) -> Result<(), StageError> {
        let Some(item) = inputs.take(0) else {
            return Ok(());
        };

        let meta = item.meta();
        let written = write_line(&mut self.output, meta, &self.filter);

        written.map_err(|error| {
            let position = format!("frame {}", meta.index);

            write_error(&position, &error)
        })
    }

    fn end_of_stream(&mut self, _context: &mut FrameContext) -> Result<(), StageError> {
        let flushed = self.output.flush();

        flushed.map_err(|error| flush_error(&error))
    }
}

fn write_error(position: &str, error: &io::Error) -> StageError {
    let message = match error.kind() {
        io::ErrorKind::BrokenPipe => {
            format!("the reader closed the metadata stream before {position} was written")
        },
        _ => format!("writing {position} of the metadata stream failed: {error}"),
    };

    StageError::msg(message)
}

fn flush_error(error: &io::Error) -> StageError {
    let message = match error.kind() {
        io::ErrorKind::BrokenPipe => {
            "the reader closed the metadata stream before it was flushed".to_string()
        },
        _ => format!("flushing the metadata stream failed: {error}"),
    };

    StageError::msg(message)
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::{fs, io};

    use av_toolbox_core::{end_meta_sink, run_meta_sink};
    use av_toolbox_gpu::{AnyItem, HostStreamItem, MaskFormat, MetaValue, SampleFormat, testing_context};

    use super::*;
    use crate::shared::testing::{SharedBuffer, filled_frame, yuv420};

    fn frame_item(index: u64, tagged: bool) -> AnyItem {
        let format = yuv420(4, 2, SampleFormat::U8, None);
        let mut frame = filled_frame(&format, index, 0);

        if tagged {
            frame.meta_mut().attach("seen", MetaValue::Bool(true));
            frame.meta_mut().attachments.stamp("filter.tag");
        }

        AnyItem::Host(HostStreamItem::from(frame))
    }

    fn run<W: Write + Send + 'static>(
        sink: &mut NdjsonSink<W>,
        items: Vec<AnyItem>,
    ) -> Result<(), StageError> {
        let mut context = testing_context();

        for item in items {
            run_meta_sink(sink, &mut context, vec![Some(item)])?;
        }

        end_meta_sink(sink, &mut context)
    }

    #[test]
    fn each_frame_is_one_line_in_order() {
        let buffer = SharedBuffer::default();
        let mut sink = NdjsonSink::new(buffer.clone(), KeyFilter::default());

        run(&mut sink, vec![frame_item(0, true), frame_item(1, false)]).expect("runs");

        let text = String::from_utf8(buffer.contents()).expect("utf-8");
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].starts_with("{\"index\":0,"));
        assert!(lines[0].ends_with("\"filter.tag\":{\"seen\":true}}"));
        assert!(lines[1].starts_with("{\"index\":1,"));
    }

    #[test]
    fn an_empty_stream_writes_nothing() {
        let buffer = SharedBuffer::default();
        let mut sink = NdjsonSink::new(buffer.clone(), KeyFilter::default());

        run(&mut sink, Vec::new()).expect("runs");

        assert!(buffer.contents().is_empty());
    }

    #[test]
    fn ending_twice_is_harmless() {
        let buffer = SharedBuffer::default();
        let mut sink = NdjsonSink::new(buffer.clone(), KeyFilter::default());
        let mut context = testing_context();

        run(&mut sink, vec![frame_item(0, false)]).expect("runs");
        end_meta_sink(&mut sink, &mut context).expect("ends again");

        let text = String::from_utf8(buffer.contents()).expect("utf-8");
        assert_eq!(text.lines().count(), 1);
    }

    #[test]
    fn a_closed_reader_gives_the_broken_pipe_error() {
        struct ClosedReader;

        impl Write for ClosedReader {
            fn write(&mut self, _buffer: &[u8]) -> io::Result<usize> {
                Err(io::ErrorKind::BrokenPipe.into())
            }

            fn flush(&mut self) -> io::Result<()> {
                Err(io::ErrorKind::BrokenPipe.into())
            }
        }

        let mut sink = NdjsonSink::new(ClosedReader, KeyFilter::default());

        let error = run(&mut sink, vec![frame_item(0, false)]).expect_err("the reader is gone");

        assert!(error.to_string().contains("reader closed"), "{error}");
    }

    #[test]
    fn a_mask_is_refused() {
        let offered = StreamFormat::Mask(MaskFormat {
            width: 4,
            height: 2,
            sample: SampleFormat::U8,
            chroma: None,
        });

        let result = accepts(offered);

        assert!(result.is_err());
    }

    #[test]
    fn both_fd_and_path_is_refused() {
        let result = open_target("metadata/ndjson", Some(3), Some(Path::new("a.ndjson")));

        let error = result.expect_err("both given");
        assert!(error.to_string().contains("metadata/ndjson"), "{error}");
    }

    #[test]
    fn a_path_is_written_through() {
        let folder = tempfile::tempdir().expect("a temp dir");
        let path = folder.path().join("out.ndjson");
        let file = open_target("metadata/ndjson", None, Some(&path)).expect("opens");
        let mut sink = NdjsonSink::new(file, KeyFilter::default());

        run(&mut sink, vec![frame_item(0, false)]).expect("runs");
        drop(sink);

        let text = fs::read_to_string(&path).expect("reads");
        assert_eq!(text.lines().count(), 1);
    }

    #[test]
    fn the_arguments_deserialize_include_and_exclude() {
        let value = serde_json::json!({
            "path": "out.ndjson",
            "include": ["index", "filter.*"],
            "exclude": ["filter.debug_*"],
        });

        let args: NdjsonArgs = serde_json::from_value(value).expect("parses");

        assert_eq!(args.include.map(|patterns| patterns.len()), Some(2));
        assert_eq!(args.exclude.map(|patterns| patterns.len()), Some(1));
    }
}
