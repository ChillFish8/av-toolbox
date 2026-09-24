use std::io;
use std::io::Write;

use av_toolbox_core::{
    BoundInputs,
    Claims,
    InputSignature,
    Sink,
    SinkDefinition,
    StageDefinition,
    StageError,
};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    FormatSet,
    HostSide,
    Offered,
    ResolvedFormats,
    Runtime,
    StreamFormat,
};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::shared::{INPUTS, RateArg, Y4mSink, accepts, writer_for};

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub(crate) struct StdoutArgs {
    /// Overrides the stream's frame rate, written as `"30000/1001"` or `"25"`.
    pub(crate) frame_rate: Option<RateArg>,
}

pub(crate) struct Stdout;

impl StageDefinition for Stdout {
    type Args = StdoutArgs;

    fn package() -> &'static str {
        "y4m"
    }

    fn identifier() -> &'static str {
        "stdout"
    }

    fn inputs() -> &'static InputSignature {
        INPUTS
    }
}

impl SinkDefinition for Stdout {
    type Side = HostSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        accepts(offered, args.frame_rate)
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        args: Self::Args,
        _inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Sink<HostSide>>, StageError> {
        let claims = Claims::process();
        let stdout = io::stdout();
        let sink = sink_from(claims, stdout, formats, args.frame_rate)?;

        Ok(Box::new(sink))
    }
}

fn sink_from<W: Write>(
    claims: &Claims,
    output: W,
    formats: &ResolvedFormats,
    frame_rate: Option<RateArg>,
) -> Result<Y4mSink<W>, StageError> {
    let writer = writer_for(output, formats, frame_rate)?;

    let claimed = claims.claim_stdout();
    claimed.map_err(|_| StageError::msg("stdout is already written by another y4m/stdout sink"))?;

    Ok(Y4mSink::new(writer))
}

#[cfg(test)]
mod tests {
    use av_toolbox_gpu::{FrameRate, SampleFormat, testing_context};

    use super::*;
    use crate::shared::tests::{SharedBuffer, consume, filled_frame, resolved, yuv420};

    #[test]
    fn stdout_writes_through_the_writer_it_is_given() {
        let claims = Claims::new();
        let rate = FrameRate::new(25, 1);
        let format = yuv420(4, 2, SampleFormat::U8, rate);
        let formats = resolved(format);
        let buffer = SharedBuffer::default();
        let frame = filled_frame(&format, 0, 5);

        let mut sink = sink_from(&claims, buffer.clone(), &formats, None).expect("builds");
        let mut context = testing_context();
        consume(&mut sink, frame);
        sink.end_of_stream(&mut context).expect("ends");

        let bytes = buffer.contents();
        assert!(bytes.starts_with(b"YUV4MPEG2 W4 H2 F25:1 C420\nFRAME\n"));
        assert_eq!(bytes.len(), 27 + 6 + 12);
    }

    #[test]
    fn a_second_stdout_sink_is_refused() {
        let claims = Claims::new();
        let rate = FrameRate::new(25, 1);
        let formats = resolved(yuv420(4, 2, SampleFormat::U8, rate));

        let first = sink_from(&claims, SharedBuffer::default(), &formats, None);
        let second = sink_from(&claims, SharedBuffer::default(), &formats, None);

        assert!(first.is_ok());

        let error = second.err().expect("stdout is taken");
        assert!(error.to_string().contains("already"));
    }
}
