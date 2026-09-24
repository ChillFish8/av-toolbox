use std::io;
use std::io::{BufReader, IsTerminal, Read};

use av_toolbox_core::{
    BoundInputs,
    Claims,
    InputSignature,
    Source,
    SourceDefinition,
    StageDefinition,
    StageError,
};
use av_toolbox_gpu::{BuildContext, Capabilities, HostSide, OutputFormats, ResolvedFormats, Runtime};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::shared::{Y4mReader, Y4mSource, produced_by};

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub(crate) struct StdinArgs {}

pub(crate) type Opened = Y4mReader<BufReader<io::Stdin>>;

pub(crate) struct Stdin;

impl StageDefinition for Stdin {
    type Args = StdinArgs;

    fn package() -> &'static str {
        "y4m"
    }

    fn identifier() -> &'static str {
        "stdin"
    }

    fn inputs() -> &'static InputSignature {
        &InputSignature::EMPTY
    }
}

impl SourceDefinition for Stdin {
    type Side = HostSide;
    type Opened = Opened;

    fn reads_stream(_args: &Self::Args) -> bool {
        true
    }

    fn open(_args: &Self::Args) -> Result<Self::Opened, StageError> {
        let claims = Claims::process();
        let stdin = io::stdin();
        let is_terminal = stdin.is_terminal();

        tracing::info!("y4m/stdin is reading a y4m stream from standard input");

        if let Some(notice) = waiting_notice(is_terminal) {
            tracing::warn!("{notice}");
        }

        let reader = BufReader::new(stdin);

        open_from(claims, reader)
    }

    fn produces(_capabilities: &Capabilities, opened: &Self::Opened) -> Result<OutputFormats, StageError> {
        let format = opened.format();
        let produced = produced_by(format);

        Ok(produced)
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        opened: Self::Opened,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Source<HostSide>>, StageError> {
        let source = Y4mSource::new(opened);

        Ok(Box::new(source))
    }
}

/// What to log before reading standard input, louder when nothing is piped in.
fn waiting_notice(is_terminal: bool) -> Option<&'static str> {
    if is_terminal {
        let notice = "y4m/stdin is waiting for a y4m stream on standard input, which is a terminal. Pipe one in, such as `< input.y4m`.";

        return Some(notice);
    }

    None
}

fn open_from<R: Read>(claims: &Claims, reader: R) -> Result<Y4mReader<R>, StageError> {
    let claimed = claims.claim_stdin();
    claimed.map_err(|_| StageError::msg("stdin is already read by another y4m/stdin source"))?;

    Y4mReader::open(reader)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    #[test]
    fn stdin_opens_through_the_reader_it_is_given() {
        let claims = Claims::new();
        let bytes = crate::shared::tests::y4m_header_only();

        let opened = open_from(&claims, Cursor::new(bytes)).expect("opens");

        assert_eq!(opened.format().width, 4);
    }

    #[test]
    fn a_second_stdin_open_is_refused() {
        let claims = Claims::new();
        let first_bytes = crate::shared::tests::y4m_header_only();
        let second_bytes = crate::shared::tests::y4m_header_only();

        let first = open_from(&claims, Cursor::new(first_bytes));
        let second = open_from(&claims, Cursor::new(second_bytes));

        assert!(first.is_ok());
        assert!(second.is_err());
    }

    #[test]
    fn a_terminal_on_stdin_gets_a_waiting_warning() {
        let notice = waiting_notice(true).expect("a notice");

        assert!(notice.contains("waiting"));
        assert!(notice.contains("< input.y4m"));
    }

    #[test]
    fn a_piped_stdin_gets_no_warning() {
        let notice = waiting_notice(false);

        assert!(notice.is_none());
    }

    #[test]
    fn stdin_reads_a_live_stream() {
        assert!(Stdin::reads_stream(&StdinArgs {}));
    }
}
