use std::io;
use std::io::{BufReader, Read};

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

    fn open(_args: &Self::Args) -> Result<Self::Opened, StageError> {
        let claims = Claims::process();
        let stdin = io::stdin();
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
}
