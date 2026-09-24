use codespan_reporting::term::termcolor::{Color, ColorSpec, StandardStream, WriteColor};

use crate::cli::ColorMode;

pub(crate) fn stderr(mode: ColorMode) -> StandardStream {
    let choice = mode.choice();
    StandardStream::stderr(choice)
}

/// Prints `error: <message>` with the level coloured.
pub(crate) fn error_line(writer: &mut dyn WriteColor, message: &str) {
    // Nothing is left to report a failed write to stderr to.
    let _ = write_line(writer, "error", Color::Red, message);
}

fn write_line(writer: &mut dyn WriteColor, level: &str, colour: Color, message: &str) -> std::io::Result<()> {
    let mut spec = ColorSpec::new();
    spec.set_fg(Some(colour)).set_bold(true);

    writer.set_color(&spec)?;
    write!(writer, "{level}")?;
    writer.reset()?;
    writeln!(writer, ": {message}")
}

#[cfg(test)]
mod tests {
    use codespan_reporting::term::termcolor::NoColor;

    use super::*;

    #[test]
    fn an_error_line_names_its_level() {
        let mut writer = NoColor::new(Vec::new());

        error_line(&mut writer, "cannot read pipeline.toml");
        let written = String::from_utf8(writer.into_inner()).expect("utf-8");

        assert_eq!(written, "error: cannot read pipeline.toml\n");
    }
}
