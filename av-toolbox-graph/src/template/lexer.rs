use std::ops::Range;

use super::TemplateError;

#[derive(Debug, Clone, PartialEq)]
pub(super) enum TokenKind {
    Integer(i64),
    Float(f64),
    String(String),
    Name(String),
    Dot,
    Comma,
    OpenParen,
    CloseParen,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    EqualEqual,
    BangEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    If,
    Then,
    Else,
    And,
    Or,
    Not,
    True,
    False,
    None,
    Is,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct Token {
    pub(super) kind: TokenKind,
    /// Byte range inside the template text.
    pub(super) range: Range<usize>,
}

/// Splits one expression into tokens.
///
/// `offset` is where `text` starts inside the template, so every range points into the template.
pub(super) fn tokenize(text: &str, offset: usize) -> Result<Vec<Token>, TemplateError> {
    let mut lexer = Lexer {
        text,
        offset,
        position: 0,
    };
    let mut tokens = Vec::new();

    while let Some(token) = lexer.next_token()? {
        tokens.push(token);
    }

    Ok(tokens)
}

struct Lexer<'a> {
    text: &'a str,
    offset: usize,
    position: usize,
}

impl Lexer<'_> {
    fn next_token(&mut self) -> Result<Option<Token>, TemplateError> {
        self.skip_whitespace();

        let Some(character) = self.peek() else {
            return Ok(None);
        };

        let start = self.position;
        let kind = if character.is_ascii_digit() {
            self.number()?
        } else if character.is_ascii_alphabetic() || character == '_' {
            self.word()
        } else if character == '"' {
            self.string()?
        } else {
            self.symbol(character)?
        };

        let range = self.range(start, self.position);
        Ok(Some(Token { kind, range }))
    }

    fn peek(&self) -> Option<char> {
        self.text[self.position..].chars().next()
    }

    fn peek_second(&self) -> Option<char> {
        let mut rest = self.text[self.position..].chars();
        rest.next();
        rest.next()
    }

    fn advance(&mut self) {
        if let Some(character) = self.peek() {
            self.position += character.len_utf8();
        }
    }

    fn advance_while(&mut self, keep: impl Fn(char) -> bool) {
        while self.peek().is_some_and(&keep) {
            self.advance();
        }
    }

    fn skip_whitespace(&mut self) {
        self.advance_while(char::is_whitespace);
    }

    fn range(&self, start: usize, end: usize) -> Range<usize> {
        self.offset + start..self.offset + end
    }

    fn error(&self, message: String, start: usize) -> TemplateError {
        let range = self.range(start, self.position);
        TemplateError {
            message,
            range,
            checked: None,
            variable: None,
        }
    }

    fn number(&mut self) -> Result<TokenKind, TemplateError> {
        let start = self.position;
        let mut is_float = false;
        self.advance_while(|character| character.is_ascii_digit());

        let fraction_follows = self
            .peek_second()
            .is_some_and(|character| character.is_ascii_digit());
        if self.peek() == Some('.') && fraction_follows {
            is_float = true;
            self.advance();
            self.advance_while(|character| character.is_ascii_digit());
        }

        if matches!(self.peek(), Some('e' | 'E')) && self.exponent_follows() {
            is_float = true;
            self.advance();
            if matches!(self.peek(), Some('+' | '-')) {
                self.advance();
            }

            self.advance_while(|character| character.is_ascii_digit());
        }

        let written = &self.text[start..self.position];
        if !is_float {
            let parsed = written.parse::<i64>();
            let Ok(number) = parsed else {
                let message = format!("{written} does not fit in a 64-bit integer");
                return Err(self.error(message, start));
            };

            return Ok(TokenKind::Integer(number));
        }

        let parsed = written.parse::<f64>();
        match parsed {
            Ok(number) if number.is_finite() => Ok(TokenKind::Float(number)),
            _ => {
                let message = format!("{written} is not a finite number");
                Err(self.error(message, start))
            },
        }
    }

    /// Whether the `e` at the cursor starts an exponent, such as `e3` or `e-3`.
    fn exponent_follows(&self) -> bool {
        let mut rest = self.text[self.position..].chars();
        rest.next();

        let mut next = rest.next();
        if matches!(next, Some('+' | '-')) {
            next = rest.next();
        }

        next.is_some_and(|character| character.is_ascii_digit())
    }

    fn word(&mut self) -> TokenKind {
        let start = self.position;
        self.advance_while(|character| character.is_ascii_alphanumeric() || character == '_');

        let written = &self.text[start..self.position];
        match written {
            "if" => TokenKind::If,
            "then" => TokenKind::Then,
            "else" => TokenKind::Else,
            "and" => TokenKind::And,
            "or" => TokenKind::Or,
            "not" => TokenKind::Not,
            "true" => TokenKind::True,
            "false" => TokenKind::False,
            "none" => TokenKind::None,
            "is" => TokenKind::Is,
            name => TokenKind::Name(name.to_string()),
        }
    }

    fn string(&mut self) -> Result<TokenKind, TemplateError> {
        let start = self.position;
        self.advance();

        let content_start = self.position;
        self.advance_while(|character| character != '"');

        if self.peek().is_none() {
            let message = "a string opened with \" is never closed".to_string();
            return Err(self.error(message, start));
        }

        let content = self.text[content_start..self.position].to_string();
        self.advance();

        Ok(TokenKind::String(content))
    }

    fn symbol(&mut self, character: char) -> Result<TokenKind, TemplateError> {
        let start = self.position;
        let followed_by_equals = self.peek_second() == Some('=');
        self.advance();

        let kind = match character {
            '.' => TokenKind::Dot,
            ',' => TokenKind::Comma,
            '(' => TokenKind::OpenParen,
            ')' => TokenKind::CloseParen,
            '+' => TokenKind::Plus,
            '-' => TokenKind::Minus,
            '*' => TokenKind::Star,
            '/' => TokenKind::Slash,
            '%' => TokenKind::Percent,
            '<' if followed_by_equals => TokenKind::LessEqual,
            '<' => TokenKind::Less,
            '>' if followed_by_equals => TokenKind::GreaterEqual,
            '>' => TokenKind::Greater,
            '=' if followed_by_equals => TokenKind::EqualEqual,
            '!' if followed_by_equals => TokenKind::BangEqual,
            _ => {
                let message = format!("unexpected character {character:?}");
                return Err(self.error(message, start));
            },
        };

        if matches!(
            kind,
            TokenKind::LessEqual | TokenKind::GreaterEqual | TokenKind::EqualEqual | TokenKind::BangEqual
        ) {
            self.advance();
        }

        Ok(kind)
    }
}
