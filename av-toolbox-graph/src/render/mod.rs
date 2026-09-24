mod diagram;
#[cfg(test)]
pub(crate) mod fixtures;
mod html;
mod text;

pub use self::html::render_html;
pub use self::text::render_ascii;

/// What a rendered graph shows.
#[derive(Debug, Clone)]
pub struct RenderOptions {
    /// Print each stage's `package/stage` identifier under its name.
    pub show_identifier: bool,
    /// Label each edge with the input param it fills.
    pub show_param_names: bool,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            show_identifier: true,
            show_param_names: true,
        }
    }
}
