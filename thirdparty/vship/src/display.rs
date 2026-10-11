use std::ffi::{CString, c_char};
use std::ptr;

use crate::error::VshipError;

/// The display CVVDP models viewing on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DisplayModel {
    Standard4k,
    StandardFhd,
    StandardHdrPq,
    StandardHdrHlg,
    StandardHdrDark,
    StandardHdrLinearZoom,
    /// Adds a display, or overrides a built-in one, from inline JSON in Vship's display config
    /// format.
    Custom {
        name: String,
        json: String,
    },
}

impl DisplayModel {
    fn key(&self) -> &str {
        match self {
            Self::Standard4k => "standard_4k",
            Self::StandardFhd => "standard_fhd",
            Self::StandardHdrPq => "standard_hdr_pq",
            Self::StandardHdrHlg => "standard_hdr_hlg",
            Self::StandardHdrDark => "standard_hdr_dark",
            Self::StandardHdrLinearZoom => "standard_hdr_linear_zoom",
            Self::Custom { name, .. } => name,
        }
    }

    /// Refuses custom JSON that Vship would read as a file path.
    pub(crate) fn check(&self) -> Result<(), VshipError> {
        let Self::Custom { json, .. } = self else {
            return Ok(());
        };

        let trimmed = json.trim_start();
        if !trimmed.starts_with('{') {
            return Err(not_inline_json_error());
        }

        Ok(())
    }

    /// The model key and config Vship's init takes.
    pub(crate) fn c_strings(&self) -> Result<DisplayStrings, VshipError> {
        let key = c_string(self.key(), "display name")?;
        let config = match self {
            Self::Custom { json, .. } => {
                let trimmed = json.trim_start();
                Some(c_string(trimmed, "display config")?)
            },
            _ => None,
        };

        Ok(DisplayStrings { key, config })
    }
}

/// A display model's key and config as C strings, kept alive across Vship's init.
pub(crate) struct DisplayStrings {
    key: CString,
    config: Option<CString>,
}

impl DisplayStrings {
    pub(crate) fn key_pointer(&self) -> *const c_char {
        self.key.as_ptr()
    }

    /// The config pointer, null for a built-in display.
    pub(crate) fn config_pointer(&self) -> *const c_char {
        match &self.config {
            Some(config) => config.as_ptr(),
            None => ptr::null(),
        }
    }
}

fn c_string(text: &str, name: &str) -> Result<CString, VshipError> {
    match CString::new(text) {
        Ok(converted) => Ok(converted),
        Err(_) => Err(nul_byte_error(name)),
    }
}

fn not_inline_json_error() -> VshipError {
    VshipError::new("a custom display config must be inline JSON starting with `{`, not a path")
}

fn nul_byte_error(name: &str) -> VshipError {
    VshipError::new(format!("the {name} carries a nul byte, which vship cannot take"))
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;

    use super::*;

    fn key_of(model: &DisplayModel) -> String {
        let strings = model.c_strings().expect("converts the model");
        // SAFETY: `key_pointer` points at a NUL terminated string `strings` owns.
        let key = unsafe { CStr::from_ptr(strings.key_pointer()) };

        key.to_string_lossy().into_owned()
    }

    fn custom(name: &str, json: &str) -> DisplayModel {
        DisplayModel::Custom {
            name: name.to_string(),
            json: json.to_string(),
        }
    }

    #[test]
    fn built_in_models_use_vships_keys() {
        let expected = [
            (DisplayModel::Standard4k, "standard_4k"),
            (DisplayModel::StandardFhd, "standard_fhd"),
            (DisplayModel::StandardHdrPq, "standard_hdr_pq"),
            (DisplayModel::StandardHdrHlg, "standard_hdr_hlg"),
            (DisplayModel::StandardHdrDark, "standard_hdr_dark"),
            (DisplayModel::StandardHdrLinearZoom, "standard_hdr_linear_zoom"),
        ];

        for (model, key) in expected {
            assert_eq!(key_of(&model), key);
        }
    }

    #[test]
    fn a_built_in_model_passes_no_config() {
        let strings = DisplayModel::StandardFhd.c_strings().expect("converts the model");

        assert!(strings.config_pointer().is_null());
    }

    #[test]
    fn a_custom_model_passes_its_name_and_config() {
        let model = custom("my_display", "{}");
        let strings = model.c_strings().expect("converts the model");

        assert_eq!(key_of(&model), "my_display");
        assert!(!strings.config_pointer().is_null());
    }

    #[test]
    fn custom_json_that_is_not_inline_is_refused() {
        let model = custom("my_display", "displays/my_display.json");

        let error = model.check().expect_err("a path passed");

        assert!(error.message().contains("inline JSON"), "{error}");
    }

    #[test]
    fn custom_json_may_start_with_whitespace() {
        let model = custom("my_display", "  \n{}");

        let result = model.check();
        let strings = model.c_strings().expect("converts the model");
        // SAFETY: `config_pointer` points at a NUL terminated string `strings` owns.
        let config = unsafe { CStr::from_ptr(strings.config_pointer()) };

        assert!(result.is_ok(), "{result:?}");
        assert_eq!(config.to_str().expect("config is utf-8"), "{}");
    }

    #[test]
    fn a_nul_byte_in_a_custom_name_is_refused() {
        let model = custom("my\0display", "{}");

        let Err(error) = model.c_strings() else {
            panic!("a nul name converted");
        };

        assert!(error.message().contains("display name"), "{error}");
    }
}
