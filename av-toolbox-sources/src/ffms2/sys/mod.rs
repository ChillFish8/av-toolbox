mod error;
mod index;
mod video;

use std::ffi::CString;
use std::path::Path;
use std::sync::Once;

pub(crate) use self::error::Ffms2Error;
pub(crate) use self::index::{Index, Indexer};
pub(crate) use self::video::{StridedPlane, VideoSource};

static INIT: Once = Once::new();

/// Initialises FFMS2 once per process.
pub(crate) fn init() {
    INIT.call_once(|| {
        // SAFETY: FFMS2 asks for both arguments to be 0, and `Once` runs this a single time.
        unsafe { ffms2_sys::FFMS_Init(0, 0) };
    });
}

/// The FFMS2 id of an ffmpeg pixel format name, or `-1` for an unknown name.
pub(crate) fn pixel_format_id(name: &str) -> i32 {
    init();

    let Ok(c_name) = CString::new(name) else {
        return -1;
    };

    // SAFETY: `c_name` is a NUL terminated string that outlives the call.
    unsafe { ffms2_sys::FFMS_GetPixFmt(c_name.as_ptr()) }
}

/// A path as the C string FFMS2 takes, which must be UTF-8 on every platform.
pub(crate) fn c_path(path: &Path) -> Result<CString, Ffms2Error> {
    let Some(text) = path.to_str() else {
        let display = path.display();
        let message = format!("{display} is not valid utf-8, which ffms2 requires");

        return Err(Ffms2Error::local(message));
    };

    let converted = CString::new(text);

    converted.map_err(|_| {
        let message = format!("{text} carries a nul byte, which ffms2 cannot turn into a c string");

        Ffms2Error::local(message)
    })
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[cfg(unix)]
    #[test]
    fn a_non_utf8_path_is_refused() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;

        let bytes = [0x66, 0x6f, 0x80, 0x6f];
        let path = PathBuf::from(OsStr::from_bytes(&bytes));

        let error = c_path(&path).expect_err("not valid utf-8");

        assert!(error.to_string().contains("utf-8"));
    }

    #[cfg(unix)]
    #[test]
    fn a_path_with_an_interior_nul_is_refused() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;

        let bytes = [0x66, 0x6f, 0x00, 0x6f];
        let path = PathBuf::from(OsStr::from_bytes(&bytes));

        let error = c_path(&path).expect_err("a nul byte");

        assert!(error.to_string().contains("nul"));
    }

    #[test]
    fn a_utf8_path_converts() {
        let path = PathBuf::from("/media/clip.mkv");

        let converted = c_path(&path).expect("converts");

        assert_eq!(converted.to_str(), Ok("/media/clip.mkv"));
    }

    #[test]
    fn known_pixel_formats_have_distinct_ids() {
        let yuv_id = pixel_format_id("yuv420p");
        let grey_id = pixel_format_id("gray");

        assert!(yuv_id >= 0);
        assert!(grey_id >= 0);
        assert_ne!(yuv_id, grey_id);
    }

    #[test]
    fn an_unknown_pixel_format_is_minus_one() {
        assert_eq!(pixel_format_id("not-a-pixel-format"), -1);
        assert_eq!(pixel_format_id("yuv\x00420p"), -1);
    }
}
