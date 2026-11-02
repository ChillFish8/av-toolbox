use std::fmt::{self, Display, Formatter};
use std::path::{Path, PathBuf};
use std::sync::Once;

/// The toolkit locations CubeCL checks when `CUDA_PATH` is unset.
///
/// CubeCL has no Windows defaults, so Windows relies on `CUDA_PATH`.
const CUDA_ROOTS: [&str; 4] = ["/usr/local/cuda", "/opt/cuda", "/usr/lib/cuda", "/usr"];

/// Where `libdevice.10.bc` was looked for and not found.
#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct LibdeviceMissing {
    searched: Vec<PathBuf>,
    /// Whether `CUDA_PATH` was set and so was the only root searched.
    from_cuda_path: bool,
}

impl Display for LibdeviceMissing {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        let roots: Vec<_> = self
            .searched
            .iter()
            .map(|root| root.display().to_string())
            .collect();
        let roots = roots.join(", ");

        if self.from_cuda_path {
            write!(
                formatter,
                "libdevice.10.bc was not found because CUDA_PATH is set to \"{roots}\". \
                 Point it at a CUDA toolkit containing nvvm/libdevice, or unset it to use the \
                 bundled copy."
            )
        } else {
            write!(
                formatter,
                "libdevice.10.bc was not found in {roots}. Install the CUDA toolkit, or set \
                 CUDA_PATH to one containing nvvm/libdevice."
            )
        }
    }
}

/// Finds `libdevice.10.bc` the way CubeCL does.
///
/// A set `CUDA_PATH` is the only root checked. Otherwise the first default root with an
/// `nvvm/libdevice` directory is used, and the file must exist inside it.
pub(crate) fn find_libdevice(
    cuda_path: Option<PathBuf>,
    defaults: &[PathBuf],
) -> Result<PathBuf, LibdeviceMissing> {
    let from_cuda_path = cuda_path.is_some();
    let root = match cuda_path {
        Some(path) => Some(path),
        None => defaults
            .iter()
            .find(|candidate| candidate.join("nvvm").join("libdevice").is_dir())
            .cloned(),
    };

    let Some(root) = root else {
        return Err(LibdeviceMissing {
            searched: defaults.to_vec(),
            from_cuda_path,
        });
    };

    let file = libdevice_file(&root);
    if file.is_file() {
        return Ok(file);
    }

    Err(LibdeviceMissing {
        searched: vec![root],
        from_cuda_path,
    })
}

fn libdevice_file(root: &Path) -> PathBuf {
    root.join("nvvm").join("libdevice").join("libdevice.10.bc")
}

/// Whether CUDA kernels can compile here, warning once per process when they cannot.
pub(crate) fn libdevice_available() -> bool {
    static WARNED: Once = Once::new();

    let cuda_path = std::env::var_os("CUDA_PATH").map(PathBuf::from);
    let defaults: Vec<PathBuf> = CUDA_ROOTS.iter().map(PathBuf::from).collect();
    let Err(missing) = find_libdevice(cuda_path, &defaults) else {
        return true;
    };

    WARNED.call_once(|| tracing::warn!("the cuda backend cannot be used, {missing}"));

    false
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::{TempDir, tempdir};

    use super::*;

    fn toolkit_with_file(root: &Path) {
        let directory = root.join("nvvm").join("libdevice");
        fs::create_dir_all(&directory).expect("should create the directory");
        fs::write(directory.join("libdevice.10.bc"), b"bc").expect("should write the file");
    }

    fn toolkit_without_file(root: &Path) {
        let directory = root.join("nvvm").join("libdevice");
        fs::create_dir_all(directory).expect("should create the directory");
    }

    fn roots(count: usize) -> (TempDir, Vec<PathBuf>) {
        let temp = tempdir().expect("should create a temp dir");
        let roots = (0..count)
            .map(|index| temp.path().join(format!("root{index}")))
            .collect();

        (temp, roots)
    }

    #[test]
    fn the_default_roots_match_cubecl() {
        assert_eq!(
            CUDA_ROOTS,
            ["/usr/local/cuda", "/opt/cuda", "/usr/lib/cuda", "/usr"]
        );
    }

    #[test]
    fn a_set_cuda_path_is_the_only_root_searched() {
        let (_temp, roots) = roots(2);
        toolkit_with_file(&roots[1]);

        let error = find_libdevice(Some(roots[0].clone()), &roots[1..]).unwrap_err();

        let expected = LibdeviceMissing {
            searched: vec![roots[0].clone()],
            from_cuda_path: true,
        };
        assert_eq!(error, expected);
    }

    #[test]
    fn a_set_cuda_path_with_the_file_is_found() {
        let (_temp, roots) = roots(2);
        toolkit_with_file(&roots[0]);
        toolkit_with_file(&roots[1]);

        let found = find_libdevice(Some(roots[0].clone()), &roots[1..]).expect("should be found");

        assert_eq!(found, libdevice_file(&roots[0]));
    }

    #[test]
    fn the_first_default_root_with_nvvm_wins() {
        let (_temp, roots) = roots(3);
        toolkit_with_file(&roots[1]);
        toolkit_with_file(&roots[2]);

        let found = find_libdevice(None, &roots).expect("should be found");

        assert_eq!(found, libdevice_file(&roots[1]));
    }

    #[test]
    fn a_root_without_the_file_is_missing() {
        let (_temp, roots) = roots(2);
        toolkit_without_file(&roots[0]);
        toolkit_with_file(&roots[1]);

        let error = find_libdevice(None, &roots).unwrap_err();

        let expected = LibdeviceMissing {
            searched: vec![roots[0].clone()],
            from_cuda_path: false,
        };
        assert_eq!(error, expected);
    }

    #[test]
    fn no_root_at_all_lists_every_default() {
        let (_temp, roots) = roots(3);

        let error = find_libdevice(None, &roots).unwrap_err();

        assert_eq!(error.searched, roots);
    }

    #[test]
    fn a_missing_libdevice_names_the_roots_searched() {
        let defaults = LibdeviceMissing {
            searched: vec![PathBuf::from("/usr/local/cuda"), PathBuf::from("/opt/cuda")],
            from_cuda_path: false,
        };
        assert_eq!(
            defaults.to_string(),
            "libdevice.10.bc was not found in /usr/local/cuda, /opt/cuda. Install the CUDA toolkit, or set \
             CUDA_PATH to one containing nvvm/libdevice."
        );

        let from_variable = LibdeviceMissing {
            searched: vec![PathBuf::from("/nonexistent")],
            from_cuda_path: true,
        };
        assert_eq!(
            from_variable.to_string(),
            "libdevice.10.bc was not found because CUDA_PATH is set to \"/nonexistent\". Point it at a CUDA \
             toolkit containing nvvm/libdevice, or unset it to use the bundled copy."
        );
    }
}
