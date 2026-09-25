use std::collections::HashMap;
use std::io;
use std::path::{Component, Path, PathBuf};

use crate::load::Files;

/// Pipeline files held in memory, keyed by their absolute path.
pub(crate) struct MemoryFiles {
    entries: HashMap<PathBuf, String>,
}

impl MemoryFiles {
    pub(crate) fn new(entries: &[(&str, &str)]) -> Self {
        let mut held = HashMap::new();

        for (path, text) in entries {
            let written = Path::new(path);
            let absolute = normalise(written);
            held.insert(absolute, text.to_string());
        }

        Self { entries: held }
    }
}

impl Files for MemoryFiles {
    fn read(&self, path: &Path) -> io::Result<String> {
        let absolute = normalise(path);
        let found = self.entries.get(&absolute);

        found
            .cloned()
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no such file"))
    }

    fn canonicalize(&self, path: &Path) -> io::Result<PathBuf> {
        let absolute = normalise(path);
        if !self.entries.contains_key(&absolute) {
            return Err(io::Error::new(io::ErrorKind::NotFound, "no such file"));
        }

        Ok(absolute)
    }
}

/// Makes `path` absolute and folds away `.` and `..` without touching the disk.
fn normalise(path: &Path) -> PathBuf {
    let absolute = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
    let mut folded = PathBuf::new();

    for component in absolute.components() {
        match component {
            Component::CurDir => {},
            Component::ParentDir => {
                folded.pop();
            },
            other => folded.push(other),
        }
    }

    folded
}
