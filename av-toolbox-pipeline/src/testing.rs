use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};

use crate::load::Files;
use crate::variables::normalise_path;

/// Pipeline files held in memory, keyed by their absolute path.
pub(crate) struct MemoryFiles {
    entries: HashMap<PathBuf, String>,
}

impl MemoryFiles {
    pub(crate) fn new(entries: &[(&str, &str)]) -> Self {
        let mut held = HashMap::new();

        for (path, text) in entries {
            let written = Path::new(path);
            let absolute = normalise_path(written);
            held.insert(absolute, text.to_string());
        }

        Self { entries: held }
    }
}

impl Files for MemoryFiles {
    fn read(&self, path: &Path) -> io::Result<String> {
        let absolute = normalise_path(path);
        let found = self.entries.get(&absolute);

        found
            .cloned()
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no such file"))
    }

    fn canonicalize(&self, path: &Path) -> io::Result<PathBuf> {
        let absolute = normalise_path(path);
        if !self.entries.contains_key(&absolute) {
            return Err(io::Error::new(io::ErrorKind::NotFound, "no such file"));
        }

        Ok(absolute)
    }
}
