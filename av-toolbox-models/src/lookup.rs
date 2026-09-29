use std::borrow::Cow;
use std::io;
use std::path::{Path, PathBuf};

use crate::root::{TAG, installed_tags, models_root};

const ARTCNN_R8F64: &[u8] = include_bytes!("../builtin/ArtCNN_R8F64.bin");

/// Where a lookup found a file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Located {
    File(PathBuf),
    Builtin(&'static [u8]),
}

impl Located {
    pub fn read(&self) -> io::Result<Cow<'static, [u8]>> {
        match self {
            Self::File(path) => {
                let bytes = std::fs::read(path)?;
                Ok(Cow::Owned(bytes))
            },
            Self::Builtin(bytes) => Ok(Cow::Borrowed(bytes)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
    "{identifier}'s {file} is not installed under {shown_root}. Run `av-toolbox pull {identifier}` to install it",
    shown_root = .root.display()
)]
pub struct MissingModel {
    pub identifier: String,
    pub file: String,
    pub root: PathBuf,
}

/// The models directory, as one build of av-toolbox sees it.
#[derive(Debug, Clone)]
pub struct Store {
    root: PathBuf,
    tag: String,
}

impl Store {
    pub fn new(root: PathBuf, tag: impl Into<String>) -> Self {
        Self {
            root,
            tag: tag.into(),
        }
    }

    /// The store at [models_root](crate::models_root), for this build's [TAG](crate::TAG).
    pub fn from_environment() -> Self {
        let root = models_root();
        Self::new(root, TAG)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn tag(&self) -> &str {
        &self.tag
    }

    pub fn pack_dir(&self, tag: &str, identifier: &str) -> PathBuf {
        self.root.join(tag).join(identifier)
    }

    /// Finds `file` of the `identifier` pack.
    ///
    /// The store's own tag is tried first, then every other installed tag from the highest version down, then the
    /// copy built into the binary. Files are used as they are, without checking their sha256.
    pub fn find(&self, identifier: &str, file: &str) -> Result<Located, MissingModel> {
        let own = self.pack_dir(&self.tag, identifier).join(file);
        if own.is_file() {
            return Ok(Located::File(own));
        }

        let listed = installed_tags(&self.root);
        let tags = match listed {
            Ok(tags) => tags,
            Err(error) => {
                tracing::warn!("could not list {}: {error}", self.root.display());
                Vec::new()
            },
        };

        for tag in &tags {
            let candidate = self.pack_dir(tag, identifier).join(file);

            if candidate.is_file() {
                return Ok(Located::File(candidate));
            }
        }

        if let Some(bytes) = builtin(identifier, file) {
            return Ok(Located::Builtin(bytes));
        }

        Err(MissingModel {
            identifier: identifier.to_string(),
            file: file.to_string(),
            root: self.root.clone(),
        })
    }
}

/// The copy of `file` built into the binary, if there is one.
pub fn builtin(identifier: &str, file: &str) -> Option<&'static [u8]> {
    match (identifier, file) {
        ("artcnn", "ArtCNN_R8F64.bin") => Some(ARTCNN_R8F64),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{registry, sha256_hex};

    fn store_in(temp: &tempfile::TempDir, tag: &str) -> Store {
        let root = temp.path().to_path_buf();
        Store::new(root, tag)
    }

    fn install_file(root: &Path, tag: &str, identifier: &str, file: &str, content: &[u8]) -> PathBuf {
        let directory = root.join(tag).join(identifier);
        std::fs::create_dir_all(&directory).expect("created");

        let path = directory.join(file);
        std::fs::write(&path, content).expect("written");
        path
    }

    #[test]
    fn the_builds_own_tag_wins_over_a_higher_one() {
        let root = tempfile::tempdir().expect("a temp dir");
        let base = root.path();
        let own = install_file(base, "v0.1.0", "fake", "w.bin", b"own");
        install_file(base, "v9.0.0", "fake", "w.bin", b"higher");
        let store = store_in(&root, "v0.1.0");

        let located = store.find("fake", "w.bin").expect("found");

        assert_eq!(located, Located::File(own));
    }

    #[test]
    fn the_highest_tag_holding_the_file_is_used_otherwise() {
        let root = tempfile::tempdir().expect("a temp dir");
        let base = root.path();
        install_file(base, "v0.2.0", "fake", "w.bin", b"older");
        let newest_holding = install_file(base, "v0.10.0", "fake", "w.bin", b"newer");
        install_file(base, "v1.0.0", "fake", "other.bin", b"not this file");
        let store = store_in(&root, "v0.1.0");

        let located = store.find("fake", "w.bin").expect("found");

        assert_eq!(located, Located::File(newest_holding));
    }

    #[test]
    fn the_builtin_copy_is_used_when_nothing_is_installed() {
        let root = tempfile::tempdir().expect("a temp dir");
        let missing_root = root.path().join("never-created");
        let store = Store::new(missing_root, "v0.1.0");

        let located = store.find("artcnn", "ArtCNN_R8F64.bin").expect("builtin");

        assert!(matches!(located, Located::Builtin(_)));
    }

    #[test]
    fn an_installed_copy_beats_the_builtin_one() {
        let root = tempfile::tempdir().expect("a temp dir");
        let base = root.path();
        let installed = install_file(base, "v0.1.0", "artcnn", "ArtCNN_R8F64.bin", b"newer weights");
        let store = store_in(&root, "v0.1.0");

        let located = store.find("artcnn", "ArtCNN_R8F64.bin").expect("found");

        assert_eq!(located, Located::File(installed));
    }

    #[test]
    fn a_missing_file_names_the_pack_the_root_and_the_pull_command() {
        let root = tempfile::tempdir().expect("a temp dir");
        let store = store_in(&root, "v0.1.0");

        let error = store
            .find("artcnn", "ArtCNN_R16F96.bin")
            .expect_err("not installed");

        let message = error.to_string();
        let shown_root = root.path().display().to_string();
        assert!(message.contains("ArtCNN_R16F96.bin"), "{message}");
        assert!(message.contains(&shown_root), "{message}");
        assert!(message.contains("`av-toolbox pull artcnn`"), "{message}");
    }

    #[test]
    fn a_located_file_and_a_builtin_both_read() {
        let root = tempfile::tempdir().expect("a temp dir");
        let base = root.path();
        let path = install_file(base, "v0.1.0", "fake", "w.bin", b"bytes");
        let on_disk = Located::File(path);
        let built_in = Located::Builtin(b"built in");

        let disk_bytes = on_disk.read().expect("read");
        let builtin_bytes = built_in.read().expect("read");

        assert_eq!(disk_bytes.as_ref(), b"bytes");
        assert_eq!(builtin_bytes.as_ref(), b"built in");
    }

    #[test]
    fn the_builtin_r8f64_matches_its_checksum_in_the_lock() {
        let pack = registry().pack("artcnn").expect("in the lock");
        let listed = pack.file("ArtCNN_R8F64.bin").expect("listed");
        let bytes = builtin("artcnn", "ArtCNN_R8F64.bin").expect("built in");

        let digest = sha256_hex(bytes);

        assert_eq!(digest, listed.sha256);
    }

    #[test]
    fn only_listed_files_are_built_in() {
        assert!(builtin("artcnn", "ArtCNN_R16F96.bin").is_none());
        assert!(builtin("other", "ArtCNN_R8F64.bin").is_none());
    }
}
