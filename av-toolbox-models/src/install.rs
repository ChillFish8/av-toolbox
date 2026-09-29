use std::fs::File;
use std::io::{self, Read, Seek, Write};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::lookup::Store;
use crate::registry::{Pack, Registry, to_hex};
use crate::root::{installed_tags, is_release_tag};

/// The one file an archive may hold beyond its pack's files.
const LICENSE: &str = "LICENSE";

#[derive(Debug, thiserror::Error)]
pub enum InstallError {
    #[error("there is no pack named `{0}`")]
    UnknownPack(String),

    #[error("the archive could not be read: {0}")]
    Archive(#[from] zip::result::ZipError),

    #[error("{}: {source}", path.display())]
    Io { path: PathBuf, source: io::Error },

    #[error("the archive holds `{0}`, which is not part of the pack")]
    ExtraFile(String),

    #[error("the archive is missing `{0}`")]
    MissingFile(String),

    #[error("`{name}` has sha256 {found}, expected {expected}")]
    ChecksumMismatch {
        name: String,
        expected: String,
        found: String,
    },

    #[error("`{0}` is not a release tag, which is `v` and a semantic version")]
    InvalidTag(String),
}

impl Store {
    /// Installs the `identifier` pack from a zip `archive` as release `tag`, returning its directory.
    ///
    /// The archive is extracted into a staging directory and every file checked against its sha256 in
    /// `registry` before the pack is moved into place, so a refused archive leaves nothing behind. A pack
    /// already installed for `tag` is replaced whole.
    pub fn install<R: Read + Seek>(
        &self,
        registry: &Registry,
        identifier: &str,
        tag: &str,
        archive: R,
    ) -> Result<PathBuf, InstallError> {
        let Some(pack) = registry.pack(identifier) else {
            return Err(InstallError::UnknownPack(identifier.to_string()));
        };

        if !is_release_tag(tag) {
            return Err(InstallError::InvalidTag(tag.to_string()));
        }

        let root = self.root();
        std::fs::create_dir_all(root).at(root)?;

        let staging = tempfile::Builder::new()
            .prefix(".staging-")
            .tempdir_in(root)
            .at(root)?;
        let mut archive = zip::ZipArchive::new(archive)?;
        let mut extracted: Vec<String> = Vec::new();

        for index in 0..archive.len() {
            let mut entry = archive.by_index(index)?;
            let name = entry.name().to_string();
            let expected = expected_sha256(pack, &name)?;
            let target = staging.path().join(&name);
            let mut written = File::create(&target).at(&target)?;
            let found = copy_hashing(&mut entry, &mut written).at(&target)?;

            if let Some(expected) = expected
                && expected != found
            {
                return Err(InstallError::ChecksumMismatch {
                    name,
                    expected: expected.to_string(),
                    found,
                });
            }

            extracted.push(name);
        }

        for file in &pack.files {
            if !extracted.contains(&file.name) {
                return Err(InstallError::MissingFile(file.name.clone()));
            }
        }

        let tag_dir = root.join(tag);
        std::fs::create_dir_all(&tag_dir).at(&tag_dir)?;

        let destination = tag_dir.join(identifier);
        if destination.exists() {
            std::fs::remove_dir_all(&destination).at(&destination)?;
        }

        let staged = staging.path();
        std::fs::rename(staged, &destination).at(&destination)?;

        Ok(destination)
    }

    /// The installed tags other than `keep_tag` that hold the `identifier` pack, highest version first.
    pub fn other_tags(&self, identifier: &str, keep_tag: &str) -> Result<Vec<String>, InstallError> {
        let root = self.root();
        let tags = installed_tags(root).at(root)?;
        let others = tags
            .into_iter()
            .filter(|tag| tag != keep_tag)
            .filter(|tag| self.pack_dir(tag, identifier).is_dir())
            .collect();

        Ok(others)
    }

    /// Removes the `identifier` pack from every tag but `keep_tag`, returning the tags it was removed from.
    ///
    /// A tag directory left empty is removed too.
    pub fn prune(&self, identifier: &str, keep_tag: &str) -> Result<Vec<String>, InstallError> {
        let others = self.other_tags(identifier, keep_tag)?;

        for tag in &others {
            let pack_dir = self.pack_dir(tag, identifier);
            std::fs::remove_dir_all(&pack_dir).at(&pack_dir)?;

            let tag_dir = self.root().join(tag);
            let mut remaining = std::fs::read_dir(&tag_dir).at(&tag_dir)?;

            if remaining.next().is_none() {
                std::fs::remove_dir(&tag_dir).at(&tag_dir)?;
            }
        }

        Ok(others)
    }
}

/// Names the path an I/O call was working on when it fails.
trait AtPath<T> {
    fn at(self, path: &Path) -> Result<T, InstallError>;
}

impl<T> AtPath<T> for io::Result<T> {
    fn at(self, path: &Path) -> Result<T, InstallError> {
        match self {
            Ok(value) => Ok(value),
            Err(source) => Err(InstallError::Io {
                path: path.to_path_buf(),
                source,
            }),
        }
    }
}

/// The sha256 the archive entry `name` must have, or `None` for the licence.
///
/// Names outside the pack, including any with a path, are refused.
fn expected_sha256<'a>(pack: &'a Pack, name: &str) -> Result<Option<&'a str>, InstallError> {
    if name == LICENSE {
        return Ok(None);
    }

    let Some(file) = pack.file(name) else {
        return Err(InstallError::ExtraFile(name.to_string()));
    };

    Ok(Some(file.sha256.as_str()))
}

/// Copies `reader` into `writer`, returning the lowercase hex sha256 of what was copied.
fn copy_hashing(reader: &mut impl Read, writer: &mut impl Write) -> io::Result<String> {
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 64 * 1024];

    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }

        let chunk = &buffer[..read];
        hasher.update(chunk);
        writer.write_all(chunk)?;
    }

    let digest = hasher.finalize();
    Ok(to_hex(&digest))
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::path::Path;

    use zip::write::SimpleFileOptions;

    use super::*;
    use crate::registry::sha256_hex;

    const WEIGHTS: &[u8] = b"weights";
    const EXTRA: &[u8] = b"extra";

    fn fake_registry() -> Registry {
        let weights_hash = sha256_hex(WEIGHTS);
        let extra_hash = sha256_hex(EXTRA);
        let lock = format!("[fake]\n\"w.bin\" = \"{weights_hash}\"\n\"x.bin\" = \"{extra_hash}\"\n");

        Registry::parse(&lock).expect("valid")
    }

    fn zip_of(entries: &[(&str, &[u8])]) -> Cursor<Vec<u8>> {
        let buffer = Cursor::new(Vec::new());
        let mut writer = zip::ZipWriter::new(buffer);

        for (name, content) in entries {
            if name.ends_with('/') {
                writer
                    .add_directory(*name, SimpleFileOptions::default())
                    .expect("added");
                continue;
            }

            writer
                .start_file(*name, SimpleFileOptions::default())
                .expect("started");
            writer.write_all(content).expect("written");
        }

        let mut finished = writer.finish().expect("finished");
        finished.set_position(0);
        finished
    }

    fn complete_pack() -> Cursor<Vec<u8>> {
        zip_of(&[("w.bin", WEIGHTS), ("x.bin", EXTRA), ("LICENSE", b"licence")])
    }

    fn store_in(temp: &tempfile::TempDir, tag: &str) -> Store {
        let root = temp.path().to_path_buf();
        Store::new(root, tag)
    }

    fn entries_of(directory: &Path) -> Vec<String> {
        let listing = std::fs::read_dir(directory).expect("listed");
        let mut names: Vec<String> = listing
            .map(|entry| {
                entry
                    .expect("an entry")
                    .file_name()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();

        names.sort();
        names
    }

    #[test]
    fn a_pack_installs_into_its_tag_and_leaves_no_staging() {
        let temp = tempfile::tempdir().expect("a temp dir");
        let root = temp.path().join("models");
        let store = Store::new(root.clone(), "v0.1.0");
        let registry = fake_registry();

        let archive = complete_pack();

        let installed = store
            .install(&registry, "fake", "v0.2.0", archive)
            .expect("installed");

        let weights_path = installed.join("w.bin");
        let weights = std::fs::read(&weights_path).expect("read");
        assert_eq!(installed, root.join("v0.2.0").join("fake"));
        assert_eq!(weights, WEIGHTS);
        assert_eq!(entries_of(&installed), ["LICENSE", "w.bin", "x.bin"]);
        assert_eq!(entries_of(&root), ["v0.2.0"]);
    }

    #[test]
    fn a_pack_without_a_licence_still_installs() {
        let temp = tempfile::tempdir().expect("a temp dir");
        let store = store_in(&temp, "v0.1.0");
        let registry = fake_registry();
        let archive = zip_of(&[("w.bin", WEIGHTS), ("x.bin", EXTRA)]);

        let installed = store
            .install(&registry, "fake", "v0.1.0", archive)
            .expect("installed");

        assert_eq!(entries_of(&installed), ["w.bin", "x.bin"]);
    }

    #[test]
    fn a_wrong_checksum_is_refused_and_leaves_nothing() {
        let temp = tempfile::tempdir().expect("a temp dir");
        let store = store_in(&temp, "v0.1.0");
        let registry = fake_registry();
        let archive = zip_of(&[("w.bin", b"tampered"), ("x.bin", EXTRA)]);

        let error = store
            .install(&registry, "fake", "v0.1.0", archive)
            .expect_err("refused");

        let expected = sha256_hex(WEIGHTS);
        assert!(
            matches!(&error, InstallError::ChecksumMismatch { name, .. } if name == "w.bin"),
            "{error}"
        );
        assert!(error.to_string().contains(&expected), "{error}");
        assert!(entries_of(temp.path()).is_empty());
    }

    #[test]
    fn a_missing_file_is_refused_and_leaves_nothing() {
        let temp = tempfile::tempdir().expect("a temp dir");
        let store = store_in(&temp, "v0.1.0");
        let registry = fake_registry();
        let archive = zip_of(&[("w.bin", WEIGHTS)]);

        let error = store
            .install(&registry, "fake", "v0.1.0", archive)
            .expect_err("refused");

        assert!(
            matches!(&error, InstallError::MissingFile(name) if name == "x.bin"),
            "{error}"
        );
        assert!(entries_of(temp.path()).is_empty());
    }

    #[test]
    fn extra_and_path_entries_are_refused_without_writing_outside() {
        let temp = tempfile::tempdir().expect("a temp dir");
        let root = temp.path().join("models");
        let store = Store::new(root.clone(), "v0.1.0");
        let registry = fake_registry();
        let refused_names = ["surprise.bin", "../w.bin", "nested/w.bin", "nested/"];

        for refused in refused_names {
            let mut entries: Vec<(&str, &[u8])> = vec![("w.bin", WEIGHTS), ("x.bin", EXTRA)];
            entries.insert(0, (refused, WEIGHTS));
            let archive = zip_of(&entries);

            let error = store
                .install(&registry, "fake", "v0.1.0", archive)
                .expect_err(refused);

            assert!(
                matches!(&error, InstallError::ExtraFile(name) if name == refused),
                "{refused}: {error}"
            );
            assert!(entries_of(&root).is_empty(), "{refused}");
        }

        let outside = temp.path().join("w.bin");
        assert!(!outside.exists());
    }

    #[test]
    fn an_unknown_pack_is_refused() {
        let temp = tempfile::tempdir().expect("a temp dir");
        let store = store_in(&temp, "v0.1.0");
        let registry = fake_registry();

        let archive = complete_pack();

        let error = store
            .install(&registry, "missing", "v0.1.0", archive)
            .expect_err("refused");

        assert!(
            matches!(&error, InstallError::UnknownPack(name) if name == "missing"),
            "{error}"
        );
    }

    #[test]
    fn installing_again_replaces_the_whole_directory() {
        let temp = tempfile::tempdir().expect("a temp dir");
        let store = store_in(&temp, "v0.1.0");
        let registry = fake_registry();
        let first_archive = complete_pack();
        let second_archive = complete_pack();
        let first = store
            .install(&registry, "fake", "v0.1.0", first_archive)
            .expect("installed");
        let stray = first.join("stray.bin");
        std::fs::write(&stray, b"left behind").expect("written");

        let second = store
            .install(&registry, "fake", "v0.1.0", second_archive)
            .expect("installed");

        assert_eq!(first, second);
        assert!(!stray.exists());
        assert_eq!(entries_of(&second), ["LICENSE", "w.bin", "x.bin"]);
    }

    #[test]
    fn pruning_removes_only_other_tags_of_the_pack() {
        let temp = tempfile::tempdir().expect("a temp dir");
        let root = temp.path().to_path_buf();
        let store = store_in(&temp, "v0.3.0");
        let registry = fake_registry();
        for tag in ["v0.1.0", "v0.2.0", "v0.3.0"] {
            let archive = complete_pack();
            store.install(&registry, "fake", tag, archive).expect("installed");
        }

        let other_pack = root.join("v0.2.0").join("another");
        std::fs::create_dir_all(&other_pack).expect("created");

        let listed = store.other_tags("fake", "v0.3.0").expect("listed");
        let pruned = store.prune("fake", "v0.3.0").expect("pruned");

        assert_eq!(listed, ["v0.2.0", "v0.1.0"]);
        assert_eq!(pruned, listed);
        assert_eq!(entries_of(&root), ["v0.2.0", "v0.3.0"]);
        let kept_tag = root.join("v0.2.0");
        assert_eq!(entries_of(&kept_tag), ["another"]);
        assert!(store.pack_dir("v0.3.0", "fake").is_dir());
    }

    #[test]
    fn a_root_that_is_a_file_is_refused_naming_it() {
        let temp = tempfile::tempdir().expect("a temp dir");
        let root = temp.path().join("models");
        std::fs::write(&root, b"not a directory").expect("written");
        let store = Store::new(root.clone(), "v0.1.0");
        let registry = fake_registry();
        let archive = complete_pack();

        let error = store
            .install(&registry, "fake", "v0.1.0", archive)
            .expect_err("refused");

        let message = error.to_string();
        let shown = root.display().to_string();
        assert!(message.contains(&shown), "{message}");
    }

    #[test]
    fn a_tag_that_is_not_a_release_is_refused_and_leaves_nothing() {
        let temp = tempfile::tempdir().expect("a temp dir");
        let store = store_in(&temp, "v0.1.0");
        let registry = fake_registry();

        for tag in ["nightly", "0.2.0", "release/v1"] {
            let archive = complete_pack();

            let error = store.install(&registry, "fake", tag, archive).expect_err(tag);

            assert!(
                matches!(&error, InstallError::InvalidTag(name) if name == tag),
                "{tag}: {error}"
            );
            assert!(entries_of(temp.path()).is_empty(), "{tag}");
        }
    }

    #[test]
    fn pruning_a_missing_root_removes_nothing() {
        let temp = tempfile::tempdir().expect("a temp dir");
        let missing = temp.path().join("never-created");
        let store = Store::new(missing, "v0.1.0");

        let pruned = store.prune("fake", "v0.1.0").expect("nothing to do");

        assert!(pruned.is_empty());
    }
}
