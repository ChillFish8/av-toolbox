use std::sync::OnceLock;

use sha2::{Digest, Sha256};

/// The lock `tools/dependencies/build.py` writes, listing every pack's files and their sha256.
const LOCK: &str = include_str!("../../dependencies/dependencies.lock");

static REGISTRY: OnceLock<Registry> = OnceLock::new();

#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    #[error("the lock is not valid TOML: {0}")]
    Toml(#[from] toml::de::Error),

    #[error("`{identifier}` in the lock is not a table of files")]
    NotATable { identifier: String },

    #[error("`{identifier}`'s `{file}` in the lock is {value}, which is not a lowercase sha256")]
    BadChecksum {
        identifier: String,
        file: String,
        value: String,
    },
}

/// A file in a pack, and the lowercase hex sha256 a download of it must have.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackFile {
    pub name: String,
    pub sha256: String,
}

/// A set of files downloaded and installed together, named by its identifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pack {
    pub identifier: String,
    pub files: Vec<PackFile>,
}

impl Pack {
    pub fn file(&self, name: &str) -> Option<&PackFile> {
        self.files.iter().find(|file| file.name == name)
    }
}

/// Every pack a lock lists, in the order it lists them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Registry {
    packs: Vec<Pack>,
}

impl Registry {
    pub fn parse(text: &str) -> Result<Self, RegistryError> {
        let table: toml::Table = toml::from_str(text)?;
        let mut packs = Vec::new();

        for (identifier, entry) in table {
            let Some(listed) = entry.as_table() else {
                return Err(RegistryError::NotATable { identifier });
            };

            let mut files = Vec::new();
            for (name, value) in listed {
                let Some(sha256) = value.as_str().filter(|text| is_sha256(text)) else {
                    return Err(RegistryError::BadChecksum {
                        identifier,
                        file: name.clone(),
                        value: format!("{value:?}"),
                    });
                };

                files.push(PackFile {
                    name: name.clone(),
                    sha256: sha256.to_string(),
                });
            }

            packs.push(Pack { identifier, files });
        }

        Ok(Self { packs })
    }

    pub fn pack(&self, identifier: &str) -> Option<&Pack> {
        self.packs.iter().find(|pack| pack.identifier == identifier)
    }

    pub fn identifiers(&self) -> impl Iterator<Item = &str> {
        self.packs.iter().map(|pack| pack.identifier.as_str())
    }
}

/// The registry read from the lock embedded at compile time.
pub fn registry() -> &'static Registry {
    REGISTRY.get_or_init(parse_embedded_lock)
}

fn parse_embedded_lock() -> Registry {
    let parsed = Registry::parse(LOCK);
    parsed.expect("the embedded dependencies.lock parses")
}

fn is_sha256(text: &str) -> bool {
    text.len() == 64 && text.bytes().all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
}

/// The lowercase hex sha256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    to_hex(&digest)
}

pub(crate) fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_embedded_lock_lists_artcnn_in_file_order() {
        let pack = registry().pack("artcnn").expect("artcnn is in the lock");
        let names: Vec<&str> = pack.files.iter().map(|file| file.name.as_str()).collect();

        assert_eq!(names, ["ArtCNN_R8F64.bin", "ArtCNN_R16F96.bin"]);
    }

    #[test]
    fn a_lock_parses_in_file_order() {
        let first = "a".repeat(64);
        let second = "b".repeat(64);
        let text =
            format!("# header\n\n[zeta]\n\"z.bin\" = \"{first}\"\n\n[alpha]\n\"a.bin\" = \"{second}\"\n");

        let parsed = Registry::parse(&text).expect("valid");

        let identifiers: Vec<&str> = parsed.identifiers().collect();
        let alpha = parsed.pack("alpha").expect("listed");
        assert_eq!(identifiers, ["zeta", "alpha"]);
        assert_eq!(
            alpha.file("a.bin").map(|file| file.sha256.as_str()),
            Some(second.as_str())
        );
        assert!(parsed.pack("missing").is_none());
    }

    #[test]
    fn an_entry_that_is_not_a_table_is_refused() {
        let error = Registry::parse("flat = \"value\"\n").expect_err("not a table");

        assert!(error.to_string().contains("flat"), "{error}");
    }

    #[test]
    fn a_checksum_that_is_not_lowercase_hex_is_refused() {
        let upper = "A".repeat(64);
        let cases = [
            format!("[pack]\n\"f.bin\" = \"{upper}\"\n"),
            "[pack]\n\"f.bin\" = \"abc\"\n".to_string(),
            "[pack]\n\"f.bin\" = 3\n".to_string(),
        ];

        for text in cases {
            let error = Registry::parse(&text).expect_err("bad checksum");
            let message = error.to_string();

            assert!(message.contains("pack") && message.contains("f.bin"), "{message}");
        }
    }

    #[test]
    fn sha256_hex_matches_the_standard_vector() {
        let digest = sha256_hex(b"abc");

        assert_eq!(
            digest,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
