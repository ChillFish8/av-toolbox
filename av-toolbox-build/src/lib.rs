//! Build script helpers that key a crate's kernel cache
//!
//! A kernel crate calls [emit] from its `build.rs`. The hash covers the crate's sources and the exact
//! versions of the GPU libraries it pins, so a change to either starts a fresh cache.
//! It reads nothing outside the crate's own files and its workspace manifest, so it gives the same result
//! from a published package.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::fs;
use std::path::{Path, PathBuf};

use toml::{Table, Value};

#[cfg(test)]
mod tests;

/// Dependency name prefixes that must be pinned to an exact version.
pub const PINNED_PREFIXES: [&str; 3] = ["cubecl", "cubek", "av-denoise-core"];

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
const DEPENDENCY_TABLES: [&str; 2] = ["dependencies", "build-dependencies"];

/// Why the pinned dependencies of a manifest could not be read.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum PinError {
    /// A pinned dependency has a version that does not start with `=`.
    NotExact { name: String, version: String },
    /// A pinned dependency has no version to read.
    MissingVersion { name: String },
    /// A pinned dependency inherits from the workspace, which has no entry for it.
    MissingWorkspaceEntry { name: String },
    /// The manifest pins no dependency at all.
    NoPins,
    /// A manifest could not be read or parsed.
    Manifest { path: PathBuf, reason: String },
}

impl Display for PinError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotExact { name, version } => write!(
                formatter,
                "`{name}` must be pinned to an exact version such as \"={version}\", not \"{version}\""
            ),
            Self::MissingVersion { name } => write!(formatter, "`{name}` has no version to pin"),
            Self::MissingWorkspaceEntry { name } => {
                write!(
                    formatter,
                    "`{name}` is inherited from the workspace, which has no entry for it"
                )
            },
            Self::NoPins => write!(
                formatter,
                "the manifest pins no dependency, so the hash would ignore them"
            ),
            Self::Manifest { path, reason } => write!(formatter, "cannot read {}: {reason}", path.display()),
        }
    }
}

impl Error for PinError {}

/// Hashes every file below `src` with FNV-1a, in path order.
///
/// Each file adds its path below `src`, its length and its contents. The path of `src` itself is not part
/// of the hash.
pub fn source_hash(src: &Path) -> u64 {
    let mut files = Vec::new();
    collect_files(src, &mut files);

    let mut relative_paths: Vec<(String, PathBuf)> = files
        .into_iter()
        .map(|path| (relative_path(src, &path), path))
        .collect();
    relative_paths.sort();

    let mut hash = FNV_OFFSET_BASIS;
    for (relative, path) in relative_paths {
        let contents = read_source(&path);
        let length = contents.len() as u64;

        hash = fnv1a(hash, relative.as_bytes());
        hash = fnv1a(hash, &[0]);
        hash = fnv1a(hash, &length.to_le_bytes());
        hash = fnv1a(hash, &contents);
    }

    hash
}

/// Every pinned dependency of `manifest` with its version, sorted by name.
///
/// A dependency inherited from the workspace takes its version from the nearest workspace manifest above.
/// Only `[dependencies]` and `[build-dependencies]` are read, not `[target.*.dependencies]` tables.
/// It is an error for the manifest to pin nothing.
pub fn pinned_versions(manifest: &Path) -> Result<Vec<(String, String)>, PinError> {
    let pins = read_pins(manifest)?;
    Ok(pins.versions)
}

/// Hashes the sources in `src` and the pinned dependency versions of `manifest` into one value.
pub fn build_hash_for(src: &Path, manifest: &Path) -> Result<u64, PinError> {
    let pins = read_pins(manifest)?;
    Ok(fold_pins(source_hash(src), &pins.versions))
}

/// Prints the cargo directives that set `env_name` to the build hash of the crate being built.
///
/// Call it from `build.rs`. It panics if a pinned dependency does not have an exact version.
pub fn emit(env_name: &str) {
    let manifest_dir = std::env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets the manifest directory");
    let manifest_dir = PathBuf::from(manifest_dir);
    let sources = manifest_dir.join("src");
    let manifest = manifest_dir.join("Cargo.toml");

    let pins = match read_pins(&manifest) {
        Ok(pins) => pins,
        Err(error) => panic!("{error}"),
    };
    let hash = fold_pins(source_hash(&sources), &pins.versions);

    println!("cargo:rustc-env={env_name}={hash:016x}");
    println!("cargo:rerun-if-changed={}", sources.display());
    println!("cargo:rerun-if-changed={}", manifest.display());
    if let Some(workspace_manifest) = pins.workspace_manifest {
        println!("cargo:rerun-if-changed={}", workspace_manifest.display());
    }
}

struct Pins {
    versions: Vec<(String, String)>,
    workspace_manifest: Option<PathBuf>,
}

fn read_pins(manifest: &Path) -> Result<Pins, PinError> {
    let table = read_manifest(manifest)?;
    let mut workspace_dependencies = None;
    let mut workspace_manifest = None;
    let mut versions = BTreeMap::new();

    for dependency_table in DEPENDENCY_TABLES {
        let Some(Value::Table(dependencies)) = table.get(dependency_table) else {
            continue;
        };

        for (name, spec) in dependencies {
            if !is_pinned_name(name) {
                continue;
            }

            let version = if inherits_workspace(spec) {
                if workspace_dependencies.is_none() {
                    let (path, dependencies) = read_workspace_dependencies(manifest)?;
                    workspace_manifest = Some(path);
                    workspace_dependencies = Some(dependencies);
                }
                let dependencies = workspace_dependencies.as_ref().expect("the workspace was read");
                let entry = dependencies.get(name);
                let entry = entry.ok_or_else(|| PinError::MissingWorkspaceEntry {
                    name: name.to_owned(),
                })?;
                version_of(entry)
            } else {
                version_of(spec)
            };
            let version = version.ok_or_else(|| PinError::MissingVersion {
                name: name.to_owned(),
            })?;

            if !version.starts_with('=') {
                return Err(PinError::NotExact {
                    name: name.to_owned(),
                    version,
                });
            }
            versions.insert(name.to_owned(), version);
        }
    }

    if versions.is_empty() {
        return Err(PinError::NoPins);
    }

    Ok(Pins {
        versions: versions.into_iter().collect(),
        workspace_manifest,
    })
}

fn is_pinned_name(name: &str) -> bool {
    PINNED_PREFIXES.iter().any(|prefix| name.starts_with(prefix))
}

fn inherits_workspace(spec: &Value) -> bool {
    spec.get("workspace").and_then(Value::as_bool) == Some(true)
}

fn version_of(spec: &Value) -> Option<String> {
    match spec {
        Value::String(version) => Some(version.clone()),
        Value::Table(table) => table.get("version")?.as_str().map(str::to_owned),
        _ => None,
    }
}

fn read_manifest(path: &Path) -> Result<Table, PinError> {
    let text = fs::read_to_string(path).map_err(|error| manifest_error(path, &error))?;
    text.parse::<Table>()
        .map_err(|error| manifest_error(path, &error))
}

fn manifest_error(path: &Path, error: &dyn Display) -> PinError {
    PinError::Manifest {
        path: path.to_path_buf(),
        reason: error.to_string(),
    }
}

/// The `[workspace.dependencies]` of the first manifest at or above `manifest` that holds `[workspace]`.
fn read_workspace_dependencies(manifest: &Path) -> Result<(PathBuf, Table), PinError> {
    let start = manifest.parent().unwrap_or(Path::new("."));

    for directory in start.ancestors() {
        let candidate = directory.join("Cargo.toml");
        if !candidate.is_file() {
            continue;
        }

        let table = read_manifest(&candidate)?;
        let Some(Value::Table(workspace)) = table.get("workspace") else {
            continue;
        };

        let dependencies = match workspace.get("dependencies") {
            Some(Value::Table(dependencies)) => dependencies.clone(),
            _ => Table::new(),
        };
        return Ok((candidate, dependencies));
    }

    Err(PinError::Manifest {
        path: manifest.to_path_buf(),
        reason: "no workspace manifest above it".to_owned(),
    })
}

fn fold_pins(source_hash: u64, versions: &[(String, String)]) -> u64 {
    let mut hash = source_hash;
    for (name, version) in versions {
        let line = format!("{name}={version}\n");
        hash = fnv1a(hash, line.as_bytes());
    }

    hash
}

fn fnv1a(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }

    hash
}

fn read_source(path: &Path) -> Vec<u8> {
    match fs::read(path) {
        Ok(contents) => contents,
        Err(error) => panic!("cannot read source file {}: {error}", path.display()),
    }
}

fn read_source_directory(directory: &Path) -> Vec<fs::DirEntry> {
    let entries = match fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error) => panic!("cannot read source directory {}: {error}", directory.display()),
    };

    entries
        .map(|entry| match entry {
            Ok(entry) => entry,
            Err(error) => panic!("cannot read an entry of {}: {error}", directory.display()),
        })
        .collect()
}

fn collect_files(directory: &Path, files: &mut Vec<PathBuf>) {
    let entries = read_source_directory(directory);
    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            collect_files(&path, files);
        } else {
            files.push(path);
        }
    }
}

/// The path of `path` below `root`, joined with `/` on every platform.
fn relative_path(root: &Path, path: &Path) -> String {
    let relative = path.strip_prefix(root).expect("path is inside the source root");
    let components: Vec<String> = relative
        .components()
        .map(|component| component.as_os_str().to_string_lossy().into_owned())
        .collect();
    components.join("/")
}
