use std::fs;
use std::path::{Path, PathBuf};

use tempfile::TempDir;

use crate::{PinError, build_hash_for, pinned_versions, source_hash};

fn write_file(root: &Path, relative: &str, contents: &str) -> PathBuf {
    let path = root.join(relative);
    fs::create_dir_all(path.parent().expect("the file has a parent")).expect("the directory is created");
    fs::write(&path, contents).expect("the file is written");
    path
}

fn manifest_with(root: &Path, dependencies: &str) -> PathBuf {
    let text = format!("[package]\nname = \"example\"\n\n{dependencies}");
    write_file(root, "Cargo.toml", &text)
}

#[test]
fn a_source_hash_changes_with_a_file() {
    let root = TempDir::new().expect("a temp dir is created");
    write_file(root.path(), "lib.rs", "fn a() {}");
    let before = source_hash(root.path());

    write_file(root.path(), "lib.rs", "fn b() {}");
    let after = source_hash(root.path());

    assert_ne!(before, after);
}

#[test]
fn a_source_hash_ignores_the_root_path() {
    let first = TempDir::new().expect("a temp dir is created");
    let second = TempDir::new().expect("a temp dir is created");
    for root in [first.path(), second.path()] {
        write_file(root, "lib.rs", "fn a() {}");
        write_file(root, "nested/mod.rs", "fn b() {}");
    }

    assert_eq!(source_hash(first.path()), source_hash(second.path()));
}

#[test]
fn an_inline_exact_pin_is_read() {
    let root = TempDir::new().expect("a temp dir is created");
    let manifest = manifest_with(
        root.path(),
        "[dependencies]\ncubecl = { version = \"=0.11.3\", features = [\"std\"] }\nserde = \"1\"\n",
    );

    let pins = pinned_versions(&manifest).expect("the pins are read");

    assert_eq!(pins, vec![("cubecl".to_owned(), "=0.11.3".to_owned())]);
}

#[test]
fn a_packaged_table_pin_is_read() {
    let root = TempDir::new().expect("a temp dir is created");
    let manifest = manifest_with(root.path(), "[dependencies.cubecl]\nversion = \"=0.11.3\"\n");

    let pins = pinned_versions(&manifest).expect("the pins are read");

    assert_eq!(pins, vec![("cubecl".to_owned(), "=0.11.3".to_owned())]);
}

#[test]
fn a_workspace_pin_is_read_from_the_workspace_manifest() {
    let root = TempDir::new().expect("a temp dir is created");
    write_file(
        root.path(),
        "Cargo.toml",
        concat!(
            "[workspace]\nmembers = [\"member\"]\n\n",
            "[workspace.dependencies]\n",
            "cubek-std = \"=0.3.1\"\n",
            "cubecl = { version = \"=0.11.3\" }\n",
        ),
    );
    let manifest = write_file(
        root.path(),
        "member/Cargo.toml",
        concat!(
            "[package]\nname = \"member\"\n\n",
            "[dependencies]\ncubecl = { workspace = true }\n\n",
            "[build-dependencies]\ncubek-std = { workspace = true }\n",
        ),
    );

    let pins = pinned_versions(&manifest).expect("the pins are read");

    let expected = vec![
        ("cubecl".to_owned(), "=0.11.3".to_owned()),
        ("cubek-std".to_owned(), "=0.3.1".to_owned()),
    ];
    assert_eq!(pins, expected);
}

#[test]
fn an_inexact_pin_is_refused() {
    let root = TempDir::new().expect("a temp dir is created");
    let manifest = manifest_with(root.path(), "[dependencies]\ncubecl = \"0.11.3\"\n");

    let error = pinned_versions(&manifest).expect_err("the pin is refused");

    let expected = PinError::NotExact {
        name: "cubecl".to_owned(),
        version: "0.11.3".to_owned(),
    };
    assert_eq!(error, expected);
}

#[test]
fn a_dependency_version_change_changes_the_hash() {
    let root = TempDir::new().expect("a temp dir is created");
    let sources = root.path().join("src");
    write_file(&sources, "lib.rs", "fn a() {}");
    let old_manifest = manifest_with(root.path(), "[dependencies]\ncubecl = \"=0.11.3\"\n");
    let old_hash = build_hash_for(&sources, &old_manifest).expect("the hash is built");

    let new_manifest = manifest_with(root.path(), "[dependencies]\ncubecl = \"=0.11.4\"\n");
    let new_hash = build_hash_for(&sources, &new_manifest).expect("the hash is built");

    assert_ne!(old_hash, new_hash);
}

#[test]
fn a_manifest_with_no_pins_is_refused() {
    let root = TempDir::new().expect("a temp dir is created");
    let manifest = manifest_with(root.path(), "[dependencies]\nserde = \"1\"\n");

    let error = pinned_versions(&manifest).expect_err("the manifest is refused");

    assert_eq!(error, PinError::NoPins);
}

#[test]
fn a_workspace_dependency_missing_from_the_workspace_is_refused() {
    let root = TempDir::new().expect("a temp dir is created");
    write_file(root.path(), "Cargo.toml", "[workspace]\nmembers = [\"member\"]\n");
    let manifest = write_file(
        root.path(),
        "member/Cargo.toml",
        "[package]\nname = \"member\"\n\n[dependencies]\ncubecl = { workspace = true }\n",
    );

    let error = pinned_versions(&manifest).expect_err("the entry is missing");

    let expected = PinError::MissingWorkspaceEntry {
        name: "cubecl".to_owned(),
    };
    assert_eq!(error, expected);
}

#[test]
fn a_workspace_dependency_without_a_workspace_manifest_is_refused() {
    let root = TempDir::new().expect("a temp dir is created");
    let manifest = manifest_with(root.path(), "[dependencies]\ncubecl = { workspace = true }\n");

    let error = pinned_versions(&manifest).expect_err("there is no workspace manifest");

    assert!(matches!(error, PinError::Manifest { .. }));
}
