use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use super::BUILD;
use crate::cache::STALE_BUILD_AGE;
use crate::cache::tidy::{build_cache_dir, is_build_hash, prune_stale_builds, tidy_cache_root};

fn long_ago() -> SystemTime {
    let margin = Duration::from_secs(60);
    SystemTime::now() - STALE_BUILD_AGE - margin
}

/// Opening a directory as a `File` to set its times only works on Unix.
#[cfg(unix)]
fn make_old_dir(path: &Path) {
    std::fs::create_dir(path).expect("create dir");

    let handle = File::open(path).expect("open dir");
    let modified = long_ago();
    handle.set_modified(modified).expect("set mtime");
}

#[test]
fn the_build_dir_is_the_root_joined_with_the_build() {
    let root = PathBuf::from("/home/u/.cache/av-toolbox");
    let expected = root.join(BUILD);

    let build_dir = build_cache_dir(&root, BUILD);

    assert_eq!(build_dir, expected);
}

#[test]
fn is_build_hash_accepts_sixteen_lowercase_hex_chars() {
    let looks_like_hash = is_build_hash(BUILD);

    assert!(looks_like_hash, "{BUILD} should look like a build hash");
}

#[test]
fn is_build_hash_rejects_other_names() {
    for name in [
        "vulkan",
        "0123456789ABCDEF",
        "0123456789abcde",
        "0123456789abcdef0",
        "warm-0123456789ab",
    ] {
        let looks_like_hash = is_build_hash(name);
        assert!(!looks_like_hash, "{name} should not look like a build hash");
    }
}

#[cfg(unix)]
#[test]
fn pruning_removes_only_stale_builds_of_other_hashes() {
    let root = tempfile::tempdir().expect("temp dir");
    let now = SystemTime::now();

    let current = "aaaaaaaaaaaaaaaa";
    let stale = "bbbbbbbbbbbbbbbb";
    let fresh = "cccccccccccccccc";
    let non_hash_dirs = ["vulkan", "hip", "warm-1234"];

    let old_dirs = [current, stale].into_iter().chain(non_hash_dirs);
    for name in old_dirs {
        let path = root.path().join(name);
        make_old_dir(&path);
    }

    let fresh_path = root.path().join(fresh);
    std::fs::create_dir(&fresh_path).expect("create dir");

    let stale_file = root.path().join("dddddddddddddddd");
    let handle = File::create(&stale_file).expect("create file");
    let modified = long_ago();
    handle.set_modified(modified).expect("set mtime");

    prune_stale_builds(root.path(), current, now);

    assert!(!root.path().join(stale).exists(), "stale build should be removed");
    assert!(root.path().join(current).exists(), "current build should be kept");
    assert!(fresh_path.exists(), "fresh build should be kept");
    assert!(stale_file.exists(), "a file named like a hash should be kept");
    for name in non_hash_dirs {
        assert!(root.path().join(name).exists(), "{name} should be kept");
    }
}

#[test]
fn pruning_a_missing_root_does_nothing() {
    let root = tempfile::tempdir().expect("temp dir");
    let missing = root.path().join("missing");
    let now = SystemTime::now();

    prune_stale_builds(&missing, BUILD, now);
}

#[cfg(unix)]
#[test]
fn tidying_marks_the_current_build_as_recently_used() {
    let root = tempfile::tempdir().expect("temp dir");
    let build_dir = build_cache_dir(root.path(), BUILD);
    make_old_dir(&build_dir);

    tidy_cache_root(root.path(), BUILD);

    let metadata = std::fs::metadata(&build_dir).expect("build dir metadata");
    let modified = metadata.modified().expect("mtime");
    let age = SystemTime::now().duration_since(modified).unwrap_or_default();
    assert!(age < Duration::from_secs(60), "build dir mtime is {age:?} old");
}

/// Pruning as if far in the future makes the fresh link look stale, so only its type keeps it.
#[cfg(unix)]
#[test]
fn tidying_skips_a_symlinked_build_directory() {
    let root = tempfile::tempdir().expect("temp dir");
    let elsewhere = tempfile::tempdir().expect("temp dir");

    let target = elsewhere.path().join("kept");
    make_old_dir(&target);
    let kept_file = target.join("default.db");
    File::create(&kept_file).expect("create file");

    let link = root.path().join("bbbbbbbbbbbbbbbb");
    std::os::unix::fs::symlink(&target, &link).expect("create symlink");

    let far_ahead = STALE_BUILD_AGE * 2;
    let future = SystemTime::now() + far_ahead;
    prune_stale_builds(root.path(), BUILD, future);

    let link_metadata = std::fs::symlink_metadata(&link);
    assert!(link_metadata.is_ok(), "the symlink should be kept");
    assert!(kept_file.exists(), "the symlink's target should be kept");
}
