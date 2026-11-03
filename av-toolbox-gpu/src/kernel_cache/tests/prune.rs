use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use super::spawn_child_test;
use crate::kernel_cache::lock::hold_liveness_lock;
use crate::kernel_cache::prune::{
    STALE_PRIVATE_AGE,
    is_private_name,
    lock_file_name,
    private_name,
    prune_stale_private_files,
    remove_database_files,
    sidecar,
};

const CHILD_DIR_ENV: &str = "AVTB_TEST_KERNEL_CACHE_DIR";
const CHILD_NAME: &str = "7-eeeeeeeeeeeeeeee";
const CHILD_TEST: &str = "kernel_cache::tests::prune::liveness_lock_child";

fn database_path(dir: &Path, name: &str) -> PathBuf {
    dir.join(cubecl::environment::file_name(name))
}

fn make_private_files(dir: &Path, name: &str, modified: SystemTime) {
    let database = database_path(dir, name);
    let wal = sidecar(&database, "-wal");
    let lock = dir.join(lock_file_name(name));
    for path in [database, wal, lock] {
        let file = File::create(&path).unwrap();
        file.set_modified(modified).unwrap();
    }
}

fn has_files(dir: &Path, name: &str) -> bool {
    let database = database_path(dir, name);
    let lock = dir.join(lock_file_name(name));
    database.exists() || lock.exists()
}

#[test]
fn a_private_name_is_recognised() {
    let name = private_name();

    assert!(is_private_name(&name), "{name}");
    for other in [
        "default",
        "123",
        "abc-0123456789abcdef",
        "123-0123456789ABCDEF",
        "123-0123",
    ] {
        assert!(!is_private_name(other), "{other}");
    }
}

#[test]
fn deleting_removes_the_private_file_and_its_sidecars() {
    let dir = tempfile::tempdir().unwrap();
    let name = private_name();
    make_private_files(dir.path(), &name, SystemTime::now());
    let database = database_path(dir.path(), &name);
    let wal = sidecar(&database, "-wal");

    remove_database_files(&database);

    assert!(!database.exists());
    assert!(!wal.exists());
}

#[test]
fn pruning_removes_only_stale_files_of_exited_processes() {
    let dir = tempfile::tempdir().unwrap();
    let now = SystemTime::now();
    let long_ago = now - STALE_PRIVATE_AGE - Duration::from_secs(60);

    let own = "1-aaaaaaaaaaaaaaaa";
    let stale = "2-bbbbbbbbbbbbbbbb";
    let live = "3-cccccccccccccccc";
    let fresh = "4-dddddddddddddddd";
    for name in [own, stale, live] {
        make_private_files(dir.path(), name, long_ago);
    }

    make_private_files(dir.path(), fresh, now);

    let foreign = dir.path().join("default.db");
    let foreign_file = File::create(&foreign).unwrap();
    foreign_file.set_modified(long_ago).unwrap();

    let live_lock = dir.path().join(lock_file_name(live));
    let held = File::options().read(true).write(true).open(live_lock).unwrap();
    held.lock().unwrap();

    prune_stale_private_files(dir.path(), own, now);

    assert!(!has_files(dir.path(), stale), "stale files should be removed");
    assert!(has_files(dir.path(), own), "this process's files should be kept");
    assert!(
        has_files(dir.path(), live),
        "a running process's files should be kept"
    );
    assert!(has_files(dir.path(), fresh), "fresh files should be kept");
    assert!(
        foreign.exists(),
        "files not named like a private file should be kept"
    );
}

#[test]
fn pruning_keeps_a_file_whose_lock_is_held_by_a_live_child() {
    let dir = tempfile::tempdir().unwrap();
    let mut child = spawn_child_test(CHILD_TEST, CHILD_DIR_ENV, dir.path());
    let ready = dir.path().join("ready");
    let deadline = Instant::now() + Duration::from_secs(30);
    while !ready.exists() {
        assert!(Instant::now() < deadline, "the child never became ready");
        std::thread::sleep(Duration::from_millis(20));
    }

    let now = SystemTime::now();
    let long_ago = now - STALE_PRIVATE_AGE - Duration::from_secs(60);
    make_private_files(dir.path(), CHILD_NAME, long_ago);

    prune_stale_private_files(dir.path(), "1-aaaaaaaaaaaaaaaa", now);
    let kept = has_files(dir.path(), CHILD_NAME);

    child.kill().unwrap();
    child.wait().unwrap();
    prune_stale_private_files(dir.path(), "1-aaaaaaaaaaaaaaaa", now);
    let removed = !has_files(dir.path(), CHILD_NAME);

    assert!(kept, "a live child's files should be kept");
    assert!(removed, "files should go once the child has exited");
}

/// Runs inside the child process of the test above, and does nothing when run on its own.
#[test]
#[ignore = "run as a child process by another test"]
fn liveness_lock_child() {
    let Some(dir) = std::env::var_os(CHILD_DIR_ENV) else {
        return;
    };
    let dir = PathBuf::from(dir);

    let lock_path = dir.join(lock_file_name(CHILD_NAME));
    let _held = hold_liveness_lock(&lock_path).expect("take the liveness lock");
    File::create(dir.join("ready")).unwrap();

    std::thread::sleep(Duration::from_secs(60));
}
