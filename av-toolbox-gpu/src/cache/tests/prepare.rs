use std::fs::File;
use std::io::Read;
use std::time::Duration;

use cubecl_environment::persistence::{Access, Database, Origin};

use crate::cache::prepare::{FIRST_OPEN_LOCK, lock_within, prepare_kernel_cache};

/// The offsets of SQLite's read and write format versions in its file header.
const FORMAT_VERSIONS: [usize; 2] = [18, 19];

/// The format version SQLite writes for a file in write-ahead logging mode.
const WAL_FORMAT: u8 = 2;

#[test]
fn preparing_a_new_cache_leaves_it_in_wal_mode() {
    let directory = tempfile::tempdir().expect("temp dir");
    let database = directory.path().join("default.db");

    prepare_kernel_cache(&database);

    let mut header = [0u8; 20];
    let mut file = File::open(&database).expect("the cache file exists");
    file.read_exact(&mut header).expect("the header is read");
    for offset in FORMAT_VERSIONS {
        assert_eq!(header[offset], WAL_FORMAT, "header byte {offset}");
    }
}

#[test]
fn preparing_an_existing_cache_keeps_its_entries() {
    let directory = tempfile::tempdir().expect("temp dir");
    let database = directory.path().join("default.db");

    {
        let handle = Database::open_at(&database, Access::ReadWrite).expect("open the database");
        handle.insert("namespace", b"key", b"value", Origin::Local);
    }

    prepare_kernel_cache(&database);

    let handle = Database::open_at(&database, Access::ReadWrite).expect("open the database");
    let stored = handle.get("namespace", b"key").expect("the entry is kept");
    assert_eq!(&stored[..], b"value");
}

#[test]
fn preparing_without_a_directory_returns_quietly() {
    let directory = tempfile::tempdir().expect("temp dir");
    let missing = directory.path().join("missing");
    let database = missing.join("default.db");

    prepare_kernel_cache(&database);

    assert!(!missing.exists());
}

#[test]
fn a_lock_held_elsewhere_runs_out_the_wait() {
    let directory = tempfile::tempdir().expect("temp dir");
    let lock_path = directory.path().join(FIRST_OPEN_LOCK);
    let elsewhere = File::create(&lock_path).expect("make the lock");
    elsewhere.lock().expect("lock it");

    let taken = lock_within(&lock_path, Duration::from_millis(50));

    assert!(taken.is_none(), "a held lock was taken");
}

#[test]
fn a_released_lock_can_be_taken_again() {
    let directory = tempfile::tempdir().expect("temp dir");
    let lock_path = directory.path().join(FIRST_OPEN_LOCK);

    let first = lock_within(&lock_path, Duration::from_millis(50));
    assert!(first.is_some(), "a free lock was refused");
    drop(first);

    let second = lock_within(&lock_path, Duration::from_millis(50));
    assert!(second.is_some(), "a released lock was refused");
}
