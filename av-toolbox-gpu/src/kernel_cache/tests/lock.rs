use std::fs::File;
use std::time::{Duration, Instant};

use crate::kernel_cache::lock::{LockKind, lock_merge, lock_merge_within};

#[test]
fn shared_locks_coexist_and_block_a_writer_until_released() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("merge.lock");

    let first = lock_merge(&path, LockKind::Shared).expect("a first reader");
    let second = lock_merge(&path, LockKind::Shared).expect("a second reader");

    let writer = File::options().read(true).write(true).open(&path).unwrap();
    assert!(writer.try_lock().is_err(), "a writer must wait for the readers");

    drop(first);
    drop(second);
    assert!(
        writer.try_lock().is_ok(),
        "a writer gets the lock once the readers are gone"
    );
}

#[test]
fn a_held_lock_times_out_within_the_limit() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("merge.lock");
    let _held = lock_merge(&path, LockKind::Exclusive).expect("take the lock");

    let waiter_path = path.clone();
    let waiter = std::thread::spawn(move || {
        let start = Instant::now();
        let lock = lock_merge_within(&waiter_path, LockKind::Exclusive, Duration::from_millis(300));
        (lock.is_some(), start.elapsed())
    });
    let (got_lock, elapsed) = waiter.join().unwrap();

    assert!(!got_lock, "a held lock must not be granted");
    assert!(elapsed >= Duration::from_millis(300), "{elapsed:?}");
    assert!(elapsed < Duration::from_secs(2), "{elapsed:?}");
}
