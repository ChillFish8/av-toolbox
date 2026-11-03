use std::fs::File;

use crate::kernel_cache::lock::{LockKind, lock_merge};

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
