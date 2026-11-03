use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError, mpsc};
use std::time::{Duration, Instant};

use cubecl::config::CubeClRuntimeConfig;
use cubecl_environment::persistence::{Access, Database, Origin};

use super::spawn_child_test;
use crate::kernel_cache::database::{fill_private_file, has_log, new_entries, write_state};
use crate::kernel_cache::lock::{LockKind, hold_liveness_lock, lock_merge};
use crate::kernel_cache::private::{
    ACTIVE,
    ActiveCache,
    MERGE_LOCK_NAME,
    PRIVATE_DIR_NAME,
    PrivateKernelCache,
    active_slot,
    finish_kernel_cache,
    finish_kernel_cache_at_exit,
    finish_waiting,
    guarded,
    merge_kernel_cache,
    merge_lock_path,
    refill_kernel_cache,
};
use crate::kernel_cache::prune::{lock_file_name, private_name, sidecar};

const NAMESPACE: &str = "cubecl/test/kernels";
const CHILD_DIR_ENV: &str = "AVTB_TEST_PRIVATE_CACHE_DIR";
const CHILD_TEST: &str = "kernel_cache::tests::private::merging_child";
const EXIT_CHILD_TEST: &str = "kernel_cache::tests::private::exit_child";
const GO_FILE: &str = "go";

/// Held by the tests that put a cache in the process-wide slot, so they do not see each other's.
static SLOT_TESTS: Mutex<()> = Mutex::new(());

fn insert(database: &Path, key: &str, origin: Origin) {
    let opened = Database::open_at(database, Access::ReadWrite).expect("open the database");
    let value = format!("value of {key}");
    opened.insert(NAMESPACE, key.as_bytes(), value.as_bytes(), origin);
}

fn entries(database: &Path) -> Vec<(String, Origin)> {
    let opened = Database::open_at(database, Access::ReadWrite).expect("open the database");
    let mut found = Vec::new();
    opened.scan_with_origin(NAMESPACE, &mut |key, _, origin| {
        let key = String::from_utf8_lossy(key).to_string();
        found.push((key, origin));
    });

    found.sort_by(|left, right| left.0.cmp(&right.0));
    found
}

fn active_cache(dir: &Path) -> ActiveCache {
    let private_dir = dir.join(PRIVATE_DIR_NAME);
    std::fs::create_dir_all(&private_dir).expect("create the private dir");

    let name = private_name();
    let file_name = cubecl::environment::file_name(&name);
    let private = private_dir.join(file_name);
    let shared = dir.join("default.db");
    let merge_lock = merge_lock_path(&shared);
    let liveness_path = private_dir.join(lock_file_name(&name));
    let liveness = hold_liveness_lock(&liveness_path);
    let written = write_state(&private);

    ActiveCache {
        shared,
        private,
        merge_lock,
        liveness_path,
        liveness,
        shared_seen: None,
        written,
        merged: HashSet::new(),
    }
}

fn fill(cache: &mut ActiveCache) {
    cache.shared_seen = fill_private_file(&cache.shared, &cache.private, &cache.merge_lock);
    cache.written = write_state(&cache.private);
}

fn standalone(database: &Path) {
    let opened = Database::open_at(database, Access::ReadWrite).expect("open the database");
    opened.into_standalone().expect("checkpoint the database");
}

/// Fails when this process still has `database` or one of its sidecars open or mapped.
#[cfg(target_os = "linux")]
fn assert_closed(database: &Path) {
    let name = database.to_string_lossy().to_string();
    let descriptors = std::fs::read_dir("/proc/self/fd").unwrap();
    for descriptor in descriptors.flatten() {
        let Ok(target) = std::fs::read_link(descriptor.path()) else {
            continue;
        };

        let target = target.to_string_lossy().to_string();
        assert!(!target.starts_with(&name), "{target} is still open");
    }

    let maps = std::fs::read_to_string("/proc/self/maps").unwrap();
    assert!(!maps.contains(&name), "{name} is still mapped");
}

#[test]
fn a_fill_copies_the_shared_entries() {
    let dir = tempfile::tempdir().unwrap();
    let mut cache = active_cache(dir.path());
    insert(&cache.shared, "a", Origin::Imported);
    insert(&cache.shared, "b", Origin::Imported);
    standalone(&cache.shared);

    fill(&mut cache);

    let copied = entries(&cache.private);
    let expected = vec![
        ("a".to_string(), Origin::Imported),
        ("b".to_string(), Origin::Imported),
    ];
    assert_eq!(copied, expected);
    assert!(cache.shared_seen.is_some());
}

#[test]
fn a_fill_checkpoints_a_shared_file_with_a_log_first() {
    let dir = tempfile::tempdir().unwrap();
    let mut cache = active_cache(dir.path());
    insert(&cache.shared, "logged", Origin::Imported);
    assert!(
        has_log(&cache.shared),
        "the insert should leave a write-ahead log"
    );

    fill(&mut cache);

    let copied = entries(&cache.private);
    let expected = vec![("logged".to_string(), Origin::Imported)];
    assert_eq!(copied, expected);
    assert!(
        !has_log(&cache.shared),
        "the fill should checkpoint the shared file"
    );

    #[cfg(target_os = "linux")]
    assert_closed(&cache.shared);
}

#[test]
fn a_fill_without_a_shared_file_leaves_the_private_file_absent() {
    let dir = tempfile::tempdir().unwrap();
    let mut cache = active_cache(dir.path());

    fill(&mut cache);

    assert!(!cache.private.exists());
}

#[test]
fn a_fill_without_the_lock_leaves_the_private_file_absent() {
    let dir = tempfile::tempdir().unwrap();
    let mut cache = active_cache(dir.path());
    insert(&cache.shared, "shared", Origin::Imported);
    standalone(&cache.shared);
    cache.merge_lock = dir.path().join("missing").join(MERGE_LOCK_NAME);

    fill(&mut cache);

    assert!(!cache.private.exists());
}

#[test]
fn a_merge_copies_only_the_entries_this_process_wrote() {
    let dir = tempfile::tempdir().unwrap();
    let mut cache = active_cache(dir.path());
    insert(&cache.shared, "shared", Origin::Imported);
    standalone(&cache.shared);
    fill(&mut cache);
    insert(&cache.private, "compiled", Origin::Local);

    cache.merge(None);

    let merged = entries(&cache.shared);
    let expected = vec![
        ("compiled".to_string(), Origin::Imported),
        ("shared".to_string(), Origin::Imported),
    ];
    assert_eq!(merged, expected);

    let merged_key = (NAMESPACE.to_string(), b"compiled".to_vec());
    assert!(cache.merged.contains(&merged_key));
    assert_eq!(cache.merged.len(), 1);
}

#[test]
fn reading_the_private_file_leaves_it_unwritten() {
    let dir = tempfile::tempdir().unwrap();
    let mut cache = active_cache(dir.path());
    insert(&cache.shared, "shared", Origin::Imported);
    standalone(&cache.shared);
    fill(&mut cache);

    let private = Database::open_at(&cache.private, Access::ReadWrite).unwrap();
    let found = private.get(NAMESPACE, b"shared");
    drop(private);

    assert!(found.is_some());
    assert_eq!(write_state(&cache.private), cache.written);
}

#[test]
fn writing_the_private_file_is_noticed() {
    let dir = tempfile::tempdir().unwrap();
    let mut cache = active_cache(dir.path());
    insert(&cache.shared, "shared", Origin::Imported);
    standalone(&cache.shared);
    fill(&mut cache);

    insert(&cache.private, "compiled", Origin::Local);

    assert_ne!(write_state(&cache.private), cache.written);
}

#[test]
fn a_merge_without_the_lock_leaves_the_shared_file_alone() {
    let dir = tempfile::tempdir().unwrap();
    let mut cache = active_cache(dir.path());
    insert(&cache.private, "compiled", Origin::Local);
    cache.merge_lock = dir.path().join("missing").join(MERGE_LOCK_NAME);

    cache.merge(None);

    assert!(!cache.shared.exists(), "nothing should have been merged");
    assert!(cache.merged.is_empty());
}

#[test]
fn a_merge_skips_the_records_namespaces() {
    let dir = tempfile::tempdir().unwrap();
    let mut cache = active_cache(dir.path());
    let private = Database::open_at(&cache.private, Access::ReadWrite).unwrap();
    private.insert("records/v1/sessions", b"session", b"value", Origin::Local);
    drop(private);

    let found = {
        let private = Database::open_at(&cache.private, Access::ReadWrite).unwrap();
        new_entries(&private, &cache.merged)
    };

    cache.merge(None);

    assert!(found.is_empty(), "{found:?}");
    assert!(!cache.shared.exists(), "nothing should have been merged");
}

#[test]
fn a_merged_shared_file_has_no_write_ahead_log_left() {
    let dir = tempfile::tempdir().unwrap();
    let mut cache = active_cache(dir.path());
    insert(&cache.private, "compiled", Origin::Local);

    cache.merge(None);

    assert!(cache.shared.exists());
    assert!(!has_log(&cache.shared));
}

#[cfg(target_os = "linux")]
#[test]
fn the_shared_file_is_closed_after_a_merge_and_a_refill() {
    let dir = tempfile::tempdir().unwrap();
    let mut cache = active_cache(dir.path());
    insert(&cache.shared, "shared", Origin::Imported);
    standalone(&cache.shared);
    insert(&cache.private, "compiled", Origin::Local);

    cache.merge(None);
    assert_closed(&cache.shared);

    cache.shared_seen = None;
    cache.refill_if_changed();
    assert_closed(&cache.shared);
}

#[test]
fn a_refill_adds_what_others_merged_without_overwriting() {
    let dir = tempfile::tempdir().unwrap();
    let mut cache = active_cache(dir.path());
    insert(&cache.shared, "first", Origin::Imported);
    standalone(&cache.shared);
    fill(&mut cache);
    insert(&cache.private, "mine", Origin::Local);

    insert(&cache.shared, "second", Origin::Imported);
    standalone(&cache.shared);
    cache.refill_if_changed();

    let refilled = entries(&cache.private);
    let expected = vec![
        ("first".to_string(), Origin::Imported),
        ("mine".to_string(), Origin::Local),
        ("second".to_string(), Origin::Imported),
    ];
    assert_eq!(refilled, expected);
}

#[test]
fn a_refill_skips_an_unchanged_shared_file() {
    let dir = tempfile::tempdir().unwrap();
    let mut cache = active_cache(dir.path());
    insert(&cache.shared, "first", Origin::Imported);
    standalone(&cache.shared);
    fill(&mut cache);
    let held = lock_merge(&cache.merge_lock, LockKind::Exclusive).expect("take the lock");

    let (sender, receiver) = mpsc::channel();
    std::thread::spawn(move || {
        cache.refill_if_changed();
        sender.send(()).unwrap();
    });
    let finished = receiver.recv_timeout(Duration::from_secs(5));
    drop(held);

    assert!(finished.is_ok(), "an unchanged file needs no lock");
}

#[test]
fn a_panic_in_cache_work_is_caught_and_leaves_the_slot_usable() {
    let _serial = SLOT_TESTS.lock().unwrap_or_else(PoisonError::into_inner);
    let caught = guarded(|| panic!("a corrupt cache"));
    assert!(caught.is_none());

    let poisoner = std::thread::spawn(|| {
        let _slot = ACTIVE.lock();
        panic!("poison the slot");
    });
    let _ = poisoner.join();
    assert!(ACTIVE.is_poisoned());

    merge_kernel_cache();
    refill_kernel_cache();
    finish_kernel_cache();
}

#[test]
fn deleting_removes_the_private_file_and_its_lock() {
    let dir = tempfile::tempdir().unwrap();
    let cache = active_cache(dir.path());
    insert(&cache.private, "compiled", Origin::Local);
    let private = cache.private.clone();
    let wal = sidecar(&private, "-wal");
    let lock = cache.liveness_path.clone();
    assert!(lock.exists());

    cache.delete();

    assert!(!private.exists());
    assert!(!wal.exists());
    assert!(!lock.exists());
}

#[test]
fn an_exit_that_cannot_get_the_lock_leaves_the_private_file() {
    let _serial = SLOT_TESTS.lock().unwrap_or_else(PoisonError::into_inner);
    let dir = tempfile::tempdir().unwrap();
    let cache = active_cache(dir.path());
    insert(&cache.private, "compiled", Origin::Local);
    let private = cache.private.clone();
    let merge_lock = cache.merge_lock.clone();
    let _held = lock_merge(&merge_lock, LockKind::Exclusive).expect("take the lock");

    *active_slot() = Some(cache);
    let start = Instant::now();
    finish_waiting(Some(Duration::from_millis(300)));

    assert!(start.elapsed() < Duration::from_secs(5));
    assert!(private.exists(), "unmerged kernels must stay on disk");
}

#[test]
fn an_exit_finish_is_bounded_while_another_thread_holds_the_slot() {
    let _serial = SLOT_TESTS.lock().unwrap_or_else(PoisonError::into_inner);
    let dir = tempfile::tempdir().unwrap();
    let cache = active_cache(dir.path());
    insert(&cache.private, "compiled", Origin::Local);
    let held = lock_merge(&cache.merge_lock, LockKind::Exclusive).expect("take the lock");
    *active_slot() = Some(cache);

    let blocked = std::thread::spawn(merge_kernel_cache);
    let deadline = Instant::now() + Duration::from_secs(5);
    while ACTIVE.try_lock().is_ok() {
        assert!(Instant::now() < deadline, "the merge never took the slot");
        std::thread::sleep(Duration::from_millis(10));
    }

    let (sender, receiver) = mpsc::channel();
    std::thread::spawn(move || {
        finish_waiting(Some(Duration::from_millis(300)));
        sender.send(()).unwrap();
    });
    let finished = receiver.recv_timeout(Duration::from_secs(5));

    drop(held);
    blocked.join().unwrap();
    finish_kernel_cache();

    assert!(finished.is_ok(), "an exit must not wait on a busy slot");
}

/// Waits until every child in `ids` has inserted its entry and written its ready file.
fn wait_for_ready(dir: &Path, ids: &[u32]) {
    let deadline = Instant::now() + Duration::from_secs(60);
    for id in ids {
        let ready = dir.join(format!("ready-{id}"));
        while !ready.exists() {
            assert!(Instant::now() < deadline, "a child never became ready");
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

/// Waits until the parent says go, so both children merge at about the same time.
fn wait_for_go(dir: &Path) {
    let go = dir.join(GO_FILE);
    let deadline = Instant::now() + Duration::from_secs(60);
    while !go.exists() {
        assert!(Instant::now() < deadline, "the parent never said go");
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn shared_path(config: &CubeClRuntimeConfig, dir: &Path) -> PathBuf {
    dir.join(cubecl::environment::file_name(&config.environment.name))
}

/// Runs inside the child processes of the test below, and does nothing when run on its own.
#[test]
#[ignore = "run as a child process by another test"]
fn merging_child() {
    let Some(dir) = std::env::var_os(CHILD_DIR_ENV) else {
        return;
    };
    let dir = PathBuf::from(dir);

    let mut config = CubeClRuntimeConfig::default();
    let cache = PrivateKernelCache::configure(&dir, &mut config);
    let private = cache.private_file();
    cache.start();

    let key = format!("key-{}", std::process::id());
    insert(&private, &key, Origin::Local);
    let ready = dir.join(format!("ready-{}", std::process::id()));
    std::fs::File::create(ready).unwrap();
    wait_for_go(&dir);

    merge_kernel_cache();
    finish_kernel_cache();
}

#[test]
fn two_processes_merging_at_once_keep_both_entries() {
    let dir = tempfile::tempdir().unwrap();
    let mut first = spawn_child_test(CHILD_TEST, CHILD_DIR_ENV, dir.path());
    let mut second = spawn_child_test(CHILD_TEST, CHILD_DIR_ENV, dir.path());
    wait_for_ready(dir.path(), &[first.id(), second.id()]);
    std::fs::File::create(dir.path().join(GO_FILE)).unwrap();

    assert!(first.wait().unwrap().success());
    assert!(second.wait().unwrap().success());

    let config = CubeClRuntimeConfig::default();
    let shared = shared_path(&config, dir.path());
    let mut expected = vec![
        (format!("key-{}", first.id()), Origin::Imported),
        (format!("key-{}", second.id()), Origin::Imported),
    ];
    expected.sort_by(|left, right| left.0.cmp(&right.0));
    assert_eq!(entries(&shared), expected);

    let private_dir = dir.path().join(PRIVATE_DIR_NAME);
    let leftovers = std::fs::read_dir(private_dir)
        .unwrap()
        .flatten()
        .filter(|entry| entry.file_name().to_string_lossy().ends_with(".db"))
        .count();
    assert_eq!(leftovers, 0, "no private database should be left");
}

#[test]
fn a_corrupt_shared_file_starts_cold() {
    let _serial = SLOT_TESTS.lock().unwrap_or_else(PoisonError::into_inner);
    let dir = tempfile::tempdir().unwrap();
    let mut config = CubeClRuntimeConfig::default();
    let shared = shared_path(&config, dir.path());
    std::fs::write(&shared, b"this is not a database, just garbage bytes").unwrap();

    let cache = PrivateKernelCache::configure(dir.path(), &mut config);
    let private = cache.private_file();
    cache.start();

    insert(&private, "compiled", Origin::Local);
    merge_kernel_cache();

    assert_eq!(entries(&private), vec![("compiled".to_string(), Origin::Local)]);

    finish_kernel_cache();
}

/// Runs inside the child process of the test below, and does nothing when run on its own.
#[test]
#[ignore = "run as a child process by another test"]
fn exit_child() {
    let Some(dir) = std::env::var_os(CHILD_DIR_ENV) else {
        return;
    };
    let dir = PathBuf::from(dir);

    let mut config = CubeClRuntimeConfig::default();
    let cache = PrivateKernelCache::configure(&dir, &mut config);
    let private = cache.private_file();
    cache.start();
    finish_kernel_cache_at_exit();

    insert(&private, "compiled", Origin::Local);
}

#[test]
fn finishing_at_exit_merges_and_deletes_the_private_file() {
    let dir = tempfile::tempdir().unwrap();
    let mut child = spawn_child_test(EXIT_CHILD_TEST, CHILD_DIR_ENV, dir.path());
    assert!(child.wait().unwrap().success());

    let config = CubeClRuntimeConfig::default();
    let shared = shared_path(&config, dir.path());
    let expected = vec![("compiled".to_string(), Origin::Imported)];
    assert_eq!(entries(&shared), expected);

    let private_dir = dir.path().join(PRIVATE_DIR_NAME);
    let leftovers = std::fs::read_dir(private_dir)
        .unwrap()
        .flatten()
        .filter(|entry| entry.file_name().to_string_lossy().ends_with(".db"))
        .count();
    assert_eq!(leftovers, 0, "no private database should be left");
}
