use cubecl::environment::{CacheOption, Namespace, Store, StoreOptions};
use cubecl_environment::persistence::{Access, Database};

use super::BUILD;
use crate::cache::{
    CacheError,
    compilation_cache_dir,
    finish_compilation_cache,
    install_compilation_cache_at,
    install_or_skip,
};

// CubeCL's config is process wide, so the tests that install rely on nextest running each test in
// its own process.

#[test]
fn install_at_creates_the_build_directory() {
    let root = tempfile::tempdir().expect("temp dir");

    let build_dir = install_compilation_cache_at(root.path(), BUILD).expect("the cache installs");

    assert_eq!(build_dir, root.path().join(BUILD));
    assert!(build_dir.is_dir(), "{build_dir:?} should exist");
    assert_eq!(compilation_cache_dir(), Some(build_dir.as_path()));
}

#[cfg(unix)]
#[test]
fn a_read_only_root_installs_nothing_and_says_so() {
    use std::fs::Permissions;
    use std::os::unix::fs::PermissionsExt;

    let root = tempfile::tempdir().expect("temp dir");
    let read_only = Permissions::from_mode(0o555);
    std::fs::set_permissions(root.path(), read_only).expect("make the root read only");

    // Mode bits do not stop root, so there is nothing to test when the root still takes a new directory.
    let probe = root.path().join("probe");
    let writable_anyway = std::fs::create_dir(&probe).is_ok();
    if writable_anyway {
        return;
    }

    let installed = install_compilation_cache_at(root.path(), BUILD);
    let skipped = install_or_skip(root.path(), BUILD);

    let writable = Permissions::from_mode(0o755);
    std::fs::set_permissions(root.path(), writable).expect("make the root writable again");

    assert!(
        matches!(installed, Err(CacheError::CreateDir { .. })),
        "{installed:?} should fail to create the build directory"
    );
    assert!(matches!(skipped, Ok(None)), "{skipped:?} should install nothing");
    assert_eq!(compilation_cache_dir(), None);
}

#[test]
fn a_finished_install_merges_its_entries_and_deletes_its_private_file() {
    let root = tempfile::tempdir().expect("temp dir");

    let build_dir = install_compilation_cache_at(root.path(), BUILD).expect("the cache installs");

    let private_dir = build_dir.join("private");
    let database = cubecl::environment::path();
    assert_eq!(database.parent(), Some(private_dir.as_path()));

    let options = StoreOptions::new()
        .storage(Namespace::new("cache-test"))
        .cache(CacheOption::Lazy);
    let mut store: Store<String, String> = Store::new(options);
    store
        .insert("key".to_string(), "value".to_string())
        .expect("the entry is stored");
    assert!(database.exists(), "{database:?} should hold the entry");

    finish_compilation_cache();

    assert!(!database.exists(), "{database:?} should be deleted once finished");

    let shared_path = build_dir.join("default.db");
    let shared = Database::open_at(&shared_path, Access::ReadOnly).expect("the shared file exists");
    let namespace = Namespace::new("cache-test");
    let mut values = Vec::new();
    shared.scan(namespace.as_str(), &mut |_, value| {
        let owned = value.to_vec();
        values.push(owned);
    });

    // The store writes values as CBOR, where a five byte text string starts with 0x65.
    let expected = b"\x65value".to_vec();
    assert_eq!(
        values,
        vec![expected],
        "the entry should be merged into {shared_path:?}"
    );
}
