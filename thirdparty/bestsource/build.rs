use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=shim/bestsource_shim.h");
    println!("cargo:rerun-if-changed=shim/bestsource_shim.cpp");
    println!("cargo:rerun-if-env-changed=BESTSOURCE_LIB_DIR");

    let expected_dir = expected_lib_dir();
    let headers = probe_bestsource(false);
    require_expected_tree(&headers, &expected_dir);

    compile_shim(&headers);

    probe_bestsource(true);
}

/// The canonical `BESTSOURCE_LIB_DIR`, which pkg-config must resolve BestSource inside.
fn expected_lib_dir() -> PathBuf {
    let lib_dir = env::var_os("BESTSOURCE_LIB_DIR");
    let Some(lib_dir) = lib_dir else {
        panic!(
            "BESTSOURCE_LIB_DIR must point at vcpkg_installed/<triplet>/lib, run `just static-deps` first"
        );
    };

    let Ok(expected_dir) = Path::new(&lib_dir).canonicalize() else {
        panic!("BESTSOURCE_LIB_DIR must point at vcpkg_installed/<triplet>/lib, but it does not resolve");
    };

    expected_dir
}

/// Probes the static BestSource description, emitting link lines only when `emit_metadata` is set.
fn probe_bestsource(emit_metadata: bool) -> pkg_config::Library {
    let probe = pkg_config::Config::new()
        .statik(true)
        .cargo_metadata(emit_metadata)
        .probe("bestsource");

    match probe {
        Ok(library) => library,
        Err(error) => panic!(
            "no static bestsource found, point PKG_CONFIG_PATH at \
             vcpkg_installed/<triplet>/lib/pkgconfig: {error}"
        ),
    }
}

/// Stops a system BestSource or FFmpeg from being linked in place of the vcpkg tree.
fn require_expected_tree(library: &pkg_config::Library, expected_dir: &Path) {
    let found_expected = library
        .link_paths
        .iter()
        .filter_map(|path| path.canonicalize().ok())
        .any(|path| path == expected_dir);

    if !found_expected {
        panic!(
            "pkg-config resolved bestsource outside BESTSOURCE_LIB_DIR, \
             point PKG_CONFIG_PATH at the same vcpkg tree"
        );
    }
}

fn compile_shim(library: &pkg_config::Library) {
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .include("shim")
        .file("shim/bestsource_shim.cpp");

    for include_path in &library.include_paths {
        build.include(include_path);
    }

    build.compile("bestsource_shim");
}
