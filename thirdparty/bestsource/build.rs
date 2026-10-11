use std::env;
use std::path::{Path, PathBuf};

/// BestSource's library sources, built without GPU support.
const BESTSOURCE_SOURCES: [&str; 5] = [
    "audiosource.cpp",
    "bsshared.cpp",
    "gpuhash.cpp",
    "tracklist.cpp",
    "videosource.cpp",
];

/// libp2p's sources built on every target.
const P2P_SOURCES: [&str; 4] = ["p2p_api.cpp", "v210.cpp", "simd/cpuinfo_x86.cpp", "simd/p2p_simd.cpp"];

/// The system libraries dynamic mode links, with the least version each needs.
const SYSTEM_LIBRARIES: [(&str, &str); 4] = [
    ("libavcodec", "63.1"),
    ("libavformat", "63.1"),
    ("libavutil", "61.1"),
    ("libxxhash", "0.8"),
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=shim");
    println!("cargo:rerun-if-changed=vendor");
    println!("cargo:rerun-if-env-changed=BESTSOURCE_LIB_DIR");
    println!("cargo:rerun-if-env-changed=DOCS_RS");

    if env::var_os("DOCS_RS").is_some() {
        skip("docs.rs builds documentation only");
        return;
    }

    let is_static = env::var_os("CARGO_FEATURE_STATIC").is_some();
    if is_static {
        link_static();
    } else {
        link_dynamic();
    }
}

/// Builds and links nothing, so only a library build of this crate can succeed.
fn skip(reason: &str) {
    println!(
        "cargo:warning=BestSource is not linked because {reason}. Anything that links avtbx-bestsource will fail to link."
    );
}

/// Whether the vendored BestSource and libp2p submodules have their sources.
fn submodules_checked_out() -> bool {
    let bestsource = Path::new("vendor/bestsource/src/videosource.cpp");
    let libp2p = Path::new("vendor/libp2p/p2p_api.h");

    bestsource.exists() && libp2p.exists()
}

/// Prints each line of `text` as its own warning, as cargo only shows the first line of one.
fn warn_lines(text: &str) {
    for line in text.lines() {
        println!("cargo:warning={line}");
    }
}

/// Links BestSource, libp2p, xxHash and FFmpeg statically from the vcpkg tree.
fn link_static() {
    let lib_dir = env::var_os("BESTSOURCE_LIB_DIR");
    let Some(lib_dir) = lib_dir else {
        skip("BESTSOURCE_LIB_DIR is unset, run `just static-deps` and point it at vcpkg_installed/<triplet>/lib");
        return;
    };

    let expected_dir = canonical_lib_dir(&lib_dir);
    let headers = probe_static_bestsource(false);
    require_expected_tree(&headers, &expected_dir);

    // The shim is compiled before the second probe so the linker sees it ahead of the BestSource
    // archives it depends on.
    compile_shim(&headers.include_paths);

    probe_static_bestsource(true);
}

/// Builds the vendored BestSource and libp2p, and links the system FFmpeg and xxHash.
fn link_dynamic() {
    if !submodules_checked_out() {
        skip("its submodules are not checked out. Run `just submodules`");
        return;
    }

    let probed = probe_system_libraries(false);
    let libraries = match probed {
        Ok(libraries) => libraries,
        Err(reason) => {
            let fix = "Install the FFmpeg 9 and xxHash development packages, or enable the `static` feature";
            skip(&format!("{reason}. {fix}"));
            return;
        },
    };

    let mut include_paths: Vec<PathBuf> = libraries
        .iter()
        .flat_map(|library| library.include_paths.iter().cloned())
        .collect();
    include_paths.push(PathBuf::from("vendor/bestsource/src"));
    include_paths.push(PathBuf::from("vendor/libp2p"));

    // Archives are linked in the order they compile, and each one only refers to those after it.
    compile_shim(&include_paths);
    compile_bestsource(&include_paths);
    compile_libp2p();

    let linked = probe_system_libraries(true);
    if let Err(reason) = linked {
        panic!("{reason}");
    }
}

/// The canonical `BESTSOURCE_LIB_DIR`, which pkg-config must resolve BestSource inside.
fn canonical_lib_dir(lib_dir: &std::ffi::OsStr) -> PathBuf {
    let Ok(expected_dir) = Path::new(lib_dir).canonicalize() else {
        panic!("BESTSOURCE_LIB_DIR must point at vcpkg_installed/<triplet>/lib, but it does not resolve");
    };

    expected_dir
}

/// Probes the static BestSource description, emitting link lines only when `emit_metadata` is set.
fn probe_static_bestsource(emit_metadata: bool) -> pkg_config::Library {
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

/// Probes the system FFmpeg and xxHash, emitting link lines only when `emit_metadata` is set.
fn probe_system_libraries(emit_metadata: bool) -> Result<Vec<pkg_config::Library>, String> {
    let mut libraries = Vec::new();

    for (name, least_version) in SYSTEM_LIBRARIES {
        let probe = pkg_config::Config::new()
            .atleast_version(least_version)
            .cargo_metadata(emit_metadata)
            .probe(name);

        match probe {
            Ok(library) => libraries.push(library),
            Err(error) => {
                warn_lines(&error.to_string());
                return Err(format!("pkg-config found no {name} {least_version} or newer"));
            },
        }
    }

    Ok(libraries)
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

fn compile_shim(include_paths: &[PathBuf]) {
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .include("shim")
        .file("shim/bestsource_shim.cpp");

    for include_path in include_paths {
        build.include(include_path);
    }

    build.compile("bestsource_shim");
}

fn compile_bestsource(include_paths: &[PathBuf]) {
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .warnings(false)
        .define("_FILE_OFFSET_BITS", "64");

    for include_path in include_paths {
        build.include(include_path);
    }

    let source_dir = Path::new("vendor/bestsource/src");
    for source in BESTSOURCE_SOURCES {
        let source_path = source_dir.join(source);
        build.file(source_path);
    }

    build.compile("bestsource");
}

/// Builds libp2p, with its SSE4.1 file and `P2P_SIMD` on x86 only, as upstream does.
fn compile_libp2p() {
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let is_x86 = matches!(target_arch.as_str(), "x86" | "x86_64");

    let mut build = cc::Build::new();
    build.cpp(true).std("c++14").warnings(false).include("vendor/libp2p");

    let source_dir = Path::new("vendor/libp2p");
    for source in P2P_SOURCES {
        let source_path = source_dir.join(source);
        build.file(source_path);
    }

    if is_x86 {
        build.define("P2P_SIMD", None);
    }

    build.compile("p2p");

    if is_x86 {
        compile_libp2p_sse41();
    }
}

fn compile_libp2p_sse41() {
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++14")
        .warnings(false)
        .include("vendor/libp2p")
        .define("P2P_SIMD", None)
        .flag_if_supported("-march=x86-64-v2")
        .file("vendor/libp2p/simd/p2p_sse41.cpp");

    build.compile("p2p_sse41");
}
