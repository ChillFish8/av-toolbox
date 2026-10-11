#[path = "build/shaders.rs"]
mod shaders;

use std::env;
use std::path::{Path, PathBuf};

/// The least system Vship version dynamic mode links.
const SYSTEM_VSHIP_VERSION: &str = "5.1.2";

/// The CUDA architectures built when `VSHIP_CUDA_ARCH` is unset.
const DEFAULT_CUDA_ARCH: &str = "all-major";

/// The single file holding all of Vship's library code.
const VSHIP_SOURCE: &str = "vendor/vship/src/VshipLib.cpp";

/// The version Vship reports, matching the submodule's Makefile.
const VSHIP_VERSION: [(&str, &str); 3] = [
    ("VSHIP_VERSION_MAJOR", "5"),
    ("VSHIP_VERSION_MINOR", "1"),
    ("VSHIP_VERSION_MINORMINOR", "2"),
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=build");
    println!("cargo:rerun-if-changed=include");
    println!("cargo:rerun-if-changed=vendor");
    println!("cargo:rerun-if-env-changed=DOCS_RS");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    println!("cargo:rerun-if-env-changed=VSHIP_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=PATH");

    generate_bindings();

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

/// Links nothing, so only a library build of this crate can succeed.
fn skip(reason: &str) {
    let message =
        format!("Vship is not linked because {reason}. Anything that links avtbx-vship will fail to link.");
    println!("cargo:warning={message}");
}

/// Prints each line of `text` as its own warning, as cargo only shows the first line of one.
fn warn_lines(text: &str) {
    for line in text.lines() {
        println!("cargo:warning={line}");
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Backend {
    Vulkan,
    Cuda,
}

/// The backend the `vulkan` or `cuda` feature picks, failing unless exactly one is on.
fn selected_backend() -> Backend {
    let vulkan = env::var_os("CARGO_FEATURE_VULKAN").is_some();
    let cuda = env::var_os("CARGO_FEATURE_CUDA").is_some();

    match (vulkan, cuda) {
        (true, false) => Backend::Vulkan,
        (false, true) => Backend::Cuda,
        (true, true) => panic!("avtbx-vship takes one backend, enable `vulkan` or `cuda` but not both"),
        (false, false) => panic!("static avtbx-vship needs a backend, enable the `vulkan` or `cuda` feature"),
    }
}

/// Links the system `libvship` through pkg-config.
fn link_dynamic() {
    let probe = pkg_config::Config::new()
        .atleast_version(SYSTEM_VSHIP_VERSION)
        .probe("libvship");

    if let Err(error) = probe {
        skip("no system libvship 5.1.2 or newer, install it or enable the `static` feature");

        let text = error.to_string();
        warn_lines(&text);
    }
}

/// Builds the vendored Vship for the selected backend and links it statically.
fn link_static() {
    let backend = selected_backend();

    let source = Path::new(VSHIP_SOURCE);
    if !source.exists() {
        skip("its submodule is not checked out. Run `just submodules`");
        return;
    }

    match backend {
        Backend::Vulkan => compile_vulkan(),
        Backend::Cuda => compile_cuda(),
    }
}

fn out_dir() -> PathBuf {
    let out_dir = env::var_os("OUT_DIR");
    let Some(out_dir) = out_dir else {
        panic!("cargo did not set OUT_DIR");
    };

    PathBuf::from(out_dir)
}

fn is_windows_target() -> bool {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    target_os == "windows"
}

/// Builds Vship's Vulkan backend with its SPIR-V shaders embedded, then links the Vulkan loader.
fn compile_vulkan() {
    let vulkan_sdk = env::var_os("VULKAN_SDK").map(PathBuf::from);
    let include_dirs = vulkan_include_dirs(vulkan_sdk.as_deref());
    let found_headers = has_vulkan_headers(&include_dirs);
    if !found_headers {
        skip("no Vulkan headers were found, install them or set VULKAN_SDK");
        return;
    }

    let out_dir = out_dir();
    let shader_dir = Path::new("vendor/vship/libvshipSpvShaders");
    let shader_header = out_dir.join("libvshipSpvShaders.hpp");
    let written = shaders::write_shader_header(shader_dir, &shader_header);
    if let Err(error) = written {
        panic!("could not write {}: {error}", shader_header.display());
    }

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .warnings(false)
        .include(&out_dir)
        .include("vendor/vship/include");

    for include_dir in &include_dirs {
        build.include(include_dir);
    }

    build
        .define("VULKANBUILD", None)
        .define("NDEBUG", None)
        .define("VSHIP_STATIC", None);

    for (name, value) in VSHIP_VERSION {
        build.define(name, value);
    }

    build.file(VSHIP_SOURCE).compile("vship");

    link_vulkan_loader(vulkan_sdk.as_deref());
}

/// The include dirs that may hold the Vulkan headers, from `VULKAN_SDK` or pkg-config.
///
/// This is empty when pkg-config finds the headers in the default system path.
fn vulkan_include_dirs(vulkan_sdk: Option<&Path>) -> Vec<PathBuf> {
    if let Some(vulkan_sdk) = vulkan_sdk {
        let include_name = if is_windows_target() { "Include" } else { "include" };
        let include_dir = vulkan_sdk.join(include_name);
        return vec![include_dir];
    }

    let probe = pkg_config::Config::new().cargo_metadata(false).probe("vulkan");

    match probe {
        Ok(library) => library.include_paths,
        Err(_) => Vec::new(),
    }
}

fn has_vulkan_headers(include_dirs: &[PathBuf]) -> bool {
    let system_dir = PathBuf::from("/usr/include");

    include_dirs
        .iter()
        .chain([&system_dir])
        .any(|include_dir| include_dir.join("vulkan/vulkan.h").exists())
}

/// Links the Vulkan loader, which is always a shared library.
fn link_vulkan_loader(vulkan_sdk: Option<&Path>) {
    if is_windows_target() {
        if let Some(vulkan_sdk) = vulkan_sdk {
            let lib_dir = vulkan_sdk.join("Lib");
            println!("cargo:rustc-link-search=native={}", lib_dir.display());
        }

        println!("cargo:rustc-link-lib=dylib=vulkan-1");
        return;
    }

    if let Some(vulkan_sdk) = vulkan_sdk {
        let lib_dir = vulkan_sdk.join("lib");
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
    }

    println!("cargo:rustc-link-lib=dylib=vulkan");
}

/// Builds Vship's CUDA backend with nvcc and links the static CUDA runtime.
fn compile_cuda() {
    let user_nvcc = user_nvcc();
    if user_nvcc.is_none() {
        let nvcc = find_nvcc();
        let Some(nvcc) = nvcc else {
            skip("no nvcc was found, install the CUDA toolkit or set CUDA_PATH");
            return;
        };

        // `cc` only looks for nvcc through `NVCC` or `PATH`, so it is pointed at the one found.
        // SAFETY: The build script is single threaded here, as bindgen's thread has already
        // joined and `cc` has not started.
        unsafe { env::set_var("NVCC", &nvcc) };
    }

    let arch = env::var("VSHIP_CUDA_ARCH").unwrap_or_else(|_| DEFAULT_CUDA_ARCH.to_string());
    let arch_flag = format!("-arch={arch}");

    // nvcc treats a `.cpp` file as host code unless `-x cu` makes it a CUDA source.
    let mut build = cc::Build::new();
    build
        .cuda(true)
        .cudart("static")
        .warnings(false)
        .flag("-x")
        .flag("cu")
        .flag("-std=c++17")
        .flag(&arch_flag)
        .include("vendor/vship/include")
        .define("NDEBUG", None)
        .define("VSHIP_STATIC", None);

    for (name, value) in VSHIP_VERSION {
        build.define(name, value);
    }

    if !is_windows_target() {
        build.flag("-Xcompiler").flag("-fPIC");
    }

    build.file(VSHIP_SOURCE).compile("vship");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "linux" {
        for library in ["dl", "rt", "pthread"] {
            println!("cargo:rustc-link-lib=dylib={library}");
        }
    }
}

/// The `NVCC` the user set, when it names an existing file.
fn user_nvcc() -> Option<PathBuf> {
    let nvcc = env::var_os("NVCC")?;
    let nvcc = PathBuf::from(nvcc);

    nvcc.is_file().then_some(nvcc)
}

/// `nvcc` from `CUDA_PATH/bin`, falling back to the first one on `PATH`.
fn find_nvcc() -> Option<PathBuf> {
    let nvcc_name = format!("nvcc{}", env::consts::EXE_SUFFIX);

    let cuda_path = env::var_os("CUDA_PATH");
    if let Some(cuda_path) = cuda_path {
        let nvcc = PathBuf::from(cuda_path).join("bin").join(&nvcc_name);
        if nvcc.is_file() {
            return Some(nvcc);
        }
    }

    let path = env::var_os("PATH")?;
    env::split_paths(&path)
        .map(|dir| dir.join(&nvcc_name))
        .find(|nvcc| nvcc.is_file())
}

fn generate_bindings() {
    let include_dir = PathBuf::from("include");
    let header = include_dir.join("VshipAPI.h");
    println!("cargo:rerun-if-changed={}", header.display());

    let header_text = header.to_string_lossy();
    let include_argument = format!("-I{}", include_dir.display());
    let generated = bindgen::Builder::default()
        .header(header_text)
        .clang_arg(include_argument)
        .clang_arg("-DVSHIP_STATIC")
        .allowlist_function("Vship_.*")
        .allowlist_type("Vship_.*")
        .allowlist_var("Vship_.*")
        .prepend_enum_name(false)
        .rust_edition(bindgen::RustEdition::Edition2024)
        .generate();

    let bindings = match generated {
        Ok(bindings) => bindings,
        Err(error) => panic!("bindgen could not read {}: {error}", header.display()),
    };

    let output = out_dir().join("bindings.rs");
    let written = bindings.write_to_file(&output);
    if let Err(error) = written {
        panic!("could not write {}: {error}", output.display());
    }
}
