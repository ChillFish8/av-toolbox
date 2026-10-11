use std::path::{Path, PathBuf};
use std::{env, fs};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=include");
    println!("cargo:rerun-if-env-changed=VSHIP_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=DOCS_RS");

    generate_bindings();

    if env::var_os("DOCS_RS").is_some() {
        skip("docs.rs builds documentation only");
        return;
    }

    let vship_dir = env::var_os("VSHIP_DIR");
    let Some(vship_dir) = vship_dir else {
        skip("VSHIP_DIR is unset, run `just vship <backend>` and point it at target/thirdparty/vship/<backend>");
        return;
    };

    let vship_dir = PathBuf::from(vship_dir);
    let backend = selected_backend();
    require_matching_build(&vship_dir, backend);

    let lib_dir = vship_dir.join("lib");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=vship");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    match backend {
        Backend::Vulkan => link_vulkan(),
        Backend::Cuda => link_cuda(),
    }
}

/// Links nothing, so only a library build of this crate can succeed.
fn skip(reason: &str) {
    println!("cargo:warning=Vship is not linked because {reason}. Anything that links avtbx-vship will fail to link.");
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Backend {
    Vulkan,
    Cuda,
}

impl Backend {
    fn name(self) -> &'static str {
        match self {
            Self::Vulkan => "vulkan",
            Self::Cuda => "cuda",
        }
    }
}

/// The backend the `vulkan` or `cuda` feature picks, failing unless exactly one is on.
fn selected_backend() -> Backend {
    let vulkan = env::var_os("CARGO_FEATURE_VULKAN").is_some();
    let cuda = env::var_os("CARGO_FEATURE_CUDA").is_some();

    match (vulkan, cuda) {
        (true, false) => Backend::Vulkan,
        (false, true) => Backend::Cuda,
        (true, true) => panic!("avtbx-vship takes one backend, enable `vulkan` or `cuda` but not both"),
        (false, false) => panic!("avtbx-vship needs a backend, enable the `vulkan` or `cuda` feature"),
    }
}

/// Stops a Vship built for one backend from being linked with the other backend's libraries.
fn require_matching_build(vship_dir: &Path, backend: Backend) {
    let marker = vship_dir.join("backend");
    println!("cargo:rerun-if-changed={}", marker.display());

    let built = match fs::read_to_string(&marker) {
        Ok(text) => text,
        Err(error) => panic!(
            "could not read {}, run `just vship {}` first: {error}",
            marker.display(),
            backend.name()
        ),
    };

    let built = built.trim();
    if built != backend.name() {
        panic!(
            "VSHIP_DIR holds a {built} build of Vship but the `{}` feature is enabled",
            backend.name()
        );
    }
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

    let out_dir = env::var_os("OUT_DIR");
    let Some(out_dir) = out_dir else {
        panic!("cargo did not set OUT_DIR");
    };

    let output = PathBuf::from(out_dir).join("bindings.rs");
    let written = bindings.write_to_file(&output);
    if let Err(error) = written {
        panic!("could not write {}: {error}", output.display());
    }
}

/// Links the Vulkan loader, which cannot be linked statically.
fn link_vulkan() {
    println!("cargo:rustc-link-lib=dylib=vulkan");
}

/// Links the static CUDA runtime, so only the driver's `libcuda` is needed at runtime.
fn link_cuda() {
    let cuda_lib_dir = cuda_lib_dir();
    println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());
    println!("cargo:rustc-link-lib=static=cudart_static");

    for library in ["dl", "rt", "pthread"] {
        println!("cargo:rustc-link-lib=dylib={library}");
    }
}

/// `CUDA_PATH/lib64`, or the first usual install folder that exists.
fn cuda_lib_dir() -> PathBuf {
    let cuda_path = env::var_os("CUDA_PATH");
    if let Some(cuda_path) = cuda_path {
        return PathBuf::from(cuda_path).join("lib64");
    }

    for candidate in ["/opt/cuda/lib64", "/usr/local/cuda/lib64"] {
        let path = PathBuf::from(candidate);
        if path.is_dir() {
            return path;
        }
    }

    panic!("no CUDA toolkit found, set CUDA_PATH to its install folder");
}
