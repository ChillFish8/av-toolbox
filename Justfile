# Backend the tests build against. Set AVTB_TEST_BACKEND=metal on macOS.
test_backend := env("AVTB_TEST_BACKEND", "vulkan")

# Backend features across every crate that forwards them.
backend_features := "av-toolbox-gpu/" + test_backend + ",av-toolbox-core/" + test_backend + ",av-toolbox-graph/" + test_backend + ",av-toolbox-runtime/" + test_backend + ",av-toolbox-sources/" + test_backend + ",av-toolbox-providers/" + test_backend + ",av-toolbox-sinks/" + test_backend + ",av-toolbox-pipeline/" + test_backend + ",av-toolbox/" + test_backend + ",av-toolbox-masks/" + test_backend + ",av-toolbox-filters/" + test_backend + ",av-toolbox-cubek/" + test_backend

# The above plus the device tests those crates gate behind gpu-tests.
device_features := backend_features + ",av-toolbox-gpu/gpu-tests,av-toolbox-core/gpu-tests,av-toolbox-graph/gpu-tests,av-toolbox-runtime/gpu-tests,av-toolbox-sources/gpu-tests,av-toolbox-providers/gpu-tests,av-toolbox-sinks/gpu-tests,av-toolbox-pipeline/gpu-tests,av-toolbox/gpu-tests,av-toolbox-masks/gpu-tests,av-toolbox-filters/gpu-tests,av-toolbox-cubek/gpu-tests"

# Crates the workspace recipes leave out, since each needs a library built outside cargo.
thirdparty_excludes := "--exclude avtbx-bestsource --exclude avtbx-vship"

# The vcpkg triplet `static-deps` installs for. Override on Intel macOS.
vcpkg_triplet := env("AVTB_VCPKG_TRIPLET", if os() == "macos" { "arm64-osx-release" } else { "x64-linux-release" })

# The backend `build-static` builds when none is given.
static_features := if os() == "macos" { "metal" } else { "vulkan" }

# Formats the whole workspace.
format:
    cargo +nightly fmt --all

# Lints the workspace. Pass `-- -D warnings` to fail on any lint.
clippy *ARGS:
    cargo clippy --workspace {{thirdparty_excludes}} --all-targets --features {{device_features}} {{ARGS}}

# Every test, accelerator and doc tests included.
test:
    cargo nextest run --cargo-profile test-release --workspace {{thirdparty_excludes}} --features {{device_features}}
    cargo test --doc --profile test-release --workspace {{thirdparty_excludes}} --features {{device_features}}
    cargo check --workspace {{thirdparty_excludes}} --features av-toolbox-gpu/{{test_backend}}

# Every test that runs without an accelerator.
test-fast:
    cargo nextest run --cargo-profile test-release --workspace {{thirdparty_excludes}} --features {{backend_features}}
    cargo test --doc --profile test-release --workspace {{thirdparty_excludes}} --features {{backend_features}}

# Lints every backend including test code, and checks the illegal feature sets are refused.
check-backends:
    cargo clippy -p av-toolbox-gpu --features vulkan,gpu-tests --all-targets -- -D warnings
    cargo clippy -p av-toolbox-gpu --features metal,gpu-tests --all-targets -- -D warnings
    cargo clippy -p av-toolbox-gpu --features cuda,gpu-tests --all-targets -- -D warnings
    cargo clippy -p av-toolbox-gpu --features rocm,gpu-tests --all-targets -- -D warnings
    cargo clippy -p av-toolbox-gpu --features cuda,rocm,vulkan,gpu-tests --all-targets -- -D warnings
    cargo check -p av-toolbox-gpu 2>&1 | grep -q "at least one accelerator backend"
    cargo check -p av-toolbox-gpu --features vulkan,metal 2>&1 | grep -q "cannot both be enabled"

# Fetches the vendored BestSource and libp2p sources at their pinned commits.
submodules:
    git submodule update --init --recursive

# Moves every submodule to its upstream's latest commit.
#
# Review the new pins and move the matching `REF`s in `vcpkg/ports/bestsource` and `vcpkg/ports/libp2p`
# after any change in revision.
update-submodules:
    git submodule update --init --remote --recursive
    git submodule status

# Builds the static ffms2, BestSource and FFmpeg tree from vcpkg.json into vcpkg_installed/.
static-deps:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ ! -d .vcpkg ]; then
        git clone https://github.com/microsoft/vcpkg.git .vcpkg
    fi
    git -C .vcpkg fetch --quiet origin
    baseline="$(jq -r '."builtin-baseline"' vcpkg.json)"
    git -C .vcpkg checkout --quiet "$baseline"
    if [ ! -x .vcpkg/vcpkg ]; then
        .vcpkg/bootstrap-vcpkg.sh -disableMetrics
    fi
    .vcpkg/vcpkg install --triplet {{vcpkg_triplet}} --host-triplet {{vcpkg_triplet}}

# Builds a release CLI with ffms2 and FFmpeg linked statically. Run `static-deps` first.
build-static features=static_features:
    #!/usr/bin/env bash
    set -euo pipefail
    prefix="$PWD/vcpkg_installed/{{vcpkg_triplet}}"
    export FFMS_INCLUDE_DIR="$prefix/include"
    export FFMS_LIB_DIR="$prefix/lib"
    export PKG_CONFIG_PATH="$prefix/lib/pkgconfig"
    cargo build --release -p av-toolbox --features {{features}},static-ffms2

# Lints and tests the BestSource wrapper in both link modes. Run `static-deps` first.
test-bestsource:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo clippy -p avtbx-bestsource --all-targets -- -D warnings
    cargo nextest run --cargo-profile test-release -p avtbx-bestsource
    cargo clippy -p av-toolbox-sources --features av-toolbox-sources/vulkan,av-toolbox-sources/bestsource --all-targets -- -D warnings
    cargo nextest run --cargo-profile test-release -p av-toolbox-sources --features av-toolbox-sources/vulkan,av-toolbox-sources/bestsource bestsource
    prefix="$PWD/vcpkg_installed/{{vcpkg_triplet}}"
    export BESTSOURCE_LIB_DIR="$prefix/lib"
    export PKG_CONFIG_PATH="$prefix/lib/pkgconfig"
    cargo clippy -p avtbx-bestsource --features static --all-targets -- -D warnings
    cargo nextest run --cargo-profile test-release -p avtbx-bestsource --features static

# The Vship commit `vship` builds, tag v5.1.2.
vship_commit := "5a627933274196219f782b3b74e13fad47cd6370"

# Builds the static Vship library for `vulkan` or `cuda` into target/thirdparty/vship/<backend>.
vship backend:
    #!/usr/bin/env bash
    set -euo pipefail
    case "{{backend}}" in
        vulkan) make_backend=Vulkan ;;
        cuda) make_backend=Cuda ;;
        *) echo "the backend must be vulkan or cuda, not {{backend}}" >&2; exit 1 ;;
    esac
    if [ "$(uname -s)" != "Linux" ]; then
        echo "just vship only builds on Linux" >&2
        exit 1
    fi
    source_dir="$PWD/target/thirdparty/vship/src"
    install_dir="$PWD/target/thirdparty/vship/{{backend}}"
    if [ ! -d "$source_dir" ]; then
        git clone https://codeberg.org/Line-fr/Vship.git "$source_dir"
    fi
    git -C "$source_dir" fetch --quiet origin
    git -C "$source_dir" checkout --quiet {{vship_commit}}
    make -C "$source_dir" build ENABLE_STATIC=1 BACKEND="$make_backend"
    mkdir -p "$install_dir/lib" "$install_dir/include"
    cp "$source_dir/libvship.a" "$install_dir/lib/"
    cp "$source_dir/src/VshipAPI.h" "$source_dir/src/VshipColor.h" "$install_dir/include/"
    echo "{{backend}}" > "$install_dir/backend"

# Lints and tests the Vship wrapper on one backend, device tests included. Run `vship <backend>` first.
test-vship backend:
    #!/usr/bin/env bash
    set -euo pipefail
    export VSHIP_DIR="$PWD/target/thirdparty/vship/{{backend}}"
    cargo clippy -p avtbx-vship --all-targets --features {{backend}},gpu-tests -- -D warnings
    cargo nextest run --cargo-profile test-release -p avtbx-vship --features {{backend}},gpu-tests

# Builds the optional dependencies under dependencies/, or only the ones named.
dependencies *IDENTIFIERS:
    uv run --script tools/dependencies/build.py {{IDENTIFIERS}}

# Rebuilds every dependency and fails if the lock or a builtin file would change.
dependencies-check:
    uv run --script tools/dependencies/build.py --check

# Tests the dependency builder and every converter.
dependencies-test:
    uv run --with pytest --with onnx==1.23.0 --with numpy==2.5.3 pytest --import-mode=importlib tools/dependencies dependencies

# Writes the ArtCNN golden fixture from the cached ONNX model. Run `just dependencies artcnn` first.
artcnn-golden:
    uv run --script dependencies/artcnn/golden.py target/dependencies/cache/artcnn av-toolbox-cubek/src/testing/fixtures

# Writes the DPIR golden fixtures from the cached ONNX models. Run `just dependencies dpir` first.
dpir-golden:
    uv run --script dependencies/dpir/golden.py target/dependencies/cache/dpir av-toolbox-cubek/src/testing/fixtures/dpir

# Publishes every crate not yet on crates.io, waiting out rate limits. Pass `--dry-run` to check without uploading.
publish *ARGS:
    uv run --script tools/publish.py {{ARGS}}
