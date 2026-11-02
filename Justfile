# Backend the tests build against. Set AVTB_TEST_BACKEND=metal on macOS.
test_backend := env("AVTB_TEST_BACKEND", "vulkan")

# ROCm crashes under more test processes than this.
test_threads := if test_backend == "rocm" { "8" } else { "num-cpus" }

# Crates that forward the backend and gpu-tests features.
backend_crates := (
    "av-toolbox-gpu av-toolbox-core av-toolbox-graph av-toolbox-runtime"
    + " av-toolbox-sources av-toolbox-providers av-toolbox-sinks av-toolbox-pipeline"
    + " av-toolbox av-toolbox-masks av-toolbox-filters av-toolbox-cubek"
)

# Third-party sources and filters the default recipes always build in.
thirdparty_features := (
    "av-toolbox-sources/bestsource,av-toolbox-filters/vship"
    + ",av-toolbox-pipeline/bestsource,av-toolbox-pipeline/vship"
    + ",av-toolbox/bestsource,av-toolbox/vship"
)

backend_list := replace_regex(backend_crates, '(\S+)', '$1/' + test_backend)
gpu_test_list := replace_regex(backend_crates + " avtbx-vship", '(\S+)', '$1/gpu-tests')

# Backend features across every crate that forwards them.
backend_features := replace(backend_list, " ", ",") + "," + thirdparty_features

# The above plus the device tests those crates gate behind gpu-tests.
device_features := backend_features + "," + replace(gpu_test_list, " ", ",")

# The vcpkg triplet `static-deps` installs for. Override on Intel macOS.
vcpkg_triplet := env("AVTB_VCPKG_TRIPLET", if os() == "macos" { "arm64-osx-release" } else { "x64-linux-release" })

# The backend `build-static` builds when none is given.
static_features := if os() == "macos" { "metal" } else { "vulkan" }

# The backend features used for the `run` alias
run_features := static_features + ",ffms2,bestsource,vship"

# Formats the whole workspace.
format:
    cargo +nightly fmt --all

# Lints the workspace. Pass `-- -D warnings` to fail on any lint.
clippy *ARGS:
    cargo clippy --workspace --all-targets --features {{device_features}} {{ARGS}}

# Every test, accelerator and doc tests included.
test:
    cargo nextest run --cargo-profile test-release --test-threads {{test_threads}} --workspace --features {{device_features}}
    cargo test --doc --profile test-release --workspace --features {{device_features}}
    cargo check --workspace --features av-toolbox-gpu/{{test_backend}}

# Helper run alias which enabled the platform default features
run *ARGS:
    cargo run --release --features {{run_features}} -- {{ARGS}}

# Execute a pipeline and pipe the output to ffmpeg into a lossless h264 file.
#
# The pipeline must write its y4m stream to standard output.
execute-to-lossless pipeline output *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo run --release --features {{run_features}} -- execute "{{pipeline}}" {{ARGS}} \
        | ffmpeg -hide_banner -nostdin -loglevel warning -i - -c:v libx264 -preset veryfast -qp 0 "{{output}}"

# Every test that runs without an accelerator.
test-fast:
    cargo nextest run --cargo-profile test-release --workspace --features {{backend_features}}
    cargo test --doc --profile test-release --workspace --features {{backend_features}}

# Lints every backend including test code, and checks the illegal feature sets are refused.
check-backends:
    #!/usr/bin/env bash
    set -eu
    cargo clippy -p av-toolbox-gpu --features vulkan,gpu-tests --all-targets -- -D warnings
    cargo clippy -p av-toolbox-gpu --features cuda,gpu-tests --all-targets -- -D warnings
    cargo clippy -p av-toolbox-gpu --features rocm,gpu-tests --all-targets -- -D warnings
    cargo clippy -p av-toolbox-gpu --features cuda,rocm,vulkan,gpu-tests --all-targets -- -D warnings
    cargo check -p av-toolbox-gpu 2>&1 | grep -q "at least one accelerator backend"
    if [ "{{os()}}" = "macos" ]; then
        cargo clippy -p av-toolbox-gpu --features metal,gpu-tests --all-targets -- -D warnings
    else
        cargo check -p av-toolbox-gpu --features metal 2>&1 | grep -q "needs an Apple target"
    fi

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

# Builds a release CLI with ffms2, BestSource and FFmpeg linked statically. Run `static-deps` first.
build-static features=static_features:
    #!/usr/bin/env bash
    set -euo pipefail
    prefix="$PWD/vcpkg_installed/{{vcpkg_triplet}}"
    export FFMS_INCLUDE_DIR="$prefix/include"
    export FFMS_LIB_DIR="$prefix/lib"
    export BESTSOURCE_LIB_DIR="$prefix/lib"
    export PKG_CONFIG_PATH="$prefix/lib/pkgconfig"
    cargo build --release -p av-toolbox --features {{features}},static-ffms2,static-bestsource

# Lints and tests the BestSource wrapper linked statically from the vcpkg tree. Run `static-deps` first.
test-bestsource:
    #!/usr/bin/env bash
    set -euo pipefail
    prefix="$PWD/vcpkg_installed/{{vcpkg_triplet}}"
    export BESTSOURCE_LIB_DIR="$prefix/lib"
    export PKG_CONFIG_PATH="$prefix/lib/pkgconfig"
    cargo clippy -p avtbx-bestsource --features static --all-targets -- -D warnings
    cargo nextest run --cargo-profile test-release -p avtbx-bestsource --features static

# Lints and tests the Vship wrapper built statically from its submodule with Vulkan.
test-vship:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo clippy -p avtbx-vship --features static,vulkan,gpu-tests --all-targets -- -D warnings
    cargo nextest run --cargo-profile test-release -p avtbx-vship --features static,vulkan,gpu-tests

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
