# Backend the tests build against. Set AVTB_TEST_BACKEND=metal on macOS.
test_backend := env("AVTB_TEST_BACKEND", "vulkan")

# Backend features across every crate that forwards them.
backend_features := "av-toolbox-gpu/" + test_backend + ",av-toolbox-core/" + test_backend + ",av-toolbox-graph/" + test_backend + ",av-toolbox-runtime/" + test_backend + ",av-toolbox-sources/" + test_backend + ",av-toolbox-transforms/" + test_backend + ",av-toolbox-sinks/" + test_backend + ",av-toolbox-pipeline/" + test_backend + ",av-toolbox/" + test_backend + ",av-toolbox-masks/" + test_backend + ",av-toolbox-filters/" + test_backend + ",av-toolbox-cubek/" + test_backend

# The above plus the device tests those crates gate behind gpu-tests.
device_features := backend_features + ",av-toolbox-gpu/gpu-tests,av-toolbox-core/gpu-tests,av-toolbox-graph/gpu-tests,av-toolbox-runtime/gpu-tests,av-toolbox-sources/gpu-tests,av-toolbox-transforms/gpu-tests,av-toolbox-sinks/gpu-tests,av-toolbox-pipeline/gpu-tests,av-toolbox/gpu-tests,av-toolbox-masks/gpu-tests,av-toolbox-filters/gpu-tests,av-toolbox-cubek/gpu-tests"

# Formats the whole workspace.
format:
    cargo +nightly fmt --all

# Lints the workspace. Pass `-- -D warnings` to fail on any lint.
clippy *ARGS:
    cargo clippy --workspace --all-targets --features {{device_features}} {{ARGS}}

# Every test, accelerator and doc tests included.
test:
    cargo nextest run --cargo-profile test-release --workspace --features {{device_features}}
    cargo test --doc --profile test-release --workspace --features {{device_features}}
    cargo check --workspace --features av-toolbox-gpu/{{test_backend}}

# Every test that runs without an accelerator.
test-fast:
    cargo nextest run --cargo-profile test-release --workspace --features {{backend_features}}
    cargo test --doc --profile test-release --workspace --features {{backend_features}}

# Lints every backend including test code, and checks the illegal feature sets are refused.
check-backends:
    cargo clippy -p av-toolbox-gpu --features vulkan,gpu-tests --all-targets -- -D warnings
    cargo clippy -p av-toolbox-gpu --features metal,gpu-tests --all-targets -- -D warnings
    cargo clippy -p av-toolbox-gpu --features cuda,gpu-tests --all-targets -- -D warnings
    cargo clippy -p av-toolbox-gpu --features rocm,gpu-tests --all-targets -- -D warnings
    cargo clippy -p av-toolbox-gpu --features cuda,rocm,vulkan,gpu-tests --all-targets -- -D warnings
    cargo check -p av-toolbox-gpu 2>&1 | grep -q "at least one accelerator backend"
    cargo check -p av-toolbox-gpu --features vulkan,metal 2>&1 | grep -q "cannot both be enabled"

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
