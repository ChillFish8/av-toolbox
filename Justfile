# Backend the tests build against. Set AVTB_TEST_BACKEND=metal on macOS.
test_backend := env("AVTB_TEST_BACKEND", "vulkan")

# Backend features across every crate that forwards them.
backend_features := "av-toolbox-gpu/" + test_backend + ",av-toolbox-core/" + test_backend + ",av-toolbox-graph/" + test_backend + ",av-toolbox-runtime/" + test_backend + ",av-toolbox-sources/" + test_backend + ",av-toolbox-transforms/" + test_backend + ",av-toolbox-sinks/" + test_backend

# The above plus the device tests those crates gate behind gpu-tests.
device_features := backend_features + ",av-toolbox-gpu/gpu-tests,av-toolbox-core/gpu-tests,av-toolbox-graph/gpu-tests,av-toolbox-runtime/gpu-tests,av-toolbox-sources/gpu-tests,av-toolbox-transforms/gpu-tests,av-toolbox-sinks/gpu-tests"

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
