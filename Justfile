# Backend the tests build against. Set AVTB_TEST_BACKEND=metal on macOS.
test_backend := env("AVTB_TEST_BACKEND", "vulkan")

# Formats the whole workspace.
format:
    cargo +nightly fmt --all

# Lints the workspace. Pass `-- -D warnings` to fail on any lint.
clippy *ARGS:
    cargo clippy --workspace --all-targets --features av-toolbox-gpu/{{test_backend}} {{ARGS}}

# Every test, accelerator and doc tests included.
test:
    cargo nextest run --release --workspace --features av-toolbox-gpu/{{test_backend}},av-toolbox-gpu/gpu-tests
    cargo test --doc --release --workspace --features av-toolbox-gpu/{{test_backend}},av-toolbox-gpu/gpu-tests
    cargo check --workspace --features av-toolbox-gpu/{{test_backend}}

# Every test that runs without an accelerator.
test-fast:
    cargo nextest run --release --workspace --features av-toolbox-gpu/{{test_backend}}
    cargo test --doc --release --workspace --features av-toolbox-gpu/{{test_backend}}

# Lints each backend and checks the illegal feature sets are refused.
check-backends:
    cargo clippy -p av-toolbox-gpu --features vulkan --all-targets -- -D warnings
    cargo clippy -p av-toolbox-gpu --features metal --all-targets -- -D warnings
    cargo clippy -p av-toolbox-gpu --features cuda --all-targets -- -D warnings
    cargo clippy -p av-toolbox-gpu --features rocm --all-targets -- -D warnings
    cargo clippy -p av-toolbox-gpu --features cuda,rocm,vulkan --all-targets -- -D warnings
    cargo check -p av-toolbox-gpu 2>&1 | grep -q "at least one accelerator backend"
    cargo check -p av-toolbox-gpu --features vulkan,metal 2>&1 | grep -q "cannot both be enabled"
