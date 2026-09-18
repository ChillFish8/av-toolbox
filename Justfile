
format:
    cargo +nightly fmt --all

clippy:
    cargo clippy

test:
    cargo nextest run