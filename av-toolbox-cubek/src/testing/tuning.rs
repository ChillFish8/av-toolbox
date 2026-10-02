use std::time::{SystemTime, UNIX_EPOCH};

use crate::artcnn::{Candidate, DirectVariant, Kernels};

/// A tuning label no earlier run used, so the tuner benchmarks `test`'s keys afresh.
pub fn fresh_tune_label(test: &str) -> String {
    let since_epoch = SystemTime::now().duration_since(UNIX_EPOCH);
    let nanos = since_epoch.expect("the clock is past the epoch").as_nanos();
    let process = std::process::id();

    format!("test-{test}-{process}-{nanos}")
}

/// Kernels whose every candidate is refused before it runs, so a body convolution on them fails.
pub fn failing_kernels() -> Kernels {
    let off_the_tile = DirectVariant {
        columns: 24,
        ..DirectVariant::default()
    };
    let empty_block = DirectVariant {
        rows: 0,
        ..DirectVariant::default()
    };
    let candidates = vec![Candidate::Direct(off_the_tile), Candidate::Direct(empty_block)];

    Kernels::Among {
        label: "test-failing".to_string(),
        candidates,
    }
}
