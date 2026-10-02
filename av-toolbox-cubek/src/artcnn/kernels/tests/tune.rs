use std::collections::HashSet;

use super::R9700;
use crate::artcnn::kernels::tune::{SizeBucket, offered_candidates};
use crate::artcnn::{Candidate, Kernels};

#[test]
fn sizes_fall_into_their_buckets_at_the_boundaries() {
    let cases = [
        (0, SizeBucket::Small),
        (65_535, SizeBucket::Small),
        (65_536, SizeBucket::Medium),
        (1_048_576, SizeBucket::Medium),
        (1_048_577, SizeBucket::Large),
        (1920 * 1080, SizeBucket::Large),
    ];

    for (samples, bucket) in cases {
        assert_eq!(SizeBucket::of(samples), bucket, "{samples} samples");
    }
}

#[test]
fn the_kernels_are_tuned_unless_pinned() {
    assert_eq!(Kernels::default(), Kernels::Tuned);
}

#[test]
fn the_offered_candidates_end_with_the_fused_cubek_convolution_and_have_distinct_names() {
    let properties = R9700.properties();

    let candidates = offered_candidates(&properties, 64);

    let names: HashSet<String> = candidates.iter().map(Candidate::name).collect();
    assert_eq!(candidates.last(), Some(&Candidate::FusedCubek));
    assert!(candidates.len() > 2, "{candidates:?}");
    assert_eq!(names.len(), candidates.len(), "{names:?}");
}
