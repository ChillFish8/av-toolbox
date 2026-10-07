use std::collections::HashSet;

use super::{R9700, TestDevice};
use crate::artcnn::Model;
use crate::cnn::kernels::tune::{Choice, SizeBucket, choose, offered_candidates};
use crate::cnn::{Candidate, Kernels, direct_variants};

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

    let candidates = offered_candidates(&properties, 64, None);

    let names: HashSet<String> = candidates.iter().map(Candidate::name).collect();
    assert_eq!(candidates.last(), Some(&Candidate::FusedCubek));
    assert!(candidates.len() > 2, "{candidates:?}");
    assert_eq!(names.len(), candidates.len(), "{names:?}");
}

#[test]
fn a_device_offering_winograd_but_no_direct_variant_runs_the_fused_cubek_convolution() {
    // Room for the smallest Winograd variant at width 64, but not for the smallest direct variant.
    let device = TestDevice {
        shared_bytes: 8 * 1024,
        ..R9700
    };
    let properties = device.properties();
    let model = Model::R8F64;
    let features = model.features();

    let direct = direct_variants(&properties, features);
    let candidates = offered_candidates(&properties, features, Some(model));
    let choice = choose(&properties, &candidates, features);

    let holds_winograd = candidates
        .iter()
        .any(|candidate| matches!(candidate, Candidate::Winograd(_)));
    assert!(direct.is_empty(), "{direct:?}");
    assert!(holds_winograd, "{candidates:?}");
    assert_eq!(choice, Choice::Once(Candidate::FusedCubek));
}
