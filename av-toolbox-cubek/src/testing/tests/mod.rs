mod based;

use super::reference_narrow;

#[test]
fn narrowing_rounds_half_up_and_clamps() {
    let values = [-3.0, 0.49, 0.5, 1.5, 254.5, 300.0];

    let narrowed = reference_narrow(&values, 255.0);

    assert_eq!(narrowed, vec![0, 0, 1, 2, 255, 255]);
}
