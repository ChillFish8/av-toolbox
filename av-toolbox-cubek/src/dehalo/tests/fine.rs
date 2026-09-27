use super::alpha::{defaults, haloed_step};
use crate::dehalo::fine::FineParams;
use crate::testing::reference_fine_dehalo;

fn fine(bits: u32) -> FineParams {
    FineParams {
        alpha: defaults(bits),
        thmi: 80.0,
        thma: 128.0,
        thlimi: 50.0,
        thlima: 100.0,
        contra: 0.0,
        excl: true,
        edgeproc: 0.0,
    }
}

#[test]
fn the_radii_round_half_away_from_zero() {
    let mut params = fine(8);
    params.alpha.rx = 1.4;
    params.alpha.ry = 2.5;

    assert_eq!(params.radii(), (1, 3));
}

#[test]
fn a_flat_plane_comes_back_flat() {
    let flat = vec![90.0f32; 64 * 32];

    for contra in [0.0, 1.0] {
        let params = FineParams { contra, ..fine(8) };
        let dehaloed = reference_fine_dehalo(&flat, 64, 32, &params);

        assert!(
            dehaloed.iter().all(|value| (value - 90.0).abs() < 1e-3),
            "contra {contra}"
        );
    }
}

#[test]
fn a_haloed_step_stays_within_its_input_range_and_changes() {
    let input = haloed_step(64, 16);
    let params = fine(8);

    let dehaloed = reference_fine_dehalo(&input, 64, 16, &params);

    let brightest = dehaloed.iter().copied().fold(f32::MIN, f32::max);
    let darkest = dehaloed.iter().copied().fold(f32::MAX, f32::min);
    let changed = dehaloed
        .iter()
        .zip(&input)
        .any(|(after, before)| (after - before).abs() > 0.5);
    assert!(
        brightest <= 240.0 && darkest >= 10.0,
        "left the range {darkest}..{brightest}"
    );
    assert!(changed, "nothing near the edge changed");
}
