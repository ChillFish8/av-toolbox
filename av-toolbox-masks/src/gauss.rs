/// The blur radius for `sigma`, `ceil(6 * sigma + 1) / 2`, as vs-jetpack rounds it.
pub(crate) fn jetpack_radius(sigma: f32) -> u32 {
    let reach = (6.0 * sigma + 1.0).ceil();

    reach as u32 / 2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_blur_radius_is_vs_jetpacks() {
        assert_eq!(jetpack_radius(1.0), 3);
        assert_eq!(jetpack_radius(5.0), 15);
    }
}
