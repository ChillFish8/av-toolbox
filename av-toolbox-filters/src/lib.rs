//! Filters that change the frames of a stream
//!
//! - `blend/masked` picks each pixel from one of two clips by a mask.
//! - `resize/point`, `resize/bilinear`, `resize/bicubic`, `resize/lanczos`, `resize/spline16`,
//!   `resize/spline36` and `resize/spline64` resize a clip, optionally from a window of it.

mod blend;
mod resize;

use av_toolbox_core::{PipelineRegistry, RegistryError};

use self::resize::{Bicubic, Bilinear, Lanczos, Point, Resize, Spline16, Spline36, Spline64};

/// Registers every filter this crate provides.
pub fn register(registry: &mut PipelineRegistry) -> Result<(), RegistryError> {
    registry.register_filter::<blend::Masked>()?;
    registry.register_filter::<Resize<Point>>()?;
    registry.register_filter::<Resize<Bilinear>>()?;
    registry.register_filter::<Resize<Bicubic>>()?;
    registry.register_filter::<Resize<Lanczos>>()?;
    registry.register_filter::<Resize<Spline16>>()?;
    registry.register_filter::<Resize<Spline36>>()?;
    registry.register_filter::<Resize<Spline64>>()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use av_toolbox_core::StageKind;

    use super::*;

    fn registered(identifier: &str) -> bool {
        let mut registry = PipelineRegistry::new();
        register(&mut registry).expect("the filters register");

        registry
            .stages()
            .any(|info| info.kind() == StageKind::Filter && info.full_identifier() == identifier)
    }

    #[test]
    fn every_filter_is_registered() {
        assert!(registered("blend/masked"));
        assert!(registered("resize/point"));
        assert!(registered("resize/bilinear"));
        assert!(registered("resize/bicubic"));
        assert!(registered("resize/lanczos"));
        assert!(registered("resize/spline16"));
        assert!(registered("resize/spline36"));
        assert!(registered("resize/spline64"));
    }
}
