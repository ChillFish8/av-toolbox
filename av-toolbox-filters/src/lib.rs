//! Filters that change the frames of a stream
//!
//! - `blend/masked` picks each pixel from one of two clips by a mask.
//! - `deband/placebo` smooths banding with libplacebo's algorithm, then adds grain.
//! - `dehalo/alpha` removes halos by pulling pixels toward a halo-free estimate made by resizing.
//! - `dehalo/edge_clean` warp-sharpens and repairs away from strong edges.
//! - `dehalo/fine` runs `dehalo/alpha` only in a band around strong edges.
//! - `dehalo/yahr` removes halos by limiting a clip's detail to the detail of its warp-sharpened
//!   self, the way havsfunc's YAHR does.
//! - `dering/hq` smooths ringing inside a mask of the area around edges.
//! - `lineart/darken` darkens thin lines, as vs-jetpack's fast_line_darken does.
//! - `lineart/thin` warp-sharpens only on lines, to thin them.
//! - `lineart/toon` darkens lines by a warp-thinned map of how dark they are, as havsfunc's Toon does.
//! - `lineart/vinverse` removes residual combing, as vs-jetpack's vinverse does.
//! - `resize/point`, `resize/bilinear`, `resize/bicubic`, `resize/lanczos`, `resize/spline16`,
//!   `resize/spline36` and `resize/spline64` resize a clip, optionally from a window of it.
//! - `resample/depth` changes a clip's bit depth, dithering when it lowers it.
//! - `resample/chroma` changes a YUV clip's chroma subsampling and siting.
//! - `planes/extract` pulls one plane, or the chroma pair, out of a clip.
//! - `planes/merge` builds a clip by picking each plane from one of several clips.
//! - `upscale/artcnn` doubles a clip's size, running its luma through the ArtCNN network.
//! - `warp/awarpsharp` sharpens by pulling samples toward nearby edges, the way aWarpSharp2 does.

mod bits;
mod blend;
mod checks;
mod deband;
mod dehalo;
mod dering;
mod lineart;
mod masked;
mod padded_warp;
mod planes;
mod planewise;
mod resample;
mod resize;
mod scratch;
mod selection;
mod sizing;
mod upscale;
mod warp;

use av_toolbox_core::{PipelineRegistry, RegistryError};

use self::resample::{Chroma, Depth};
use self::resize::{Bicubic, Bilinear, Lanczos, Point, Resize, Spline16, Spline36, Spline64};

/// Registers every filter this crate provides.
pub fn register(registry: &mut PipelineRegistry) -> Result<(), RegistryError> {
    registry.register_filter::<blend::Masked>()?;
    registry.register_filter::<deband::Placebo>()?;
    registry.register_filter::<dehalo::Alpha>()?;
    registry.register_filter::<dehalo::EdgeClean>()?;
    registry.register_filter::<dehalo::Fine>()?;
    registry.register_filter::<dehalo::Yahr>()?;
    registry.register_filter::<dering::Hq>()?;
    registry.register_filter::<lineart::Darken>()?;
    registry.register_filter::<lineart::Thin>()?;
    registry.register_filter::<lineart::Toon>()?;
    registry.register_filter::<lineart::Vinverse>()?;
    registry.register_filter::<Resize<Point>>()?;
    registry.register_filter::<Resize<Bilinear>>()?;
    registry.register_filter::<Resize<Bicubic>>()?;
    registry.register_filter::<Resize<Lanczos>>()?;
    registry.register_filter::<Resize<Spline16>>()?;
    registry.register_filter::<Resize<Spline36>>()?;
    registry.register_filter::<Resize<Spline64>>()?;
    registry.register_filter::<Depth>()?;
    registry.register_filter::<Chroma>()?;
    registry.register_filter::<planes::Extract>()?;
    registry.register_filter::<planes::Merge>()?;
    registry.register_filter::<warp::Awarpsharp>()?;
    registry.register_filter::<upscale::Artcnn>()?;

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
        assert!(registered("deband/placebo"));
        assert!(registered("dehalo/alpha"));
        assert!(registered("dehalo/edge_clean"));
        assert!(registered("dehalo/fine"));
        assert!(registered("dehalo/yahr"));
        assert!(registered("dering/hq"));
        assert!(registered("lineart/darken"));
        assert!(registered("lineart/thin"));
        assert!(registered("lineart/toon"));
        assert!(registered("lineart/vinverse"));
        assert!(registered("resize/point"));
        assert!(registered("resize/bilinear"));
        assert!(registered("resize/bicubic"));
        assert!(registered("resize/lanczos"));
        assert!(registered("resize/spline16"));
        assert!(registered("resize/spline36"));
        assert!(registered("resize/spline64"));
        assert!(registered("resample/depth"));
        assert!(registered("resample/chroma"));
        assert!(registered("planes/extract"));
        assert!(registered("planes/merge"));
        assert!(registered("warp/awarpsharp"));
        assert!(registered("upscale/artcnn"));
    }
}
