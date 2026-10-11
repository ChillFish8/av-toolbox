#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, doc(auto_cfg))]

//! Filters that change the frames of a stream
//!
//! - `av-denoise/nlmeans`, `av-denoise/nlmeans-hq` and `av-denoise/nl4d` denoise frames with av-denoise's
//!   engines, run directly on the device planes.
//! - `aa/based` anti-aliases luma by running EEDI3 on an ArtCNN supersample and merging the result
//!   back through an edge mask, the way vs-jetpack's based_aa does.
//! - `aa/eedi2` anti-aliases by re-interpolating each field along a smoothed map of edge directions, the
//!   way EEDI2 does.
//! - `aa/eedi3` anti-aliases by re-interpolating each field along the cheapest path of edge directions
//!   across each row, the way EEDI3 does.
//! - `aa/sangnom` anti-aliases by rebuilding each field along its edges, the way SangNom2 does.
//! - `blend/masked` picks each pixel from one of two clips by a mask.
//! - `deband/placebo` smooths banding with libplacebo's algorithm, then adds grain.
//! - `deblock/dpir` removes compression blocking with the DPIR network, the way vs-jetpack's dpir does.
//! - `deblock/h264` removes compression blocking with H.264's in-loop deblocking filter, the way
//!   VapourSynth's deblock.Deblock does.
//! - `deblock/qed` deblocks block borders gently and spreads a stronger pass into each block's interior, the
//!   way vsdenoise's deblock_qed does.
//! - `deblock/pp7` removes compression blocking by dropping each sample's weak local frequencies, the way
//!   VapourSynth's pp7.DeblockPP7 does.
//! - `dehalo/alpha` removes halos by pulling pixels toward a halo-free estimate made by resizing.
//! - `dehalo/edge_clean` warp-sharpens and repairs away from strong edges.
//! - `dehalo/fine` runs `dehalo/alpha` only in a band around strong edges.
//! - `dehalo/yahr` removes halos by limiting a clip's detail to the detail of its warp-sharpened
//!   self, the way havsfunc's YAHR does.
//! - `dering/hq` smooths ringing inside a mask of the area around edges.
//! - `crop/auto` finds the black bars around a clip's picture and removes them.
//! - `crop/manual` removes a given number of samples from each edge of a clip.
//! - `deinterlace/bwdif` rebuilds progressive frames from interlaced fields, the way bwdif does, at the
//!   input's rate or at one frame per field.
//! - `deinterlace/qtgmc` rebuilds progressive frames from interlaced fields, the way QTGMC does, smoothing
//!   and sharpening them along motion.
//! - `ivtc/match` rebuilds telecined film frames by field matching, the way VIVTC's VFM does.
//! - `ivtc/decimate` drops one frame in every cycle, the one most likely to be a duplicate, the way VIVTC's
//!   VDecimate does.
//! - `lineart/darken` darkens thin lines, as vs-jetpack's fast_line_darken does.
//! - `lineart/thin` warp-sharpens only on lines, to thin them.
//! - `lineart/toon` darkens lines by a warp-thinned map of how dark they are, as havsfunc's Toon does.
//! - `lineart/vinverse` removes residual combing, as vs-jetpack's vinverse does.
//! - `resize/point`, `resize/bilinear`, `resize/bicubic`, `resize/lanczos`, `resize/spline16`,
//!   `resize/spline36` and `resize/spline64` resize a clip, optionally from a window of it.
//! - `resample/depth` changes a clip's bit depth, dithering when it lowers it.
//! - `resample/chroma` changes a YUV clip's chroma subsampling and siting.
//! - `resample/matrix` converts a 4:4:4 YUV clip to RGB and back.
//! - `planes/extract` pulls one plane, or the chroma pair, out of a clip.
//! - `planes/merge` builds a clip by picking each plane from one of several clips.
//! - `upscale/artcnn` doubles a clip's size, running its luma through the ArtCNN network.
//! - `warp/awarpsharp` sharpens by pulling samples toward nearby edges, the way aWarpSharp2 does.

mod aa;
mod artcnn;
mod bits;
mod blend;
mod checks;
mod cnn;
mod crop;
mod deband;
mod deblock;
mod dehalo;
mod deinterlace;
mod denoise;
mod dering;
mod ivtc;
mod lineart;
mod masked;
mod padded_warp;
mod planes;
mod planewise;
mod resample;
mod resize;
#[cfg(test)]
mod sample_checks;
mod scratch;
mod selection;
mod sizing;
#[cfg(feature = "test-support")]
pub mod testing;
mod upscale;
mod warp;

use av_toolbox_core::{PipelineRegistry, RegistryError};

use self::denoise::{Denoise, Nl4dArgs, NlmeansArgs, NlmeansHqArgs};
use self::resample::{Chroma, Depth, Matrix};
use self::resize::{Bicubic, Bilinear, Lanczos, Point, Resize, Spline16, Spline36, Spline64};

/// Registers every filter this crate provides.
pub fn register(registry: &mut PipelineRegistry) -> Result<(), RegistryError> {
    registry.register_filter::<Denoise<NlmeansArgs>>()?;
    registry.register_filter::<Denoise<NlmeansHqArgs>>()?;
    registry.register_filter::<Denoise<Nl4dArgs>>()?;
    registry.register_filter::<aa::Based>()?;
    registry.register_filter::<aa::Eedi2>()?;
    registry.register_filter::<aa::Eedi3>()?;
    registry.register_filter::<aa::Sangnom>()?;
    registry.register_filter::<blend::Masked>()?;
    registry.register_filter::<crop::Auto>()?;
    registry.register_filter::<crop::Manual>()?;
    registry.register_probe::<crop::Crop>()?;
    registry.register_filter::<deband::Placebo>()?;
    registry.register_filter::<deblock::Dpir>()?;
    registry.register_filter::<deblock::H264>()?;
    registry.register_filter::<deblock::Qed>()?;
    registry.register_filter::<deblock::Pp7>()?;
    registry.register_filter::<dehalo::Alpha>()?;
    registry.register_filter::<dehalo::EdgeClean>()?;
    registry.register_filter::<dehalo::Fine>()?;
    registry.register_filter::<dehalo::Yahr>()?;
    registry.register_filter::<dering::Hq>()?;
    registry.register_filter::<deinterlace::Bwdif>()?;
    registry.register_filter::<deinterlace::Qtgmc>()?;
    registry.register_filter::<ivtc::Match>()?;
    registry.register_filter::<ivtc::Decimate>()?;
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
    registry.register_filter::<Matrix>()?;
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
        assert!(registered("av-denoise/nlmeans"));
        assert!(registered("av-denoise/nlmeans-hq"));
        assert!(registered("av-denoise/nl4d"));
        assert!(registered("aa/based"));
        assert!(registered("aa/eedi2"));
        assert!(registered("aa/eedi3"));
        assert!(registered("aa/sangnom"));
        assert!(registered("blend/masked"));
        assert!(registered("crop/auto"));
        assert!(registered("crop/manual"));
        assert!(registered("deband/placebo"));
        assert!(registered("deblock/dpir"));
        assert!(registered("deblock/h264"));
        assert!(registered("deblock/qed"));
        assert!(registered("deblock/pp7"));
        assert!(registered("dehalo/alpha"));
        assert!(registered("dehalo/edge_clean"));
        assert!(registered("dehalo/fine"));
        assert!(registered("dehalo/yahr"));
        assert!(registered("dering/hq"));
        assert!(registered("deinterlace/bwdif"));
        assert!(registered("deinterlace/qtgmc"));
        assert!(registered("ivtc/match"));
        assert!(registered("ivtc/decimate"));
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
        assert!(registered("resample/matrix"));
        assert!(registered("planes/extract"));
        assert!(registered("planes/merge"));
        assert!(registered("warp/awarpsharp"));
        assert!(registered("upscale/artcnn"));
    }
}
