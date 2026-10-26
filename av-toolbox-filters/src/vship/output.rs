use av_toolbox_core::StageError;
use av_toolbox_gpu::{
    Channel,
    ColourInfo,
    ColourRange,
    FrameContext,
    FrameFormat,
    FrameLayout,
    FrameMeta,
    HostFrame,
    ResolvedFormats,
    SampleFormat,
    StreamFormat,
};
use avtbx_vship::DistortionMap;
use schemars::JsonSchema;
use serde::Deserialize;

/// What a Vship metric stage outputs.
#[derive(Debug, Copy, Clone, Default, Eq, PartialEq, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Output {
    /// The distorted clip, unchanged.
    #[default]
    Clip,
    /// The per-pixel distortion map, as a single plane `F32` frame.
    Map,
}

/// An `F32` `Grey` frame of `width` by `height`, keeping the distorted clip's timing.
pub(crate) fn map_format(distorted: &FrameFormat, width: u32, height: u32) -> FrameFormat {
    let colour = ColourInfo {
        range: ColourRange::Full,
        ..ColourInfo::default()
    };

    FrameFormat {
        width,
        height,
        layout: FrameLayout::Grey,
        sample: SampleFormat::F32,
        colour,
        frame_rate: distorted.frame_rate,
        field_order: distorted.field_order,
    }
}

/// The map Vship writes into, and the frame format it is output as.
pub(crate) struct MapOutput {
    pub(crate) map: DistortionMap,
    pub(crate) format: FrameFormat,
}

/// Pairs `map` with the output format `stage` declared, refusing a map of another size.
pub(crate) fn declared_map(
    stage: &str,
    formats: &ResolvedFormats,
    map: DistortionMap,
) -> Result<MapOutput, StageError> {
    let output = formats.output();
    let Some(format) = output.as_ref().and_then(StreamFormat::as_frames).copied() else {
        return Err(missing_output_error(stage));
    };

    let width = u32::try_from(map.width());
    let height = u32::try_from(map.height());
    let fits = width == Ok(format.width) && height == Ok(format.height);
    if !fits || format.sample != SampleFormat::F32 {
        return Err(map_size_error(stage, &format, &map));
    }

    Ok(MapOutput { map, format })
}

/// A frame of `format` carrying `meta`, holding every value of `map` times `multiplier` in its `Y` plane.
pub(crate) fn map_frame(
    context: &mut FrameContext,
    format: &FrameFormat,
    meta: FrameMeta,
    map: &DistortionMap,
    multiplier: f32,
) -> Result<HostFrame, StageError> {
    let mut frame = context.alloc_host_frame(format, meta);
    let Some(plane) = frame.plane_mut(Channel::Y) else {
        return Err(missing_plane_error());
    };

    write_values(plane, map.data(), multiplier)?;

    Ok(frame.freeze())
}

/// Writes `values` times `multiplier` as little-endian `f32`s over the whole of `plane`.
fn write_values(plane: &mut [u8], values: &[f32], multiplier: f32) -> Result<(), StageError> {
    let needed = size_of_val(values);
    if plane.len() != needed {
        return Err(plane_size_error(plane.len(), needed));
    }

    let (slots, _) = plane.as_chunks_mut::<4>();
    for (slot, value) in slots.iter_mut().zip(values) {
        *slot = (value * multiplier).to_le_bytes();
    }

    Ok(())
}

fn missing_plane_error() -> StageError {
    StageError::msg("a distortion map frame has no Y plane")
}

fn missing_output_error(stage: &str) -> StageError {
    StageError::msg(format!("{stage} was not given a frame output format"))
}

fn map_size_error(stage: &str, format: &FrameFormat, map: &DistortionMap) -> StageError {
    StageError::msg(format!(
        "{stage} writes a {} by {} distortion map but its output is a {} by {} {} frame",
        map.width(),
        map.height(),
        format.width,
        format.height,
        format.sample,
    ))
}

fn plane_size_error(length: usize, needed: usize) -> StageError {
    StageError::msg(format!(
        "a distortion map needs {needed} bytes but its frame's plane holds {length}"
    ))
}

#[cfg(test)]
mod tests {
    use av_toolbox_gpu::{FieldOrder, FrameRate, Matrix, Subsampling};

    use super::*;

    fn distorted() -> FrameFormat {
        FrameFormat {
            width: 64,
            height: 32,
            layout: FrameLayout::Yuv {
                subsampling: Subsampling::Yuv420,
            },
            sample: SampleFormat::U8,
            colour: ColourInfo {
                range: ColourRange::Limited,
                matrix: Matrix::Bt709,
                ..ColourInfo::default()
            },
            frame_rate: FrameRate::new(24, 1),
            field_order: Some(FieldOrder::Progressive),
        }
    }

    #[test]
    fn the_map_format_is_full_range_f32_grey_at_the_map_size() {
        let source = distorted();

        let format = map_format(&source, 100, 50);

        assert_eq!((format.width, format.height), (100, 50));
        assert_eq!(format.layout, FrameLayout::Grey);
        assert_eq!(format.sample, SampleFormat::F32);
        assert_eq!(format.colour.range, ColourRange::Full);
        assert_eq!(format.colour.matrix, Matrix::Unspecified);
        assert_eq!(format.frame_rate, source.frame_rate);
        assert_eq!(format.field_order, source.field_order);
    }

    #[test]
    fn every_byte_of_the_plane_is_written() {
        let mut plane = vec![0xAA; 12];
        let values = [1.5, -2.0, 0.0];

        write_values(&mut plane, &values, 1.0).expect("the plane fits");

        let mut expected = Vec::new();
        for value in values {
            expected.extend_from_slice(&f32::to_le_bytes(value));
        }
        assert_eq!(plane, expected);
    }

    #[test]
    fn values_are_multiplied_as_they_are_written() {
        let mut plane = vec![0; 8];

        write_values(&mut plane, &[0.25, -0.5], 10.0).expect("the plane fits");

        let mut expected = Vec::new();
        expected.extend_from_slice(&f32::to_le_bytes(2.5));
        expected.extend_from_slice(&f32::to_le_bytes(-5.0));
        assert_eq!(plane, expected);
    }

    #[test]
    fn a_plane_of_the_wrong_size_is_refused() {
        let mut plane = vec![0; 8];

        let error = write_values(&mut plane, &[1.0, 2.0, 3.0], 1.0).expect_err("the plane is short");

        assert!(error.to_string().contains("needs 12 bytes"), "{error}");
    }

    #[test]
    fn output_parses_clip_and_map() {
        let clip: Output = serde_json::from_value(serde_json::json!("clip")).expect("parses");
        let map: Output = serde_json::from_value(serde_json::json!("map")).expect("parses");

        assert_eq!(clip, Output::Clip);
        assert_eq!(map, Output::Map);
        assert_eq!(Output::default(), Output::Clip);
    }
}
