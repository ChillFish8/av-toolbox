use av_toolbox_gpu::{
    ChromaLocation,
    ColourRange,
    FieldOrder,
    FrameFormat,
    FrameLayout,
    FrameRate,
    Matrix,
    Primaries,
    SampleFormat,
    StreamFormat,
    Subsampling,
    Transfer,
};

use crate::template::Literal;

/// Every field a frames stream has.
const FRAME_FIELDS: &[&str] = &[
    "width",
    "height",
    "bit_depth",
    "is_float",
    "layout",
    "frame_rate",
    "frame_rate_num",
    "frame_rate_den",
    "field_order",
    "range",
    "matrix",
    "primaries",
    "transfer",
    "chroma_location",
];

/// Every field a mask stream has.
const MASK_FIELDS: &[&str] = &[
    "width",
    "height",
    "bit_depth",
    "is_float",
    "frame_rate",
    "frame_rate_num",
    "frame_rate_den",
];

/// Reads one named field of a stream's format.
///
/// The error is a phrase meant to follow the stage's name, such as `has no field widht`.
pub fn field_of(format: &StreamFormat, field: &str) -> Result<Literal, String> {
    let (width, height) = format.dimensions();
    let sample = format.sample();
    let frame_rate = match format {
        StreamFormat::Frames(frames) => frames.frame_rate,
        StreamFormat::Mask(mask) => mask.frame_rate,
    };

    match field {
        "width" => {
            let width = i64::from(width);
            Ok(Literal::Integer(width))
        },
        "height" => {
            let height = i64::from(height);
            Ok(Literal::Integer(height))
        },
        "bit_depth" => {
            let bits = bit_depth(sample);
            Ok(Literal::Integer(bits))
        },
        "is_float" => {
            let is_float = sample == SampleFormat::F32;
            Ok(Literal::Bool(is_float))
        },
        "frame_rate" => {
            let rate = known_rate(frame_rate)?;
            let numerator = f64::from(rate.numerator());
            let denominator = f64::from(rate.denominator());

            Ok(Literal::Float(numerator / denominator))
        },
        "frame_rate_num" => {
            let rate = known_rate(frame_rate)?;
            let numerator = i64::from(rate.numerator());

            Ok(Literal::Integer(numerator))
        },
        "frame_rate_den" => {
            let rate = known_rate(frame_rate)?;
            let denominator = i64::from(rate.denominator());

            Ok(Literal::Integer(denominator))
        },
        _ => frame_field(format, field),
    }
}

/// Reads a field only a frames stream has.
fn frame_field(format: &StreamFormat, field: &str) -> Result<Literal, String> {
    let StreamFormat::Frames(frames) = format else {
        if FRAME_FIELDS.contains(&field) {
            let message = format!("is a mask, which has no {field}");
            return Err(message);
        }

        let message = unknown_field(field, MASK_FIELDS);
        return Err(message);
    };

    let name = match field {
        "layout" => layout_name(frames.layout),
        "field_order" => field_order_name(frames)?,
        "range" => range_name(frames.colour.range),
        "matrix" => matrix_name(frames.colour.matrix),
        "primaries" => primaries_name(frames.colour.primaries),
        "transfer" => transfer_name(frames.colour.transfer),
        "chroma_location" => chroma_location_name(frames.colour.chroma_location),
        _ => {
            let message = unknown_field(field, FRAME_FIELDS);
            return Err(message);
        },
    };

    let name = String::from(name);
    Ok(Literal::String(name))
}

fn unknown_field(field: &str, fields: &[&str]) -> String {
    let listed = fields.join(", ");

    format!("has no field {field}, it has {listed}")
}

fn known_rate(frame_rate: Option<FrameRate>) -> Result<FrameRate, String> {
    frame_rate.ok_or_else(unknown_rate)
}

fn unknown_rate() -> String {
    String::from("has an unknown frame rate")
}

fn bit_depth(sample: SampleFormat) -> i64 {
    match sample {
        SampleFormat::U8 => 8,
        SampleFormat::U16 { depth } => i64::from(depth.bits()),
        SampleFormat::F32 => 32,
    }
}

fn layout_name(layout: FrameLayout) -> &'static str {
    match layout {
        FrameLayout::Yuv { subsampling } => match subsampling {
            Subsampling::Yuv420 => "yuv420",
            Subsampling::Yuv422 => "yuv422",
            Subsampling::Yuv444 => "yuv444",
        },
        FrameLayout::Chroma { subsampling } => match subsampling {
            Subsampling::Yuv420 => "chroma420",
            Subsampling::Yuv422 => "chroma422",
            Subsampling::Yuv444 => "chroma444",
        },
        FrameLayout::Rgb => "rgb",
        FrameLayout::Grey => "grey",
    }
}

fn field_order_name(frames: &FrameFormat) -> Result<&'static str, String> {
    match frames.field_order {
        Some(FieldOrder::Progressive) => Ok("progressive"),
        Some(FieldOrder::TopFirst) => Ok("top_first"),
        Some(FieldOrder::BottomFirst) => Ok("bottom_first"),
        None => Err(String::from("has an unknown field order")),
    }
}

fn range_name(range: ColourRange) -> &'static str {
    match range {
        ColourRange::Unspecified => "unspecified",
        ColourRange::Limited => "limited",
        ColourRange::Full => "full",
    }
}

fn matrix_name(matrix: Matrix) -> &'static str {
    match matrix {
        Matrix::Unspecified => "unspecified",
        Matrix::Bt709 => "bt709",
        Matrix::Bt601 => "bt601",
        Matrix::Bt2020Ncl => "bt2020_ncl",
        Matrix::Identity => "identity",
    }
}

fn primaries_name(primaries: Primaries) -> &'static str {
    match primaries {
        Primaries::Unspecified => "unspecified",
        Primaries::Bt709 => "bt709",
        Primaries::Bt601 => "bt601",
        Primaries::Bt2020 => "bt2020",
    }
}

fn transfer_name(transfer: Transfer) -> &'static str {
    match transfer {
        Transfer::Unspecified => "unspecified",
        Transfer::Bt709 => "bt709",
        Transfer::Srgb => "srgb",
        Transfer::Pq => "pq",
        Transfer::Hlg => "hlg",
    }
}

fn chroma_location_name(location: ChromaLocation) -> &'static str {
    match location {
        ChromaLocation::Left => "left",
        ChromaLocation::Center => "center",
        ChromaLocation::TopLeft => "top_left",
        ChromaLocation::Top => "top",
        ChromaLocation::BottomLeft => "bottom_left",
        ChromaLocation::Bottom => "bottom",
    }
}
