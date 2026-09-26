mod args;
mod colour;
mod convert;
mod image;
mod like;
mod picture;
mod repeat;
mod solid;

use self::convert::Converted;
pub(crate) use self::image::{File, ImageLike};
pub(crate) use self::solid::{Fill, SolidLike};

/// What opening a constant source yields, its frame and how many times to hand it out.
pub(crate) struct Prepared {
    pub(crate) converted: Converted,
    pub(crate) frames: u64,
}
