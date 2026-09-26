mod args;
mod colour;
mod convert;
mod image;
mod picture;
mod repeat;
mod solid;

use self::convert::Converted;
pub(crate) use self::image::File;
pub(crate) use self::solid::Fill;

/// What opening a constant source yields, its frame and how many times to hand it out.
pub(crate) struct Prepared {
    pub(crate) converted: Converted,
    pub(crate) frames: u64,
}
