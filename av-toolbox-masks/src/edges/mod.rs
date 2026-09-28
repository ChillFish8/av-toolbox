mod operator;
mod retinex;

pub(crate) use self::operator::{Edges, Kirsch, Prewitt, Sobel};
pub(crate) use self::retinex::Retinex;
