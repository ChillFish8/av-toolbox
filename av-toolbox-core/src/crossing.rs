use crate::Residency;

/// Moving a stream between host and device.
///
/// A pipeline never declares one. The graph inserts one wherever a stream's producer and a
/// consumer run on different sides, once per producer and direction, so every transfer the run
/// pays for is a node that can be seen.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum Crossing {
    Upload,
    Unload,
}

impl Crossing {
    /// The side the stream is on before it crosses.
    pub const fn from_side(self) -> Residency {
        match self {
            Self::Upload => Residency::Host,
            Self::Unload => Residency::Device,
        }
    }

    /// The side the stream is on after it crosses.
    pub const fn to_side(self) -> Residency {
        match self {
            Self::Upload => Residency::Device,
            Self::Unload => Residency::Host,
        }
    }

    /// The identifier this crossing is named by wherever a built-in stage needs one, such as a
    /// diagnostic or a rendered graph.
    pub const fn identifier(self) -> &'static str {
        match self {
            Self::Upload => "builtin/upload",
            Self::Unload => "builtin/unload",
        }
    }

    /// The verb this crossing is described by in a message meant for a person.
    pub const fn verb(self) -> &'static str {
        match self {
            Self::Upload => "upload",
            Self::Unload => "unload",
        }
    }

    /// The crossing that takes a stream from `from` to `to`, or `None` when no crossing is needed.
    pub fn between(from: Residency, to: Residency) -> Option<Self> {
        match (from, to) {
            (Residency::Host, Residency::Device) => Some(Self::Upload),
            (Residency::Device, Residency::Host) => Some(Self::Unload),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_crossing_is_found_between_two_different_sides_only() {
        assert_eq!(
            Crossing::between(Residency::Host, Residency::Device),
            Some(Crossing::Upload)
        );
        assert_eq!(
            Crossing::between(Residency::Device, Residency::Host),
            Some(Crossing::Unload)
        );
        assert_eq!(Crossing::between(Residency::Device, Residency::Device), None);
    }

    #[test]
    fn a_crossing_names_itself_as_a_builtin() {
        assert_eq!(Crossing::Upload.identifier(), "builtin/upload");
        assert_eq!(Crossing::Unload.identifier(), "builtin/unload");
        assert_eq!(Crossing::Unload.to_side(), Residency::Host);
    }
}
