use crate::DemandDefinition;

/// A demand kind for test fixtures that need a chunk request but no particular breaks.
pub struct FixtureCuts;

impl DemandDefinition for FixtureCuts {
    type Request = ();

    fn kind() -> &'static str {
        "fixture-cuts"
    }
}
