mod lock;
mod prune;

use std::path::Path;
use std::process::{Child, Command};

/// Starts a copy of the test binary that runs only the ignored test `test_name`.
///
/// The copy reads `env_name` for its directory, which lets a test run code in a second process.
pub(crate) fn spawn_child_test(test_name: &str, env_name: &str, dir: &Path) -> Child {
    let binary = std::env::current_exe().expect("the test binary path");

    Command::new(binary)
        .args([test_name, "--exact", "--ignored", "--nocapture"])
        .env(env_name, dir)
        .spawn()
        .expect("spawn the child test")
}
