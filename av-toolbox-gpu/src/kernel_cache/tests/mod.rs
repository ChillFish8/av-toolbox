mod lock;
mod private;
mod prune;

use std::ops::{Deref, DerefMut};
use std::path::Path;
use std::process::{Child, Command};

/// A child process that is killed and reaped when dropped, so a failing test leaves none behind.
pub(crate) struct ChildGuard(Child);

impl Deref for ChildGuard {
    type Target = Child;

    fn deref(&self) -> &Child {
        &self.0
    }
}

impl DerefMut for ChildGuard {
    fn deref_mut(&mut self) -> &mut Child {
        &mut self.0
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Starts a copy of the test binary that runs only the ignored test `test_name`.
///
/// The copy reads `env_name` for its directory, which lets a test run code in a second process.
pub(crate) fn spawn_child_test(test_name: &str, env_name: &str, dir: &Path) -> ChildGuard {
    let binary = std::env::current_exe().expect("the test binary path");

    let child = Command::new(binary)
        .args([test_name, "--exact", "--ignored", "--nocapture"])
        .env(env_name, dir)
        .spawn()
        .expect("spawn the child test");

    ChildGuard(child)
}
