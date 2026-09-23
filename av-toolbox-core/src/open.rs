use std::any::Any;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::StageError;

/// What a source learned by opening its input, shared by every clone of the slot.
#[derive(Clone, Default)]
pub struct OpenSlot {
    state: Arc<Mutex<OpenState>>,
}

#[derive(Default)]
enum OpenState {
    #[default]
    Unopened,
    Opened(Box<dyn Any + Send>),
    Taken,
}

impl OpenSlot {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reads the opened value, opening it first when nothing has.
    pub(crate) fn read<T, R>(
        &self,
        open: impl FnOnce() -> Result<T, StageError>,
        read: impl FnOnce(&T) -> R,
    ) -> Result<R, StageError>
    where
        T: Send + 'static,
    {
        let mut state = self.lock();

        if let OpenState::Unopened = *state {
            let opened = open()?;
            *state = OpenState::Opened(Box::new(opened));
        }

        match &*state {
            OpenState::Opened(opened) => {
                let opened = opened
                    .downcast_ref::<T>()
                    .expect("a slot only ever holds its own source's opened value");

                Ok(read(opened))
            },
            OpenState::Taken => Err(already_built()),
            OpenState::Unopened => unreachable!("an unopened slot was opened above"),
        }
    }

    /// Takes the opened value out, opening it first when nothing has.
    pub(crate) fn take<T>(&self, open: impl FnOnce() -> Result<T, StageError>) -> Result<T, StageError>
    where
        T: Send + 'static,
    {
        let mut state = self.lock();
        let previous = std::mem::replace(&mut *state, OpenState::Taken);

        match previous {
            OpenState::Unopened => open().inspect_err(|_| *state = OpenState::Unopened),
            OpenState::Opened(opened) => {
                let opened = opened
                    .downcast::<T>()
                    .expect("a slot only ever holds its own source's opened value");

                Ok(*opened)
            },
            OpenState::Taken => Err(already_built()),
        }
    }

    fn lock(&self) -> MutexGuard<'_, OpenState> {
        self.state.lock().unwrap_or_else(|error| error.into_inner())
    }
}

fn already_built() -> StageError {
    StageError::msg("this source was already built, and its input can only be read once")
}

impl fmt::Debug for OpenSlot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = match &*self.lock() {
            OpenState::Unopened => "unopened",
            OpenState::Opened(_) => "opened",
            OpenState::Taken => "taken",
        };

        formatter.debug_tuple("OpenSlot").field(&state).finish()
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    #[test]
    fn reading_twice_opens_once() {
        let slot = OpenSlot::new();
        let opens = Cell::new(0);
        let open = || {
            opens.set(opens.get() + 1);
            Ok(7u32)
        };

        let first = slot.read(open, |value: &u32| *value).expect("opens");
        let second = slot.read(open, |value: &u32| *value).expect("reads");

        assert_eq!((first, second), (7, 7));
        assert_eq!(opens.get(), 1);
    }

    #[test]
    fn a_clone_shares_what_was_opened() {
        let slot = OpenSlot::new();
        let clone = slot.clone();

        slot.read(|| Ok(3u32), |_: &u32| ()).expect("opens");
        let taken: u32 = clone.take(|| Ok(99)).expect("takes");

        assert_eq!(taken, 3);
    }

    #[test]
    fn taking_an_unopened_slot_opens_it_first() {
        let slot = OpenSlot::new();

        let taken: u32 = slot.take(|| Ok(5)).expect("opens and takes");

        assert_eq!(taken, 5);
    }

    #[test]
    fn a_slot_can_be_taken_once() {
        let slot = OpenSlot::new();
        let _: u32 = slot.take(|| Ok(1)).expect("takes");

        let second = slot.take(|| Ok(2u32));

        assert!(second.is_err());
    }

    #[test]
    fn a_failed_open_leaves_the_slot_unopened() {
        let slot = OpenSlot::new();

        let failed = slot.read(|| Err::<u32, _>(StageError::msg("no input")), |_| ());
        let retried: u32 = slot.take(|| Ok(4)).expect("opens on retry");

        assert!(failed.is_err());
        assert_eq!(retried, 4);
    }
}
