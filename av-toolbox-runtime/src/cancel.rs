use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

/// Something cancellation has to wake by hand.
type Waker = Box<dyn Fn() + Send + Sync>;

/// The stop signal every thread of one run shares.
///
/// Stopping sets a flag, which a loop checks between steps, and then wakes everything registered
/// through [Cancel::on_stop]. A thread already parked cannot see a flag, so whatever it is parked
/// on has to be woken for it. Stopping twice does nothing the second time.
#[derive(Default)]
pub struct Cancel {
    stopped: AtomicBool,
    wakers: Mutex<Vec<Waker>>,
}

impl Cancel {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_stopped(&self) -> bool {
        self.stopped.load(Ordering::SeqCst)
    }

    /// Stops the run, then wakes everything registered against it.
    ///
    /// The flag is set under the same lock [Cancel::on_stop] registers under, so a waker
    /// registered at the same moment is either run by this call or run immediately by its own.
    pub fn stop(&self) {
        let wakers = {
            let mut registered = self.wakers.lock().expect("not poisoned");
            self.stopped.store(true, Ordering::SeqCst);
            std::mem::take(&mut *registered)
        };

        for wake in &wakers {
            wake();
        }
    }

    /// Registers something to wake when the run stops, or wakes it now if it already has.
    ///
    /// A condition variable, a released budget or a channel send belongs here. Whatever it does
    /// runs on the thread that calls [Cancel::stop].
    pub fn on_stop<W>(&self, waker: W)
    where
        W: Fn() + Send + Sync + 'static,
    {
        let mut registered = self.wakers.lock().expect("not poisoned");

        if self.stopped.load(Ordering::SeqCst) {
            drop(registered);
            waker();
            return;
        }

        registered.push(Box::new(waker));
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicU32;
    use std::sync::{Arc, mpsc};
    use std::thread;
    use std::time::Duration;

    use super::*;

    #[test]
    fn a_fresh_cancel_is_not_stopped() {
        assert!(!Cancel::new().is_stopped());
    }

    #[test]
    fn stopping_wakes_everything_registered() {
        let cancel = Cancel::new();
        let woken = Arc::new(AtomicU32::new(0));

        let counter = woken.clone();
        cancel.on_stop(move || {
            counter.fetch_add(1, Ordering::SeqCst);
        });
        cancel.stop();

        assert!(cancel.is_stopped());
        assert_eq!(woken.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn registering_after_a_stop_wakes_immediately() {
        let cancel = Cancel::new();
        let woken = Arc::new(AtomicU32::new(0));
        cancel.stop();

        let counter = woken.clone();
        cancel.on_stop(move || {
            counter.fetch_add(1, Ordering::SeqCst);
        });

        assert_eq!(woken.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn stopping_twice_wakes_everything_once() {
        let cancel = Cancel::new();
        let woken = Arc::new(AtomicU32::new(0));

        let counter = woken.clone();
        cancel.on_stop(move || {
            counter.fetch_add(1, Ordering::SeqCst);
        });
        cancel.stop();
        cancel.stop();

        assert_eq!(woken.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_waker_can_hand_a_stop_to_another_thread() {
        let cancel = Arc::new(Cancel::new());
        let (sender, receiver) = mpsc::channel();

        cancel.on_stop(move || {
            sender.send(()).ok();
        });

        let stopper = cancel.clone();
        thread::spawn(move || stopper.stop());

        receiver
            .recv_timeout(Duration::from_secs(10))
            .expect("the waker never ran on the thread that stopped the run");
    }
}
