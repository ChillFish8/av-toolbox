use av_toolbox_gpu::{Accelerator, Backend, Device, Tuning};

/// No accelerator could open the device asked for.
#[derive(Debug, thiserror::Error)]
#[error("no accelerator could open device {device}, tried {}", describe(.tried))]
pub struct OpenBackendError {
    device: Device,
    tried: Vec<Accelerator>,
}

impl OpenBackendError {
    pub fn device(&self) -> Device {
        self.device
    }

    pub fn tried(&self) -> &[Accelerator] {
        &self.tried
    }
}

/// Opens the first of `accelerators` that starts on `device`, trying them in order.
///
/// Stages built against the backend pick their kernels by `tuning`.
pub fn open_backend(
    accelerators: &[Accelerator],
    device: Device,
    tuning: Tuning,
) -> Result<Backend, OpenBackendError> {
    for accelerator in accelerators {
        let opened = Backend::open(*accelerator, device);
        if let Some(backend) = opened {
            return Ok(backend.with_tuning(tuning));
        }
    }

    Err(OpenBackendError {
        device,
        tried: accelerators.to_vec(),
    })
}

fn describe(tried: &[Accelerator]) -> String {
    if tried.is_empty() {
        return "none".to_string();
    }

    let names: Vec<String> = tried.iter().map(ToString::to_string).collect();
    names.join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_accelerators_fails_naming_the_device() {
        let device = Device::Discrete { index: 3 };

        let result = open_backend(&[], device, Tuning::Cached);
        let Err(error) = result else {
            panic!("nothing to try");
        };

        assert_eq!(error.device(), device);
        assert!(error.tried().is_empty());
        assert!(error.to_string().contains("discrete:3"));
    }
}
