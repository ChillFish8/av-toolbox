use std::process::ExitCode;

use av_toolbox_gpu::{Accelerator, BackendDevices, Device, enumerate_devices};

/// Prints the devices every backend in `accelerators` can see.
pub(crate) fn run(accelerators: &[Accelerator]) -> ExitCode {
    let reported = enumerate_devices(accelerators);
    let drawn = format_table(&reported);
    print!("{drawn}");

    ExitCode::SUCCESS
}

/// Renders what the backends reported as a table.
///
/// The table opens with a blank line. Graphics drivers write their own notices to stderr while
/// the backends start, and the header is hard to pick out of that without a gap in front of it.
fn format_table(reported: &[BackendDevices]) -> String {
    let rows = build_rows(reported);
    let unavailable: Vec<&BackendDevices> = reported.iter().filter(|backend| !backend.available).collect();

    let mut drawn = String::from("\n");

    if rows.is_empty() {
        drawn.push_str("No usable devices found.\n");
    } else {
        let width = rows
            .iter()
            .map(|(device, _)| device.len())
            .chain(std::iter::once("DEVICE".len()))
            .max()
            .unwrap_or(0);

        let header = format!("{:<width$}  BACKENDS\n", "DEVICE");
        drawn.push_str(&header);

        for (device, backends) in &rows {
            let names = backends.join(", ");
            let row = format!("{device:<width$}  {names}\n");

            drawn.push_str(&row);
        }
    }

    if !unavailable.is_empty() {
        let names: Vec<String> = unavailable
            .iter()
            .map(|backend| backend.accelerator.to_string())
            .collect();
        let joined = names.join(", ");
        let footer = format!("\nEnabled but unavailable on this machine: {joined}\n");

        drawn.push_str(&footer);
    }

    drawn
}

/// Turns per-backend device lists into per-device backend lists.
///
/// Rows are sorted by device kind then ordinal. A backend that started always gets a `default`
/// row, and two backends sharing a row are not necessarily the same physical card, since ordinals
/// are counted per backend.
fn build_rows(reported: &[BackendDevices]) -> Vec<(String, Vec<String>)> {
    let mut rows: Vec<(Device, Vec<String>)> = Vec::new();

    let mut record = |device: Device, accelerator: Accelerator| {
        let name = accelerator.to_string();
        let existing = rows.iter_mut().find(|(seen, _)| *seen == device);

        match existing {
            Some((_, backends)) if backends.contains(&name) => {},
            Some((_, backends)) => backends.push(name),
            None => rows.push((device, vec![name])),
        }
    };

    for backend in reported.iter().filter(|backend| backend.available) {
        record(Device::Default, backend.accelerator);

        for device in &backend.devices {
            record(*device, backend.accelerator);
        }
    }

    rows.sort_by_key(|(device, _)| sort_key(device));
    rows.into_iter()
        .map(|(device, backends)| (device.to_string(), backends))
        .collect()
}

/// Orders devices by kind first and ordinal second.
///
/// `default` leads because it is what a run without `--device` uses.
fn sort_key(device: &Device) -> (u8, usize) {
    match device {
        Device::Default => (0, 0),
        Device::Discrete { index } => (1, *index),
        Device::Integrated { index } => (2, *index),
        Device::Virtual { index } => (3, *index),
        Device::Cpu => (4, 0),
    }
}

#[cfg(test)]
mod tests {
    use av_toolbox_gpu::default_accelerators;

    use super::*;

    fn any_accelerator() -> Accelerator {
        default_accelerators()[0]
    }

    fn backend(accelerator: Accelerator, devices: Vec<Device>) -> BackendDevices {
        BackendDevices {
            accelerator,
            available: true,
            devices,
        }
    }

    /// Which `Accelerator` variants exist depends on the enabled features, so the same one twice
    /// stands in for two backends that see the same device. Merging keys on the device, and a
    /// backend is named once per row however often it reports that device.
    #[test]
    fn rows_merge_backends_offering_the_same_device() {
        let reported = vec![
            backend(any_accelerator(), vec![Device::Discrete { index: 0 }]),
            backend(any_accelerator(), vec![Device::Discrete { index: 0 }]),
        ];

        let name = any_accelerator().to_string();

        let rows = build_rows(&reported);
        assert_eq!(
            rows,
            vec![
                ("default".to_string(), vec![name.clone()]),
                ("discrete:0".to_string(), vec![name]),
            ],
        );
    }

    #[test]
    fn rows_are_sorted_by_kind_then_ordinal() {
        let reported = vec![backend(
            any_accelerator(),
            vec![
                Device::Cpu,
                Device::Discrete { index: 1 },
                Device::Integrated { index: 0 },
                Device::Discrete { index: 0 },
            ],
        )];

        let names: Vec<String> = build_rows(&reported)
            .into_iter()
            .map(|(device, _)| device)
            .collect();
        assert_eq!(
            names,
            ["default", "discrete:0", "discrete:1", "integrated:0", "cpu"],
        );
    }

    #[test]
    fn a_backend_that_did_not_start_contributes_no_rows() {
        let reported = vec![BackendDevices {
            accelerator: any_accelerator(),
            available: false,
            devices: Vec::new(),
        }];

        assert!(build_rows(&reported).is_empty());
    }

    #[test]
    fn the_table_opens_with_a_blank_line() {
        let reported = vec![backend(any_accelerator(), vec![Device::Cpu])];

        assert!(
            format_table(&reported).starts_with('\n'),
            "driver notices on stderr run straight into the header",
        );
    }

    #[test]
    fn table_pads_the_device_column() {
        let reported = vec![backend(any_accelerator(), vec![Device::Integrated { index: 0 }])];

        let name = any_accelerator();
        let expected = format!("\nDEVICE        BACKENDS\ndefault       {name}\nintegrated:0  {name}\n");

        assert_eq!(format_table(&reported), expected);
    }

    #[test]
    fn table_names_backends_that_did_not_start() {
        let reported = vec![BackendDevices {
            accelerator: any_accelerator(),
            available: false,
            devices: Vec::new(),
        }];

        let name = any_accelerator();
        let expected =
            format!("\nNo usable devices found.\n\nEnabled but unavailable on this machine: {name}\n");

        assert_eq!(format_table(&reported), expected);
    }

    #[test]
    fn nothing_enabled_says_so() {
        assert_eq!(format_table(&[]), "\nNo usable devices found.\n");
    }
}
