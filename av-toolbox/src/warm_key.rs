use std::fmt::Write;

use av_toolbox_core::StageKind;
use av_toolbox_gpu::Backend;
use av_toolbox_pipeline::CompiledPipeline;

/// The FNV-1a 128 offset basis, which seeds every key.
const FNV_OFFSET: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;

const FNV_PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;

/// The key [WarmUp](av_toolbox_gpu::WarmUp) claims for the kernels `compiled` makes on `backend`.
///
/// It hashes each stage's type, args and solved formats, and the backend's accelerator and device.
/// A source adds only its solved output format, and a sink only its type and solved formats, so runs
/// of one pipeline over different input and output files share a key.
pub(crate) fn pipeline_key(compiled: &CompiledPipeline<'_, '_>, backend: &Backend) -> u128 {
    let rendered = render(compiled, backend);
    fnv1a(rendered.as_bytes())
}

/// The `Debug` form of everything the key covers, in node order.
fn render(compiled: &CompiledPipeline<'_, '_>, backend: &Backend) -> String {
    let graph = compiled.graph();
    let formats = compiled.formats();
    let mut rendered = String::new();

    for id in graph.order() {
        let node = graph.node(*id);

        if node.kind() == StageKind::Source {
            let output = formats.output(*id);
            let _ = writeln!(rendered, "{output:?}");
            continue;
        }

        let identifier = node.identifier();
        let resolved = formats.get(*id);

        if node.kind() == StageKind::Sink {
            let _ = writeln!(rendered, "{identifier:?} {resolved:?}");
            continue;
        }

        let args = node.args();
        let _ = writeln!(rendered, "{identifier:?} {args:?} {resolved:?}");
    }

    let accelerator = backend.accelerator();
    let device = backend.device();
    let _ = write!(rendered, "{accelerator:?} {device:?}");

    rendered
}

fn fnv1a(bytes: &[u8]) -> u128 {
    let mut hash = FNV_OFFSET;
    for byte in bytes {
        hash ^= u128::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }

    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv1a_matches_the_published_vectors() {
        assert_eq!(fnv1a(b""), FNV_OFFSET);
        assert_eq!(fnv1a(b"a"), 0xd228_cb69_6f1a_8caf_7891_2b70_4e4a_8964);
    }
}
