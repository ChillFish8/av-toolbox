# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["onnx==1.23.0", "onnxruntime==1.23.2", "numpy==2.5.3"]
# ///
"""Runs DPIR's DRUNet deblocking models through onnxruntime on fixed test planes."""

import sys
from pathlib import Path

import numpy
import onnx
import onnxruntime

TAIL_NAME = "/m_tail/Conv"
TAIL_INDEX = 123
STRENGTHS = (5, 10, 50)


def test_plane(width: int, height: int, seed: int, offset: int) -> numpy.ndarray:
    """A plane of 8x8 blocks of flat values with a little noise, the shape of compression blocking."""
    rows = numpy.arange(height).reshape(height, 1)
    columns = numpy.arange(width).reshape(1, width)
    blocks = ((columns // 8) * 37 + (rows // 8) * 91 + offset) % 64
    generator = numpy.random.default_rng(seed)
    noise = generator.normal(0.0, 0.02, (height, width))
    plane = blocks / 63.0 * 0.8 + 0.1 + noise
    return numpy.clip(plane, 0.0, 1.0).astype(numpy.float32)


def padding(size: int) -> tuple[int, int]:
    extra = (8 - size % 8) % 8
    before = extra // 2
    return before, extra - before


def run(session, planes: numpy.ndarray, strength: float) -> numpy.ndarray:
    """Runs `planes` ([channels, height, width]) padded by the reference's rule, cropped back and clamped."""
    _, height, width = planes.shape
    top, bottom = padding(height)
    left, right = padding(width)
    padded = numpy.pad(planes, ((0, 0), (top, bottom), (left, right)), mode="symmetric")
    strength_plane = numpy.full((1, *padded.shape[1:]), strength / 255.0, dtype=numpy.float32)
    batch = numpy.concatenate([padded, strength_plane])[numpy.newaxis]
    result = session.run(None, {"input": batch})[0][0]
    cropped = result[:, top:top + height, left:left + width]
    return numpy.clip(cropped, 0.0, 1.0).astype("<f4")


def check_tail(model: onnx.ModelProto, name: str) -> bool:
    nodes = model.graph.node
    index = next((position for position, node in enumerate(nodes) if node.name == TAIL_NAME), None)
    print(f"{name}: {TAIL_NAME} is node {index} of {len(nodes)}")
    return index == TAIL_INDEX


def write(output: Path, name: str, planes: numpy.ndarray) -> None:
    target = output / name
    target.write_bytes(planes.astype("<f4").tobytes())
    print(f"wrote {target}")


def write_runs(output: Path, session, prefix: str, planes: numpy.ndarray) -> None:
    write(output, f"{prefix}.input.f32", planes)

    for strength in STRENGTHS:
        result = run(session, planes, strength)
        write(output, f"{prefix}_s{strength}.f32", result)


def main(arguments: list[str]) -> int:
    cache = Path(arguments[0])
    output = Path(arguments[1])
    gray_path = cache / "drunet_deblocking_gray.onnx"
    colour_path = cache / "drunet_deblocking_color.onnx"
    gray_model = onnx.load(gray_path)
    colour_model = onnx.load(colour_path)
    gray_ok = check_tail(gray_model, gray_path.name)
    colour_ok = check_tail(colour_model, colour_path.name)

    if not (gray_ok and colour_ok):
        return 1

    providers = ["CPUExecutionProvider"]
    gray_session = onnxruntime.InferenceSession(gray_path.read_bytes(), providers=providers)
    colour_session = onnxruntime.InferenceSession(colour_path.read_bytes(), providers=providers)
    output.mkdir(parents=True, exist_ok=True)

    gray = test_plane(48, 40, 1, 0)[numpy.newaxis]
    write_runs(output, gray_session, "gray_48x40", gray)

    red = test_plane(40, 48, 2, 0)
    green = test_plane(40, 48, 3, 21)
    blue = test_plane(40, 48, 4, 42)
    colour = numpy.stack([red, green, blue])
    write_runs(output, colour_session, "colour_40x48", colour)

    odd = test_plane(50, 38, 5, 7)[numpy.newaxis]
    write(output, "gray_50x38.input.f32", odd)
    write(output, "gray_50x38_s10.f32", run(gray_session, odd, 10))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
