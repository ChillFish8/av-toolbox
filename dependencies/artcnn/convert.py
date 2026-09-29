# /// script
# requires-python = ">=3.11"
# dependencies = ["onnx==1.23.0", "numpy==2.5.3"]
# ///
"""Converts ArtCNN's ONNX models into av-toolbox's AVTBCNN1 weights format."""

import io
import struct
import sys
from pathlib import Path

import numpy
import onnx
from onnx import helper, numpy_helper

MAGIC = b"AVTBCNN1"
MODELS = {
    "ArtCNN_R8F64.onnx": ("ArtCNN_R8F64.bin", 64, 8),
    "ArtCNN_R16F96.onnx": ("ArtCNN_R16F96.bin", 96, 16),
}


class ConvertError(Exception):
    pass


def expected_shapes(features: int, blocks: int) -> list[tuple[int, int]]:
    body = [(features, features)] * (3 * blocks + 1)
    return [(features, 1), *body, (4, features)]


def attribute(node, name: str, default):
    for entry in node.attribute:
        if entry.name == name:
            return helper.get_attribute_value(entry)

    return default


def check_convolution(node, model_name: str) -> None:
    raw_strides = attribute(node, "strides", [1, 1])
    raw_dilations = attribute(node, "dilations", [1, 1])
    raw_pads = attribute(node, "pads", [0, 0, 0, 0])
    strides = list(raw_strides)
    dilations = list(raw_dilations)
    pads = list(raw_pads)
    group = attribute(node, "group", 1)
    auto_pad = attribute(node, "auto_pad", b"NOTSET")

    if strides != [1, 1] or dilations != [1, 1] or group != 1:
        raise ConvertError(f"{model_name}: {node.name} is not a stride 1 convolution")

    same = auto_pad in (b"SAME_UPPER", b"SAME_LOWER")
    explicit = auto_pad == b"NOTSET" and pads == [1, 1, 1, 1]
    if not (same or explicit):
        raise ConvertError(f"{model_name}: {node.name} does not pad by one on every side")


def to_f16(values, model_name: str, node) -> numpy.ndarray:
    converted = values.astype("<f2")
    finite = numpy.isfinite(converted)

    if not finite.all():
        raise ConvertError(f"{model_name}: {node.name} has values f16 cannot hold")

    return converted


def convert_model(source: Path, target: Path, features: int, blocks: int) -> None:
    model = onnx.load(source)
    initializers = {entry.name: numpy_helper.to_array(entry) for entry in model.graph.initializer}
    convolutions = [node for node in model.graph.node if node.op_type == "Conv"]
    expected = expected_shapes(features, blocks)

    if len(convolutions) != len(expected):
        raise ConvertError(f"{source.name}: has {len(convolutions)} convolutions, expected {len(expected)}")

    written = io.BytesIO()
    written.write(MAGIC)
    count = struct.pack("<I", len(convolutions))
    written.write(count)

    for node, (out_channels, in_channels) in zip(convolutions, expected):
        check_convolution(node, source.name)

        weight = initializers[node.input[1]]
        bias = initializers[node.input[2]]
        expected_weight_shape = (out_channels, in_channels, 3, 3)
        expected_bias_shape = (out_channels,)

        if weight.shape != expected_weight_shape or bias.shape != expected_bias_shape:
            found = f"weight {weight.shape} and bias {bias.shape}"
            wanted = f"weight {expected_weight_shape} and bias {expected_bias_shape}"
            raise ConvertError(f"{source.name}: {node.name} has shapes {found}, expected {wanted}")

        ohwi = weight.transpose(0, 2, 3, 1)
        contiguous = numpy.ascontiguousarray(ohwi)
        weight_f16 = to_f16(contiguous, source.name, node)
        bias_f16 = to_f16(bias, source.name, node)
        header = struct.pack("<4I", out_channels, in_channels, 3, 3)
        weight_bytes = weight_f16.tobytes()
        bias_bytes = bias_f16.tobytes()

        written.write(header)
        written.write(weight_bytes)
        written.write(bias_bytes)

    content = written.getvalue()
    target.write_bytes(content)


def main(arguments: list[str]) -> int:
    sources = Path(arguments[0])
    output = Path(arguments[1])

    for source_name, (target_name, features, blocks) in MODELS.items():
        source = sources / source_name
        target = output / target_name
        convert_model(source, target, features, blocks)

    return 0


if __name__ == "__main__":
    try:
        status = main(sys.argv[1:])
    except ConvertError as error:
        print(f"error: {error}", file=sys.stderr)
        status = 1

    sys.exit(status)
