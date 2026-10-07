# /// script
# requires-python = ">=3.11"
# dependencies = ["onnx==1.23.0", "numpy==2.5.3"]
# ///
"""Converts DPIR's DRUNet deblocking models into av-toolbox's AVTBDPR1 weights format."""

import io
import struct
import sys
from pathlib import Path

import numpy
import onnx
from onnx import helper, numpy_helper

MAGIC = b"AVTBDPR1"
CONV, DOWN, UP = 0, 1, 2
WIDTHS = (64, 128, 256, 512)
BLOCK_CONVOLUTIONS = 8
MODELS = {
    "drunet_deblocking_gray.onnx": ("drunet_deblock_gray.bin", 0, 1),
    "drunet_deblocking_color.onnx": ("drunet_deblock_color.bin", 1, 3),
}


class ConvertError(Exception):
    pass


def layer_plan(image_channels: int, widths=WIDTHS) -> list[tuple[int, int, int]]:
    plan = [(CONV, image_channels + 1, widths[0])]

    for width, next_width in zip(widths, widths[1:]):
        plan.extend([(CONV, width, width)] * BLOCK_CONVOLUTIONS)
        plan.append((DOWN, width, next_width))

    plan.extend([(CONV, widths[-1], widths[-1])] * BLOCK_CONVOLUTIONS)

    for width, next_width in zip(reversed(widths), reversed(widths[:-1])):
        plan.append((UP, width, next_width))
        plan.extend([(CONV, next_width, next_width)] * BLOCK_CONVOLUTIONS)

    plan.append((CONV, widths[0], image_channels))
    return plan


def attribute(node, name: str, default):
    for entry in node.attribute:
        if entry.name == name:
            return helper.get_attribute_value(entry)

    return default


def check_node(node, kind: int, source_name: str) -> None:
    where = f"{source_name}: {node.name}"

    if len(node.input) != 2:
        raise ConvertError(f"{where} has a bias")

    if (node.op_type == "ConvTranspose") != (kind == UP):
        raise ConvertError(f"{where} is a {node.op_type}, which does not match its place in the network")

    group = attribute(node, "group", 1)
    dilations = list(attribute(node, "dilations", [1, 1]))
    strides = list(attribute(node, "strides", [1, 1]))
    pads = list(attribute(node, "pads", [0, 0, 0, 0]))
    output_padding = list(attribute(node, "output_padding", [0, 0]))

    if group != 1 or dilations != [1, 1]:
        raise ConvertError(f"{where} is grouped or dilated")

    if kind == CONV:
        if strides != [1, 1] or pads != [1, 1, 1, 1]:
            raise ConvertError(f"{where} is not a stride 1 convolution padded by one")
    else:
        if strides != [2, 2] or pads != [0, 0, 0, 0]:
            raise ConvertError(f"{where} is not a stride 2, unpadded 2x2 layer")

    if output_padding != [0, 0]:
        raise ConvertError(f"{where} has output padding")


def expected_weight_shape(kind: int, in_channels: int, out_channels: int) -> tuple[int, ...]:
    if kind == CONV:
        return (out_channels, in_channels, 3, 3)

    if kind == DOWN:
        return (out_channels, in_channels, 2, 2)

    return (in_channels, out_channels, 2, 2)


def layer_values(kind: int, weight: numpy.ndarray) -> numpy.ndarray:
    if kind == CONV:
        return weight.transpose(0, 2, 3, 1)

    if kind == DOWN:
        out_channels, in_channels = weight.shape[:2]
        return weight.transpose(2, 3, 1, 0).reshape(4 * in_channels, out_channels)

    in_channels, out_channels = weight.shape[:2]
    return weight.transpose(0, 2, 3, 1).reshape(in_channels, 4 * out_channels)


def convert_model(source: Path, target: Path, variant: int, image_channels: int, widths=WIDTHS) -> None:
    model = onnx.load(source)
    initializers = {entry.name: numpy_helper.to_array(entry) for entry in model.graph.initializer}
    kinds = ("Conv", "ConvTranspose")
    nodes = [node for node in model.graph.node if node.op_type in kinds]
    plan = layer_plan(image_channels, widths)

    if len(nodes) != len(plan):
        raise ConvertError(f"{source.name}: has {len(nodes)} convolutions, expected {len(plan)}")

    written = io.BytesIO()
    written.write(MAGIC)
    written.write(struct.pack("<II", variant, len(plan)))

    for node, (kind, in_channels, out_channels) in zip(nodes, plan):
        check_node(node, kind, source.name)

        weight = initializers[node.input[1]]
        wanted = expected_weight_shape(kind, in_channels, out_channels)

        if weight.shape != wanted:
            raise ConvertError(f"{source.name}: {node.name} has weight {weight.shape}, expected {wanted}")

        values = layer_values(kind, weight)
        converted = numpy.ascontiguousarray(values).astype("<f4")

        if not numpy.isfinite(converted).all():
            raise ConvertError(f"{source.name}: {node.name} has values that are not finite")

        header = struct.pack("<3I", kind, in_channels, out_channels)
        written.write(header)
        written.write(converted.tobytes())

    content = written.getvalue()
    target.write_bytes(content)


def main(arguments: list[str]) -> int:
    sources = Path(arguments[0])
    output = Path(arguments[1])

    for source_name, (target_name, variant, image_channels) in MODELS.items():
        source = sources / source_name
        target = output / target_name
        convert_model(source, target, variant, image_channels)

    return 0


if __name__ == "__main__":
    try:
        status = main(sys.argv[1:])
    except ConvertError as error:
        print(f"error: {error}", file=sys.stderr)
        status = 1

    sys.exit(status)
