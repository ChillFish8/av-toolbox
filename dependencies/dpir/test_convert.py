import importlib.util
import struct
from pathlib import Path

import numpy
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

CONVERTER_PATH = Path(__file__).parent / "convert.py"
CONVERTER_SPEC = importlib.util.spec_from_file_location("dpir_convert", CONVERTER_PATH)
convert = importlib.util.module_from_spec(CONVERTER_SPEC)
CONVERTER_SPEC.loader.exec_module(convert)

TINY_WIDTHS = (2, 4, 8, 16)


def tiny_drunet(path: Path, image_channels: int, widths=TINY_WIDTHS, mutate=None, with_bias=None, drop_last=False):
    """Writes a DRUNet graph with KAIR's node order and returns each layer's ONNX weight in forward order."""
    plan = convert.layer_plan(image_channels, widths)
    generator = numpy.random.default_rng(11)
    nodes, initializers, weights = [], [], []
    previous = "input"

    for index, (kind, in_channels, out_channels) in enumerate(plan):
        if kind == convert.CONV:
            shape = (out_channels, in_channels, 3, 3)
            attributes = {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1], "strides": [1, 1]}
            op = "Conv"
        elif kind == convert.DOWN:
            shape = (out_channels, in_channels, 2, 2)
            attributes = {"kernel_shape": [2, 2], "pads": [0, 0, 0, 0], "strides": [2, 2]}
            op = "Conv"
        else:
            shape = (in_channels, out_channels, 2, 2)
            attributes = {"kernel_shape": [2, 2], "pads": [0, 0, 0, 0], "strides": [2, 2]}
            op = "ConvTranspose"

        if mutate is not None:
            shape, attributes, op = mutate(index, shape, attributes, op)

        weight = generator.standard_normal(shape).astype(numpy.float32)
        name = f"w{index}"
        output = f"n{index}"
        inputs = [previous, name]
        initializers.append(numpy_helper.from_array(weight, name))

        if with_bias == index:
            bias = numpy.zeros(out_channels, dtype=numpy.float32)
            initializers.append(numpy_helper.from_array(bias, f"b{index}"))
            inputs.append(f"b{index}")

        nodes.append(helper.make_node(op, inputs, [output], name=f"layer{index}", **attributes))
        weights.append(weight)
        previous = output

    if drop_last:
        nodes.pop()
        previous = f"n{len(nodes) - 1}"

    # The graph's Relu and Add nodes do not change the converter's output, so the tiny graph is a plain chain.
    graph_input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, image_channels + 1, None, None])
    graph_output = helper.make_tensor_value_info(previous, TensorProto.FLOAT, None)
    graph = helper.make_graph(nodes, "tiny", [graph_input], [graph_output], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, path)

    return weights


def convert_tiny(tmp_path: Path, image_channels: int = 1, **options):
    source = tmp_path / "tiny.onnx"
    target = tmp_path / "tiny.bin"
    weights = tiny_drunet(source, image_channels, **options)
    variant = 0 if image_channels == 1 else 1
    convert.convert_model(source, target, variant, image_channels, TINY_WIDTHS)
    return weights, target.read_bytes()


def parse_layers(content: bytes):
    magic, variant, count = struct.unpack_from("<8sII", content, 0)
    offset = 16
    layers = []

    for _ in range(count):
        kind, in_channels, out_channels = struct.unpack_from("<3I", content, offset)
        offset += 12
        length = {
            convert.CONV: out_channels * 9 * in_channels,
            convert.DOWN: 4 * in_channels * out_channels,
            convert.UP: 4 * in_channels * out_channels,
        }[kind]
        values = numpy.frombuffer(content, dtype="<f4", count=length, offset=offset)
        offset += 4 * length
        layers.append((kind, in_channels, out_channels, values))

    return magic, variant, layers, offset


def test_layer_plan_has_64_layers_in_forward_order():
    gray = convert.layer_plan(1)
    colour = convert.layer_plan(3)

    assert len(gray) == 64
    assert gray[0] == (convert.CONV, 2, 64)
    assert gray[9] == (convert.DOWN, 64, 128)
    assert gray[36] == (convert.UP, 512, 256)
    assert gray[63] == (convert.CONV, 64, 1)
    assert colour[0] == (convert.CONV, 4, 64)
    assert colour[63] == (convert.CONV, 64, 3)


def test_writes_the_header_and_every_layer(tmp_path):
    _, content = convert_tiny(tmp_path, 1)
    magic, variant, layers, end = parse_layers(content)
    plan = convert.layer_plan(1, TINY_WIDTHS)

    assert magic == b"AVTBDPR1"
    assert variant == 0
    assert len(layers) == 64
    assert [layer[:3] for layer in layers] == plan
    assert end == len(content)


def test_colour_variant_is_one(tmp_path):
    _, content = convert_tiny(tmp_path, 3)
    _, variant, layers, end = parse_layers(content)

    assert variant == 1
    assert len(layers) == 64
    assert end == len(content)


def test_three_by_three_layers_are_ohwi(tmp_path):
    weights, content = convert_tiny(tmp_path, 1)
    _, _, layers, _ = parse_layers(content)

    expected = weights[0].transpose(0, 2, 3, 1).ravel()
    assert numpy.array_equal(layers[0][3], expected)


def test_down_matrix_computes_the_strided_convolution(tmp_path):
    weights, content = convert_tiny(tmp_path, 1)
    _, _, layers, _ = parse_layers(content)
    kind, in_channels, out_channels, values = layers[9]
    assert kind == convert.DOWN

    generator = numpy.random.default_rng(3)
    image = generator.standard_normal((in_channels, 4, 6)).astype(numpy.float32)
    weight = weights[9]
    direct = numpy.zeros((out_channels, 2, 3), dtype=numpy.float32)
    for y in range(2):
        for x in range(3):
            patch = image[:, 2 * y:2 * y + 2, 2 * x:2 * x + 2]
            direct[:, y, x] = numpy.einsum("oiyx,iyx->o", weight, patch)

    matrix = values.reshape(4 * in_channels, out_channels)
    gathered = numpy.zeros((2, 3, 4 * in_channels), dtype=numpy.float32)
    for y in range(2):
        for x in range(3):
            for dy in range(2):
                for dx in range(2):
                    for ci in range(in_channels):
                        gathered[y, x, (dy * 2 + dx) * in_channels + ci] = image[ci, 2 * y + dy, 2 * x + dx]

    result = (gathered @ matrix).transpose(2, 0, 1)
    assert numpy.allclose(result, direct, atol=1e-5)


def test_up_matrix_computes_the_transposed_convolution(tmp_path):
    weights, content = convert_tiny(tmp_path, 1)
    _, _, layers, _ = parse_layers(content)
    kind, in_channels, out_channels, values = layers[36]
    assert kind == convert.UP

    generator = numpy.random.default_rng(4)
    image = generator.standard_normal((in_channels, 2, 3)).astype(numpy.float32)
    weight = weights[36]
    direct = numpy.zeros((out_channels, 4, 6), dtype=numpy.float32)
    for y in range(2):
        for x in range(3):
            for dy in range(2):
                for dx in range(2):
                    direct[:, 2 * y + dy, 2 * x + dx] = image[:, y, x] @ weight[:, :, dy, dx]

    matrix = values.reshape(in_channels, 4 * out_channels)
    product = image.transpose(1, 2, 0) @ matrix
    result = numpy.zeros((out_channels, 4, 6), dtype=numpy.float32)
    for y in range(2):
        for x in range(3):
            for dy in range(2):
                for dx in range(2):
                    start = (dy * 2 + dx) * out_channels
                    result[:, 2 * y + dy, 2 * x + dx] = product[y, x, start:start + out_channels]

    assert numpy.allclose(result, direct, atol=1e-5)


def refuses(tmp_path, message_part: str, **options):
    with pytest.raises(convert.ConvertError) as caught:
        convert_tiny(tmp_path, 1, **options)

    assert "tiny.onnx" in str(caught.value)
    assert message_part in str(caught.value)


def test_refuses_a_bias(tmp_path):
    refuses(tmp_path, "layer3", with_bias=3)


def test_refuses_a_stride_one_down(tmp_path):
    def mutate(index, shape, attributes, op):
        if index == 9:
            attributes = {**attributes, "strides": [1, 1]}
        return shape, attributes, op

    refuses(tmp_path, "layer9", mutate=mutate)


def test_refuses_padding_on_an_up(tmp_path):
    def mutate(index, shape, attributes, op):
        if index == 36:
            attributes = {**attributes, "pads": [1, 1, 1, 1]}
        return shape, attributes, op

    refuses(tmp_path, "layer36", mutate=mutate)


def test_refuses_a_missing_layer(tmp_path):
    refuses(tmp_path, "63", drop_last=True)


def test_refuses_the_wrong_input_channels(tmp_path):
    source = tmp_path / "tiny.onnx"
    tiny_drunet(source, 1 + 2)

    with pytest.raises(convert.ConvertError) as caught:
        convert.convert_model(source, tmp_path / "tiny.bin", 0, 1, TINY_WIDTHS)

    assert "layer0" in str(caught.value)
