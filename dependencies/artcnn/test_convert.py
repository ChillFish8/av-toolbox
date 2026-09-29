import importlib.util
import struct
from pathlib import Path

import numpy
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

CONVERTER_PATH = Path(__file__).parent / "convert.py"
CONVERTER_SPEC = importlib.util.spec_from_file_location("artcnn_convert", CONVERTER_PATH)
convert = importlib.util.module_from_spec(CONVERTER_SPEC)
CONVERTER_SPEC.loader.exec_module(convert)

PLAIN = {"strides": [1, 1], "dilations": [1, 1], "auto_pad": "SAME_UPPER"}


def tiny_model(
    path: Path,
    features: int = 2,
    blocks: int = 1,
    attributes=PLAIN,
    first_weight=None,
    first_bias_length=None,
):
    """Writes a chain of convolutions with ArtCNN's layer shapes, returning each layer's weight and bias."""
    shapes = convert.expected_shapes(features, blocks)
    generator = numpy.random.default_rng(7)
    nodes = []
    initializers = []
    layers = []
    previous = "input"

    for index, (out_channels, in_channels) in enumerate(shapes):
        weight = generator.standard_normal((out_channels, in_channels, 3, 3)).astype(numpy.float32)
        bias_length = out_channels

        if index == 0 and first_bias_length is not None:
            bias_length = first_bias_length

        bias = generator.standard_normal(bias_length).astype(numpy.float32)

        if index == 0 and first_weight is not None:
            weight[0, 0, 0, 0] = first_weight

        weight_name = f"w{index}"
        bias_name = f"b{index}"
        output = f"c{index}"
        weight_tensor = numpy_helper.from_array(weight, weight_name)
        bias_tensor = numpy_helper.from_array(bias, bias_name)
        node = helper.make_node("Conv", [previous, weight_name, bias_name], [output], **attributes)

        initializers.extend([weight_tensor, bias_tensor])
        nodes.append(node)
        layers.append((weight, bias))
        previous = output

    graph_input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, None, None])
    graph_output = helper.make_tensor_value_info(previous, TensorProto.FLOAT, None)
    graph = helper.make_graph(nodes, "tiny", [graph_input], [graph_output], initializers)
    model = helper.make_model(graph)
    onnx.save(model, path)

    return layers


def read_layers(path: Path):
    data = path.read_bytes()
    assert data[:8] == b"AVTBCNN1"
    (count,) = struct.unpack_from("<I", data, 8)
    offset = 12
    layers = []

    for _ in range(count):
        out_channels, in_channels, kernel_height, kernel_width = struct.unpack_from("<4I", data, offset)
        offset += 16
        weight_count = out_channels * kernel_height * kernel_width * in_channels
        weight = numpy.frombuffer(data, "<f2", weight_count, offset)
        offset += weight_count * 2
        bias = numpy.frombuffer(data, "<f2", out_channels, offset)
        offset += out_channels * 2
        shape = (out_channels, in_channels, kernel_height, kernel_width)
        ohwi = weight.reshape(out_channels, kernel_height, kernel_width, in_channels)
        layers.append((shape, ohwi, bias))

    assert offset == len(data)
    return layers


def test_the_shapes_follow_the_width_and_block_count():
    shapes = convert.expected_shapes(64, 8)

    assert len(shapes) == 27
    assert shapes[0] == (64, 1)
    assert shapes[1:26] == [(64, 64)] * 25
    assert shapes[26] == (4, 64)


def test_layers_are_written_as_ohwi_f16_in_graph_order(tmp_path):
    source = tmp_path / "tiny.onnx"
    target = tmp_path / "tiny.bin"
    layers = tiny_model(source)

    convert.convert_model(source, target, 2, 1)

    written = read_layers(target)
    assert len(written) == len(layers)
    for (shape, ohwi, bias), (weight, expected_bias) in zip(written, layers):
        expected_ohwi = weight.transpose(0, 2, 3, 1).astype(numpy.float16)
        expected_bias_f16 = expected_bias.astype(numpy.float16)
        assert shape == weight.shape
        assert numpy.array_equal(ohwi, expected_ohwi)
        assert numpy.array_equal(bias, expected_bias_f16)


def test_explicit_padding_of_one_is_accepted(tmp_path):
    source = tmp_path / "tiny.onnx"
    target = tmp_path / "tiny.bin"
    explicit = {"strides": [1, 1], "dilations": [1, 1], "pads": [1, 1, 1, 1]}
    tiny_model(source, attributes=explicit)

    convert.convert_model(source, target, 2, 1)

    assert target.exists()


@pytest.mark.parametrize(
    "attributes",
    [
        {"strides": [2, 2], "dilations": [1, 1], "auto_pad": "SAME_UPPER"},
        {"strides": [1, 1], "dilations": [2, 2], "auto_pad": "SAME_UPPER"},
        {"strides": [1, 1], "dilations": [1, 1], "auto_pad": "VALID"},
        {"strides": [1, 1], "dilations": [1, 1], "pads": [0, 0, 0, 0]},
    ],
)
def test_a_convolution_that_is_not_stride_one_same_padded_is_refused(tmp_path, attributes):
    source = tmp_path / "tiny.onnx"
    target = tmp_path / "tiny.bin"
    tiny_model(source, attributes=attributes)

    with pytest.raises(convert.ConvertError, match="tiny.onnx"):
        convert.convert_model(source, target, 2, 1)

    assert not target.exists()


def test_a_model_with_the_wrong_layer_count_is_refused(tmp_path):
    source = tmp_path / "tiny.onnx"
    target = tmp_path / "tiny.bin"
    tiny_model(source, blocks=2)

    with pytest.raises(convert.ConvertError, match="convolutions"):
        convert.convert_model(source, target, 2, 1)


def test_a_model_with_the_wrong_width_is_refused(tmp_path):
    source = tmp_path / "tiny.onnx"
    target = tmp_path / "tiny.bin"
    tiny_model(source, features=3)

    with pytest.raises(convert.ConvertError, match="shape"):
        convert.convert_model(source, target, 2, 1)


def test_a_wrong_bias_shape_is_reported(tmp_path):
    source = tmp_path / "tiny.onnx"
    target = tmp_path / "tiny.bin"
    tiny_model(source, first_bias_length=3)

    with pytest.raises(convert.ConvertError, match=r"bias \(3,\)"):
        convert.convert_model(source, target, 2, 1)


def test_a_weight_f16_cannot_hold_is_refused(tmp_path):
    source = tmp_path / "tiny.onnx"
    target = tmp_path / "tiny.bin"
    tiny_model(source, first_weight=1.0e6)

    with pytest.raises(convert.ConvertError, match="f16"):
        convert.convert_model(source, target, 2, 1)

    assert not target.exists()


def test_main_converts_every_listed_model(tmp_path, monkeypatch):
    sources = tmp_path / "sources"
    output = tmp_path / "output"
    sources.mkdir()
    output.mkdir()
    tiny_model(sources / "a.onnx")
    monkeypatch.setattr(convert, "MODELS", {"a.onnx": ("a.bin", 2, 1)})

    status = convert.main([str(sources), str(output)])

    written = sorted(path.name for path in output.iterdir())
    assert status == 0
    assert written == ["a.bin"]
