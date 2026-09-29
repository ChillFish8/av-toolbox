# /// script
# requires-python = ">=3.11"
# dependencies = ["onnx==1.23.0", "onnxruntime==1.23.2", "numpy==2.5.3"]
# ///
"""Runs ArtCNN R8F64 through onnxruntime, with its weights rounded to f16, on a fixed 16x16 input."""

import sys
from pathlib import Path

import numpy
import onnx
import onnxruntime
from onnx import numpy_helper

SIZE = 16


def golden_input() -> numpy.ndarray:
    rows = numpy.arange(SIZE).reshape(SIZE, 1)
    columns = numpy.arange(SIZE).reshape(1, SIZE)
    pattern = (columns * 7 + rows * 13) % 16
    return (pattern / 15.0).astype(numpy.float32)


def rounded_to_f16(model: onnx.ModelProto) -> onnx.ModelProto:
    convolution_inputs = set()
    for node in model.graph.node:
        if node.op_type == "Conv":
            convolution_inputs.update(node.input[1:])

    for index, initializer in enumerate(model.graph.initializer):
        if initializer.name not in convolution_inputs:
            continue

        values = numpy_helper.to_array(initializer)
        rounded = values.astype(numpy.float16).astype(numpy.float32)
        replacement = numpy_helper.from_array(rounded, initializer.name)
        model.graph.initializer[index].CopyFrom(replacement)

    return model


def main(arguments: list[str]) -> int:
    cache = Path(arguments[0])
    output = Path(arguments[1])
    source = cache / "ArtCNN_R8F64.onnx"

    model = onnx.load(source)
    rounded = rounded_to_f16(model)
    serialized = rounded.SerializeToString()
    session = onnxruntime.InferenceSession(serialized, providers=["CPUExecutionProvider"])

    image = golden_input()
    batch = image.reshape(1, 1, SIZE, SIZE)
    input_name = session.get_inputs()[0].name
    results = session.run(None, {input_name: batch})
    doubled = results[0].reshape(SIZE * 2, SIZE * 2).astype("<f4")

    output.mkdir(parents=True, exist_ok=True)
    target = output / "golden_r8f64.f32"
    target.write_bytes(doubled.tobytes())
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
