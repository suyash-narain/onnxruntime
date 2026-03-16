#!/usr/bin/env python3
"""
create_test_model.py
Generates a minimal ONNX model with a single Add node.

Graph:
    A (float32, [3,2]) ─┐
                         Add ── C (float32, [3,2])
    B (float32, [3,2]) ─┘

Usage:
    python create_test_model.py                    # saves to add_model.onnx
    python create_test_model.py --out my.onnx      # custom path
    python create_test_model.py --shape 4 4        # 4×4 tensors
"""

import argparse
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto, helper


def make_add_model(shape: list[int], output_path: str) -> None:
    # Input type descriptor
    float_tensor_type = helper.make_tensor_type_proto(
        TensorProto.FLOAT, shape
    )

    # Graph inputs
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, shape)
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, shape)

    # Graph output
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, shape)

    # Single Add node: C = A + B
    add_node = helper.make_node(
        op_type="Add",
        inputs=["A", "B"],
        outputs=["C"],
        name="Add_0",
    )

    # Assemble graph
    graph = helper.make_graph(
        nodes=[add_node],
        name="single_add",
        inputs=[A, B],
        outputs=[C],
    )

    # Model with opset 13 (Add is available from opset 1 onwards)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    # Validate
    onnx.checker.check_model(model)

    onnx.save(model, output_path)
    print(f"Model saved to: {output_path}")
    print(f"  Input  A : float32 {shape}")
    print(f"  Input  B : float32 {shape}")
    print(f"  Output C : float32 {shape}")


def make_add_with_constant_model(shape: list[int], output_path: str) -> None:
    """
    Variant: B is a constant initializer (weight).
    This exercises the drop_constant_initializers path in the EP.
    """
    # A constant weight tensor for B
    weight_data = np.ones(shape, dtype=np.float32) * 0.5

    # Graph input: only A is live at runtime
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, shape)
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, shape)

    # Initializer (constant weight)
    B_init = numpy_helper.from_array(weight_data, name="B_const")

    add_node = helper.make_node("Add", inputs=["A", "B_const"], outputs=["C"], name="Add_0")

    graph = helper.make_graph(
        nodes=[add_node],
        name="add_with_constant",
        inputs=[A],
        outputs=[C],
        initializer=[B_init],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, output_path)

    print(f"Model (with constant B) saved to: {output_path}")
    print(f"  Input  A      : float32 {shape}")
    print(f"  Initializer B : float32 {shape} (all 0.5)")
    print(f"  Output C      : float32 {shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a single-Add ONNX test model.")
    parser.add_argument("--out",   default="add_model.onnx",
                        help="Output ONNX file path (default: add_model.onnx)")
    parser.add_argument("--shape", nargs="+", type=int, default=[3, 2],
                        help="Tensor shape (default: 3 2)")
    parser.add_argument("--with-constant", action="store_true",
                        help="Make the second input a constant initializer")
    parser.add_argument("--const-out", default="add_model_const.onnx",
                        help="Output path for the constant-B variant")
    args = parser.parse_args()

    make_add_model(args.shape, args.out)

    if args.with_constant:
        make_add_with_constant_model(args.shape, args.const_out)
