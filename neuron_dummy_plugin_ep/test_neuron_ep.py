#!/usr/bin/env python3
"""
test_neuron_ep.py
Tests the NeuronDummyPluginEP by:
  1. Registering the .so with ORT.
  2. Discovering the EP device.
  3. Creating an InferenceSession that uses the EP.
  4. Running the single-Add model and verifying the result.

Usage (x86_64 host or aarch64 target):
    python test_neuron_ep.py --so ./neuron_dummy_plugin_ep.so --model add_model.onnx
    python test_neuron_ep.py --so ./neuron_dummy_plugin_ep.so --model add_model_const.onnx

Prerequisites:
    pip install onnxruntime numpy onnx
    # On target: copy libonnxruntime.so and neuron_dummy_plugin_ep.so to the same directory
    #            or ensure LD_LIBRARY_PATH includes both.
"""

import argparse
import os
import sys
import numpy as np

import onnxruntime as ort

# ---------------------------------------------------------------------------
# EP registration name – must match what you pass to
# register_execution_provider_library().  This becomes the `registration_name`
# argument to CreateEpFactories() inside the .so.
# ---------------------------------------------------------------------------
EP_REGISTRATION_NAME = "NeuronDummyEP"


def test_ep(so_path: str, model_path: str, verbose: bool) -> None:
    so_path    = os.path.realpath(so_path)
    model_path = os.path.realpath(model_path)

    if not os.path.exists(so_path):
        sys.exit(f"ERROR: .so not found: {so_path}")
    if not os.path.exists(model_path):
        sys.exit(f"ERROR: model not found: {model_path}")

    print(f"\n=== NeuronDummyPluginEP test ===")
    print(f"  SO   : {so_path}")
    print(f"  Model: {model_path}")

    # ------------------------------------------------------------------
    # 1. Register the plugin EP with the ORT environment.
    # ------------------------------------------------------------------
    print(f"\n[1] Registering EP '{EP_REGISTRATION_NAME}' ...")
    ort.register_execution_provider_library(EP_REGISTRATION_NAME, so_path)
    print("    OK")

    # ------------------------------------------------------------------
    # 2. List all OrtEpDevices and find ours.
    # ------------------------------------------------------------------
    print(f"\n[2] Querying EP devices ...")
    ep_devices = ort.get_ep_devices()
    neuron_device = None

    for d in ep_devices:
        if verbose:
            print(f"    EP: {d.ep_name}  vendor: {d.ep_vendor}  "
                  f"hw_type: {d.device.type}")
        if d.ep_name == EP_REGISTRATION_NAME:
            neuron_device = d

    if neuron_device is None:
        sys.exit(f"ERROR: NeuronDummyEP device not found after registration.")

    print(f"    Found NeuronDummyEP device:")
    print(f"      ep_name   = {neuron_device.ep_name}")
    print(f"      ep_vendor = {neuron_device.ep_vendor}")
    print(f"      hw_type   = {neuron_device.device.type}")
    meta = neuron_device.ep_metadata
    if meta:
        print(f"      metadata  = {dict(meta)}")

    # ------------------------------------------------------------------
    # 3. Build SessionOptions that use NeuronDummyEP.
    # ------------------------------------------------------------------
    print(f"\n[3] Creating SessionOptions with NeuronDummyEP ...")
    sess_opts = ort.SessionOptions()
    if verbose:
        sess_opts.log_severity_level = 1   # INFO
    sess_opts.add_provider_for_devices([neuron_device], {})
    assert sess_opts.has_providers(), "Session options have no providers!"
    print("    OK")

    # ------------------------------------------------------------------
    # 4. Load the model (triggers GetCapability + Compile).
    # ------------------------------------------------------------------
    print(f"\n[4] Loading model (GetCapability + Compile) ...")
    sess = ort.InferenceSession(model_path, sess_options=sess_opts)
    print("    OK")

    # ------------------------------------------------------------------
    # 5. Inspect inputs / outputs.
    # ------------------------------------------------------------------
    inputs  = sess.get_inputs()
    outputs = sess.get_outputs()
    print(f"\n[5] Model I/O:")
    for inp in inputs:
        print(f"    Input  '{inp.name}': shape={inp.shape}  type={inp.type}")
    for out in outputs:
        print(f"    Output '{out.name}': shape={out.shape}  type={out.type}")

    # ------------------------------------------------------------------
    # 6. Run inference and verify result.
    # ------------------------------------------------------------------
    print(f"\n[6] Running inference ...")

    # Build feed_dict based on the model's actual inputs.
    feed = {}
    shapes = {inp.name: [d if isinstance(d, int) else 3 for d in inp.shape]
              for inp in inputs}

    if len(inputs) == 2:
        # Two live inputs: A and B.
        shape_A = shapes[inputs[0].name]
        shape_B = shapes[inputs[1].name]
        A = np.array(range(1, int(np.prod(shape_A)) + 1),
                     dtype=np.float32).reshape(shape_A)
        B = np.ones(shape_B, dtype=np.float32) * 2.0
        feed[inputs[0].name] = A
        feed[inputs[1].name] = B
        expected = A + B
    else:
        # One live input (A); B is a constant initializer (value = 0.5).
        shape_A = shapes[inputs[0].name]
        A = np.array(range(1, int(np.prod(shape_A)) + 1),
                     dtype=np.float32).reshape(shape_A)
        feed[inputs[0].name] = A
        expected = A + 0.5   # matches the initializer in create_test_model.py

    result = sess.run(None, feed)
    C = result[0]

    print(f"    Input  A  = \n{A}")
    if len(inputs) == 2:
        print(f"    Input  B  = \n{B}")
    else:
        print(f"    Constant B = 0.5 (from initializer)")
    print(f"    Output C  = \n{C}")
    print(f"    Expected  = \n{expected}")

    np.testing.assert_allclose(C, expected, rtol=1e-5, atol=1e-6)
    print("\n    PASS – output matches expected A + B ✓")

    # ------------------------------------------------------------------
    # 7. Clean up.
    # ------------------------------------------------------------------
    del sess
    ort.unregister_execution_provider_library(EP_REGISTRATION_NAME)
    print("\n[7] EP unregistered. Test complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test NeuronDummyPluginEP")
    parser.add_argument("--so",    required=True,
                        help="Path to neuron_dummy_plugin_ep.so")
    parser.add_argument("--model", default="add_model.onnx",
                        help="Path to ONNX model (default: add_model.onnx)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable INFO-level ORT logging")
    args = parser.parse_args()

    test_ep(args.so, args.model, args.verbose)
