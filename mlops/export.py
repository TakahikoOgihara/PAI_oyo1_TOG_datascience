# -*- coding: utf-8 -*-
"""ONNX export + inference-latency benchmarking.

Deploying to a robot means leaving the training framework behind: the policy is
exported to ONNX (the portable graph consumed by TensorRT / ONNX Runtime on the
edge).  We export, numerically validate against the PyTorch reference, and
benchmark per-inference latency for both backends so the report can quote a real
control-loop budget.
"""

from __future__ import annotations
import time
import numpy as np
import torch


def _onnx_export(model, dummies, path: str, names: list) -> None:
    """torch.onnx.export with a dynamic batch dim, across torch versions.

    torch >= 2.9 defaults to the dynamo/torch.export-based exporter, which can
    silently SPECIALIZE the batch dimension to the dummy's batch=1 even when
    dynamic_axes is given (seen with the ACT policy: its Reshape kept a baked
    shape {1,256,49} and any batch>1 inference failed in onnxruntime). The
    legacy tracer handles dynamic_axes correctly, so request it explicitly;
    older torch without the `dynamo` kwarg raises TypeError and already uses
    the legacy path by default.
    """
    kwargs = dict(
        input_names=names, output_names=["action"],
        dynamic_axes={**{n: {0: "batch"} for n in names}, "action": {0: "batch"}},
        opset_version=17,
    )
    try:
        torch.onnx.export(model, dummies, path, dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, dummies, path, **kwargs)


def export_onnx(model: torch.nn.Module, obs_dim: int, path: str,
                device: str = "cpu"):
    model = model.eval().to(device)
    dummy = torch.zeros(1, obs_dim, device=device)
    _onnx_export(model, dummy, path, ["obs"])
    return path


def validate_onnx(model: torch.nn.Module, obs_dim: int, path: str,
                  n: int = 256, device: str = "cpu"):
    """Return max abs diff between torch and onnxruntime outputs."""
    try:
        import onnxruntime as ort
    except Exception as e:
        return {"onnx_available": False, "error": str(e)}
    model = model.eval().to(device)
    x = torch.randn(n, obs_dim, device=device)
    with torch.no_grad():
        y_torch = model(x).cpu().numpy()
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    y_onnx = sess.run(None, {"obs": x.cpu().numpy().astype(np.float32)})[0]
    return {"onnx_available": True,
            "max_abs_diff": float(np.abs(y_torch - y_onnx).max())}


def benchmark_latency(model: torch.nn.Module, obs_dim: int, onnx_path: str | None,
                      device: str = "cpu", iters: int = 2000):
    """Single-sample (control-loop) latency for torch and onnxruntime."""
    out = {}
    model = model.eval().to(device)
    x = torch.randn(1, obs_dim, device=device)
    with torch.no_grad():
        for _ in range(50):
            model(x)
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        out["torch_ms"] = (time.perf_counter() - t0) / iters * 1e3

    if onnx_path:
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            xn = np.random.randn(1, obs_dim).astype(np.float32)
            for _ in range(50):
                sess.run(None, {"obs": xn})
            t0 = time.perf_counter()
            for _ in range(iters):
                sess.run(None, {"obs": xn})
            out["onnx_ms"] = (time.perf_counter() - t0) / iters * 1e3
        except Exception as e:
            out["onnx_error"] = str(e)
    return out


# ---------------------------------------------------------------------------
# ARM_BCRL001: multi-input (multimodal) variants. A visuomotor policy has more than
# one named input -- e.g. "flat" (state+lidar vector) and "rgb" (H,W,3 image)
# -- so export/validation/benchmark take a list of (name, shape) specs and
# call model.forward(*tensors) in spec order. The single-input helpers above
# stay untouched for state-only models.
# ---------------------------------------------------------------------------

def _dummy_inputs(input_specs: list, batch: int, device: str) -> tuple:
    ts = []
    for name, shape in input_specs:
        if name == "rgb":
            # image-scaled random input: values in [0, 255] like a real frame
            ts.append(torch.rand(batch, *shape, device=device) * 255.0)
        else:
            ts.append(torch.randn(batch, *shape, device=device))
    return tuple(ts)


def export_onnx_multi(model: torch.nn.Module, input_specs: list, path: str,
                      device: str = "cpu"):
    """input_specs: list of (input_name, per-sample shape tuple), e.g.
    [("flat", (80,)), ("rgb", (84, 84, 3))]."""
    model = model.eval().to(device)
    dummies = _dummy_inputs(input_specs, 1, device)
    names = [n for n, _ in input_specs]
    _onnx_export(model, dummies, path, names)
    return path


def validate_onnx_multi(model: torch.nn.Module, input_specs: list, path: str,
                        n: int = 32, device: str = "cpu"):
    try:
        import onnxruntime as ort
    except Exception as e:
        return {"onnx_available": False, "error": str(e)}
    model = model.eval().to(device)
    xs = _dummy_inputs(input_specs, n, device)
    with torch.no_grad():
        y_torch = model(*xs).cpu().numpy()
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    feed = {name: x.cpu().numpy().astype(np.float32) for (name, _), x in zip(input_specs, xs)}
    y_onnx = sess.run(None, feed)[0]
    return {"onnx_available": True,
            "max_abs_diff": float(np.abs(y_torch - y_onnx).max())}


def benchmark_latency_multi(model: torch.nn.Module, input_specs: list,
                            onnx_path: str | None, device: str = "cpu",
                            iters: int = 300):
    """Single-sample control-loop latency; fewer iters than the state-only
    benchmark because a ResNet18 forward is ~2 orders slower than an MLP."""
    out = {}
    model = model.eval().to(device)
    xs = _dummy_inputs(input_specs, 1, device)
    with torch.no_grad():
        for _ in range(10):
            model(*xs)
        t0 = time.perf_counter()
        for _ in range(iters):
            model(*xs)
        out["torch_ms"] = (time.perf_counter() - t0) / iters * 1e3

    if onnx_path:
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            feed = {name: x.cpu().numpy().astype(np.float32)
                    for (name, _), x in zip(input_specs, xs)}
            for _ in range(10):
                sess.run(None, feed)
            t0 = time.perf_counter()
            for _ in range(iters):
                sess.run(None, feed)
            out["onnx_ms"] = (time.perf_counter() - t0) / iters * 1e3
        except Exception as e:
            out["onnx_error"] = str(e)
    return out
