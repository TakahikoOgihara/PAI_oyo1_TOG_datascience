# -*- coding: utf-8 -*-
"""ONNX の書き出しと、推論遅延の計測。

ロボットへ配備するとは、学習の枠組みを置いていくこと。方策は ONNX
（エッジの TensorRT / ONNX Runtime が食う移植可能なグラフ）に書き出す。
書き出し、PyTorch の基準実装と数値で突き合わせ、両方のバックエンドで
1回の推論の遅延を測る。だからレポートは、実際の制御ループの予算を
数字で言える。
"""

from __future__ import annotations
import time
import numpy as np
import torch


def _onnx_export(model, dummies, path: str, names: list) -> None:
    """動的なバッチ次元付きの torch.onnx.export を、torch の版をまたいで使う。

    torch 2.9 以降は dynamo / torch.export 系の書き出しが既定で、
    dynamic_axes を渡しても、ダミーの batch=1 にバッチ次元を黙って
    固定してしまうことがある（ACT の方策で実際に起きた。Reshape が
    {1,256,49} の形を焼き込み、batch>1 の推論が onnxruntime で全部
    失敗した）。旧来のトレーサは dynamic_axes を正しく扱うので、明示的に
    そちらを要求する。`dynamo` 引数を知らない古い torch は TypeError を
    出すが、そちらは既定で旧来の経路を使っている。
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
    """torch と onnxruntime の出力の、最大絶対差を返す。"""
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
    """1標本（制御ループ）の遅延を、torch と onnxruntime で測る。"""
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
# ARM_BCRL001 で入れた多入力（マルチモーダル）版。視覚運動方策は名前付き
# の入力を複数持つ。たとえば "flat"（状態+LiDARのベクトル）と
# "rgb"（H,W,3 の画像）。だから書き出し／検証／計測は (name, shape) の
# リストを取り、その順で model.forward(*tensors) を呼ぶ。上の単入力の
# 補助関数は、状態だけのモデル用にそのまま残してある。
# ---------------------------------------------------------------------------

def _dummy_inputs(input_specs: list, batch: int, device: str) -> tuple:
    ts = []
    for name, shape in input_specs:
        if name == "rgb":
            # 画像の尺度に合わせた乱数入力。実際のフレームと同じ [0, 255]
            ts.append(torch.rand(batch, *shape, device=device) * 255.0)
        else:
            ts.append(torch.randn(batch, *shape, device=device))
    return tuple(ts)


def export_onnx_multi(model: torch.nn.Module, input_specs: list, path: str,
                      device: str = "cpu"):
    """input_specs は (入力名, 1標本ぶんの形) のリスト。たとえば
    [("flat", (80,)), ("rgb", (84, 84, 3))]。"""
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
    """1標本の制御ループ遅延。ResNet18 の順伝播は MLP より2桁ほど遅いので、
    状態だけの計測より反復回数は少なくしてある。"""
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
