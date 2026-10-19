# -*- coding: utf-8 -*-
"""Jetson Orin Nano 用の推論バックエンド。まず TensorRT のエンジンを使い、
駄目なら onnxruntime に落ちる（どこでも動く退避路）。

Usage:
    backend = InferenceBackend("act_policy_fp16.engine")   # TensorRT
    backend = InferenceBackend("arm_bcrl006_act_policy.onnx")  # ORT の退避路
    out = backend.run({"rgb": rgb_f32, "qpos": qpos_f32})[0]

TensorRT 側は JetPack 6 付属の TensorRT 10.x の python API（名前付き
テンソル + execute_async_v3、バッファは cuda-python）を狙い、8.x の
バインディング添字のAPIにも落ちられる。エンジンは ONNX から
deploy/jetson/build_trt_engine.sh（trtexec --fp16）で作る。
"""

from __future__ import annotations
import numpy as np


class _TrtBackend:
    def __init__(self, engine_path: str):
        import tensorrt as trt
        from cuda import cudart

        self.trt = trt
        self.cudart = cudart
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        _, self.stream = cudart.cudaStreamCreate()

        self.inputs, self.outputs = {}, {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            _, dev = cudart.cudaMalloc(nbytes)
            host = np.empty(shape, dtype=dtype)
            entry = {"dev": dev, "host": host, "nbytes": nbytes, "shape": shape,
                     "dtype": dtype}
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs[name] = entry
            else:
                self.outputs[name] = entry
            self.context.set_tensor_address(name, dev)

    def run(self, feed: dict) -> list:
        cudart = self.cudart
        for name, entry in self.inputs.items():
            x = np.ascontiguousarray(feed[name].astype(entry["dtype"]).reshape(entry["shape"]))
            cudart.cudaMemcpyAsync(entry["dev"], x.ctypes.data, entry["nbytes"],
                                   cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)
        self.context.execute_async_v3(self.stream)
        outs = []
        for entry in self.outputs.values():
            cudart.cudaMemcpyAsync(entry["host"].ctypes.data, entry["dev"], entry["nbytes"],
                                   cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream)
        cudart.cudaStreamSynchronize(self.stream)
        for entry in self.outputs.values():
            outs.append(entry["host"].copy())
        return outs

    @property
    def input_names(self):
        return list(self.inputs.keys())


class _OrtBackend:
    def __init__(self, onnx_path: str):
        import onnxruntime as ort
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self._input_names = [i.name for i in self.sess.get_inputs()]
        self.metadata = self.sess.get_modelmeta().custom_metadata_map or {}

    def run(self, feed: dict) -> list:
        return self.sess.run(None, {k: v for k, v in feed.items()
                                    if k in self._input_names})

    @property
    def input_names(self):
        return self._input_names


def InferenceBackend(path: str):
    """工場関数。.engine/.plan なら TensorRT、.onnx なら onnxruntime。"""
    if path.endswith((".engine", ".plan")):
        return _TrtBackend(path)
    return _OrtBackend(path)
