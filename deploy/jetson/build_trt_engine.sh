#!/bin/bash
# =====================================================================
# ONNX -> TensorRT エンジンのビルド（Jetson Orin Nano、JetPack 6.x）。
# trtexec は TensorRT 付属で、/usr/src/tensorrt/bin/trtexec にある。
#
#   bash build_trt_engine.sh ~/edge_bundle            # ACT（既定）
#   bash build_trt_engine.sh ~/edge_bundle diffusion  # DP のエンコーダ+デノイザ
#
# FP16 は Orin Nano で遅延を半分にし、この種の方策では精度の劣化は
# 無視できる。あとで必ず検証すること（policy_runtime が遅延を表示し、
# ORT との保留データでの比較も手順に入れてある）。
# =====================================================================
set -euo pipefail
BUNDLE="${1:?usage: build_trt_engine.sh <edge_bundle_dir> [diffusion]}"
VARIANT="${2:-act}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"

if [ "$VARIANT" = "diffusion" ]; then
  ONNX="$BUNDLE/arm_bcrl006_dp_policy.onnx"
  DEN="$BUNDLE/arm_bcrl006_dp_policy_denoiser.onnx"
  IMG=$(python3 -c "import json;print(json.load(open('$BUNDLE/edge_config.json'))['onnx_metadata'].get('image_size',224))")
  CHUNK=$(python3 -c "import json;print(json.load(open('$BUNDLE/edge_config.json'))['onnx_metadata'].get('chunk',20))")
  "$TRTEXEC" --onnx="$ONNX" --saveEngine="$BUNDLE/dp_encoder_fp16.engine" --fp16 \
    --shapes=rgb:1x${IMG}x${IMG}x3,qpos:1x7
  "$TRTEXEC" --onnx="$DEN" --saveEngine="$BUNDLE/dp_denoiser_fp16.engine" --fp16 \
    --shapes=noisy_chunk:1x${CHUNK}x7,timestep:1,obs_emb:1x256
  echo "engines: $BUNDLE/dp_encoder_fp16.engine + dp_denoiser_fp16.engine"
  echo "run: python3 policy_runtime.py --bundle $BUNDLE --engine $BUNDLE/dp_encoder_fp16.engine --denoiser-engine $BUNDLE/dp_denoiser_fp16.engine --pi <pi-ip>:9000"
else
  ONNX="$BUNDLE/arm_bcrl006_act_policy.onnx"
  IMG=$(python3 -c "import json;print(json.load(open('$BUNDLE/edge_config.json'))['onnx_metadata'].get('image_size',224))")
  "$TRTEXEC" --onnx="$ONNX" --saveEngine="$BUNDLE/act_policy_fp16.engine" --fp16 \
    --shapes=rgb:1x${IMG}x${IMG}x3,qpos:1x7
  echo "engine: $BUNDLE/act_policy_fp16.engine"
  echo "run: python3 policy_runtime.py --bundle $BUNDLE --engine $BUNDLE/act_policy_fp16.engine --pi <pi-ip>:9000"
fi
