# -*- coding: utf-8 -*-
"""ARM_BCRL006 の方策ランタイム。Jetson Orin Nano 8GB の上で走る。

シミュレータで学習した、実機の制御ループを閉じる:

  USB の手首カメラ（Jetson 側）                20Hz の制御刻み
      -> 学習したカメラ仕様へ中央切り出し + 縮小
      -> TensorRT（または onnxruntime）の視覚運動方策。ACT のチャンク / DP の DDIM
      -> 単位の関節差分指令を LAN（JSON/TCP）で送る
      -> Raspberry Pi 4B のブリッジ（deploy/raspi/mycobot_bridge_server.py）
      -> myCobot 280 Pi のサーボ

同時に WebSocket のテレメトリ／指令の口（既定 :8765）を開き、操作用PCの
Webアプリ（deploy/operator_pc/webapp）が繋ぐ。カメラのJPEGと関節状態を
出し、開始／一時停止／ジョグ／グリッパ／原点復帰／非常停止を受ける。

入力は、パイプラインが書き出すエッジ配布物（src/infer.py）から来る:
  edge_bundle/
    arm_bcrl006_act_policy.onnx   （または _dp_policy.onnx + _dp_policy_denoiser.onnx）
    edge_config.json              （制御の契約 + カメラ仕様 + チャンク設定）
    deploy_report.json

Run:
  python3 policy_runtime.py --bundle ~/edge_bundle --pi 192.168.10.10:9000 \
      --engine act_policy_fp16.engine        # --engine を省くと ORT に落ちる
"""

from __future__ import annotations
import argparse
import base64
import json
import math
import os
import socket
import threading
import time

import numpy as np

from trt_runner import InferenceBackend


# --------------------------------------------------------------------------
# DDIM（拡散版）。arm_policy/diffusion.py のスケジュールと必ず一致させる
# --------------------------------------------------------------------------
def ddim_alphas(diffusion_steps: int) -> np.ndarray:
    s = 0.008
    t = np.arange(diffusion_steps + 1, dtype=np.float64) / diffusion_steps
    f = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    alphas_bar = f / f[0]
    betas = np.clip(1.0 - alphas_bar[1:] / alphas_bar[:-1], 0.0, 0.999)
    return np.cumprod(1.0 - betas)


def ddim_sample(denoiser, obs_emb, chunk, act_dim, alphas_cumprod, ddim_steps):
    B = obs_emb.shape[0]
    T = len(alphas_cumprod)
    ts = np.linspace(T - 1, 0, ddim_steps).round().astype(np.int64)
    x = np.random.RandomState(0).randn(B, chunk, act_dim).astype(np.float32)
    for i, t in enumerate(ts):
        a_t = alphas_cumprod[t]
        a_prev = alphas_cumprod[ts[i + 1]] if i + 1 < len(ts) else 1.0
        eps = denoiser.run({"noisy_chunk": x,
                            "timestep": np.full(B, t, dtype=np.int64),
                            "obs_emb": obs_emb})[0]
        x0 = np.clip((x - math.sqrt(1 - a_t) * eps) / math.sqrt(a_t), -1.0, 1.0)
        x = math.sqrt(a_prev) * x0 + math.sqrt(1 - a_prev) * eps
    return np.clip(x, -1.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------
# Pi ブリッジのクライアント（TCP 上の改行区切り JSON）
# --------------------------------------------------------------------------
class PiClient:
    def __init__(self, host: str, port: int):
        self.addr = (host, port)
        self.lock = threading.Lock()
        self.sock = None
        self._connect()

    def _connect(self):
        self.sock = socket.create_connection(self.addr, timeout=2.0)
        self.rfile = self.sock.makefile("r")

    def call(self, msg: dict) -> dict:
        with self.lock:
            try:
                self.sock.sendall((json.dumps(msg) + "\n").encode())
                return json.loads(self.rfile.readline())
            except (OSError, ValueError):
                self._connect()
                self.sock.sendall((json.dumps(msg) + "\n").encode())
                return json.loads(self.rfile.readline())

    def state(self):
        return self.call({"cmd": "state"})

    def deltas(self, d, grip):
        return self.call({"cmd": "deltas", "d": [float(v) for v in d],
                          "grip": float(grip)})

    def send_q(self, q, speed=30):
        return self.call({"cmd": "send_q", "q": [float(v) for v in q], "speed": speed})

    def gripper(self, open_: bool):
        return self.call({"cmd": "gripper", "open": bool(open_)})

    def home(self):
        return self.call({"cmd": "home"})

    def safety_reset(self):
        return self.call({"cmd": "safety_reset"})

    def stop(self):
        return self.call({"cmd": "stop"})

    def home_and_wait(self, timeout_s: float = 8.0) -> dict:
        """学習時の 'ready' 姿勢を指令し、アームが到達したと報告するまで待つ。
        最後の状態の辞書を返す（``homed`` で分かる）。"""
        self.home()
        deadline = time.time() + timeout_s
        st = self.state()
        while time.time() < deadline:
            st = self.state()
            if st.get("homed"):
                return st
            time.sleep(0.1)
        return st


# --------------------------------------------------------------------------
# camera
# --------------------------------------------------------------------------
class Camera:
    def __init__(self, index: int, width: int, height: int, policy_size: int):
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.policy_size = policy_size
        self.lock = threading.Lock()
        self.frame_bgr = None
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame_bgr = frame
            else:
                time.sleep(0.05)

    def policy_rgb(self) -> np.ndarray | None:
        """正方の中央切り出し + 学習した入力への縮小。シミュレータのカメラ仕様が
        前提にしている変換そのもの（実画像から正方の 224x224 を作る）。
        4:3 sensor)."""
        with self.lock:
            f = None if self.frame_bgr is None else self.frame_bgr.copy()
        if f is None:
            return None
        h, w = f.shape[:2]
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        sq = f[y0:y0 + s, x0:x0 + s]
        sq = self.cv2.resize(sq, (self.policy_size, self.policy_size),
                             interpolation=self.cv2.INTER_AREA)
        return self.cv2.cvtColor(sq, self.cv2.COLOR_BGR2RGB).astype(np.float32)

    def jpeg_b64(self) -> str | None:
        with self.lock:
            f = None if self.frame_bgr is None else self.frame_bgr.copy()
        if f is None:
            return None
        ok, buf = self.cv2.imencode(".jpg", f, [int(self.cv2.IMWRITE_JPEG_QUALITY), 70])
        return base64.b64encode(buf.tobytes()).decode() if ok else None


# --------------------------------------------------------------------------
# runtime
# --------------------------------------------------------------------------
class Runtime:
    def __init__(self, args):
        with open(os.path.join(args.bundle, "edge_config.json"), encoding="utf-8") as f:
            self.cfg = json.load(f)
        self.control_dt = float(args.control_dt or self.cfg["control_dt"])
        self.execute_horizon = int(self.cfg["execute_horizon"])
        meta = self.cfg.get("onnx_metadata", {})
        self.policy_type = meta.get("policy_type", "act")
        self.chunk = int(meta.get("chunk", 20))
        self.act_dim = int(meta.get("act_dim", 7))
        self.image_size = int(meta.get("image_size", self.cfg["camera"]["width"]))

        onnx_path = os.path.join(args.bundle, self.cfg["onnx"])
        self.backend = InferenceBackend(args.engine or onnx_path)
        self.denoiser = None
        if self.policy_type == "diffusion":
            den = args.denoiser_engine or onnx_path.replace(".onnx", "_denoiser.onnx")
            self.denoiser = InferenceBackend(den)
            self.alphas = ddim_alphas(int(meta.get("diffusion_steps", 100)))
            self.ddim_steps = int(meta.get("ddim_steps", 8))

        host, port = args.pi.split(":")
        self.pi = PiClient(host, int(port))
        self.camera = Camera(args.camera, args.cam_width, args.cam_height,
                             self.image_size)

        self.mode = "pause"          # pause | auto
        self.grip_cmd = 1.0          # 最後に送ったグリッパ指令（2値）
        # 方策自身が出したグリッパのチャンネル値を、qpos[6] として戻す。
        # シミュレータはそこへ生の（クランプした）行動を入れているので、
        # ここで2値化すると（ARM_BCRL002 がそうしていた）、方策が見る自己
        # 受容感覚が、何の得も無く学習時の分布からずれる
        # （実機の失敗記録の原因G）。
        self.grip_qpos = 1.0
        self.buf = None
        self.cursor = self.chunk     # 空なら最初の刻みで推論する
        self.latency_ms = 0.0
        self.loop_hz = 0.0
        self.safety = {"latched": False, "reason": "ok"}
        self.homed = False
        self.status_msg = "idle"
        self._stop = False

    # ---- policy -------------------------------------------------------
    def _qpos(self, state: dict) -> np.ndarray:
        return np.array(state["q"] + [self.grip_qpos], dtype=np.float32)

    def _infer_chunk(self, rgb: np.ndarray, qpos: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        feed = {"rgb": rgb[None], "qpos": qpos[None]}
        if self.policy_type == "diffusion":
            emb = self.backend.run(feed)[0]
            chunk = ddim_sample(self.denoiser, emb, self.chunk, self.act_dim,
                                self.alphas, self.ddim_steps)[0]
        else:
            chunk = self.backend.run(feed)[0][0]
        self.latency_ms = (time.perf_counter() - t0) * 1e3
        return chunk

    # ---- 制御ループ -----------------------------------------------------
    def control_loop(self):
        next_t = time.perf_counter()
        ticks, t_rate = 0, time.perf_counter()
        while not self._stop:
            next_t += self.control_dt
            if self.mode == "auto":
                try:
                    state = self.pi.state()
                    self.safety = state.get("safety", self.safety)
                    self.homed = bool(state.get("homed", self.homed))
                    # 権限を持つのは Pi のガード。ラッチしたなら、アームはすでに
                    # 保持されている。方策は送信を止め、操作者が安定した状態を見て
                    # 何をするか決められるようにする。
                    if self.safety.get("latched"):
                        self.status_msg = (f"safety stop: "
                                           f"{self.safety.get('reason')} "
                                           f"{self.safety.get('detail', '')}".strip())
                        print(f"[runtime] {self.status_msg} -> pause")
                        self.mode = "pause"
                        continue
                    if self.cursor >= self.execute_horizon:
                        rgb = self.camera.policy_rgb()
                        if rgb is None:
                            # 新しい画像が無いとき。古いチャンクを開ループで実行し続けるのは
                            # 絶対にしない。保持する。
                            self.status_msg = "no camera frame -> holding"
                            self.pi.deltas([0.0] * 6, self.grip_cmd)
                            continue
                        self.buf = self._infer_chunk(rgb, self._qpos(state))
                        self.cursor = 0
                    if self.buf is not None and self.cursor < self.chunk:
                        a = self.buf[self.cursor]
                        self.cursor += 1
                        self.grip_qpos = float(a[6])
                        self.grip_cmd = 1.0 if a[6] > 0 else -1.0
                        rep = self.pi.deltas(a[:6], self.grip_cmd)
                        self.safety = rep.get("safety", self.safety)
                        self.status_msg = "running"
                except Exception as e:
                    print(f"[runtime] control error: {e} -> pause")
                    self.status_msg = f"control error: {e}"
                    self.mode = "pause"
            ticks += 1
            if ticks % 40 == 0:
                now = time.perf_counter()
                self.loop_hz = 40.0 / (now - t_rate)
                t_rate = now
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.perf_counter()  # 刻みを超過。追いかけて悪化させない

    # ---- 操作者からの指令 -----------------------------------------------
    def handle_cmd(self, msg: dict) -> None:
        action = msg.get("action")
        if action == "start":
            # 原点復帰は任意ではない。学習の全エピソードは "ready" 姿勢から
            # 始まるので、別の場所から方策を動かすのは、見たことのない状態
            # （とカメラ視点）を与えること。ARM_BCRL002 が実機で10回中10回
            # 失敗した系統的な原因がこれ（実機の失敗記録の原因A）。
            # 原点復帰して、確認して、それから走らせる。できなければ
            # 拒否する。
            try:
                st = self.pi.state()
                self.safety = st.get("safety", self.safety)
                if self.safety.get("latched"):
                    self.status_msg = ("safety latched -- press Safety reset "
                                       "after clearing the obstruction")
                    return
                if not st.get("homed"):
                    self.status_msg = "homing to the training ready pose..."
                    self.grip_cmd = self.grip_qpos = 1.0
                    self.pi.gripper(True)
                    st = self.pi.home_and_wait()
                self.homed = bool(st.get("homed"))
                if not self.homed:
                    self.status_msg = (
                        f"REFUSED to start: the arm did not reach the ready "
                        f"pose (q={[round(v, 3) for v in st.get('q', [])]}, "
                        f"expected [0.25,-1.0,-0.1,-0.5,0,0]). Check the servo "
                        f"zero calibration -- the deployment notes.")
                    print(f"[runtime] {self.status_msg}")
                    return
            except Exception as e:
                self.status_msg = f"REFUSED to start: bridge unreachable ({e})"
                return
            self.cursor = self.chunk  # 推論を強制的にやり直す
            self.buf = None
            self.status_msg = "running"
            self.mode = "auto"
        elif action == "pause":
            self.mode = "pause"
            self.status_msg = "paused"
        elif action == "estop":
            self.mode = "pause"
            self.pi.stop()
            self.status_msg = "E-STOP: holding position"
        elif action == "safety_reset":
            self.mode = "pause"
            try:
                rep = self.pi.safety_reset()
                self.safety = rep.get("safety", self.safety)
                self.status_msg = "safety cleared -- Home, then Start"
            except Exception as e:
                self.status_msg = f"safety reset failed: {e}"
        elif action == "home":
            self.mode = "pause"
            self.grip_cmd = self.grip_qpos = 1.0
            self.pi.gripper(True)
            st = self.pi.home_and_wait()
            self.homed = bool(st.get("homed"))
            self.safety = st.get("safety", self.safety)
            self.status_msg = ("at the ready pose" if self.homed else
                               "home did NOT converge -- check zero calibration")
        elif action == "gripper":
            self.grip_cmd = 1.0 if msg.get("open", True) else -1.0
            self.pi.gripper(bool(msg.get("open", True)))
        elif action == "jog" and self.mode == "pause":
            j = int(msg.get("joint", 0))
            step = 0.05 * (1 if int(msg.get("dir", 1)) > 0 else -1)
            q = list(self.pi.state()["q"])
            q[j] += step
            self.pi.send_q(q, speed=20)

    def telemetry(self) -> dict:
        try:
            state = self.pi.state()
            self.safety = state.get("safety", self.safety)
            self.homed = bool(state.get("homed", self.homed))
        except Exception:
            state = {"q": [0.0] * 6, "grip": 1}
        return {"type": "telemetry", "mode": self.mode,
                "q": [round(float(v), 4) for v in state["q"]],
                "grip_open": bool(self.grip_cmd > 0),
                "policy": self.policy_type,
                "latency_ms": round(self.latency_ms, 1),
                "loop_hz": round(self.loop_hz, 1),
                "homed": self.homed,
                "safety": self.safety,
                "tcp": state.get("tcp"),
                "tool_z": state.get("tool_z"),
                "status": self.status_msg,
                "jpeg": self.camera.jpeg_b64()}


# --------------------------------------------------------------------------
# WebSocket サーバ（操作用PCがここへ繋ぐ）
# --------------------------------------------------------------------------
async def ws_main(rt: Runtime, port: int):
    import asyncio
    import websockets

    clients = set()

    async def handler(ws):
        clients.add(ws)
        print(f"[runtime] operator connected ({len(clients)})")
        try:
            async for raw in ws:
                try:
                    rt.handle_cmd(json.loads(raw))
                except Exception as e:
                    await ws.send(json.dumps({"type": "error", "error": str(e)}))
        finally:
            clients.discard(ws)
            print(f"[runtime] operator disconnected ({len(clients)})")

    async def broadcaster():
        while True:
            if clients:
                msg = json.dumps(rt.telemetry())
                await asyncio.gather(*(c.send(msg) for c in list(clients)),
                                     return_exceptions=True)
            await asyncio.sleep(0.1)  # 10 Hz telemetry

    async with websockets.serve(handler, "0.0.0.0", port, max_size=None):
        print(f"[runtime] websocket telemetry/command server on :{port}")
        await broadcaster()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="edge_bundle directory (scp'd from the GPU server)")
    ap.add_argument("--engine", default=None, help="TensorRT engine (else ORT on the bundle ONNX)")
    ap.add_argument("--denoiser-engine", default=None)
    ap.add_argument("--pi", default="192.168.10.10:9000", help="Raspberry Pi bridge host:port")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--cam-width", type=int, default=640)
    ap.add_argument("--cam-height", type=int, default=480)
    ap.add_argument("--ws-port", type=int, default=8765)
    ap.add_argument("--control-dt", type=float, default=None)
    args = ap.parse_args()

    rt = Runtime(args)
    threading.Thread(target=rt.control_loop, daemon=True).start()
    print(f"[runtime] policy={rt.policy_type} chunk={rt.chunk} "
          f"execute_horizon={rt.execute_horizon} control_dt={rt.control_dt}")
    print("[runtime] mode=pause -- press Start in the web app to run the policy")
    print("[runtime] Start homes the arm to the training ready pose first and "
          "refuses to run if it does not get there; the Pi's safety layer "
          "(floor plane + impact stop) can pause the policy at any time")
    import asyncio
    asyncio.run(ws_main(rt, args.ws_port))


if __name__ == "__main__":
    main()
