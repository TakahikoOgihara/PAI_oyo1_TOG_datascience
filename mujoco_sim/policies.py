# -*- coding: utf-8 -*-
"""学習アルゴリズム以外で行動を作るもの、2つ:

  * ``ExpertController``。フェーズ1のデモを生成するスクリプトIKの
    pick&place エキスパート（MimicGen の元データにもなる）。
  * ``OnnxPolicy``。このパイプラインが配備する ONNX を全部こなす。
    だからシミュレータ評価も、エッジのランタイムと同じグラフを走らせる。

どちらも ARM_BCRL005 のクラスの移植で、置き換えたのは1点だけ。
Isaac Lab の ``DifferentialIKController`` + PhysX のヤコビアンが、
``arm_common.ik`` の解析ヤコビアンに対する減衰最小二乗になった。
状態機械（``arm_common/expert.py``）、目標のレート制限、行動の符号化、
チャンク後退ホライズンの実行は変えていない。
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from arm_common.expert import PickPlacePhaseMachine  # noqa: E402
from arm_common.ik import DLSPositionIK  # noqa: E402

ACT_DIM = 7


class ExpertController:
    """状態機械が出す直交座標の経由点 -> 関節差分の行動。

    分担は ARM_BCRL005 から変えていない:
      PickPlacePhaseMachine  「pick&place のどの段にいて、TCP をどこへ
                              向けるか、グリッパは開くか」
      ExpertController       「その TCP 目標を、環境と実機の両方が受け取る
                              7次元の行動に変える」
    """

    def __init__(self, env):
        self.env = env
        e = env.cfg.env.expert
        self.machine = PickPlacePhaseMachine(
            env.num_envs, str(env.device),
            approach_height=float(e.approach_height),
            grasp_z_offset=float(e.grasp_z_offset),
            pos_tol=float(e.pos_tol),
            grasp_steps=int(e.grasp_steps),
            release_steps=int(e.release_steps),
            transport_height=float(env.cfg.env.lift_height) + float(e.lift_extra),
            place_z=float(env.cfg.env.object.height) / 2.0 + float(e.grasp_z_offset),
        )
        self.ik = DLSPositionIK(env.tool_kin, damping=float(e.get("ik_damping", 0.05)))
        self._carrot: torch.Tensor | None = None
        self._carrot_step = float(e.carrot_step)      # 1制御ステップあたりのm

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        self.machine.reset_idx(env_ids)
        if self._carrot is not None:
            self._carrot[env_ids] = self.env.eef_pos_env()[env_ids]

    def act(self, obs=None) -> torch.Tensor:
        env = self.env
        eef = env.eef_pos_env()
        tgt, gripper_open = self.machine.targets(eef, env.obj_pos_env(), env.place_target)

        # レート制限した中間目標（人参）。IK は局所的な線形化なので、
        # 20cm 先の目標を渡すのは、ヤコビアンが妥当な領域の外まで
        # 外挿させることになる。
        if self._carrot is None:
            self._carrot = eef.clone()
        delta = tgt - self._carrot
        norm = delta.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        self._carrot = self._carrot + delta * (self._carrot_step / norm).clamp(max=1.0)

        q = env.arm_q()
        q_des = self.ik.solve(q, self._carrot)
        arm_action = ((q_des - q) / env.action_scale).clamp(-1.0, 1.0)
        grip_action = torch.where(gripper_open, 1.0, -1.0).unsqueeze(1)
        return torch.cat([arm_action, grip_action], dim=-1)


def _ddim_alphas(diffusion_steps: int) -> np.ndarray:
    """squaredcos_cap_v2 のスケジュール（Diffusion Policy の既定）。"""
    s = 0.008
    t = np.arange(diffusion_steps + 1, dtype=np.float64) / diffusion_steps
    f = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    alphas_bar = f / f[0]
    betas = np.clip(1.0 - alphas_bar[1:] / alphas_bar[:-1], 0.0, 0.999)
    return np.cumprod(1.0 - betas)


def _ddim_sample(denoiser, obs_emb: np.ndarray, chunk: int, act_dim: int,
                 alphas_cumprod: np.ndarray, ddim_steps: int) -> np.ndarray:
    """書き出したイプシロン予測グラフを使う、決定的な DDIM サンプリング。"""
    B = obs_emb.shape[0]
    T = len(alphas_cumprod)
    ts = np.linspace(T - 1, 0, ddim_steps).round().astype(np.int64)
    rng = np.random.RandomState(0)
    x = rng.randn(B, chunk, act_dim).astype(np.float32)
    for i, t in enumerate(ts):
        a_t = alphas_cumprod[t]
        a_prev = alphas_cumprod[ts[i + 1]] if i + 1 < len(ts) else 1.0
        eps = denoiser.run(None, {"noisy_chunk": x,
                                  "timestep": np.full(B, t, dtype=np.int64),
                                  "obs_emb": obs_emb})[0]
        x0 = np.clip((x - math.sqrt(1.0 - a_t) * eps) / math.sqrt(a_t), -1.0, 1.0)
        x = math.sqrt(a_prev) * x0 + math.sqrt(1.0 - a_prev) * eps
    return np.clip(x, -1.0, 1.0).astype(np.float32)


class OnnxPolicy:
    """このパイプラインが配備する ONNX を全部こなす。グラフが宣言する
    入力とメタデータで振り分ける:

      * PPO の書き出し   : 入力は "obs" 1つ、行動が1つ出る
      * robomimic BC     : 入力は "flat"（+ 任意で "rgb"）
      * ACT 視覚運動     : "rgb" + "qpos" -> (B, chunk, act) のチャンク。
                                 後退ホライズンで実行する
      * Diffusion Policy : "rgb" + "qpos" のエンコーダ + 対になる
                                 *_denoiser.onnx、DDIM ループ -> 行動チャンク

    チャンク方式の方策は環境ごとに状態（バッファとカーソル）を持つので、
    エピソードの切れ目で ``reset_idx`` を呼ぶ。スクリプトエキスパートと同じ。
    """

    def __init__(self, onnx_path: str, env, execute_horizon: int | None = None):
        import onnxruntime as ort
        self.env = env
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_names = [i.name for i in self.sess.get_inputs()]
        meta = self.sess.get_modelmeta().custom_metadata_map or {}
        self.policy_type = meta.get(
            "policy_type", "visuomotor" if "qpos" in self.input_names else "state")
        self.chunk = int(meta.get("chunk", 0))
        horizon = execute_horizon if execute_horizon is not None \
            else int(meta.get("execute_horizon", 10))
        self.execute_horizon = min(horizon, self.chunk) if self.chunk else 0
        self.wants_rgb = "rgb" in self.input_names
        if self.wants_rgb and not env.with_camera:
            raise RuntimeError(f"{onnx_path} has an 'rgb' input; run in a camera-bearing mode")

        self.denoiser = None
        if self.policy_type == "diffusion":
            den_path = onnx_path.replace(".onnx", "_denoiser.onnx")
            if not os.path.exists(den_path):
                raise FileNotFoundError(f"diffusion policy needs {den_path}")
            self.denoiser = ort.InferenceSession(den_path, providers=["CPUExecutionProvider"])
            self.ddim_steps = int(meta.get("ddim_steps", 8))
            self.diffusion_steps = int(meta.get("diffusion_steps", 100))
            self.alphas_cumprod = _ddim_alphas(self.diffusion_steps)

        if self.chunk:
            n = env.num_envs
            self._buf = np.zeros((n, self.chunk, ACT_DIM), dtype=np.float32)
            self._cursor = np.full(n, self.chunk, dtype=np.int64)   # 空なら推論する

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        if self.chunk:
            self._cursor[np.asarray(env_ids.cpu())] = self.chunk

    def _feed_common(self) -> dict:
        feed = {}
        if self.wants_rgb:
            feed["rgb"] = self.env.camera_rgb_uint8().numpy().astype(np.float32)
        if "qpos" in self.input_names:
            feed["qpos"] = self.env.qpos_policy().cpu().numpy().astype(np.float32)
        return feed

    def _infer_chunk(self) -> np.ndarray:
        feed = self._feed_common()
        if self.policy_type == "diffusion":
            emb = self.sess.run(None, feed)[0]
            return _ddim_sample(self.denoiser, emb, self.chunk, ACT_DIM,
                                self.alphas_cumprod, self.ddim_steps)
        return self.sess.run(None, feed)[0]        # ACT: (B, chunk, act)

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        if not self.chunk:                          # 状態方策は1ステップ
            feed = self._feed_common()
            flat = "flat" if "flat" in self.input_names else (
                "obs" if "obs" in self.input_names else self.input_names[0])
            if flat not in feed:
                feed[flat] = obs.cpu().numpy().astype(np.float32)
            return torch.from_numpy(self.sess.run(None, feed)[0]).to(self.env.device)

        # 後退ホライズン。どれか1つの環境で使い切ったら、全部まとめて
        # 推論し直す。ここでのバッチ化は無料で、高いのはフレームの描画。
        # それは既に全環境ぶん済んでいる。
        if (self._cursor >= self.execute_horizon).any():
            fresh = self._infer_chunk()
            need = self._cursor >= self.execute_horizon
            self._buf[need] = fresh[need]
            self._cursor[need] = 0
        act = self._buf[np.arange(self.env.num_envs), self._cursor]
        self._cursor += 1
        return torch.from_numpy(act).to(self.env.device)
