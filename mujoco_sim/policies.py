# -*- coding: utf-8 -*-
"""The two things that produce actions besides a learning algorithm:

  * ``ExpertController`` -- the scripted IK pick&place expert that generates
    phase-1 demonstrations (and is the MimicGen source).
  * ``OnnxPolicy``       -- serves every ONNX this pipeline deploys, so sim
    evaluation runs the SAME graph the edge runtime will.

Both are ports of ARM_BCRL005's classes with one substitution: Isaac Lab's
``DifferentialIKController`` + PhysX Jacobian becomes ``arm_common.ik``'s
damped least squares on the analytic Jacobian. The phase machine
(``arm_common/expert.py``), the carrot rate limiter, the action encoding and
the chunked receding-horizon execution are unchanged.
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
    """Cartesian waypoints from the phase machine -> joint-delta actions.

    Division of labour, unchanged from ARM_BCRL005:
      PickPlacePhaseMachine  "which stage of pick&place am I in, and where
                              should the TCP go / should the gripper be open"
      ExpertController       "turn that TCP target into the 7-dim action the
                              environment and the real robot both accept"
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
        self._carrot_step = float(e.carrot_step)      # m per control step

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        self.machine.reset_idx(env_ids)
        if self._carrot is not None:
            self._carrot[env_ids] = self.env.eef_pos_env()[env_ids]

    def act(self, obs=None) -> torch.Tensor:
        env = self.env
        eef = env.eef_pos_env()
        tgt, gripper_open = self.machine.targets(eef, env.obj_pos_env(), env.place_target)

        # rate-limited intermediate target ("carrot"): the IK is a LOCAL
        # linearisation, so handing it a target 20 cm away asks it to
        # extrapolate far outside the region where the Jacobian is valid.
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
    """squaredcos_cap_v2 schedule (the Diffusion Policy default)."""
    s = 0.008
    t = np.arange(diffusion_steps + 1, dtype=np.float64) / diffusion_steps
    f = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    alphas_bar = f / f[0]
    betas = np.clip(1.0 - alphas_bar[1:] / alphas_bar[:-1], 0.0, 0.999)
    return np.cumprod(1.0 - betas)


def _ddim_sample(denoiser, obs_emb: np.ndarray, chunk: int, act_dim: int,
                 alphas_cumprod: np.ndarray, ddim_steps: int) -> np.ndarray:
    """Deterministic DDIM sampling with the exported epsilon-denoiser graph."""
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
    """Serves every ONNX this pipeline deploys, dispatching on the graph's
    declared inputs and metadata:

      * PPO export             : single "obs" input, one action out
      * robomimic BC export    : "flat" (+ optional "rgb") inputs
      * ACT visuomotor export  : "rgb" + "qpos" -> (B, chunk, act) chunk,
                                 executed with a receding horizon
      * Diffusion Policy export: "rgb" + "qpos" encoder + companion
                                 *_denoiser.onnx, DDIM loop -> action chunk

    Chunked policies are stateful per env (buffer + cursor); call ``reset_idx``
    on episode boundaries, exactly like the scripted expert.
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
            self._cursor = np.full(n, self.chunk, dtype=np.int64)   # empty -> infer

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
        if not self.chunk:                          # state policies: single-step
            feed = self._feed_common()
            flat = "flat" if "flat" in self.input_names else (
                "obs" if "obs" in self.input_names else self.input_names[0])
            if flat not in feed:
                feed[flat] = obs.cpu().numpy().astype(np.float32)
            return torch.from_numpy(self.sess.run(None, feed)[0]).to(self.env.device)

        # receding horizon: re-infer the whole batch whenever ANY env ran out.
        # Batching is free here -- the expensive part is rendering the frames,
        # which already happened for every env.
        if (self._cursor >= self.execute_horizon).any():
            fresh = self._infer_chunk()
            need = self._cursor >= self.execute_horizon
            self._buf[need] = fresh[need]
            self._cursor[need] = 0
        act = self._buf[np.arange(self.env.num_envs), self._cursor]
        self._cursor += 1
        return torch.from_numpy(act).to(self.env.device)
