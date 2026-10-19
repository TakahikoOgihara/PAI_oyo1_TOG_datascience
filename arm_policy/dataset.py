# -*- coding: utf-8 -*-
"""Chunked visuomotor dataset for ACT / Diffusion Policy training.

Reads the trajectory dict written by mujoco_sim/train_arm_mujoco.py
(--mode distill -> artifacts/data/visuomotor_demos.pt, or --mode collect ->
bc_demos.pt as the expert-source fallback):

  {"obs": [T_i, 31] float32 ..., "act": [T_i, 7] ..., "rgb": [T_i, H, W, 3] uint8 ...}

Each sample t of trajectory i yields:
  rgb    : (H, W, 3) float32 in [0, 255]  -- raw frame, augmented
  qpos   : (7,)  = [q_arm(6), prev_gripper_cmd(1)] -- the DEPLOYABLE
           proprioception (sliced out of the 31-dim sim obs; the real robot
           reports exactly this via pymycobot get_radians + the runtime's
           last gripper command)
  chunk  : (K, 7) actions act[t : t+K], padded by repeating the last action
  is_pad : (K,) bool padding mask (ACT masks the L1 loss with it)

LIGHT domain randomization happens here, at training time (the standard
low-cost practice): color jitter (brightness/contrast/saturation/hue),
RAD-style random crop-shift, and a small random ROLL (ARM_BCRL005). All three
cover "lighting / angle / perspective may vary a little" with ZERO extra sim
rendering -- which is what keeps the PDCA cycle short. The physical axes
(object mass / friction / actuator gains / object colour) are randomised in
the simulator instead; see configs/env/mycobot_pickplace.yaml domain_rand.
"""

from __future__ import annotations
import math

import numpy as np
import torch
from torch.utils.data import Dataset

# obs layout (mujoco_sim/train_arm_mujoco.py):
# [q(8), eef_pos(3), eef_quat(4), obj(3), obj-eef(3), place_target(3), prev_action(7)]
_QPOS_ARM = slice(0, 6)
_PREV_GRIP = slice(30, 31)


def qpos_from_obs(obs: torch.Tensor) -> torch.Tensor:
    return torch.cat([obs[..., _QPOS_ARM], obs[..., _PREV_GRIP]], dim=-1)


class ChunkDataset(Dataset):
    def __init__(self, demos: dict, chunk: int, aug_cfg=None, indices=None):
        if "rgb" not in demos:
            raise ValueError("visuomotor training needs camera frames -- the demo "
                             "file has no 'rgb' key (collect/distill with CAMERA_ON)")
        self.obs = demos["obs"]
        self.act = demos["act"]
        self.rgb = demos["rgb"]
        self.chunk = int(chunk)
        self.aug = aug_cfg if (aug_cfg is not None and bool(aug_cfg.enable)) else None
        traj_ids = range(len(self.obs)) if indices is None else indices
        self.index = [(i, t) for i in traj_ids for t in range(self.obs[i].shape[0])]

    def __len__(self):
        return len(self.index)

    def _augment(self, rgb_u8: torch.Tensor) -> torch.Tensor:
        """rgb_u8: (H, W, 3) uint8 -> float32 [0,255], jittered + shifted."""
        from torchvision.transforms import functional as TF
        a = self.aug
        img = rgb_u8.permute(2, 0, 1)  # CHW uint8
        if float(a.brightness) > 0:
            img = TF.adjust_brightness(img, 1.0 + (torch.rand(1).item() * 2 - 1) * float(a.brightness))
        if float(a.contrast) > 0:
            img = TF.adjust_contrast(img, 1.0 + (torch.rand(1).item() * 2 - 1) * float(a.contrast))
        if float(a.saturation) > 0:
            img = TF.adjust_saturation(img, 1.0 + (torch.rand(1).item() * 2 - 1) * float(a.saturation))
        if float(a.hue) > 0:
            img = TF.adjust_hue(img, (torch.rand(1).item() * 2 - 1) * float(a.hue))
        # ARM_BCRL005: small random ROLL. The wrist camera is bolted to a
        # flange by hand and a few degrees of roll is normal -- and roll is
        # precisely what a single wrist camera cannot recover from, because
        # its only depth cue is how the scene looks. Replicate-padding by
        # ceil(h * sin(max_angle)) before rotating guarantees the rotated
        # centre crop never contains a black corner, so the policy never gets
        # to learn "black corner => I am near the edge of an augmentation".
        rot = float(a.get("rotate_deg", 0.0) or 0.0)
        if rot > 0:
            ang = (torch.rand(1).item() * 2 - 1) * rot
            _, h, w = img.shape
            m = int(math.ceil(max(h, w) * math.sin(math.radians(rot)))) + 2
            padded = torch.nn.functional.pad(
                img.float().unsqueeze(0), (m, m, m, m), mode="replicate").squeeze(0)
            padded = TF.rotate(padded, ang,
                               interpolation=TF.InterpolationMode.BILINEAR)
            img = TF.center_crop(padded, [h, w])

        # RAD-style random shift: replicate-pad then crop back at a random offset
        s = int(a.shift_px)
        if s > 0:
            _, h, w = img.shape
            img = torch.nn.functional.pad(img.float().unsqueeze(0), (s, s, s, s),
                                          mode="replicate").squeeze(0)
            ox = int(torch.randint(0, 2 * s + 1, (1,)).item())
            oy = int(torch.randint(0, 2 * s + 1, (1,)).item())
            img = img[:, oy:oy + h, ox:ox + w]
        return img.float().permute(1, 2, 0).contiguous()

    def __getitem__(self, idx):
        i, t = self.index[idx]
        obs_i, act_i, rgb_i = self.obs[i], self.act[i], self.rgb[i]
        T = act_i.shape[0]
        K = self.chunk

        rgb = self._augment(rgb_i[t]) if self.aug is not None else rgb_i[t].float()
        qpos = qpos_from_obs(obs_i[t].float())

        end = min(t + K, T)
        chunk = act_i[t:end].float()
        n_pad = K - chunk.shape[0]
        is_pad = torch.zeros(K, dtype=torch.bool)
        if n_pad > 0:
            chunk = torch.cat([chunk, chunk[-1:].repeat(n_pad, 1)], dim=0)
            is_pad[K - n_pad:] = True
        return {"rgb": rgb, "qpos": qpos, "chunk": chunk, "is_pad": is_pad}


def train_val_split(demos: dict, val_split: float, seed: int):
    """Split by TRAJECTORY (never by frame -- frames within a trajectory are
    heavily correlated, frame-level splits leak)."""
    n = len(demos["obs"])
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_split))
    return list(perm[n_val:]), list(perm[:n_val])
