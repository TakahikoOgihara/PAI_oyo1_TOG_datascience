# -*- coding: utf-8 -*-
"""ACT / Diffusion Policy 学習用の、チャンク化した視覚運動データセット。

mujoco_sim/train_arm_mujoco.py が書いた軌跡の辞書を読む
（--mode distill -> artifacts/data/visuomotor_demos.pt、または
--mode collect -> エキスパート元データとしての bc_demos.pt）:

  {"obs": [T_i, 31] float32 ..., "act": [T_i, 7] ..., "rgb": [T_i, H, W, 3] uint8 ...}

軌跡 i の各標本 t が返すもの:
  rgb    : (H, W, 3) float32 の [0, 255]。生の画像に増補をかけたもの
  qpos   : (7,) = [q_arm(6), prev_gripper_cmd(1)]。配備できる自己受容
           感覚（31次元の観測から切り出す。実機は pymycobot の
           get_radians と、ランタイムが最後に出したグリッパ指令で
           これと全く同じものを報告する）
  chunk  : (K, 7) の行動 act[t : t+K]。足りない分は最後の行動で埋める
  is_pad : (K,) bool の埋めマスク（ACT は L1 損失をこれで覆う）

軽いドメインランダム化はここ、つまり学習時にかける（低コストで標準的な
やり方）。色の揺らぎ（明度・コントラスト・彩度・色相）、RAD 流の
ランダムな切り出しずらし、小さなランダムロール（ARM_BCRL005）。
3つとも「照明・角度・遠近は少し変わり得る」を、シミュレータの描画を
1枚も増やさずに覆う。それが PDCA の周期を短く保っている。物理側の軸
（ワークの質量・摩擦・アクチュエータのゲイン・ワークの色）は
シミュレータでランダム化する。configs/env/mycobot_pickplace.yaml を参照。
"""

from __future__ import annotations
import math

import numpy as np
import torch
from torch.utils.data import Dataset

# 観測の並び（mujoco_sim/train_arm_mujoco.py）:
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
        """rgb_u8: (H, W, 3) uint8 -> float32 [0,255]。色を揺らし、ずらす。"""
        from torchvision.transforms import functional as TF
        a = self.aug
        img = rgb_u8.permute(2, 0, 1)  # CHW の uint8
        if float(a.brightness) > 0:
            img = TF.adjust_brightness(img, 1.0 + (torch.rand(1).item() * 2 - 1) * float(a.brightness))
        if float(a.contrast) > 0:
            img = TF.adjust_contrast(img, 1.0 + (torch.rand(1).item() * 2 - 1) * float(a.contrast))
        if float(a.saturation) > 0:
            img = TF.adjust_saturation(img, 1.0 + (torch.rand(1).item() * 2 - 1) * float(a.saturation))
        if float(a.hue) > 0:
            img = TF.adjust_hue(img, (torch.rand(1).item() * 2 - 1) * float(a.hue))
        # ARM_BCRL005 で入れた、小さなランダムロール。手首カメラは手で
        # フランジに留めるので、数度のロールは普通に起きる。しかもロールは、
        # 単眼の手首カメラが復元できない量。奥行きの手がかりがシーンの
        # 見え方だけだから。回す前に ceil(h * sin(max_angle)) ぶん端の値で
        # 埋めておけば、回転後の中央切り出しに黒い角が絶対に入らない。
        # だから方策が「黒い角 = 増補の端にいる」を学ぶ余地も無くなる。
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

        # RAD 流のランダムずらし。端の値で埋めてから、ランダムな位置で切り直す
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
    """分割は軌跡単位で行う（フレーム単位では絶対にやらない。1本の軌跡の
    中のフレームは強く相関していて、フレーム単位の分割は漏れる）。"""
    n = len(demos["obs"])
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_split))
    return list(perm[n_val:]), list(perm[:n_val])
