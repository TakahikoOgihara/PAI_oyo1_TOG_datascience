# -*- coding: utf-8 -*-
"""Diffusion Policy（Chi et al., RSS 2023）の、ARM_BCRL005 向けの小さな
自己完結版（設定の切り替えは visuomotor.type=diffusion）。

役割は ACT の隣に置く比較と発展の経路。拡散方策は多峰の行動分布
（把持へ寄るやり方が複数ある状況）を、決定的な CVAE のデコードより
うまく表せる。代償は1回の判断につき `ddim_steps` 回のデノイザ順伝播で、
ACT の1回に対してエッジ側の計算が概ね1桁増える。だから 8GB の Jetson
Orin Nano で配備する既定は ACT の方。

構造（忠実かつ最小限）:
  * ObsEncoder: ResNet18（ACT とコードを共有）+ qpos の MLP -> 観測埋め込み
  * eps デノイザ: 行動チャンクに対する1次元時間 UNet。(正弦の時刻
    埋め込み + 観測埋め込み) で FiLM 条件付け
  * 学習: DDPM のイプシロン予測、squaredcos_cap_v2 のノイズスケジュール
  * 推論: 決定的な DDIM を ddim_steps 回

ONNX の配備はグラフ2つ。<name>.onnx（エンコーダ。rgb,qpos->obs_emb。
高い部分はチャンクにつき1回だけ走る）と <name>_denoiser.onnx（DDIM の
ループ本体）。利用側（シミュレータ評価の OnnxPolicy と
deploy/jetson/policy_runtime）は、ONNX のメタデータ
（diffusion_steps / ddim_steps）と下のスケジュール関数で歩調を合わせる。
"""

from __future__ import annotations
import math

import numpy as np
import torch
import torch.nn as nn

from .act import _ImagePreproc, _resnet18_backbone


def cosine_alphas_cumprod(diffusion_steps: int) -> np.ndarray:
    """squaredcos_cap_v2。mujoco_sim/train_arm_mujoco.py と
    deploy/jetson/policy_runtime.py の DDIM サンプラと必ず一致させること。"""
    s = 0.008
    t = np.arange(diffusion_steps + 1, dtype=np.float64) / diffusion_steps
    f = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    alphas_bar = f / f[0]
    betas = np.clip(1.0 - alphas_bar[1:] / alphas_bar[:-1], 0.0, 0.999)
    return np.cumprod(1.0 - betas)


class ObsEncoder(nn.Module):
    def __init__(self, qpos_dim: int, obs_embed_dim: int = 256,
                 require_pretrained_backbone: bool = True):
        super().__init__()
        self.preproc = _ImagePreproc()
        self.backbone = _resnet18_backbone(require_pretrained_backbone)
        self.head = nn.Sequential(
            nn.Linear(512 + qpos_dim, obs_embed_dim), nn.ReLU(),
            nn.Linear(obs_embed_dim, obs_embed_dim))

    def forward(self, rgb: torch.Tensor, qpos: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(self.preproc(rgb)).mean(dim=(2, 3))  # (B,512)
        return self.head(torch.cat([feat, qpos], dim=-1))


class _SinusoidalTimeEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        ang = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([ang.sin(), ang.cos()], dim=-1)


class _FiLMBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, cond_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, 3, padding=1)
        self.conv2 = nn.Conv1d(c_out, c_out, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, c_out)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.film = nn.Linear(cond_dim, 2 * c_out)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.norm1(self.conv1(x)))
        scale, shift = self.film(cond).unsqueeze(-1).chunk(2, dim=1)
        h = h * (1 + scale) + shift
        h = torch.relu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class EpsDenoiser(nn.Module):
    """1次元時間 UNet。(B, chunk, act) のノイズ入り行動 + 時刻 + obs_emb
    -> 予測したノイズ (B, chunk, act)。"""

    def __init__(self, act_dim: int, obs_embed_dim: int = 256,
                 down_dims=(256, 512), time_dim: int = 128):
        super().__init__()
        self.time_embed = nn.Sequential(
            _SinusoidalTimeEmbed(time_dim), nn.Linear(time_dim, time_dim), nn.ReLU())
        cond_dim = time_dim + obs_embed_dim
        d0, d1 = int(down_dims[0]), int(down_dims[1])
        self.enc1 = _FiLMBlock(act_dim, d0, cond_dim)
        self.down = nn.Conv1d(d0, d0, 3, stride=2, padding=1)
        self.enc2 = _FiLMBlock(d0, d1, cond_dim)
        self.mid = _FiLMBlock(d1, d1, cond_dim)
        self.up = nn.ConvTranspose1d(d1, d0, 4, stride=2, padding=1)
        self.dec1 = _FiLMBlock(d0 + d0, d0, cond_dim)
        self.out = nn.Conv1d(d0, act_dim, 1)

    def forward(self, noisy_chunk: torch.Tensor, timestep: torch.Tensor,
                obs_emb: torch.Tensor) -> torch.Tensor:
        cond = torch.cat([self.time_embed(timestep), obs_emb], dim=-1)
        x = noisy_chunk.transpose(1, 2)          # (B, act, K)
        h1 = self.enc1(x, cond)                  # (B, d0, K)
        h2 = self.enc2(self.down(h1), cond)      # (B, d1, K/2)
        h2 = self.mid(h2, cond)
        u = self.up(h2)                          # (B, d0, K)
        u = self.dec1(torch.cat([u, h1], dim=1), cond)
        return self.out(u).transpose(1, 2)       # (B, K, act)


class DiffusionPolicy(nn.Module):
    def __init__(self, qpos_dim: int, act_dim: int, chunk: int,
                 obs_embed_dim: int = 256, down_dims=(256, 512),
                 diffusion_steps: int = 100, ddim_steps: int = 8,
                 require_pretrained_backbone: bool = True):
        super().__init__()
        self.chunk = chunk
        self.act_dim = act_dim
        self.diffusion_steps = diffusion_steps
        self.ddim_steps = ddim_steps
        self.encoder = ObsEncoder(qpos_dim, obs_embed_dim,
                                  require_pretrained_backbone)
        self.denoiser = EpsDenoiser(act_dim, obs_embed_dim, down_dims)
        ac = torch.from_numpy(cosine_alphas_cumprod(diffusion_steps)).float()
        self.register_buffer("alphas_cumprod", ac)

    def loss(self, rgb, qpos, actions, is_pad):
        B = actions.shape[0]
        obs_emb = self.encoder(rgb, qpos)
        t = torch.randint(0, self.diffusion_steps, (B,), device=actions.device)
        a = self.alphas_cumprod[t].view(B, 1, 1)
        eps = torch.randn_like(actions)
        noisy = a.sqrt() * actions + (1 - a).sqrt() * eps
        eps_hat = self.denoiser(noisy, t, obs_emb)
        err = ((eps_hat - eps) ** 2 * (~is_pad).unsqueeze(-1)).sum() \
            / ((~is_pad).sum() * self.act_dim + 1e-6)
        return err, {"mse": float(err)}

    @torch.no_grad()
    def sample(self, rgb: torch.Tensor, qpos: torch.Tensor,
               generator: torch.Generator | None = None) -> torch.Tensor:
        """決定的な DDIM サンプリング（配備する numpy のループを torch で
        写したもの。検証用）。"""
        B = rgb.shape[0]
        obs_emb = self.encoder(rgb, qpos)
        ts = np.linspace(self.diffusion_steps - 1, 0, self.ddim_steps).round().astype(np.int64)
        x = torch.randn(B, self.chunk, self.act_dim, device=rgb.device, generator=generator)
        for i, t in enumerate(ts):
            a_t = float(self.alphas_cumprod[int(t)])
            a_prev = float(self.alphas_cumprod[int(ts[i + 1])]) if i + 1 < len(ts) else 1.0
            tt = torch.full((B,), int(t), device=rgb.device, dtype=torch.long)
            eps = self.denoiser(x, tt, obs_emb)
            x0 = (x - math.sqrt(1 - a_t) * eps) / math.sqrt(a_t)
            x0 = x0.clamp(-1.0, 1.0)
            x = math.sqrt(a_prev) * x0 + math.sqrt(1 - a_prev) * eps
        return x.clamp(-1.0, 1.0)
