# -*- coding: utf-8 -*-
"""Diffusion Policy (Chi et al., RSS 2023) -- compact self-contained variant
for ARM_BCRL005 (config switch: visuomotor.type=diffusion).

Role: the comparison/upgrade path next to ACT. Diffusion policies model
MULTIMODAL action distributions (several valid ways to approach a grasp)
better than a deterministic CVAE decode, at the cost of `ddim_steps`
denoiser forwards per decision -- roughly an order of magnitude more edge
compute than ACT's single pass, which is exactly why ACT is the deployed
default on the 8 GB Jetson Orin Nano (design notes).

Structure (faithful, minimal):
  * ObsEncoder: ResNet18 (shared code with ACT) + qpos MLP -> obs embedding
  * eps-denoiser: 1D temporal UNet over the action chunk, FiLM-conditioned
    on (sinusoidal timestep embedding + obs embedding)
  * training: DDPM epsilon-prediction, squaredcos_cap_v2 noise schedule
  * inference: deterministic DDIM with ddim_steps steps

ONNX deployment is TWO graphs -- <name>.onnx (encoder: rgb,qpos->obs_emb; the
expensive part runs ONCE per chunk) and <name>_denoiser.onnx (the DDIM loop
body). The consumers (sim eval's OnnxPolicy and deploy/jetson/policy_runtime)
share the same schedule via the ONNX metadata (diffusion_steps / ddim_steps)
and the schedule function below.
"""

from __future__ import annotations
import math

import numpy as np
import torch
import torch.nn as nn

from .act import _ImagePreproc, _resnet18_backbone


def cosine_alphas_cumprod(diffusion_steps: int) -> np.ndarray:
    """squaredcos_cap_v2 -- MUST stay identical to the DDIM samplers in
    mujoco_sim/train_arm_mujoco.py and deploy/jetson/policy_runtime.py."""
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
    """1D temporal UNet: (B, chunk, act) noisy actions + timestep + obs_emb
    -> predicted noise (B, chunk, act)."""

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
        """Deterministic DDIM sampling (torch mirror of the deployed numpy
        loops -- used for validation)."""
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
