# -*- coding: utf-8 -*-
"""ACT -- Action Chunking with Transformers (Zhao et al., RSS 2023) -- compact
self-contained implementation for ARM_BCRL005.

Why self-contained instead of importing lerobot's trainer: the deployed
artifact must be a clean single-graph ONNX (rgb + qpos -> action chunk) that
TensorRT on the Jetson Orin Nano consumes; owning the ~250 lines keeps the
export deterministic and pins zero fast-moving dependencies (the same reason
ARM_BCRL001 kept its LeRobot phase optional). The architecture follows the
paper: ResNet18 visual backbone -> transformer encoder over image tokens (+
proprioception token + CVAE latent token) -> transformer decoder with `chunk`
learned queries -> per-step action head. Trained with L1 reconstruction +
KL(z || N(0,1)); inference uses z = 0 (the CVAE mean), one forward pass per
chunk.

Image preprocessing (/255 + ImageNet normalization + HWC->CHW) is BAKED INTO
forward(), so sim eval, ONNX validation and the Jetson runtime all feed raw
uint8-scale frames -- the ARM_BCRL001 export convention.
"""

from __future__ import annotations


import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _resnet18_backbone(require_pretrained: bool = True) -> nn.Module:
    """torchvision ResNet18 up to layer4 (B,512,H/32,W/32), ImageNet-pretrained.

    ARM_BCRL005 change (field-failure notes §2, cause H):
    ARM_BCRL002 printed a WARNING and silently fell back to random weights
    when the torch hub was unreachable -- which is the normal state of an
    offline GPU container. A randomly-initialised ResNet18 cannot be trained
    to convergence on a few hundred wrist-camera trajectories, so that
    fallback quietly turns the deployed policy into a much weaker one while
    every metric in the pipeline still looks "fine". A silent quality
    downgrade in the artifact that ships is not an acceptable default.

    setup_env.sh pre-caches the weights into the container. Pass
    require_pretrained=False (visuomotor.train.require_pretrained_backbone)
    only for a deliberate no-pretraining ablation.
    """
    import torchvision
    try:
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        net = torchvision.models.resnet18(weights=weights)
        print("[act] ResNet18 backbone: ImageNet-pretrained")
    except Exception as e:  # offline / hub unreachable
        if require_pretrained:
            raise RuntimeError(
                f"ImageNet weights for the ResNet18 backbone are unavailable ({e}).\n"
                "  The visuomotor policy trains on a few hundred trajectories; a\n"
                "  from-scratch backbone silently produces a much weaker deployed\n"
                "  model. Fix it, do not ignore it:\n"
                "    docker exec isaac-lab-train /isaac-sim/IsaacLab/_isaac_sim/python.sh \\\n"
                "      -c \"import torchvision as tv; tv.models.resnet18("
                "weights=tv.models.ResNet18_Weights.IMAGENET1K_V1)\"\n"
                "  (setup_env.sh step 8 does this). To train WITHOUT pretraining on\n"
                "  purpose: visuomotor.train.require_pretrained_backbone=false") from e
        print(f"[act] WARNING: ImageNet weights unavailable ({e}); random init "
              "(explicitly allowed by visuomotor.train.require_pretrained_backbone=false)")
        net = torchvision.models.resnet18(weights=None)
    return nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                         net.layer1, net.layer2, net.layer3, net.layer4)


class _ImagePreproc(nn.Module):
    """(B,H,W,3) float in [0,255] -> normalized (B,3,H,W). Baked into ONNX."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        x = rgb.permute(0, 3, 1, 2) / 255.0
        return (x - self.mean) / self.std


class ACTPolicy(nn.Module):
    def __init__(self, qpos_dim: int, act_dim: int, chunk: int,
                 dim_model: int = 256, n_heads: int = 8, dim_feedforward: int = 1024,
                 enc_layers: int = 3, dec_layers: int = 3, latent_dim: int = 32,
                 kl_weight: float = 10.0, dropout: float = 0.1, image_size: int = 224,
                 require_pretrained_backbone: bool = True):
        super().__init__()
        self.chunk = chunk
        self.act_dim = act_dim
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight

        self.preproc = _ImagePreproc()
        self.backbone = _resnet18_backbone(require_pretrained_backbone)
        self.input_proj = nn.Conv2d(512, dim_model, kernel_size=1)
        n_tokens = (image_size // 32) ** 2
        self.img_pos = nn.Parameter(torch.randn(1, n_tokens, dim_model) * 0.02)

        self.qpos_proj = nn.Linear(qpos_dim, dim_model)
        self.latent_proj = nn.Linear(latent_dim, dim_model)

        enc_layer = nn.TransformerEncoderLayer(dim_model, n_heads, dim_feedforward,
                                               dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, enc_layers)
        dec_layer = nn.TransformerDecoderLayer(dim_model, n_heads, dim_feedforward,
                                               dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, dec_layers)
        self.query_embed = nn.Parameter(torch.randn(1, chunk, dim_model) * 0.02)
        self.action_head = nn.Linear(dim_model, act_dim)

        # CVAE encoder (training only; ONNX export drops it): infers the
        # style latent z from the ground-truth action chunk
        self.cvae_act_proj = nn.Linear(act_dim, dim_model)
        self.cvae_qpos_proj = nn.Linear(qpos_dim, dim_model)
        self.cvae_cls = nn.Parameter(torch.randn(1, 1, dim_model) * 0.02)
        self.cvae_pos = nn.Parameter(torch.randn(1, chunk + 2, dim_model) * 0.02)
        cvae_layer = nn.TransformerEncoderLayer(dim_model, n_heads, dim_feedforward,
                                                dropout, batch_first=True)
        self.cvae_encoder = nn.TransformerEncoder(cvae_layer, 2)
        self.cvae_head = nn.Linear(dim_model, 2 * latent_dim)

    # -------------------------------------------------------------- pieces
    def _encode_obs(self, rgb: torch.Tensor, qpos: torch.Tensor,
                    z: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(self.preproc(rgb))                 # (B,512,h,w)
        tok = self.input_proj(feat).flatten(2).transpose(1, 2)  # (B,hw,D)
        tok = tok + self.img_pos
        qtok = self.qpos_proj(qpos).unsqueeze(1)                # (B,1,D)
        ztok = self.latent_proj(z).unsqueeze(1)                 # (B,1,D)
        memory = torch.cat([ztok, qtok, tok], dim=1)
        return self.encoder(memory)

    def _decode(self, memory: torch.Tensor) -> torch.Tensor:
        q = self.query_embed.expand(memory.shape[0], -1, -1)
        h = self.decoder(q, memory)
        return self.action_head(h)                              # (B,chunk,act)

    def _cvae_latent(self, qpos: torch.Tensor, actions: torch.Tensor):
        B = actions.shape[0]
        seq = torch.cat([self.cvae_cls.expand(B, -1, -1),
                         self.cvae_qpos_proj(qpos).unsqueeze(1),
                         self.cvae_act_proj(actions)], dim=1)
        h = self.cvae_encoder(seq + self.cvae_pos)[:, 0]
        mu, logvar = self.cvae_head(h).chunk(2, dim=-1)
        return mu, logvar

    # -------------------------------------------------------------- api
    def forward(self, rgb: torch.Tensor, qpos: torch.Tensor,
                actions: torch.Tensor | None = None):
        """Training (actions given): returns (a_hat, mu, logvar).
        Inference (actions None): returns a_hat with z = 0."""
        B = rgb.shape[0]
        if actions is not None:
            mu, logvar = self._cvae_latent(qpos, actions)
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        else:
            mu = logvar = None
            z = torch.zeros(B, self.latent_dim, device=rgb.device, dtype=rgb.dtype)
        a_hat = self._decode(self._encode_obs(rgb, qpos, z))
        return a_hat, mu, logvar

    def loss(self, rgb, qpos, actions, is_pad):
        a_hat, mu, logvar = self.forward(rgb, qpos, actions)
        l1 = (a_hat - actions).abs()
        l1 = (l1 * (~is_pad).unsqueeze(-1)).sum() / ((~is_pad).sum() * self.act_dim + 1e-6)
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()
        return l1 + self.kl_weight * kl, {"l1": float(l1), "kl": float(kl)}


class ACTInference(nn.Module):
    """ONNX export wrapper: (rgb, qpos) -> action chunk, z = 0 (CVAE mean).
    Output is clamped to the env's action range."""

    def __init__(self, policy: ACTPolicy):
        super().__init__()
        self.policy = policy

    def forward(self, rgb: torch.Tensor, qpos: torch.Tensor) -> torch.Tensor:
        a_hat, _, _ = self.policy(rgb, qpos, actions=None)
        return a_hat.clamp(-1.0, 1.0)
