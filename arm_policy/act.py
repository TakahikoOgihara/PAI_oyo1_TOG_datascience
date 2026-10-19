# -*- coding: utf-8 -*-
"""ACT（Action Chunking with Transformers、Zhao et al., RSS 2023）の、
ARM_BCRL005 向けの小さな自己完結実装。

lerobot の学習器を取り込まず自前で持つ理由。配備する成果物は、Jetson
Orin Nano の TensorRT が食える綺麗な単一グラフの ONNX（rgb + qpos ->
行動チャンク）でなければならない。約250行を自分で持てば書き出しは
決定的になり、動きの速い依存を1つも固定しなくて済む（ARM_BCRL001 が
LeRobot のフェーズを任意にしていたのと同じ理由）。構造は論文どおり。
ResNet18 の視覚バックボーン -> 画像トークン（+ 自己受容トークン +
CVAE の潜在トークン）に対する transformer エンコーダ -> `chunk` 個の
学習クエリを持つ transformer デコーダ -> ステップごとの行動ヘッド。
学習は L1 再構成 + KL(z || N(0,1))。推論では z = 0（CVAE の平均）を
使い、1チャンクにつき順伝播1回。

画像の前処理（/255 + ImageNet 正規化 + HWC->CHW）は forward() に
焼き込んである。だからシミュレータ評価、ONNX の検証、Jetson の
ランタイムが、どれも生の uint8 スケールの画像を渡せる。
"""

from __future__ import annotations


import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _resnet18_backbone(require_pretrained: bool = True) -> nn.Module:
    """torchvision の ResNet18 を layer4 まで（B,512,H/32,W/32）。ImageNet 事前学習。

    ARM_BCRL005 での変更（実機の失敗記録の原因Hに対応）:
    ARM_BCRL002 は、torch hub に届かないとき警告を出して黙って
    ランダム初期化に落ちていた。オフラインのGPUコンテナでは、それが
    普通の状態。ランダム初期化の ResNet18 は、手首カメラの軌跡
    数百本では収束まで学習できない。つまりその退避動作は、配備する
    方策を黙って弱いものに変えるのに、パイプラインの指標は全部
    「問題なし」に見える。出荷する成果物が静かに劣化する既定値は、
    受け入れられない。

    setup_env.sh が重みをあらかじめキャッシュする。
    require_pretrained=False（visuomotor.train.require_pretrained_backbone）
    は、意図的な「事前学習なし」の比較のときだけ渡す。
    """
    import torchvision
    try:
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        net = torchvision.models.resnet18(weights=weights)
        print("[act] ResNet18 backbone: ImageNet-pretrained")
    except Exception as e:  # オフライン、または hub に届かない
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
    """(B,H,W,3) の [0,255] float -> 正規化した (B,3,H,W)。ONNX に焼き込む。"""

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

        # CVAE のエンコーダ（学習時のみ。ONNX の書き出しでは落とす）。
        # 正解の行動チャンクから、作法を表す潜在 z を推定する
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
        """学習時（actions あり）は (a_hat, mu, logvar) を返す。
        推論時（actions が None）は z = 0 で a_hat を返す。"""
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
    """ONNX 書き出し用のラッパー。(rgb, qpos) -> 行動チャンク、z = 0（CVAE の平均）。
    出力は環境の行動範囲にクランプする。"""

    def __init__(self, policy: ACTPolicy):
        super().__init__()
        self.policy = policy

    def forward(self, rgb: torch.Tensor, qpos: torch.Tensor) -> torch.Tensor:
        a_hat, _, _ = self.policy(rgb, qpos, actions=None)
        return a_hat.clamp(-1.0, 1.0)
