# -*- coding: utf-8 -*-
"""シミュレータ用の、バッチ対応（torch）の順運動学。Raspberry Pi が
強制するのと同じ安全包絡。

``deploy/raspi/mycobot_fk.py`` が、連鎖とツールのサンプル点の正本で、
依存ゼロで書いてある（ハードウェアのガードの中で走るので、そうでないと
いけない）。このモジュールはその定数をそのまま取り込み、ベクトル化した
torch 実装を足す。それで ``mujoco_sim/train_arm_mujoco.py`` は、同じ
仮想床を num_envs 個の環境へ一度に適用できる。

シミュレータ側にもガードが要る理由。ロボットにしか無いフェイルセーフは、
方策が学習中に一度も当たらないフェイルセーフで、配備した方策は
見たことのない制限と戦い続けることになる。両方で強制すれば、方策が
学ぶ制御の契約が、実機が走らせる契約とバイト単位で同じになる。

``assert_matches_reference()`` が2つの実装の一致を証明する。環境の構築時
に1回走るので、ずれたら黙って実機とのギャップが広がるのではなく、
その場で失敗する。
"""

from __future__ import annotations
import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RASPI_DIR = os.path.join(_REPO_ROOT, "deploy", "raspi")
if _RASPI_DIR not in sys.path:
    sys.path.append(_RASPI_DIR)

from mycobot_fk import (  # noqa: E402  -- 正本は1つ
    CHAIN, TOOL_POINTS, TCP_Z, HOME_Q, NUM_ARM_JOINTS,
    _rpy as _rpy_rowmajor, tool_min_z as _tool_min_z_ref,
)

__all__ = ["ToolKinematics", "TCP_Z", "HOME_Q", "NUM_ARM_JOINTS"]


class ToolKinematics:
    """ベクトル化したツールFK。(N, 6) の関節角 -> ツールのサンプル点の高さ。"""

    def __init__(self, device, dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        # 固定の関節原点の変換、(6, 4, 4)
        origins = torch.zeros(len(CHAIN), 4, 4, dtype=dtype)
        for i, (xyz, rpy) in enumerate(CHAIN):
            # 回転は正本の実装が作ったものを使う。URDF の rpy の規約で
            # 2つが食い違うことがあり得ないように
            origins[i, :3, :3] = torch.tensor(_rpy_rowmajor(*rpy), dtype=dtype).view(3, 3)
            origins[i, :3, 3] = torch.tensor(xyz, dtype=dtype)
            origins[i, 3, 3] = 1.0
        self.origins = origins.to(device)
        # フランジ座標でのツールのサンプル点、(P, 3)
        self.tool_pts = torch.tensor(TOOL_POINTS, dtype=dtype, device=device)
        self.tcp_local = torch.tensor([0.0, 0.0, TCP_Z], dtype=dtype, device=device)

    def flange_pose(self, q: torch.Tensor) -> torch.Tensor:
        """q: (N, 6) rad -> 基準座標での joint6_flange の姿勢 (N, 4, 4)。"""
        n = q.shape[0]
        T = torch.eye(4, dtype=self.dtype, device=self.device).expand(n, 4, 4).contiguous()
        for i in range(NUM_ARM_JOINTS):
            c, s = torch.cos(q[:, i]), torch.sin(q[:, i])
            rot = torch.zeros(n, 4, 4, dtype=self.dtype, device=self.device)
            rot[:, 0, 0], rot[:, 0, 1] = c, -s
            rot[:, 1, 0], rot[:, 1, 1] = s, c
            rot[:, 2, 2] = 1.0
            rot[:, 3, 3] = 1.0
            T = T @ self.origins[i] @ rot
        return T

    def tool_points(self, q: torch.Tensor) -> torch.Tensor:
        """-> ロボット基準座標でのサンプル点 (N, P, 3)。"""
        T = self.flange_pose(q)
        return (self.tool_pts @ T[:, :3, :3].transpose(1, 2)) + T[:, None, :3, 3]

    def tool_min_z(self, q: torch.Tensor) -> torch.Tensor:
        """-> 基準面から見た、ツール最下点の高さ (N,)。"""
        return self.tool_points(q)[..., 2].min(dim=-1).values

    def tcp_pos(self, q: torch.Tensor) -> torch.Tensor:
        T = self.flange_pose(q)
        return (self.tcp_local @ T[:, :3, :3].transpose(1, 2)) + T[:, :3, 3]

    # ------------------------------------------------------------------
    def limit_to_floor(self, q_from: torch.Tensor, q_to: torch.Tensor,
                       min_z: float, iters: int = 8):
        """ツールが ``min_z`` を確保するまで、q_from -> q_to の各行を短くする。

        アルゴリズムは ``mycobot_fk.FloorGuard.limit`` と同一（関節空間の
        線分上の二分探索）で、それを全環境ぶん一度に評価するだけ。
        返すのは (q_safe, blocked_mask)。
        """
        unsafe = self.tool_min_z(q_to) < min_z
        if not bool(unsafe.any()):
            return q_to, unsafe
        start_unsafe = self.tool_min_z(q_from) < min_z
        lo = torch.zeros_like(unsafe, dtype=q_to.dtype)
        hi = torch.ones_like(lo)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            q_mid = q_from + (q_to - q_from) * mid.unsqueeze(-1)
            ok = self.tool_min_z(q_mid) >= min_z
            lo = torch.where(ok, mid, lo)
            hi = torch.where(ok, hi, mid)
        q_cut = q_from + (q_to - q_from) * lo.unsqueeze(-1)
        # すでに面より下（校正ずれ、または押され中）。その場で保持する
        q_cut = torch.where(start_unsafe.unsqueeze(-1), q_from, q_cut)
        return torch.where(unsafe.unsqueeze(-1), q_cut, q_to), unsafe

    # ------------------------------------------------------------------
    # ARM_BCRL006 での追加。解析的な TCP ヤコビアン。
    #
    # ARM_BCRL005 は PhysX に多関節ヤコビアン
    # （``robot.root_physx_view.get_jacobians()``）を聞いて、Isaac Lab の
    # DifferentialIKController に渡していた。Isaac Sim の外にはどちらも
    # 無い。MuJoCo に ``mj_jacSite`` を聞くと、仮想床と MimicGen が既に
    # 自前で計算している輪の中に、3本目の運動学連鎖（MuJoCo のもの）を
    # 持ち込むことになる。
    #
    # 回転関節の直列連鎖なら、位置のヤコビアンは厳密で2行で書ける。
    # J_i = z_i x (p_tcp - p_i)。z_i / p_i は基準座標での関節 i の軸と
    # 原点。床ガードと同じ連鎖から導けば、IK、ガード、MimicGen の姿勢
    # 転写、そして実機が、全部同じアームを動かす。
    # ``assert_matches_reference`` が差分近似と突き合わせ、
    # assets/build_mycobot_mjcf.py が連鎖そのものを MuJoCo と
    # 突き合わせる。
    # ------------------------------------------------------------------
    def tcp_jacobian(self, q: torch.Tensor) -> torch.Tensor:
        """q: (N, 6) -> (N, 3, 6) の d(TCP位置) / d(関節角)。"""
        n = q.shape[0]
        T = torch.eye(4, dtype=self.dtype, device=self.device).expand(n, 4, 4).contiguous()
        axes, origins = [], []
        for i in range(NUM_ARM_JOINTS):
            T = T @ self.origins[i]
            axes.append(T[:, :3, 2])            # 基準座標での関節軸（ローカルの +Z）
            origins.append(T[:, :3, 3])
            c, s = torch.cos(q[:, i]), torch.sin(q[:, i])
            rot = torch.zeros(n, 4, 4, dtype=self.dtype, device=self.device)
            rot[:, 0, 0], rot[:, 0, 1] = c, -s
            rot[:, 1, 0], rot[:, 1, 1] = s, c
            rot[:, 2, 2] = 1.0
            rot[:, 3, 3] = 1.0
            T = T @ rot
        p_tcp = (self.tcp_local @ T[:, :3, :3].transpose(1, 2)) + T[:, :3, 3]
        cols = [torch.cross(z, p_tcp - p, dim=-1) for z, p in zip(axes, origins)]
        return torch.stack(cols, dim=-1)        # (N, 3, 6)

    # ------------------------------------------------------------------
    def assert_matches_reference(self, tol: float = 1e-5) -> None:
        """torch の FK が Pi の FK からずれたら、その場で失敗させる。"""
        probes = [list(HOME_Q),
                  [0.0] * NUM_ARM_JOINTS,
                  [0.25, -1.6, -1.0, 0.2, 0.0, 0.0],
                  [-0.4, -0.8, 0.6, -1.1, 0.3, 0.7]]
        q = torch.tensor(probes, dtype=self.dtype, device=self.device)
        mine = self.tool_min_z(q).cpu().tolist()
        ref = [_tool_min_z_ref(p) for p in probes]
        for got, want, p in zip(mine, ref, probes):
            if abs(got - want) > tol:
                raise AssertionError(
                    "tool FK mismatch between arm_common/kinematics.py and "
                    f"deploy/raspi/mycobot_fk.py at q={p}: {got:.6f} vs {want:.6f}. "
                    "The simulator and the robot guard MUST share one envelope.")
        # 解析ヤコビアンを、同じ連鎖上の中心差分と突き合わせる。
        # float64 で行う。float32 では差分商自身の丸め誤差
        # （約 1e-8 m / 2e-5 rad ≒ 1e-3）が、確認したい量を飲み込んで
        # しまい、自分のノイズを測るだけになる。
        kin64 = ToolKinematics(self.device, dtype=torch.float64)
        q64 = q.to(torch.float64)
        eps = 1e-6
        j_an = kin64.tcp_jacobian(q64)
        j_fd = torch.zeros_like(j_an)
        for i in range(NUM_ARM_JOINTS):
            dq = torch.zeros_like(q64)
            dq[:, i] = eps
            j_fd[:, :, i] = (kin64.tcp_pos(q64 + dq) - kin64.tcp_pos(q64 - dq)) / (2 * eps)
        jerr = float((j_an - j_fd).abs().max())
        if jerr > 1e-7:
            raise AssertionError(
                f"tcp_jacobian disagrees with finite differences by {jerr:.2e}; "
                "the IK would drive the arm along the wrong direction.")
        print(f"[kinematics] tool-FK agrees with deploy/raspi/mycobot_fk.py "
              f"on {len(probes)} poses (max err < {tol}); analytic TCP Jacobian "
              f"matches finite differences (max err {jerr:.1e})")
