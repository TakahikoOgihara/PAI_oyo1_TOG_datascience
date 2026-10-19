# -*- coding: utf-8 -*-
"""減衰最小二乗（DLS）の微分IK。位置のみ、バッチ対応。

``isaaclab.controllers.DifferentialIKController(ik_method="dls",
command_type="position")`` の置き換え。ARM_BCRL005 では2箇所で使って
いた。スクリプトエキスパート（フェーズ1）と MimicGen の姿勢転写
（フェーズ1.5）。どちらも同じ関節差分の行動空間に流すので、制御器は
当時と同じく位置のみ・レート制限付きのままにしておく必要がある。
Raspberry Pi の安全層と、書き出す ONNX は全部その行動空間の上で定義
されているから。
アルゴリズムは標準のもの（Nakamura & Hanafusa 1986 / Wampler 1986）:

    dq = J^T (J J^T + lambda^2 I)^-1 * (p_target - p_current)

特異点の近くで使えるのは減衰項のおかげ。減衰のない擬似逆行列は、
アームが伸びきると無制限の関節速度を出す。280mm の6自由度アームが
250mm へ手を伸ばすこのタスクでは、かなりの時間を境界の近くで過ごす。
``lambda = 0.05`` は Isaac Lab 自身の既定値。

ヤコビアンの出どころは ``arm_common.kinematics.ToolKinematics.tcp_jacobian``。
解析的で、``deploy/raspi/mycobot_fk.py`` から導いている。この
プロジェクトに運動学の連鎖は1本だけで、全員がそれを読む。

自己テスト（torch だけ。シミュレータ不要）:
    python3 arm_common/ik.py
"""

from __future__ import annotations

import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from arm_common.kinematics import ToolKinematics, NUM_ARM_JOINTS  # noqa: E402


class DLSPositionIK:
    """(N,6) の関節 + (N,3) の目標TCP位置 -> (N,6) の目標関節。"""

    def __init__(self, kin: ToolKinematics, damping: float = 0.05,
                 max_step: float = 0.20):
        self.kin = kin
        self.damping = float(damping)
        # 1回の呼び出しでの |dq| の上限。エキスパート側も直交座標の目標を
        # レート制限（人参）しているが、そもそも届かない目標を渡されると、
        # いくらでも大きな関節ステップを要求してしまう。そして行動項は、
        # 物理的に意味のなかった指令を解くのに制御ステップを何回も
        # 費やすことになる。
        self.max_step = float(max_step)

    def solve(self, q: torch.Tensor, target_pos: torch.Tensor) -> torch.Tensor:
        err = target_pos - self.kin.tcp_pos(q)                 # (N,3)
        J = self.kin.tcp_jacobian(q)                           # (N,3,6)
        JT = J.transpose(1, 2)
        eye = torch.eye(3, dtype=J.dtype, device=J.device).expand(J.shape[0], 3, 3)
        A = J @ JT + (self.damping ** 2) * eye
        dq = (JT @ torch.linalg.solve(A, err.unsqueeze(-1))).squeeze(-1)   # (N,6)
        scale = (self.max_step / dq.norm(dim=-1, keepdim=True).clamp(min=1e-9)).clamp(max=1.0)
        return q + dq * scale


def _self_test() -> None:
    torch.manual_seed(0)
    kin = ToolKinematics("cpu")
    kin.assert_matches_reference()
    ik = DLSPositionIK(kin)

    # ready 姿勢から、到達できる目標のバッチへ収束させる
    from arm_common.kinematics import HOME_Q
    n = 16
    q = torch.tensor([list(HOME_Q)] * n, dtype=torch.float32)
    tgt = kin.tcp_pos(q) + (torch.rand(n, 3) * 2 - 1) * torch.tensor([0.05, 0.05, 0.04])
    for _ in range(60):
        q = ik.solve(q, tgt)
    err = (kin.tcp_pos(q) - tgt).norm(dim=-1)
    print(f"[ik] DLS convergence over {n} random targets: "
          f"mean {err.mean() * 1000:.3f} mm, max {err.max() * 1000:.3f} mm")
    assert float(err.max()) < 1e-3, err

    # 特異姿勢。有限に留まらなければならず、それが減衰項の目的そのもの
    q_sing = torch.zeros(1, NUM_ARM_JOINTS)
    q_out = ik.solve(q_sing, kin.tcp_pos(q_sing) + torch.tensor([[0.3, 0.0, 0.0]]))
    assert torch.isfinite(q_out).all() and float((q_out - q_sing).norm()) <= 0.21
    print("[ik] damped solve stays bounded at a singular configuration")
    print("\nik self-test OK")


if __name__ == "__main__":
    _self_test()
