# -*- coding: utf-8 -*-
"""Damped least-squares (DLS) differential IK -- position only, batched.

Replaces ``isaaclab.controllers.DifferentialIKController(ik_method="dls",
command_type="position")``, which ARM_BCRL005 used in two places: the scripted
expert (phase 1) and the MimicGen pose warp (phase 1.5). Both feed the SAME
joint-delta action space, so the controller has to stay position-only and
rate-limited exactly as it was -- the Raspberry Pi safety layer and every
exported ONNX graph are defined on that action space.

The algorithm is the standard one (Nakamura & Hanafusa 1986 / Wampler 1986):

    dq = J^T (J J^T + lambda^2 I)^-1 * (p_target - p_current)

The damping term is what makes it usable near singularities -- an undamped
pseudo-inverse produces unbounded joint velocities when the arm straightens,
and a 280 mm 6-DoF arm reaching to 250 mm is close to its boundary for much of
this task. ``lambda = 0.05`` is Isaac Lab's own default.

Jacobian source: ``arm_common.kinematics.ToolKinematics.tcp_jacobian`` --
analytic, derived from ``deploy/raspi/mycobot_fk.py``. There is exactly one
kinematic chain in this project and every consumer reads it.

Self-test (torch only, no simulator):
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
    """(N,6) joints + (N,3) target TCP position -> (N,6) desired joints."""

    def __init__(self, kin: ToolKinematics, damping: float = 0.05,
                 max_step: float = 0.20):
        self.kin = kin
        self.damping = float(damping)
        # cap on |dq| per call. The expert already rate-limits its cartesian
        # target ("carrot"), but a target the arm cannot reach at all would
        # otherwise ask for an arbitrarily large joint step, and the action
        # term would spend several control steps un-winding a command that was
        # never physically meaningful.
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

    # converge from the ready pose onto a batch of reachable targets
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

    # singular configuration: must stay finite, which is the whole point of
    # the damping term
    q_sing = torch.zeros(1, NUM_ARM_JOINTS)
    q_out = ik.solve(q_sing, kin.tcp_pos(q_sing) + torch.tensor([[0.3, 0.0, 0.0]]))
    assert torch.isfinite(q_out).all() and float((q_out - q_sing).norm()) <= 0.21
    print("[ik] damped solve stays bounded at a singular configuration")
    print("\nik self-test OK")


if __name__ == "__main__":
    _self_test()
