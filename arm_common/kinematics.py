# -*- coding: utf-8 -*-
"""Batched (torch) forward kinematics for the simulator -- the SAME safety
envelope the Raspberry Pi enforces.

``deploy/raspi/mycobot_fk.py`` is the authoritative, zero-dependency
definition of the chain and of the tool sample points (it has to be: it runs
inside the hardware guard). This module imports those constants verbatim and
adds a vectorised torch implementation so ``mujoco_sim/train_arm_mujoco.py``
can apply the identical virtual floor to num_envs environments at once.

Why the simulator needs the guard at all: a fail-safe that exists only on the
robot is a fail-safe the policy was never trained against, so the deployed
policy would spend its whole episode fighting a limit it has never seen.
Enforcing it in BOTH places means the control contract the policy learns is
byte-for-byte the contract the hardware runs (design notes §3).

``assert_matches_reference()`` proves the two implementations agree; it runs
once at env construction, so a drift between them fails loudly instead of
silently widening the sim-to-real gap.
"""

from __future__ import annotations
import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RASPI_DIR = os.path.join(_REPO_ROOT, "deploy", "raspi")
if _RASPI_DIR not in sys.path:
    sys.path.append(_RASPI_DIR)

from mycobot_fk import (  # noqa: E402  -- single source of truth
    CHAIN, TOOL_POINTS, TCP_Z, HOME_Q, NUM_ARM_JOINTS,
    _rpy as _rpy_rowmajor, tool_min_z as _tool_min_z_ref,
)

__all__ = ["ToolKinematics", "TCP_Z", "HOME_Q", "NUM_ARM_JOINTS"]


class ToolKinematics:
    """Vectorised tool FK: (N, 6) joint angles -> tool sample-point heights."""

    def __init__(self, device, dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        # fixed joint-origin transforms, (6, 4, 4)
        origins = torch.zeros(len(CHAIN), 4, 4, dtype=dtype)
        for i, (xyz, rpy) in enumerate(CHAIN):
            # rotation built by the REFERENCE implementation, so the two can
            # never disagree on the URDF rpy convention
            origins[i, :3, :3] = torch.tensor(_rpy_rowmajor(*rpy), dtype=dtype).view(3, 3)
            origins[i, :3, 3] = torch.tensor(xyz, dtype=dtype)
            origins[i, 3, 3] = 1.0
        self.origins = origins.to(device)
        # tool sample points in the flange frame, (P, 3)
        self.tool_pts = torch.tensor(TOOL_POINTS, dtype=dtype, device=device)
        self.tcp_local = torch.tensor([0.0, 0.0, TCP_Z], dtype=dtype, device=device)

    def flange_pose(self, q: torch.Tensor) -> torch.Tensor:
        """q: (N, 6) rad -> (N, 4, 4) pose of joint6_flange in the base frame."""
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
        """-> (N, P, 3) sample points in the robot base frame."""
        T = self.flange_pose(q)
        return (self.tool_pts @ T[:, :3, :3].transpose(1, 2)) + T[:, None, :3, 3]

    def tool_min_z(self, q: torch.Tensor) -> torch.Tensor:
        """-> (N,) height of the lowest tool point above the base plane."""
        return self.tool_points(q)[..., 2].min(dim=-1).values

    def tcp_pos(self, q: torch.Tensor) -> torch.Tensor:
        T = self.flange_pose(q)
        return (self.tcp_local @ T[:, :3, :3].transpose(1, 2)) + T[:, :3, 3]

    # ------------------------------------------------------------------
    def limit_to_floor(self, q_from: torch.Tensor, q_to: torch.Tensor,
                       min_z: float, iters: int = 8):
        """Shorten each row of q_from -> q_to until the tool clears ``min_z``.

        Identical algorithm to ``mycobot_fk.FloorGuard.limit`` (bisection on
        the joint-space segment), just evaluated for every env at once.
        Returns (q_safe, blocked_mask).
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
        # already under the plane (bad calibration / push in progress): hold
        q_cut = torch.where(start_unsafe.unsqueeze(-1), q_from, q_cut)
        return torch.where(unsafe.unsqueeze(-1), q_cut, q_to), unsafe

    # ------------------------------------------------------------------
    # ARM_BCRL006 addition: the analytic TCP Jacobian.
    #
    # ARM_BCRL005 asked PhysX for the articulation Jacobian
    # (``robot.root_physx_view.get_jacobians()``) and fed it to Isaac Lab's
    # DifferentialIKController. Outside Isaac Sim neither exists, and asking
    # MuJoCo for ``mj_jacSite`` would put a THIRD kinematic chain (MuJoCo's)
    # into the loop that the virtual floor and MimicGen already compute from
    # ``mycobot_fk.py``.
    #
    # For a serial chain of revolute joints the position Jacobian is exact and
    # two lines long:  J_i = z_i x (p_tcp - p_i), where z_i / p_i are joint i's
    # axis and origin in the base frame. Deriving it from the SAME chain the
    # floor guard uses means the IK, the guard, the MimicGen pose warp and the
    # robot all move the same arm. ``assert_matches_reference`` cross-checks it
    # against finite differences, and assets/build_mycobot_mjcf.py cross-checks
    # the chain itself against MuJoCo.
    # ------------------------------------------------------------------
    def tcp_jacobian(self, q: torch.Tensor) -> torch.Tensor:
        """q: (N, 6) -> (N, 3, 6) d(TCP position) / d(joint angle)."""
        n = q.shape[0]
        T = torch.eye(4, dtype=self.dtype, device=self.device).expand(n, 4, 4).contiguous()
        axes, origins = [], []
        for i in range(NUM_ARM_JOINTS):
            T = T @ self.origins[i]
            axes.append(T[:, :3, 2])            # joint axis (local +Z) in base frame
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
        """Fail loudly if the torch FK ever drifts from the Pi's FK."""
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
        # the analytic Jacobian against central differences on the same chain.
        # In float64: at float32 the difference quotient's own round-off
        # (~1e-8 m / 2e-5 rad ~= 1e-3) swamps the quantity being checked, so a
        # float32 comparison would only be measuring its own noise.
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
