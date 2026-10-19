# -*- coding: utf-8 -*-
"""NumPy quaternion/pose primitives shared by the MimicGen tooling.

Three primitives, used by three places that must agree on what a pose is:

* ``mimicgen/annotate.py``      -- builds the object-centric dataset
* ``mimicgen/generation.py``    -- equation 1, SLERP, source selection
* ``mimicgen/import_real_demos.py`` -- converts a REAL myCobot recording on the
  operator PC, next to the robot, with numpy only

The last one is why these are numpy and not torch: it has to run before a
simulator exists, on whatever laptop is plugged into the arm.

Convention is wxyz throughout, matching the frozen 31-dim observation layout
and MuJoCo's own ``xquat``.

Self-test (numpy only):
    python3 mimicgen/pose_np.py
"""

from __future__ import annotations

import numpy as np


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """(..., 4) wxyz -> (..., 3, 3) rotation matrix."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True).clip(1e-12)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    m[..., 0, 0] = 1 - 2 * (y * y + z * z)
    m[..., 0, 1] = 2 * (x * y - z * w)
    m[..., 0, 2] = 2 * (x * z + y * w)
    m[..., 1, 0] = 2 * (x * y + z * w)
    m[..., 1, 1] = 1 - 2 * (x * x + z * z)
    m[..., 1, 2] = 2 * (y * z - x * w)
    m[..., 2, 0] = 2 * (x * z - y * w)
    m[..., 2, 1] = 2 * (y * z + x * w)
    m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def mat_to_quat(m: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 4) wxyz. Shepperd's method (numerically stable:
    it picks the largest of the four denominators instead of always using w,
    which degenerates at 180 deg rotations)."""
    m = np.asarray(m, dtype=np.float64)
    t = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    q = np.empty(m.shape[:-2] + (4,), dtype=np.float64)

    def _fill(mask, w, x, y, z):
        if not np.any(mask):
            return
        q[mask, 0], q[mask, 1], q[mask, 2], q[mask, 3] = w[mask], x[mask], y[mask], z[mask]

    m = np.broadcast_to(m, m.shape)
    d0 = t > 0
    s = np.sqrt(np.maximum(t + 1.0, 1e-12)) * 2
    _fill(d0, 0.25 * s, (m[..., 2, 1] - m[..., 1, 2]) / s,
          (m[..., 0, 2] - m[..., 2, 0]) / s, (m[..., 1, 0] - m[..., 0, 1]) / s)

    d1 = (~d0) & (m[..., 0, 0] > m[..., 1, 1]) & (m[..., 0, 0] > m[..., 2, 2])
    s = np.sqrt(np.maximum(1.0 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2], 1e-12)) * 2
    _fill(d1, (m[..., 2, 1] - m[..., 1, 2]) / s, 0.25 * s,
          (m[..., 0, 1] + m[..., 1, 0]) / s, (m[..., 0, 2] + m[..., 2, 0]) / s)

    d2 = (~d0) & (~d1) & (m[..., 1, 1] > m[..., 2, 2])
    s = np.sqrt(np.maximum(1.0 + m[..., 1, 1] - m[..., 0, 0] - m[..., 2, 2], 1e-12)) * 2
    _fill(d2, (m[..., 0, 2] - m[..., 2, 0]) / s, (m[..., 0, 1] + m[..., 1, 0]) / s,
          0.25 * s, (m[..., 1, 2] + m[..., 2, 1]) / s)

    d3 = (~d0) & (~d1) & (~d2)
    s = np.sqrt(np.maximum(1.0 + m[..., 2, 2] - m[..., 0, 0] - m[..., 1, 1], 1e-12)) * 2
    _fill(d3, (m[..., 1, 0] - m[..., 0, 1]) / s, (m[..., 0, 2] + m[..., 2, 0]) / s,
          (m[..., 1, 2] + m[..., 2, 1]) / s, 0.25 * s)

    q /= np.linalg.norm(q, axis=-1, keepdims=True).clip(1e-12)
    # canonical sign (w >= 0) so interpolation never takes the long way round
    return np.where(q[..., :1] < 0, -q, q)


def make_pose(pos: np.ndarray, quat: np.ndarray | None = None) -> np.ndarray:
    """(..., 3) [+ (..., 4) wxyz] -> (..., 4, 4) homogeneous transform.
    Omitting the quaternion gives identity rotation -- which is exactly the
    case for this task's cylinder (it always spawns axis-aligned, and it is a
    solid of revolution about Z so its yaw is unobservable anyway)."""
    pos = np.asarray(pos, dtype=np.float64)
    T = np.zeros(pos.shape[:-1] + (4, 4), dtype=np.float64)
    T[..., :3, :3] = np.eye(3) if quat is None else quat_to_mat(quat)
    T[..., :3, 3] = pos
    T[..., 3, 3] = 1.0
    return T


# --------------------------------------------------------------------------
def _self_test() -> None:
    rng = np.random.RandomState(0)
    q = rng.randn(64, 4)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    q = np.where(q[:, :1] < 0, -q, q)
    err = np.abs(mat_to_quat(quat_to_mat(q)) - q).max()
    print(f"quaternion round-trip         : {err:.2e}")
    assert err < 1e-9

    # 180 deg about X -- the case a naive w-based conversion divides by zero on
    q180 = np.array([[0.0, 1.0, 0.0, 0.0]])
    err = np.abs(mat_to_quat(quat_to_mat(q180)) - q180).max()
    print(f"180 deg rotation              : {err:.2e}")
    assert err < 1e-9

    T = make_pose(np.array([0.2, -0.03, 0.09]), np.array([0.0, 1.0, 0.0, 0.0]))
    assert T.shape == (4, 4) and np.isclose(T[3, 3], 1.0)
    axis = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
    print(f"tool axis of the down-pose    : {np.round(axis, 3)}")
    assert axis[2] < -0.99, "tool must point down"

    print("\npose_np self-test OK")


if __name__ == "__main__":
    _self_test()
