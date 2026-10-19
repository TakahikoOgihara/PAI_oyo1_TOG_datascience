# -*- coding: utf-8 -*-
"""MimicGen step 2 -- GENERATE: warp annotated demos onto new scenes.

The algorithm (Mandlekar et al., "MimicGen: A Data Generation System for
Scalable Robot Learning using Human Demonstrations", CoRL 2023) in full:

  1. Split every source demonstration into SUBTASKS at the gripper transition
     (mimicgen/annotate.py). Each subtask is ABOUT one reference object:
        subtask 0  reach & grasp   -> reference = the block
        subtask 1  transport & place -> reference = the place target
  2. Reset the environment. It now has a new block pose and a new target.
  3. For each subtask, pick a source demonstration (nearest neighbour on the
     reference pose, sampled among the k nearest so the output keeps variety).
  4. Re-express that subtask's END-EFFECTOR TARGET trajectory in the reference
     object's frame, then put that frame where the object is NOW:

         T_eef_new = T_ref_new @ inv(T_ref_src) @ T_eef_src          (eq. 1)

  5. Interpolate from wherever the arm actually is to the first warped pose,
     so the segments join without a jump.
  6. EXECUTE it in the simulator (mujoco_sim/train_arm_mujoco.py --mode
     mimicgen) through the same IK and the same 7-dim action space the
     scripted expert uses, and keep the episode ONLY if the environment's own
     success term fires.

Step 6 is what makes this data generation rather than data interpolation: a
warped trajectory is a HYPOTHESIS, and physics is the referee. Typical yields
are 30-80 %, and a collapsing yield is a signal (the source demos are not
transferable to the new poses), which is why ``mimicgen.min_yield`` stops the
run instead of quietly training on 30 demonstrations.

Rotation is carried through the warp in full SE(3), but the executor is
POSITION-ONLY IK (``mimicgen.ik_mode`` was "position" in ARM_BCRL005 too): for
this task the tool always points down and the block is a solid of revolution,
so the warp's rotation is the source's rotation and tracking it adds nothing.
The pose maths is written for the general case so that stops being true the day
an asymmetric object is added, not the day someone rewrites this file.

Self-test (numpy only, no simulator):
    python3 mimicgen/generation.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mimicgen.pose_np import mat_to_quat, quat_to_mat  # noqa: E402

GRASP_SIGNAL = "grasp"
EEF_NAME = "eef"
# subtask index -> the object its trajectory is expressed relative to
SUBTASK_REFERENCE = ("object", "place_target")


def _inv(T: np.ndarray) -> np.ndarray:
    """Inverse of a homogeneous transform (transpose, not a linear solve)."""
    out = np.eye(4)
    R = T[:3, :3]
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ T[:3, 3]
    return out


def _slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Shortest-arc quaternion interpolation, wxyz, t: (K,) -> (K, 4)."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    if float(q0 @ q1) < 0.0:
        q1 = -q1                                # take the short way round
    dot = float(np.clip(q0 @ q1, -1.0, 1.0))
    if dot > 0.9995:                            # nearly parallel: lerp is exact
        q = q0[None, :] + t[:, None] * (q1 - q0)[None, :]
    else:
        th = np.arccos(dot)
        s = np.sin(th)
        q = (np.sin((1 - t)[:, None] * th) * q0[None, :]
             + np.sin(t[:, None] * th) * q1[None, :]) / s
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def interpolate_poses(T_from: np.ndarray, T_to: np.ndarray, steps: int) -> np.ndarray:
    """(steps, 4, 4) linear-in-position / SLERP-in-rotation path, EXCLUDING
    ``T_from`` and INCLUDING ``T_to``."""
    if steps <= 0:
        return T_to[None, ...].copy()
    t = np.linspace(0.0, 1.0, steps + 1)[1:]
    pos = T_from[:3, 3][None, :] + t[:, None] * (T_to[:3, 3] - T_from[:3, 3])[None, :]
    quat = _slerp(mat_to_quat(T_from[:3, :3]), mat_to_quat(T_to[:3, :3]), t)
    out = np.zeros((steps, 4, 4))
    out[:, :3, :3] = quat_to_mat(quat)
    out[:, :3, 3] = pos
    out[:, 3, 3] = 1.0
    return out


def warp(src_targets: np.ndarray, T_ref_src: np.ndarray,
         T_ref_new: np.ndarray) -> np.ndarray:
    """EQUATION 1. ``src_targets``: (T, 4, 4) -> (T, 4, 4) in the new scene."""
    delta = T_ref_new @ _inv(T_ref_src)
    return delta[None, ...] @ src_targets


# ---------------------------------------------------------------------------
class DataGenInfoPool:
    """The annotated source demonstrations, split into subtasks.

    One pool is built per process and then queried once per generation attempt,
    so everything expensive (HDF5 read, segmentation) happens once.
    """

    def __init__(self, episodes: list):
        self.episodes = episodes
        self.num_subtasks = len(SUBTASK_REFERENCE)

    @classmethod
    def load(cls, path: str, max_demos: int = 0) -> "DataGenInfoPool":
        import h5py
        eps = []
        with h5py.File(path, "r") as f:
            data = f["data"]
            keys = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))
            if max_demos:
                keys = keys[:int(max_demos)]
            for k in keys:
                g = data[k]
                dg = g["obs"]["datagen_info"]
                sig = np.asarray(dg["subtask_term_signals"][GRASP_SIGNAL])[:, 0]
                # the 0 -> 1 transition is the subtask boundary; MimicGen's
                # convention puts the end of subtask 0 one step after it
                boundary = int(np.argmax(sig > 0)) + 1
                T = int(g["actions"].shape[0])
                eps.append({
                    "actions": np.asarray(g["actions"]),
                    "target": np.asarray(dg["target_eef_pose"][EEF_NAME], dtype=np.float64),
                    "ref": {name: np.asarray(dg["object_pose"][name], dtype=np.float64)
                            for name in SUBTASK_REFERENCE},
                    "bounds": [(0, boundary), (boundary, T)],
                })
        if not eps:
            raise RuntimeError(f"{path} contains no annotated demonstrations")
        return cls(eps)

    def reference_positions(self, subtask: int) -> np.ndarray:
        """(n_demos, 3) reference-object position at the START of ``subtask``.

        This is the key the nearest-neighbour source selection uses: "which
        recorded demonstration started this subtask with the object closest to
        where it is now".
        """
        name = SUBTASK_REFERENCE[subtask]
        s0 = [ep["bounds"][subtask][0] for ep in self.episodes]
        return np.array([ep["ref"][name][s, :3, 3] for ep, s in zip(self.episodes, s0)])

    def segment(self, demo: int, subtask: int) -> dict:
        ep = self.episodes[demo]
        lo, hi = ep["bounds"][subtask]
        name = SUBTASK_REFERENCE[subtask]
        return {
            "target": ep["target"][lo:hi],
            "gripper": ep["actions"][lo:hi, 6],
            "T_ref_src": ep["ref"][name][lo],
        }


# ---------------------------------------------------------------------------
class TrajectoryGenerator:
    """Plans ONE generation attempt: new scene -> a target-pose sequence."""

    def __init__(self, pool: DataGenInfoPool, selection: str = "nearest_neighbor_object",
                 nn_k: int = 3, interpolation_steps: int = 12,
                 action_noise: float = 0.0, seed: int = 0):
        self.pool = pool
        self.selection = str(selection)
        self.nn_k = max(1, int(nn_k))
        self.interp = max(0, int(interpolation_steps))
        self.action_noise = float(action_noise)
        self.rng = np.random.RandomState(int(seed))
        self._ref_pos = [pool.reference_positions(s) for s in range(pool.num_subtasks)]

    def select_source(self, subtask: int, ref_pos_new: np.ndarray) -> int:
        n = len(self.pool.episodes)
        if self.selection == "random" or n <= self.nn_k:
            return int(self.rng.randint(n))
        d = np.linalg.norm(self._ref_pos[subtask] - ref_pos_new[None, :], axis=1)
        # uniform among the k nearest, NOT the single nearest: always taking
        # the argmin makes every generated demo in a region a copy of the same
        # source trajectory, which is data volume without data diversity.
        return int(self.rng.choice(np.argpartition(d, self.nn_k - 1)[:self.nn_k]))

    def plan_subtask(self, subtask: int, T_ref_new: np.ndarray,
                     T_eef_now: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """-> (target poses (K,4,4), gripper commands (K,), source demo index)."""
        src = self.select_source(subtask, T_ref_new[:3, 3])
        seg = self.pool.segment(src, subtask)
        warped = warp(seg["target"], seg["T_ref_src"], T_ref_new)
        grip = seg["gripper"]
        if self.interp > 0 and len(warped) > 0:
            bridge = interpolate_poses(T_eef_now, warped[0], self.interp)
            warped = np.concatenate([bridge, warped[1:]], axis=0)
            # hold the subtask's first gripper command across the bridge: the
            # bridge is a repositioning move, not part of the demonstrated
            # manipulation, so the gripper must not change during it
            grip = np.concatenate([np.full(self.interp, grip[0]), grip[1:]])
        if self.action_noise > 0:
            warped = warped.copy()
            warped[:, :3, 3] += self.rng.randn(len(warped), 3) * self.action_noise
        return warped, grip, src


# ---------------------------------------------------------------------------
def _self_test() -> None:
    # eq. 1 on a synthetic subtask: a straight descent onto a block
    T_ref_src = np.eye(4)
    T_ref_src[:3, 3] = [0.20, -0.03, 0.015]
    src = np.zeros((10, 4, 4))
    src[:, :3, :3] = np.diag([1.0, -1.0, -1.0])          # tool pointing down
    src[:, 3, 3] = 1.0
    src[:, :3, 3] = np.linspace([0.20, -0.03, 0.09], [0.20, -0.03, 0.023], 10)

    T_ref_new = np.eye(4)
    T_ref_new[:3, 3] = [0.17, 0.02, 0.015]
    out = warp(src, T_ref_src, T_ref_new)
    shift = T_ref_new[:3, 3] - T_ref_src[:3, 3]
    print(f"warp: pure translation? max err "
          f"{np.abs(out[:, :3, 3] - (src[:, :3, 3] + shift)).max():.2e}")
    assert np.abs(out[:, :3, 3] - (src[:, :3, 3] + shift)).max() < 1e-12
    assert np.abs(out[:, :3, :3] - src[:, :3, :3]).max() < 1e-12
    # the RELATIVE geometry the demonstration encodes is what is preserved:
    # the tool ends the segment exactly as far above the block as before
    rel_src = src[-1, :3, 3] - T_ref_src[:3, 3]
    rel_new = out[-1, :3, 3] - T_ref_new[:3, 3]
    print(f"grasp offset preserved        : {np.round(rel_src, 5)} -> {np.round(rel_new, 5)}")
    assert np.allclose(rel_src, rel_new)

    # a rotated reference must rotate the trajectory about it (general SE(3))
    a = 0.7
    T_rot = np.eye(4)
    T_rot[:3, :3] = [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
    T_rot[:3, 3] = T_ref_src[:3, 3]
    out_rot = warp(src, T_ref_src, T_rot)
    rel = out_rot[-1, :3, 3] - T_rot[:3, 3]
    assert np.allclose(np.linalg.norm(rel), np.linalg.norm(rel_src))
    print(f"rotated reference             : |offset| preserved ({np.linalg.norm(rel):.5f} m)")

    # interpolation joins without a jump and lands exactly on the target
    T_from = np.eye(4)
    T_from[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T_from[:3, 3] = [0.25, 0.0, 0.10]
    path = interpolate_poses(T_from, out[0], 8)
    assert path.shape == (8, 4, 4)
    assert np.allclose(path[-1], out[0], atol=1e-9)
    step = np.linalg.norm(np.diff(np.vstack([T_from[:3, 3][None], path[:, :3, 3]]), axis=0), axis=1)
    print(f"interpolation                 : {len(path)} steps, "
          f"max {step.max() * 1000:.1f} mm each, lands on target")
    assert step.std() < 1e-9                   # uniform spacing

    # SLERP over a real rotation stays unit-norm and monotone
    q = _slerp(np.array([1.0, 0, 0, 0]), np.array([0.0, 1.0, 0, 0]), np.linspace(0, 1, 5))
    assert np.allclose(np.linalg.norm(q, axis=-1), 1.0)
    print(f"slerp norms                   : {np.round(np.linalg.norm(q, axis=-1), 6)}")

    # source selection keeps variety
    pool = DataGenInfoPool([{
        "actions": np.zeros((20, 7)),
        "target": np.tile(np.eye(4), (20, 1, 1)),
        "ref": {"object": np.tile(np.eye(4), (20, 1, 1)) * 1.0,
                "place_target": np.tile(np.eye(4), (20, 1, 1))},
        "bounds": [(0, 10), (10, 20)],
    } for _ in range(8)])
    for i, ep in enumerate(pool.episodes):
        ep["ref"]["object"][:, :3, 3] = [0.16 + 0.01 * i, 0.0, 0.015]
    gen = TrajectoryGenerator(pool, nn_k=3, seed=0)
    picks = {gen.select_source(0, np.array([0.185, 0.0, 0.015])) for _ in range(40)}
    print(f"nearest-neighbour picks (k=3) : {sorted(picks)}")
    assert 1 < len(picks) <= 3, picks

    print("\ngeneration self-test OK")


if __name__ == "__main__":
    _self_test()
