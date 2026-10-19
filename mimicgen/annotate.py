# -*- coding: utf-8 -*-
"""MimicGen step 1 -- ANNOTATE: phase-1 demos -> an object-centric dataset.

MimicGen does not read raw demonstrations. It reads a dataset in which every
episode carries a ``datagen_info`` group:

    data/demo_<i>/
        actions                              (T, 7)
        obs/datagen_info/eef_pose/eef        (T, 4, 4)   where the tool WAS
        obs/datagen_info/target_eef_pose/eef (T, 4, 4)   where it was TOLD to go
        obs/datagen_info/object_pose/object       (T, 4, 4)
        obs/datagen_info/object_pose/place_target (T, 4, 4)
        obs/datagen_info/subtask_term_signals/grasp (T, 1)  0 -> 1 at the grasp

That is exactly the layout NVIDIA's ``isaaclab_mimic`` annotator produces and
``DataGenInfoPool.load_from_dataset_file()`` consumes -- ARM_BCRL006 keeps the
SCHEMA and writes it with plain h5py instead of Isaac Lab's dataset handler, so
the file stays inspectable and portable while the tool chain stays local
(change log §6). Writing it here -- rather than replaying every demo
through the simulator to have the env emit the signals -- is why phase 1.5 is
one pipeline step: **every field above is already recorded in phase 1's 31-dim
observation and 7-dim action.**

WHERE THE SUBTASK BOUNDARY COMES FROM -- the gripper channel:

    open ... open | closed ... closed | open ... open
                  ^                   ^
                  grasp established    object released
    <-- subtask 0 -><------- subtask 1 ------------->
      ref: object      ref: place_target

``arm_common/expert.py`` has an exact 8-phase index and recording it would be
trivially accurate -- for demos that come from the scripted expert. But the
point of MimicGen here is that phase 1 must be replaceable by a handful of REAL
demonstrations, and a human hand-guiding the arm produces no phase index: it
produces joint angles and a gripper open/close signal. Segmenting on the signal
that exists in BOTH worlds is what makes the sim source and the real source
interchangeable, and it is the standard heuristic for real teleop data. It also
costs phase 1 exactly zero changes.

TARGET POSES come from the recorded action, through the SAME forward kinematics
the robot's virtual floor runs on (``deploy/raspi/mycobot_fk.py``):
``target = FK(q + action[:6] * action_scale)``. MimicGen warps the TARGET pose
sequence, not the measured one, so a controller lag in the source demo is not
baked into every generated demo.

Run standalone to annotate a dataset:
    python3 mimicgen/annotate.py artifacts/data/bc_demos.pt \\
                                 artifacts/data/bc_demos_annotated.hdf5
Self-test (numpy only, no simulator, no torch):
    python3 mimicgen/annotate.py
"""

from __future__ import annotations
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "deploy", "raspi"))

from mimicgen.pose_np import make_pose, mat_to_quat  # noqa: E402

# Observation layout of mujoco_sim/train_arm_mujoco.py (31-dim):
#   q(8) | eef_pos(3) | eef_quat(4) | obj_pos(3) | obj-eef(3) |
#   place_target(3) | prev_action(7)
OBS_ARM_Q = slice(0, 6)
OBS_EEF_POS = slice(8, 11)
OBS_EEF_QUAT = slice(11, 15)     # wxyz, robot base frame (base == env origin)
OBS_OBJ_POS = slice(15, 18)
OBS_PLACE_TARGET = slice(21, 24)
ACT_GRIPPER = 6

# Reject a demo whose grasp sits closer than this to either end. It also
# bounds how far configs/mimicgen/default.yaml may push
# subtask_term_offset_range without Isaac Lab Mimic's own
# "got empty subtasks!" assertion firing during generation.
MIN_SUBTASK_STEPS = 10


def grasp_signal(gripper: np.ndarray,
                 min_steps: int = MIN_SUBTASK_STEPS) -> np.ndarray | None:
    """(T,) gripper channel (>0 = open) -> (T,1) 0/1 grasp signal, or None.

    The signal is 0 while the gripper is still open and latches to 1 from the
    step it first closes. Isaac Lab Mimic locates the subtask boundary at the
    0->1 transition (``DataGenInfoPool._add_episode``), so a latching signal
    is the right shape: it is a "has completed" flag, not a "is happening"
    flag.

    Returns None -- rather than raising -- when the demo has no clean
    close/re-open cycle. MimicGen's philosophy throughout is "generate a lot,
    keep what is valid"; a malformed source demo is dropped and counted.
    """
    g = np.asarray(gripper).reshape(-1)
    T = int(g.shape[0])
    closed = np.flatnonzero(g <= 0)
    if closed.size == 0:
        return None                                   # never grasped
    c = int(closed[0])
    if not np.any(g[c:] > 0):
        return None                                   # never released
    if c < min_steps or (T - c) < min_steps:
        return None                                   # degenerate segment
    sig = np.zeros((T, 1), dtype=np.float32)
    sig[c:] = 1.0
    return sig


def target_eef_poses(obs: np.ndarray, act: np.ndarray,
                     action_scale: float) -> np.ndarray:
    """(T,31) + (T,7) -> (T,4,4): the TCP pose each recorded action commanded.

    ``action_to_target_eef_pose`` in the language of the Isaac Lab Mimic
    contract. Uses ``deploy/raspi/mycobot_fk.py`` -- the very forward
    kinematics the robot's virtual floor runs on -- so a disagreement here is
    a disagreement that would also matter on the hardware.
    """
    from mycobot_fk import TCP_Z, flange_pose      # numpy only, no simulator

    q_cmd = obs[:, OBS_ARM_Q] + act[:, :6] * float(action_scale)
    T = q_cmd.shape[0]
    pos = np.zeros((T, 3))
    quat = np.zeros((T, 4))
    for t in range(T):
        R, p = flange_pose(q_cmd[t])
        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        pos[t] = np.asarray(p, dtype=np.float64) + R[:, 2] * TCP_Z
        quat[t] = mat_to_quat(R)
    return make_pose(pos, quat)


def build_episode_arrays(obs: np.ndarray, act: np.ndarray, action_scale: float,
                         eef_name: str = "eef", grasp_signal_name: str = "grasp",
                         min_steps: int = MIN_SUBTASK_STEPS) -> dict | None:
    """One recorded trajectory -> the nested dict of one annotated episode.

    Returns None when the demo cannot be segmented (no clean grasp cycle).
    """
    obs = np.asarray(obs, dtype=np.float64)
    act = np.asarray(act, dtype=np.float64)
    sig = grasp_signal(act[:, ACT_GRIPPER], min_steps)
    if sig is None:
        return None
    # Reference-object poses are recorded per timestep to match the Isaac Lab
    # Mimic layout. The cylinder spawns axis-aligned and is a solid of
    # revolution, so identity rotation is exact; make_pose takes a quaternion
    # the day an object with a meaningful orientation is added -- the warp
    # itself is full SE(3) either way.
    return {
        "actions": act.astype(np.float32),
        "obs": {
            "datagen_info": {
                "eef_pose": {
                    eef_name: make_pose(obs[:, OBS_EEF_POS],
                                        obs[:, OBS_EEF_QUAT]).astype(np.float32),
                },
                "target_eef_pose": {
                    eef_name: target_eef_poses(obs, act, action_scale
                                               ).astype(np.float32),
                },
                "object_pose": {
                    "object": make_pose(obs[:, OBS_OBJ_POS]).astype(np.float32),
                    "place_target": make_pose(obs[:, OBS_PLACE_TARGET]
                                              ).astype(np.float32),
                },
                "subtask_term_signals": {grasp_signal_name: sig},
            }
        },
    }


def _write_group(grp, d) -> None:
    for k, v in d.items():
        if isinstance(v, dict):
            _write_group(grp.create_group(k), v)
        else:
            grp.create_dataset(k, data=np.asarray(v))


def write_annotated_dataset(demos: dict, out_path: str, action_scale: float,
                            max_demos: int | None = None, eef_name: str = "eef",
                            grasp_signal_name: str = "grasp",
                            verbose: bool = True) -> dict:
    """{"obs": [...], "act": [...]} -> the annotated HDF5 described above.

    Plain h5py. The layout is the isaaclab_mimic one, so the file can still be
    inspected, diffed or handed to NVIDIA's tooling; the dependency on Isaac Sim
    is what was dropped, not the format.
    """
    import h5py

    obs_list, act_list = demos["obs"], demos["act"]
    n = len(obs_list) if max_demos is None else min(len(obs_list), int(max_demos))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    kept, rejected, lens, grasp_at = 0, 0, [], []
    with h5py.File(out_path, "w") as f:
        data = f.create_group("data")
        data.attrs["env_name"] = "arm_bcrl006_pickplace"
        data.attrs["action_scale"] = float(action_scale)
        for i in range(n):
            arrays = build_episode_arrays(np.asarray(obs_list[i]), np.asarray(act_list[i]),
                                          action_scale, eef_name, grasp_signal_name)
            if arrays is None:
                rejected += 1
                continue
            ep = data.create_group(f"demo_{kept}")
            ep.attrs["num_samples"] = int(arrays["actions"].shape[0])
            ep.attrs["success"] = True          # phase 1 keeps successes only
            _write_group(ep, arrays)
            kept += 1
            lens.append(int(arrays["actions"].shape[0]))
            sig = arrays["obs"]["datagen_info"]["subtask_term_signals"][grasp_signal_name]
            grasp_at.append(int(np.argmax(sig > 0)))
        data.attrs["num_demos"] = kept

    stats = {
        "source_demos_seen": n,
        "annotated": kept,
        "rejected_no_clean_grasp": rejected,
        "mean_steps": float(np.mean(lens)) if lens else 0.0,
        "mean_grasp_step": float(np.mean(grasp_at)) if grasp_at else 0.0,
        "subtasks": ["grasp (ref: object)", "place (ref: place_target)"],
        "annotated_dataset": out_path,
    }
    if verbose:
        print("[mimicgen/annotate] " + json.dumps(stats, indent=2, ensure_ascii=False))
        if rejected:
            print(f"[mimicgen/annotate] NOTE: {rejected} demo(s) had no clean "
                  "open->closed->open gripper cycle and were dropped. A large "
                  "fraction here means the source demos are not actually "
                  "completing the task -- fix phase 1 before generating.")
    if kept == 0:
        raise RuntimeError(
            "no demo could be annotated: not one had a clean "
            "open->closed->open gripper cycle. Check phase 1's "
            "artifacts/data/collect_summary.json (source_success_rate).")
    return stats


def load_source_demos(path: str) -> dict:
    """Load a phase-1 (.pt) dataset. torch is only needed for this one call,
    which keeps the rest of the module testable without a simulator."""
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


# --------------------------------------------------------------------------
def _self_test() -> None:
    """numpy only: signal shape, boundary location, pose plumbing, rejection."""
    T = 90
    grasp_at = 40
    obs = np.zeros((T, 31))
    act = np.zeros((T, 7))
    obj = np.array([0.20, -0.03, 0.015])
    tgt = obj + np.array([0.0, 0.05, 0.0])
    obs[:, OBS_OBJ_POS] = obj
    obs[:, OBS_PLACE_TARGET] = tgt
    obs[:, OBS_EEF_QUAT] = np.array([0.0, 1.0, 0.0, 0.0])       # tool down
    obs[:grasp_at, OBS_EEF_POS] = np.linspace([0.20, -0.03, 0.09],
                                              [0.20, -0.03, 0.023], grasp_at)
    obs[grasp_at:, OBS_EEF_POS] = np.linspace([0.20, -0.03, 0.023],
                                              [0.20, 0.02, 0.023], T - grasp_at)
    obs[:, OBS_ARM_Q] = np.array([0.25, -1.0, -0.1, -0.5, 0.0, 0.0])
    act[:, ACT_GRIPPER] = 1.0
    act[grasp_at:T - 8, ACT_GRIPPER] = -1.0                     # closed, released

    sig = grasp_signal(act[:, ACT_GRIPPER])
    assert sig is not None and sig.shape == (T, 1)
    edge = int(np.argmax(np.diff(sig[:, 0]) != 0)) + 1
    print(f"grasp signal 0->1 at step     : {edge}  (expected {grasp_at})")
    assert edge == grasp_at
    # Isaac Lab Mimic parses the boundary as (transition index + 1); the two
    # subtasks it derives must be non-empty and must cover the demo.
    end0 = edge + 1
    assert 0 < end0 < T, (end0, T)
    print(f"Mimic subtask boundaries      : [0, {end0}) and [{end0}, {T})")

    ep = build_episode_arrays(obs, act, action_scale=0.05)
    assert ep is not None
    dg = ep["obs"]["datagen_info"]
    assert dg["eef_pose"]["eef"].shape == (T, 4, 4)
    assert dg["target_eef_pose"]["eef"].shape == (T, 4, 4)
    assert np.allclose(dg["object_pose"]["object"][0, :3, 3], obj)
    assert np.allclose(dg["object_pose"]["place_target"][0, :3, 3], tgt)
    assert np.allclose(dg["eef_pose"]["eef"][0, :3, 3], [0.20, -0.03, 0.09])
    axis = dg["eef_pose"]["eef"][0, :3, :3] @ np.array([0, 0, 1.0])
    print(f"tool axis from recorded quat  : {np.round(axis, 3)}")
    assert axis[2] < -0.99, "tool axis must point down"

    # target poses: zero action => the FK of the measured joints, and the
    # gripper channel must not leak into the arm target
    tgt_pose = dg["target_eef_pose"]["eef"]
    print(f"target TCP at t=0             : {np.round(tgt_pose[0, :3, 3], 4)}  "
          "(home reference [0.2556, -0.0004, 0.1001])")
    assert np.abs(tgt_pose[0, :3, 3] - np.array([0.2556, -0.0004, 0.1001])).max() < 1e-3

    # rejection paths
    assert grasp_signal(np.ones(T)) is None                      # never closes
    assert grasp_signal(-np.ones(T)) is None                     # never opens
    short = np.ones(T); short[1:] = -1.0
    assert grasp_signal(short) is None                           # closes too early
    print("rejection of malformed demos  : OK")

    print("\nannotate self-test OK")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        write_annotated_dataset(load_source_demos(sys.argv[1]), sys.argv[2],
                                action_scale=0.05)
    elif len(sys.argv) == 1:
        _self_test()
    else:
        print(__doc__)
        raise SystemExit(2)
