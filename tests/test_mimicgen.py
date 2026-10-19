# -*- coding: utf-8 -*-
"""MimicGen: the equation, the segmentation, and the round trip through HDF5.

These are the pieces that make phase 1.5 correct rather than merely plausible.
The generation LOOP is validated by the pipeline itself (the yield gate), but
the geometry has to be right before a yield number means anything.
"""

import numpy as np
import torch

from mimicgen.annotate import grasp_signal, write_annotated_dataset
from mimicgen.generation import (DataGenInfoPool, TrajectoryGenerator,
                                 interpolate_poses, warp)
from mimicgen.pose_np import make_pose


def _fake_demo(T=90, grasp_at=40, obj=(0.20, -0.03, 0.015)):
    obs = np.zeros((T, 31), dtype=np.float32)
    act = np.zeros((T, 7), dtype=np.float32)
    obj = np.array(obj)
    obs[:, 15:18] = obj
    obs[:, 21:24] = obj + np.array([0.0, 0.05, 0.0])
    obs[:, 11:15] = np.array([0.0, 1.0, 0.0, 0.0])
    obs[:, 0:6] = np.array([0.25, -1.0, -0.1, -0.5, 0.0, 0.0])
    obs[:grasp_at, 8:11] = np.linspace([0.20, -0.03, 0.09], [0.20, -0.03, 0.023], grasp_at)
    obs[grasp_at:, 8:11] = np.linspace([0.20, -0.03, 0.023], [0.20, 0.02, 0.023], T - grasp_at)
    act[:, 6] = 1.0
    act[grasp_at:T - 8, 6] = -1.0
    return torch.from_numpy(obs), torch.from_numpy(act)


def test_equation_1_preserves_the_relative_geometry():
    """The whole method is: what the demonstration encodes is the tool's pose
    RELATIVE to the object. Moving the object must move the trajectory with it
    and change nothing else."""
    T_src = make_pose(np.array([0.20, -0.03, 0.015]))
    traj = np.tile(np.eye(4), (12, 1, 1))
    traj[:, :3, :3] = np.diag([1.0, -1.0, -1.0])
    traj[:, :3, 3] = np.linspace([0.20, -0.03, 0.09], [0.20, -0.03, 0.023], 12)
    T_new = make_pose(np.array([0.17, 0.02, 0.015]))
    out = warp(traj, T_src, T_new)
    rel_src = traj[:, :3, 3] - T_src[:3, 3]
    rel_new = out[:, :3, 3] - T_new[:3, 3]
    assert np.abs(rel_src - rel_new).max() < 1e-12
    assert np.abs(out[:, :3, :3] - traj[:, :3, :3]).max() < 1e-12


def test_subtask_segmentation_uses_the_gripper_signal():
    _obs, act = _fake_demo(grasp_at=40)
    sig = grasp_signal(act[:, 6].numpy())
    assert sig is not None and sig.shape == (90, 1)
    assert int(np.argmax(sig[:, 0] > 0)) == 40
    # a demo without a clean open->closed->open cycle is DROPPED, not repaired:
    # MimicGen's philosophy is "generate a lot, keep what is valid"
    assert grasp_signal(np.ones(90)) is None
    assert grasp_signal(-np.ones(90)) is None


def test_annotated_hdf5_round_trip(tmp_path):
    obs, act = _fake_demo()
    demos = {"obs": [obs, obs.clone()], "act": [act, act.clone()]}
    out = str(tmp_path / "ann.hdf5")
    stats = write_annotated_dataset(demos, out, action_scale=0.05, verbose=False)
    assert stats["annotated"] == 2
    pool = DataGenInfoPool.load(out)
    assert len(pool.episodes) == 2 and pool.num_subtasks == 2
    lo0, hi0 = pool.episodes[0]["bounds"][0]
    lo1, hi1 = pool.episodes[0]["bounds"][1]
    assert lo0 == 0 and hi0 == lo1 and hi1 == obs.shape[0]      # subtasks tile the demo
    assert hi0 - lo0 > 0 and hi1 - lo1 > 0                      # and neither is empty


def test_planner_bridges_without_a_jump(tmp_path):
    obs, act = _fake_demo()
    out = str(tmp_path / "ann.hdf5")
    write_annotated_dataset({"obs": [obs], "act": [act]}, out, 0.05, verbose=False)
    gen = TrajectoryGenerator(DataGenInfoPool.load(out), nn_k=1, interpolation_steps=10)
    here = make_pose(np.array([0.26, 0.0, 0.10]))
    poses, grip, _ = gen.plan_subtask(0, make_pose(np.array([0.17, 0.02, 0.015])), here)
    assert len(poses) == len(grip)
    steps = np.linalg.norm(np.diff(
        np.vstack([here[:3, 3][None], poses[:, :3, 3]]), axis=0), axis=1)
    # the bridge must not ask the arm to teleport: at 20 Hz and the expert's
    # 10 mm/step carrot, a 60 mm jump is a command the IK cannot track
    assert steps.max() < 0.06, f"{steps.max() * 1000:.0f} mm"
    # the gripper must not change state during the bridge
    assert np.all(grip[:10] == grip[0])


def test_interpolation_is_uniform_and_lands_exactly():
    a = make_pose(np.array([0.25, 0.0, 0.10]))
    b = make_pose(np.array([0.17, 0.02, 0.08]))
    p = interpolate_poses(a, b, 8)
    assert p.shape == (8, 4, 4) and np.allclose(p[-1], b, atol=1e-9)
    d = np.linalg.norm(np.diff(np.vstack([a[:3, 3][None], p[:, :3, 3]]), axis=0), axis=1)
    assert d.std() < 1e-9
