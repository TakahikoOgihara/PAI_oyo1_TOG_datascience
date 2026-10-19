# -*- coding: utf-8 -*-
"""What ships: the ONNX graph and the safety envelope that travels with it.

ARM_BCRL002 deployed a policy that measured 0.55 success and 64 mm placement
error in simulation and went 0-for-10 on hardware. Two of the three causes are
covered here: the graph must compute what the trained model computes, and the
release gate must actually refuse.
"""

import os

import numpy as np
import torch
import torch.nn as nn

from mlops import export as onnx_export


def test_onnx_export_matches_torch_and_supports_batching(tmp_path):
    """dynamic_axes has to survive the export. torch >= 2.9's dynamo exporter
    silently SPECIALISES the batch dimension, which produces a graph that works
    for the one-sample validation and fails the moment the sim evaluates 64
    environments at once -- mlops/export.py forces the legacy tracer for this
    reason, and this is the test that notices if that stops working."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(31, 64), nn.ReLU(), nn.Linear(64, 7), nn.Tanh())
    path = str(tmp_path / "p.onnx")
    onnx_export.export_onnx(model, 31, path)
    v = onnx_export.validate_onnx(model, 31, path, n=64)
    assert v["onnx_available"] and v["max_abs_diff"] < 1e-5

    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    for batch in (1, 7, 64):
        y = sess.run(None, {"obs": np.random.randn(batch, 31).astype(np.float32)})[0]
        assert y.shape == (batch, 7)


def test_release_gate_refuses_a_policy_that_cannot_work():
    """The gate is the cheapest control in the pipeline and the one thing that
    turns 'we have an MLOps stack' into 'our MLOps stack decides what ships'."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from omegaconf import OmegaConf
    from src.infer import _release_gate
    cfg = OmegaConf.create({"deploy": {"min_success_rate": 0.55, "max_place_err": 0.035,
                                       "min_lift": 0.055, "force": False},
                            "env": {"floor": {"z": 0.0, "clearance": 0.005}}})
    good = {"success_rate": 0.62, "mean_place_err": 0.031, "mean_max_lift": 0.071}
    assert _release_gate(cfg, good, "ACT")
    # the exact ARM_BCRL002 artifact: 55 % success, 64 mm error, 69 mm lift
    bad = {"success_rate": 0.553, "mean_place_err": 0.064, "mean_max_lift": 0.069}
    assert not _release_gate(cfg, bad, "ACT")
    cfg.deploy.force = True
    assert _release_gate(cfg, bad, "ACT")      # explicit override still possible


def test_edge_config_carries_the_safety_envelope():
    """The Pi must READ the floor plane and rate limit, never have them typed
    in again: two copies of a safety constant is one copy of a safety constant
    plus a bug waiting to happen."""
    import sys
    from omegaconf import OmegaConf
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    safety = OmegaConf.load(os.path.join(root, "configs", "safety", "default.yaml"))
    env = OmegaConf.load(os.path.join(root, "configs", "env", "mycobot_pickplace.yaml"))
    # the simulator's plane and the robot's plane are the same two numbers
    assert float(safety.floor_z) == float(env.floor.z)
    assert float(safety.floor_clearance) == float(env.floor.clearance)
    # the guard's per-tick joint-speed limit IS the simulator's joint velocity
    # limit expressed per control step: the policy is trained against exactly
    # the rate the robot will enforce, so it never learns a motion the hardware
    # will clip
    assert np.isclose(float(safety.max_joint_step),
                      float(env.robot.arm_velocity_limit) * float(env.control_dt))
    # ... and it must not be TIGHTER than the largest step the action space can
    # command, or the guard would be fighting the policy on every saturated step
    assert float(safety.max_joint_step) >= float(env.action_scale) - 1e-9
