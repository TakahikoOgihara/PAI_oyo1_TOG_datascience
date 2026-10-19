# -*- coding: utf-8 -*-
"""Shared fixtures. Every test here runs on CPU in seconds and needs no
trained artifact, because a test suite that needs the pipeline to have run is
a test suite nobody runs."""

import os
import sys

import pytest
from omegaconf import OmegaConf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.environ.setdefault("MUJOCO_GL", "osmesa")


@pytest.fixture(scope="session")
def cfg():
    c = OmegaConf.create({
        "env": OmegaConf.load(os.path.join(REPO_ROOT, "configs", "env",
                                           "mycobot_pickplace.yaml")),
        "mimicgen": OmegaConf.load(os.path.join(REPO_ROOT, "configs", "mimicgen",
                                                "default.yaml")),
        "seed": 0,
    })
    return c


@pytest.fixture(scope="session")
def env(cfg):
    from mujoco_sim.env import MuJoCoPickPlaceEnv
    e = MuJoCoPickPlaceEnv(cfg, num_envs=4, with_camera=False, seed=0)
    yield e
    e.close()
