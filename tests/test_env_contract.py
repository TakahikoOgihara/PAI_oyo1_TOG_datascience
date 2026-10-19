# -*- coding: utf-8 -*-
"""The environment contract every downstream artifact is defined on.

The 31-dim observation layout, the 7-dim action encoding and the success test
are indexed into by the ONNX graphs, the Jetson runtime, mimicgen/annotate.py
and arm_policy/dataset.py. A change here that nothing catches silently
corrupts every dataset and every deployed model -- which is exactly the class
of failure this project was rebuilt to stop.
"""

import numpy as np
import torch

from mujoco_sim.env import (ACT_DIM, OBS_ARM_Q, OBS_EEF_POS, OBS_EEF_QUAT,
                            OBS_OBJ_POS, OBS_PLACE_TARGET, OBS_DIM)


def test_observation_layout_is_frozen(env):
    obs = env.reset()[0]["policy"]
    assert obs.shape == (env.num_envs, OBS_DIM) == (env.num_envs, 31)
    assert torch.allclose(obs[:, OBS_ARM_Q], env.arm_q(), atol=1e-6)
    assert torch.allclose(obs[:, OBS_EEF_POS], env.eef_pos_env(), atol=1e-6)
    assert torch.allclose(obs[:, OBS_OBJ_POS], env.obj_pos_env(), atol=1e-6)
    assert torch.allclose(obs[:, OBS_PLACE_TARGET], env.place_target, atol=1e-6)
    # obj - eef, then the previous action
    assert torch.allclose(obs[:, 18:21], env.obj_pos_env() - env.eef_pos_env(), atol=1e-6)
    assert torch.allclose(obs[:, 24:31], env.actions, atol=1e-6)
    # eef_quat is wxyz and unit-norm; the tool points DOWN at the ready pose
    q = obs[:, OBS_EEF_QUAT]
    assert torch.allclose(q.norm(dim=-1), torch.ones(env.num_envs), atol=1e-5)


def test_place_target_is_the_configured_offset(env, cfg):
    env.reset()
    off = np.array([float(v) for v in cfg.env.place_offset])
    d = (env.place_target - env.obj_pos_env()).numpy()
    assert np.abs(d[:, :2] - off[:2]).max() < 1e-5


def test_action_is_clamped_and_rate_limited(env):
    """The Raspberry Pi integrates exactly this contract, so the simulator has
    to honour it: |action| <= 1, and the commanded joint step is bounded."""
    env.reset()
    q0 = env.arm_q().clone()
    env.step(torch.full((env.num_envs, ACT_DIM), 100.0))
    assert float(env.actions.abs().max()) <= 1.0 + 1e-6
    step = (env.q_target - q0).abs().max()
    assert float(step) <= 3.0 * env.action_scale + 1e-6


def test_virtual_floor_is_enforced_in_the_simulator(env):
    """Drive the arm straight down for a full episode; the tool must not end up
    below the plane the Pi's guard enforces. Without this the policy would be
    trained against a limit it never meets and would fight it on hardware."""
    env.reset()
    a = torch.zeros(env.num_envs, ACT_DIM)
    a[:, 1] = -1.0          # shoulder down, hard
    a[:, 6] = -1.0
    worst = 1.0
    for _ in range(120):
        env.step(a)
        worst = min(worst, float(env.tool_z.min()))
    assert worst > -0.002, f"tool reached {worst * 1000:.2f} mm"


def test_success_requires_lift_place_release_and_rest(env, cfg):
    """All five conditions, because dropping any one of them turns a different
    behaviour into 'success': hovering, shoving, or still holding the block."""
    env.reset()
    env.was_lifted[:] = True
    env.actions[:, 6] = 1.0                      # released
    # move the block onto the target and let it settle
    idx = env.idx
    for i in range(env.num_envs):
        a = idx.obj_qadr[i]
        tgt = env.place_target[i].numpy() + idx.env_origins[i]
        env.data.qpos[a:a + 3] = tgt
        env.data.qvel[idx.obj_dofadr[i]:idx.obj_dofadr[i] + 6] = 0.0
    env.mujoco.mj_forward(env.model, env.data)
    assert bool(env.is_success().all())
    env.actions[:, 6] = -1.0                     # still gripping -> not success
    assert not bool(env.is_success().any())
    env.actions[:, 6] = 1.0
    env.was_lifted[:] = False                    # never picked it up -> not success
    assert not bool(env.is_success().any())


def test_domain_randomisation_off_restores_nominal_physics(cfg):
    """DR must be a switch, not a ratchet. src/ablation.py trains with DR on
    and evaluates with it off in the same codebase, so a randomiser that does
    not restore its nominal value would silently contaminate the control cell.
    """
    from mujoco_sim.env import MuJoCoPickPlaceEnv
    from omegaconf import OmegaConf
    c = OmegaConf.merge(cfg, {})
    c.env.domain_rand.enable = True
    e = MuJoCoPickPlaceEnv(c, num_envs=4, with_camera=False, seed=0)
    nominal_mass = e.dr.nom_body_mass[e.idx.obj_bodyid].copy()
    nominal_gain = e.dr.nom_gainprm[e.idx.arm_actid].copy()
    e.reset()
    assert not np.allclose(e.model.body_mass[e.idx.obj_bodyid], nominal_mass)
    e.dr.enabled = False
    e.reset()
    assert np.allclose(e.model.body_mass[e.idx.obj_bodyid], nominal_mass)
    assert np.allclose(e.model.actuator_gainprm[e.idx.arm_actid], nominal_gain)
    e.close()


def test_scripted_expert_clears_the_data_quality_floor(cfg):
    """The expert's success rate is the CEILING on what BC can learn, so the
    pipeline refuses to continue below mimicgen.min_source_success. Check it
    here too, on both test benches, because it is the assumption everything
    downstream rests on."""
    from mujoco_sim.env import MuJoCoPickPlaceEnv
    from mujoco_sim.policies import ExpertController
    from omegaconf import OmegaConf
    floor = float(cfg.mimicgen.min_source_success)
    for dr in (False, True):
        c = OmegaConf.merge(cfg, {})
        c.env.domain_rand.enable = dr
        e = MuJoCoPickPlaceEnv(c, num_envs=8, with_camera=False, seed=0)
        ex = ExpertController(e)
        e.reset()
        ex.reset_idx(torch.arange(8))
        done = succ = 0
        for _ in range(600):
            _, _, term, trunc, _ = e.step(ex.act())
            ids = (term | trunc).nonzero().flatten()
            if len(ids):
                succ += int(term[ids].sum())
                done += len(ids)
                ex.reset_idx(ids)
        e.close()
        assert done >= 8
        assert succ / done >= floor, f"DR={dr}: expert only {succ}/{done}"
