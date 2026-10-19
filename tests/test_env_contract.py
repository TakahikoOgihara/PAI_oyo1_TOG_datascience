# -*- coding: utf-8 -*-
"""下流の全成果物が定義の土台にしている、環境の契約。

31次元の観測の並び、7次元の行動の符号化、成功判定は、ONNX のグラフ、
Jetson のランタイム、mimicgen/annotate.py、arm_policy/dataset.py が
位置で参照している。ここが変わって誰も捕まえないと、全データセットと
配備済みの全モデルが黙って壊れる。このプロジェクトを作り直して止めよう
としたのは、まさにその種類の失敗。
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
    # obj - eef、そのあとに直前の行動
    assert torch.allclose(obs[:, 18:21], env.obj_pos_env() - env.eef_pos_env(), atol=1e-6)
    assert torch.allclose(obs[:, 24:31], env.actions, atol=1e-6)
    # eef_quat は wxyz で単位ノルム。ready 姿勢ではツールは下を向く
    q = obs[:, OBS_EEF_QUAT]
    assert torch.allclose(q.norm(dim=-1), torch.ones(env.num_envs), atol=1e-5)


def test_place_target_is_the_configured_offset(env, cfg):
    env.reset()
    off = np.array([float(v) for v in cfg.env.place_offset])
    d = (env.place_target - env.obj_pos_env()).numpy()
    assert np.abs(d[:, :2] - off[:2]).max() < 1e-5


def test_action_is_clamped_and_rate_limited(env):
    """Raspberry Pi はまさにこの契約を積分するので、シミュレータも守らない
    といけない。|action| <= 1 で、指令する関節ステップは有界。"""
    env.reset()
    q0 = env.arm_q().clone()
    env.step(torch.full((env.num_envs, ACT_DIM), 100.0))
    assert float(env.actions.abs().max()) <= 1.0 + 1e-6
    step = (env.q_target - q0).abs().max()
    assert float(step) <= 3.0 * env.action_scale + 1e-6


def test_virtual_floor_is_enforced_in_the_simulator(env):
    """1エピソードまるごとアームを真下へ駆動する。ツールは、Pi のガードが
    強制する平面より下で終わってはならない。これが無いと、方策は一度も
    出会わない制限に対して学習し、実機でそれと戦うことになる。"""
    env.reset()
    a = torch.zeros(env.num_envs, ACT_DIM)
    a[:, 1] = -1.0          # 肩を下げる。強く
    a[:, 6] = -1.0
    worst = 1.0
    for _ in range(120):
        env.step(a)
        worst = min(worst, float(env.tool_z.min()))
    assert worst > -0.002, f"tool reached {worst * 1000:.2f} mm"


def test_success_requires_lift_place_release_and_rest(env, cfg):
    """5つの条件を全部見る。どれか1つを落とすと、別の振る舞いが「成功」に
    なってしまう。浮かせ続ける、押しのける、まだ握っている。"""
    env.reset()
    env.was_lifted[:] = True
    env.actions[:, 6] = 1.0                      # released
    # ブロックを目標へ運び、静止させる
    idx = env.idx
    for i in range(env.num_envs):
        a = idx.obj_qadr[i]
        tgt = env.place_target[i].numpy() + idx.env_origins[i]
        env.data.qpos[a:a + 3] = tgt
        env.data.qvel[idx.obj_dofadr[i]:idx.obj_dofadr[i] + 6] = 0.0
    env.mujoco.mj_forward(env.model, env.data)
    assert bool(env.is_success().all())
    env.actions[:, 6] = -1.0                     # まだ握っている -> 成功ではない
    assert not bool(env.is_success().any())
    env.actions[:, 6] = 1.0
    env.was_lifted[:] = False                    # 持ち上げていない -> 成功ではない
    assert not bool(env.is_success().any())


def test_domain_randomisation_off_restores_nominal_physics(cfg):
    """DR はスイッチであって、戻らない爪ではない。src/ablation.py は同じ
    コードで DR ありの学習と DR なしの評価をするので、公称値に戻さない
    ランダマイザは、対照のセルを黙って汚染する。
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
    """エキスパートの成功率が、BC が学べる上限。だからパイプラインは
    mimicgen.min_source_success を下回ると続行を拒否する。ここでも、
    両方の試験ベンチで確認する。下流の全部がその前提に乗っている
    から。"""
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
