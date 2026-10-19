# -*- coding: utf-8 -*-
"""PPO の安定化策を、それぞれが存在する理由の性質で試験する。

ARM_BCRL005 は、正しく見えて何もしていなかった BC アンカーで、学習1本を
丸ごと失った（減衰を損失の係数に当てていて、Adam はそれに対して不変）。
試験できない安定化策は、あると仮定しているだけの安定化策。
"""

import numpy as np
import torch
import torch.nn as nn

from mujoco_sim.rl_ppo import BCAnchor


def _actor(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(31, 16), nn.ReLU(), nn.Linear(16, 7), nn.Tanh())


def test_anchor_decay_moves_the_step_size_not_the_loss():
    """ARM_BCRL005 のあのバグ。Adam の更新は lr * m / (sqrt(v) + eps) で、
    損失の定数倍に対して不変。だから係数を 1.0 -> 0.0 と落としても、
    正確に0になるまで何も変わらない。減衰は学習率に当てないといけないし、
    誰かがそれを戻したらこのテストが落ちる。"""
    actor, ref = _actor(0), _actor(1).eval()
    a = BCAnchor(actor, ref, lr=1e-3, coef=1.0, decay_iters=10, minibatch=32,
                 max_grad_norm=1.0, start_iter=0)
    obs = torch.randn(64, 31)
    a.step(obs, 0)
    lr_early = a.optimizer.param_groups[0]["lr"]
    a.step(obs, 8)
    lr_late = a.optimizer.param_groups[0]["lr"]
    assert lr_early > lr_late > 0
    assert np.isclose(lr_late / lr_early, a.coef_at(8) / a.coef_at(0), atol=1e-9)


def test_anchor_is_inactive_before_the_critic_warm_up_ends():
    """アクターが凍結している間にアンカーをかけると、凍結と喧嘩する。"""
    actor, ref = _actor(0), _actor(1).eval()      # 本当に違う重み
    a = BCAnchor(actor, ref, lr=1e-3, coef=1.0, decay_iters=10, minibatch=32,
                 max_grad_norm=1.0, start_iter=5)
    before = [p.detach().clone() for p in actor.parameters()]
    assert a.step(torch.randn(64, 31), 3) == 0.0
    assert all(torch.equal(p, q) for p, q in zip(actor.parameters(), before))
    assert a.step(torch.randn(64, 31), 6) > 0.0


def test_anchor_pulls_the_actor_toward_the_reference():
    actor, ref = _actor(0), _actor(1).eval()
    a = BCAnchor(actor, ref, lr=1e-2, coef=1.0, decay_iters=10_000, minibatch=128,
                 max_grad_norm=10.0, start_iter=0)
    obs = torch.randn(256, 31)
    with torch.no_grad():
        d0 = float((actor(obs) - ref(obs)).pow(2).mean())
    for i in range(50):
        a.step(obs, i)
    with torch.no_grad():
        assert float((actor(obs) - ref(obs)).pow(2).mean()) < d0


def test_anchor_reaches_min_coef_and_stops():
    actor, ref = _actor(), _actor().eval()
    a = BCAnchor(actor, ref, lr=1e-3, coef=1.0, decay_iters=10, minibatch=8,
                 max_grad_norm=1.0, start_iter=0, min_coef=0.0)
    assert a.coef_at(0) == 1.0
    assert a.coef_at(100) == 0.0             # RL は最終的に BC から自由になる
    a2 = BCAnchor(actor, ref, lr=1e-3, coef=1.0, decay_iters=10, minibatch=8,
                  max_grad_norm=1.0, start_iter=0, min_coef=0.25)
    assert a2.coef_at(100) == 0.25


def test_bc_weight_transfer_reproduces_the_bc_policy():
    """フェーズ6の前提。複製が厳密でなければ、このフェーズは「BC からの
    微調整」ではなく「少し温まった状態からの PPO」になる。別の手法で、
    別の結果になる。"""
    from mujoco_sim.rl_ppo import _linears
    src = nn.Sequential(nn.Linear(31, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
                        nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 7), nn.Tanh())
    dst = nn.Sequential(nn.Linear(31, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
                        nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 7), nn.Tanh())
    a, b = _linears(src), _linears(dst)
    assert len(a) == len(b) == 4
    x = torch.randn(128, 31)
    assert float((src(x) - dst(x)).abs().max()) > 1e-3       # 最初は違っている
    with torch.no_grad():
        for u, v in zip(a, b):
            v.weight.copy_(u.weight)
            v.bias.copy_(u.bias)
    assert float((src(x) - dst(x)).abs().max()) < 1e-6
