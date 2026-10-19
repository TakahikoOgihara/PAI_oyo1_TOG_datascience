# -*- coding: utf-8 -*-
"""The PPO stabilisers, tested for the property each one exists to provide.

ARM_BCRL005 lost a full training run to a BC anchor that looked correct and did
nothing (the decay was applied to the loss coefficient, which Adam is invariant
to). A stabiliser you cannot test is a stabiliser you are assuming.
"""

import numpy as np
import torch
import torch.nn as nn

from mujoco_sim.rl_ppo import BCAnchor


def _actor(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(31, 16), nn.ReLU(), nn.Linear(16, 7), nn.Tanh())


def test_anchor_decay_moves_the_step_size_not_the_loss():
    """THE bug from ARM_BCRL005. Adam's update is lr * m / (sqrt(v) + eps),
    which is INVARIANT to scaling the loss -- so a coefficient ramped 1.0 -> 0.0
    changes nothing until it is exactly 0. The decay must land on the learning
    rate, and this test fails if anyone ever moves it back."""
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
    """Anchoring while the actor is frozen would fight the freeze."""
    actor, ref = _actor(0), _actor(1).eval()      # genuinely different weights
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
    assert a.coef_at(100) == 0.0             # RL is eventually free of BC
    a2 = BCAnchor(actor, ref, lr=1e-3, coef=1.0, decay_iters=10, minibatch=8,
                  max_grad_norm=1.0, start_iter=0, min_coef=0.25)
    assert a2.coef_at(100) == 0.25


def test_bc_weight_transfer_reproduces_the_bc_policy():
    """The premise of phase 6. If the copy is not exact the phase is not
    'fine-tuning from BC', it is 'PPO from a warm-ish start' -- a different
    method with different results."""
    from mujoco_sim.rl_ppo import _linears
    src = nn.Sequential(nn.Linear(31, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
                        nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 7), nn.Tanh())
    dst = nn.Sequential(nn.Linear(31, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
                        nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 7), nn.Tanh())
    a, b = _linears(src), _linears(dst)
    assert len(a) == len(b) == 4
    x = torch.randn(128, 31)
    assert float((src(x) - dst(x)).abs().max()) > 1e-3       # different to start with
    with torch.no_grad():
        for u, v in zip(a, b):
            v.weight.copy_(u.weight)
            v.bias.copy_(u.bias)
    assert float((src(x) - dst(x)).abs().max()) < 1e-6
