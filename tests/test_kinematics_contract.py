# -*- coding: utf-8 -*-
"""安全包絡が本物なのは、その全ての実装が一致しているときだけ。

ツールの位置を計算するコードが、独立に4つある:
  deploy/raspi/mycobot_fk.py   Raspberry Pi のガード（標準ライブラリのみ）。これが基準
  arm_common/kinematics.py     シミュレータが強制する、バッチ対応の torch FK
  assets/build_mycobot_mjcf.py MuJoCo が実際に解く MJCF
  arm_common/ik.py             エキスパートと MimicGen が操るヤコビアン

ずれると、方策が学習するアームはガードが守るアームではなくなり、その
差は実機でしか出てこない。これらのテストは、そうならない理由。
"""

import numpy as np
import torch


def test_torch_fk_matches_reference():
    from arm_common.kinematics import ToolKinematics
    ToolKinematics("cpu").assert_matches_reference()        # ずれたら例外


def test_mjcf_fk_matches_reference(tmp_path):
    from assets.build_mycobot_mjcf import build_single, verify
    verify(build_single(str(tmp_path / "arm.xml")), tol=1e-9)


def test_env_tcp_matches_reference_fk(env):
    """MuJoCo が解く TCP サイトと解析の連鎖を、実際の状態で比べる。"""
    env.reset()
    for _ in range(20):
        env.step(torch.rand(env.num_envs, 7) * 2 - 1)
    err = (env.tool_kin.tcp_pos(env.arm_q()) - env.eef_pos_env()).norm(dim=-1).max()
    assert float(err) < 1e-5, f"{float(err) * 1000:.4f} mm"


def test_floor_guard_shortens_never_rejects():
    """ガードは台に食い込む指令を短くするのであって、拒否するのではない。
    拒否すればアームは作業の途中で止まるが、短くすれば方策に接線方向では
    追従し続け、ただ平面より下へは降りられない。"""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "deploy", "raspi"))
    from mycobot_fk import FloorGuard, HOME_Q, tool_min_z
    g = FloorGuard(floor_z=0.0, clearance=0.005)
    dive = [0.25, -1.6, -1.0, 0.2, 0.0, 0.0]
    assert not g.is_safe(dive)
    q, blocked = g.limit(list(HOME_Q), dive)
    assert blocked and tool_min_z(q) >= g.min_z - 1e-9
    # 動いた。指令は切られただけで、捨てられていない
    assert np.linalg.norm(np.array(q) - np.array(HOME_Q)) > 1e-3


def test_batched_guard_matches_scalar_guard():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "deploy", "raspi"))
    from mycobot_fk import FloorGuard, HOME_Q
    from arm_common.kinematics import ToolKinematics
    kin = ToolKinematics("cpu")
    g = FloorGuard(floor_z=0.0, clearance=0.005)
    targets = [[0.25, -1.6, -1.0, 0.2, 0.0, 0.0],
               [0.25, -1.0, -0.1, -0.5, 0.0, 0.0],
               [0.0, -1.8, -0.6, 0.4, 0.2, 0.0]]
    q_from = torch.tensor([list(HOME_Q)] * len(targets))
    q_to = torch.tensor(targets, dtype=torch.float32)
    q_batch, blocked = kin.limit_to_floor(q_from, q_to, 0.005)
    for i, t in enumerate(targets):
        q_ref, blk = g.limit(list(HOME_Q), t)
        assert bool(blocked[i]) == blk
        assert np.abs(q_batch[i].numpy() - np.array(q_ref)).max() < 1e-5


def test_ik_converges_and_stays_bounded():
    from arm_common.ik import DLSPositionIK
    from arm_common.kinematics import ToolKinematics, HOME_Q
    kin = ToolKinematics("cpu")
    ik = DLSPositionIK(kin)
    q = torch.tensor([list(HOME_Q)] * 8, dtype=torch.float32)
    tgt = kin.tcp_pos(q) + (torch.rand(8, 3) * 2 - 1) * 0.04
    for _ in range(60):
        q = ik.solve(q, tgt)
    assert float((kin.tcp_pos(q) - tgt).norm(dim=-1).max()) < 1e-3
    # 特異姿勢では、減衰がステップを有限に保たないといけない
    qs = torch.zeros(1, 6)
    out = ik.solve(qs, kin.tcp_pos(qs) + torch.tensor([[0.5, 0.0, 0.0]]))
    assert torch.isfinite(out).all() and float((out - qs).norm()) <= 0.21
