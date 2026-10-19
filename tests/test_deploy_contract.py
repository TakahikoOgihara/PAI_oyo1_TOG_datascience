# -*- coding: utf-8 -*-
"""出荷するもの。ONNX のグラフと、それと一緒に運ばれる安全包絡。

ARM_BCRL002 は、シミュレータで成功率0.55・設置誤差64mmの方策を配備して、
実機で10回中0回だった。3つの原因のうち2つをここで覆う。グラフは
学習したモデルと同じものを計算しないといけないし、リリースゲートは
実際に拒否しないといけない。
"""

import os

import numpy as np
import torch
import torch.nn as nn

from mlops import export as onnx_export


def test_onnx_export_matches_torch_and_supports_batching(tmp_path):
    """dynamic_axes は書き出しを越えて生き残らないといけない。torch 2.9 以降の
    dynamo の書き出しはバッチ次元を黙って固定し、1標本の検証は通るのに
    シミュレータが64環境を同時に評価した瞬間に落ちるグラフを作る。
    mlops/export.py がこの理由で旧来のトレーサを強制していて、それが
    効かなくなったら気づくのがこのテスト。"""
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
    """ゲートはパイプラインでいちばん安い管理策で、「MLOps の仕組みがある」を
    「うちの MLOps が出荷を決めている」に変える唯一の部品。"""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from omegaconf import OmegaConf
    from src.infer import _release_gate
    cfg = OmegaConf.create({"deploy": {"min_success_rate": 0.55, "max_place_err": 0.035,
                                       "min_lift": 0.055, "force": False},
                            "env": {"floor": {"z": 0.0, "clearance": 0.005}}})
    good = {"success_rate": 0.62, "mean_place_err": 0.031, "mean_max_lift": 0.071}
    assert _release_gate(cfg, good, "ACT")
    # ARM_BCRL002 の成果物そのもの。成功率55%、誤差64mm、持ち上げ69mm
    bad = {"success_rate": 0.553, "mean_place_err": 0.064, "mean_max_lift": 0.069}
    assert not _release_gate(cfg, bad, "ACT")
    cfg.deploy.force = True
    assert _release_gate(cfg, bad, "ACT")      # 明示的な上書きは今もできる


def test_edge_config_carries_the_safety_envelope():
    """Pi は床の平面とレート制限を読まないといけない。二度打ちはしない。
    安全の定数が2つあるのは、安全の定数が1つあることと、いつか起きるバグが
    1つあることの合計。"""
    import sys
    from omegaconf import OmegaConf
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    safety = OmegaConf.load(os.path.join(root, "configs", "safety", "default.yaml"))
    env = OmegaConf.load(os.path.join(root, "configs", "env", "mycobot_pickplace.yaml"))
    # シミュレータの平面とロボットの平面は、同じ2つの数
    assert float(safety.floor_z) == float(env.floor.z)
    assert float(safety.floor_clearance) == float(env.floor.clearance)
    # ガードの刻みごとの関節速度の上限は、シミュレータの関節速度の上限を
    # 制御ステップあたりで表したもの。方策はロボットが強制するのと同じ
    # レートに対して学習するので、ハードウェアが切る動きを学ぶことがない
    assert np.isclose(float(safety.max_joint_step),
                      float(env.robot.arm_velocity_limit) * float(env.control_dt))
    # …そして、行動空間が指令できる最大のステップより厳しくてもいけない。
    # 厳しいと、飽和した刻みごとにガードが方策と戦うことになる
    assert float(safety.max_joint_step) >= float(env.action_scale) - 1e-9
