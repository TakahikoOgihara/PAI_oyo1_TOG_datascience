# -*- coding: utf-8 -*-
"""実機の myCobot のデモを、MimicGen の元デモに変換する。

「フェーズ1は実機デモ10本程度で置き換えられる」という主張を、願望では
なく具体的なものにしているファイル。

入力 : deploy/raspi/record_demo.py が書く JSON のエピソード
        {"control_dt", "object_pos", "place_target_pos",
         "steps": [{"q": [6 rad], "grip": 0|1}, ...]}
出力 : フェーズ1と完全に同じ形式の .pt
        {"obs": [(T,31) float32 ...], "act": [(T,7) float32 ...]}
        -> mimicgen.source_file をこれに向ければ、フェーズ1.5はそのまま動く。

手先の姿勢は、ロボットの床ガードが走らせるのと同じ順運動学
（deploy/raspi/mycobot_fk.py）から出す。だからシミュレータと実機は、
「グリッパがどこにいたか」の意味で、作りの上から一致する。

31チャンネルの観測のうち、MimicGen が読むのは eef_pos、eef_quat、
obj_pos、place_target と、行動のグリッパチャンネルだけ。残りは
フェーズ1と同じ形式を保つために埋める。実機の記録からは分からない
ものは、黙って捏造せず下に明記する。

実機の記録から得られないものが2つ。その扱い:
  * カメラ画像。実機デモにシミュレータのRGBは無いので、生成した
    データセットに混ぜられない。mimicgen.keep_source=false を使う。
    生成したデモはシミュレータで再生するので、そちらは画像を持てる。
    視覚運動段に見せたいなら mimicgen.camera=true にする（既定は
    false。配備する画像方策は、代わりにフェーズ7の蒸留データで
    学習するから）。
  * 物体の姿勢。record_demo.py のタッチオフ手順で与える（TCP を
    ブロックに当てて 'o' を押す）。実機で MimicGen を回すときに必ず
    どこかから用意しないといけない注釈がこれで、タッチオフ、
    AprilTag、姿勢推定器が普通の3択。

Usage:
    python3 mimicgen/import_real_demos.py ~/real_demos artifacts/data/real_demos.pt
    # then
    bash run_all.sh mimicgen.source_file=real_demos.pt mimicgen.keep_source=false
"""

from __future__ import annotations
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "deploy", "raspi"))

from mimicgen.pose_np import mat_to_quat  # noqa: E402
from mycobot_fk import TCP_Z, flange_pose  # noqa: E402

OBS_DIM = 31
ACT_DIM = 7
NUM_ARM_JOINTS = 6


def _tcp_pose(q) -> tuple:
    """-> 基準座標での TCP の (位置 (3,), クォータニオン wxyz (4,))。"""
    R, p = flange_pose(q)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    pos = np.asarray(p, dtype=np.float64) + R[:, 2] * TCP_Z
    return pos, mat_to_quat(R)


def convert_episode(ep: dict, action_scale: float = 0.05,
                    gripper_open: float = 0.028,
                    finger_closed: float = 0.0126) -> tuple:
    """記録したエピソード1件 -> フェーズ1の並びでの (obs (T,31), act (T,7))。"""
    steps = ep["steps"]
    T = len(steps)
    q = np.array([s["q"] for s in steps], dtype=np.float64)          # (T, 6)
    grip_open = np.array([int(s["grip"]) for s in steps], dtype=np.float64)
    obj = np.asarray(ep["object_pos"], dtype=np.float64)
    tgt = np.asarray(ep["place_target_pos"], dtype=np.float64)

    eef_pos = np.zeros((T, 3))
    eef_quat = np.zeros((T, 4))
    for t in range(T):
        eef_pos[t], eef_quat[t] = _tcp_pose(q[t])

    # 行動。q[t] を q[t+1] へ動かす関節差分を、環境の単位で表したもの。
    # それに2値のグリッパチャンネル（+1 開 / -1 閉）を足す。
    # deploy/raspi/mycobot_bridge_server.py が実機で積分するのと同じ契約。
    act = np.zeros((T, ACT_DIM))
    dq = np.diff(q, axis=0, append=q[-1:])   # 最後のステップは繰り返し。その後に動きは無い
    act[:, :NUM_ARM_JOINTS] = np.clip(dq / action_scale, -1.0, 1.0)
    act[:, 6] = np.where(grip_open > 0, 1.0, -1.0)

    fingers = np.where(grip_open > 0, gripper_open, finger_closed)
    obs = np.zeros((T, OBS_DIM))
    obs[:, 0:NUM_ARM_JOINTS] = q                     # アームの関節（実測）
    obs[:, 6] = fingers                              # 指の関節（2値の grip から
    obs[:, 7] = fingers                              #   推定）
    obs[:, 8:11] = eef_pos                           # <- MimicGen が使う
    obs[:, 11:15] = eef_quat                         # <- MimicGen が使う
    obs[:, 15:18] = obj                              # <- MimicGen が使う
    obs[:, 18:21] = obj[None] - eef_pos
    obs[:, 21:24] = tgt                              # <- MimicGen が使う
    obs[1:, 24:31] = act[:-1]                        # prev_action
    return obs.astype(np.float32), act.astype(np.float32)


def convert_directory(in_dir: str, out_path: str, action_scale: float = 0.05) -> dict:
    files = sorted(glob.glob(os.path.join(in_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"no *.json episodes in {in_dir}")
    obs_list, act_list, lens, rejected = [], [], [], []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            ep = json.load(fh)
        if not ep.get("object_pos") or not ep.get("place_target_pos"):
            rejected.append((os.path.basename(f), "missing object/place touch-off"))
            continue
        g = {int(s["grip"]) for s in ep["steps"]}
        if g != {0, 1}:
            rejected.append((os.path.basename(f),
                             "gripper never both closed and re-opened"))
            continue
        o, a = convert_episode(ep, action_scale=action_scale)
        obs_list.append(o)
        act_list.append(a)
        lens.append(int(o.shape[0]))

    if not obs_list:
        raise RuntimeError(
            "no usable episodes. MimicGen needs, per demo: an object touch-off, "
            "a place touch-off, and a gripper close followed by a re-open.\n  "
            + "\n  ".join(f"{n}: {why}" for n, why in rejected))

    import torch
    demos = {"obs": [torch.from_numpy(o) for o in obs_list],
             "act": [torch.from_numpy(a) for a in act_list]}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp = out_path + ".tmp"
    torch.save(demos, tmp)
    os.replace(tmp, out_path)

    stats = {"episodes_found": len(files), "converted": len(obs_list),
             "rejected": rejected, "transitions": int(sum(lens)),
             "mean_steps": float(np.mean(lens)), "output": out_path}
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print("\nNext:\n"
          f"  bash run_all.sh mimicgen.source_file={os.path.basename(out_path)} \\\n"
          "                  mimicgen.keep_source=false \\\n"
          "                  mimicgen.selection=random\n"
          "  (keep_source=false because real demos carry no simulator camera\n"
          "   frames; selection=random because with a handful of demos the\n"
          "   'nearest' choice would keep reusing the same one.)")
    return stats


def _self_test() -> None:
    """torch も実機も要らない。幾何の往復を検証する。"""
    q = [0.25, -1.0, -0.1, -0.5, 0.0, 0.0]
    T = 60
    ep = {"control_dt": 0.05,
          "object_pos": [0.20, -0.03, 0.015],
          "place_target_pos": [0.20, 0.02, 0.015],
          "steps": [{"q": q, "grip": 1 if (t < 20 or t >= 50) else 0}
                    for t in range(T)]}
    obs, act = convert_episode(ep)
    assert obs.shape == (T, OBS_DIM) and act.shape == (T, ACT_DIM)

    # obs に書く手先の姿勢は、実機自身のFKと一致しないといけない
    pos, quat = _tcp_pose(q)
    print(f"TCP from FK                   : {np.round(pos, 4)}  "
          f"(home reference [0.2556, -0.0004, 0.1001])")
    assert np.abs(pos - np.array([0.2556, -0.0004, 0.1001])).max() < 1e-3
    assert np.allclose(obs[0, 8:11], pos, atol=1e-6)
    assert np.allclose(obs[0, 11:15], quat, atol=1e-6)
    # ツールは下を向くこと。記録したクォータニオンで +Z を回すと -Z になる
    from mimicgen.pose_np import quat_to_mat
    axis = quat_to_mat(quat) @ np.array([0.0, 0.0, 1.0])
    print(f"tool axis from quaternion     : {np.round(axis, 3)}")
    assert axis[2] < -0.99

    # 作ったものを、注釈器が MimicGen の読む datagen_info に
    # できないといけない
    from mimicgen.annotate import build_episode_arrays, grasp_signal
    sig = grasp_signal(act[:, 6])
    assert sig is not None, "converted real demo must be annotatable"
    edge = int(np.argmax(np.diff(sig[:, 0]) != 0)) + 1
    print(f"grasp signal 0->1 at step     : {edge} (expected 20)")
    assert edge == 20
    arrays = build_episode_arrays(obs, act, action_scale=0.05)
    assert arrays is not None
    dg = arrays["obs"]["datagen_info"]
    assert dg["eef_pose"]["eef"].shape == (T, 4, 4)
    assert dg["target_eef_pose"]["eef"].shape == (T, 4, 4)
    assert np.allclose(dg["object_pose"]["object"][0, :3, 3], ep["object_pos"])
    assert np.allclose(dg["object_pose"]["place_target"][0, :3, 3],
                       ep["place_target_pos"])
    print("datagen_info for MimicGen     : eef/target/object poses OK")
    print("\nimport_real_demos self-test OK")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        convert_directory(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 1:
        _self_test()
    else:
        print(__doc__)
        raise SystemExit(2)
