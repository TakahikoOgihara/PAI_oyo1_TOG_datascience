# -*- coding: utf-8 -*-
"""MimicGen の1段目。注釈。フェーズ1のデモを物体中心のデータセットにする。

MimicGen は生のデモを読まない。各エピソードが ``datagen_info`` 群を
持つデータセットを読む:

    data/demo_<i>/
        actions                              (T, 7)
        obs/datagen_info/eef_pose/eef        (T, 4, 4)   ツールが実際にいた場所
        obs/datagen_info/target_eef_pose/eef (T, 4, 4)   行けと指示された場所
        obs/datagen_info/object_pose/object       (T, 4, 4)
        obs/datagen_info/object_pose/place_target (T, 4, 4)
        obs/datagen_info/subtask_term_signals/grasp (T, 1)  把持の瞬間に 0 -> 1

これは NVIDIA の ``isaaclab_mimic`` の注釈器が作り、
``DataGenInfoPool.load_from_dataset_file()`` が食う並びそのもの。
ARM_BCRL006 はスキーマを保ったまま、Isaac Lab のデータセットハンドラ
ではなく素の h5py で書く。だからファイルは中を見られて移植もでき、
道具立てはローカルのままになる。ここで書く（環境に信号を出させるため
に全デモをシミュレータで再生し直すのではなく）から、フェーズ1.5は
パイプラインの1段で済む。上の全フィールドは、フェーズ1の31次元の観測と
7次元の行動に、すでに記録されている。

サブタスクの境界はどこから来るか。グリッパのチャンネル:

    開 ... 開 | 閉 ... 閉 | 開 ... 開
                  ^                   ^
                  把持が成立        ワークを放す
    <-- サブタスク0 -><------- サブタスク1 ------------->
      基準: ワーク      基準: 設置目標

``arm_common/expert.py`` は8段の正確な番号を持っていて、それを記録すれば
正確なのは当たり前。ただしそれはスクリプトエキスパート由来のデモの話。
ここで MimicGen を置く意味は、フェーズ1が実機デモ10本程度で置き換え
られることにある。そして人が手で誘導したアームは段の番号を出さない。
出すのは関節角と、グリッパの開閉信号。両方の世界に存在する信号で
区切ることが、シミュレータ由来と実機由来を入れ替え可能にする。実機の
遠隔操作データに対する標準的な経験則でもあり、フェーズ1側の変更は
ゼロで済む。

目標姿勢は記録した行動から、ロボットの仮想床が走らせるのと同じ順運動学
（``deploy/raspi/mycobot_fk.py``）を通して作る。
``target = FK(q + action[:6] * action_scale)``。MimicGen が転写するのは
目標姿勢の列で、実測の列ではない。だから元デモの制御遅れが、生成した
全デモに焼き付くことはない。

単体で走らせてデータセットに注釈を付ける:
    python3 mimicgen/annotate.py artifacts/data/bc_demos.pt \\
                                 artifacts/data/bc_demos_annotated.hdf5
自己テスト（numpy だけ。シミュレータも torch も不要）:
    python3 mimicgen/annotate.py
"""

from __future__ import annotations
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "deploy", "raspi"))

from mimicgen.pose_np import make_pose, mat_to_quat  # noqa: E402

# mujoco_sim/train_arm_mujoco.py の観測の並び（31次元）:
#   q(8) | eef_pos(3) | eef_quat(4) | obj_pos(3) | obj-eef(3) |
#   place_target(3) | prev_action(7)
OBS_ARM_Q = slice(0, 6)
OBS_EEF_POS = slice(8, 11)
OBS_EEF_QUAT = slice(11, 15)     # wxyz、ロボット基準座標（base == 環境原点）
OBS_OBJ_POS = slice(15, 18)
OBS_PLACE_TARGET = slice(21, 24)
ACT_GRIPPER = 6

# 把持がデモの両端からこれより近いデモは捨てる。これは
# configs/mimicgen/default.yaml の subtask_term_offset_range を
# どこまで押せるかの上限にもなっていて、生成中に Isaac Lab Mimic 自身の
# 「got empty subtasks!」の例外が出ないようにしている。
MIN_SUBTASK_STEPS = 10


def grasp_signal(gripper: np.ndarray,
                 min_steps: int = MIN_SUBTASK_STEPS) -> np.ndarray | None:
    """(T,) のグリッパチャンネル（>0 が開）-> (T,1) の 0/1 把持信号、または None。

    信号はグリッパが開いている間は0で、最初に閉じたステップから1に
    固定される。Isaac Lab Mimic はサブタスクの境界を 0->1 の遷移に置く
    （``DataGenInfoPool._add_episode``）ので、固定する信号が正しい形。
    「起きている」ではなく「完了した」の旗だから、そういう意味付けに
    してある。

    綺麗な閉じ→再開のサイクルが無いデモでは、例外ではなく None を返す。
    MimicGen の考え方は一貫して「たくさん作って、妥当なものを残す」。
    形の崩れた元デモは捨てて、数える。
    """
    g = np.asarray(gripper).reshape(-1)
    T = int(g.shape[0])
    closed = np.flatnonzero(g <= 0)
    if closed.size == 0:
        return None                                   # 一度も掴んでいない
    c = int(closed[0])
    if not np.any(g[c:] > 0):
        return None                                   # 一度も放していない
    if c < min_steps or (T - c) < min_steps:
        return None                                   # 退化した区間
    sig = np.zeros((T, 1), dtype=np.float32)
    sig[c:] = 1.0
    return sig


def target_eef_poses(obs: np.ndarray, act: np.ndarray,
                     action_scale: float) -> np.ndarray:
    """(T,31) + (T,7) -> (T,4,4)。記録した各行動が指令した TCP 姿勢。

    Isaac Lab Mimic の契約の言葉で言えば ``action_to_target_eef_pose``。
    使うのは ``deploy/raspi/mycobot_fk.py``、つまりロボットの仮想床が
    走らせているまさにその順運動学。だからここでの食い違いは、実機でも
    問題になる食い違いということ。
    """
    from mycobot_fk import TCP_Z, flange_pose      # numpy だけ。シミュレータ不要

    q_cmd = obs[:, OBS_ARM_Q] + act[:, :6] * float(action_scale)
    T = q_cmd.shape[0]
    pos = np.zeros((T, 3))
    quat = np.zeros((T, 4))
    for t in range(T):
        R, p = flange_pose(q_cmd[t])
        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        pos[t] = np.asarray(p, dtype=np.float64) + R[:, 2] * TCP_Z
        quat[t] = mat_to_quat(R)
    return make_pose(pos, quat)


def build_episode_arrays(obs: np.ndarray, act: np.ndarray, action_scale: float,
                         eef_name: str = "eef", grasp_signal_name: str = "grasp",
                         min_steps: int = MIN_SUBTASK_STEPS) -> dict | None:
    """記録した軌跡1本 -> 注釈済みエピソード1件の入れ子の辞書。

    区切れないデモ（綺麗な把持サイクルが無い）では None を返す。
    """
    obs = np.asarray(obs, dtype=np.float64)
    act = np.asarray(act, dtype=np.float64)
    sig = grasp_signal(act[:, ACT_GRIPPER], min_steps)
    if sig is None:
        return None
    # 基準物体の姿勢は、Isaac Lab Mimic の並びに合わせて時刻ごとに
    # 記録する。円柱は軸を揃えて出現し、回転体なので単位回転で厳密。
    # 向きに意味のある物体が入った日のために、make_pose は
    # クォータニオンも取れる。いずれにしても転写そのものは完全な
    # SE(3)。
    return {
        "actions": act.astype(np.float32),
        "obs": {
            "datagen_info": {
                "eef_pose": {
                    eef_name: make_pose(obs[:, OBS_EEF_POS],
                                        obs[:, OBS_EEF_QUAT]).astype(np.float32),
                },
                "target_eef_pose": {
                    eef_name: target_eef_poses(obs, act, action_scale
                                               ).astype(np.float32),
                },
                "object_pose": {
                    "object": make_pose(obs[:, OBS_OBJ_POS]).astype(np.float32),
                    "place_target": make_pose(obs[:, OBS_PLACE_TARGET]
                                              ).astype(np.float32),
                },
                "subtask_term_signals": {grasp_signal_name: sig},
            }
        },
    }


def _write_group(grp, d) -> None:
    for k, v in d.items():
        if isinstance(v, dict):
            _write_group(grp.create_group(k), v)
        else:
            grp.create_dataset(k, data=np.asarray(v))


def write_annotated_dataset(demos: dict, out_path: str, action_scale: float,
                            max_demos: int | None = None, eef_name: str = "eef",
                            grasp_signal_name: str = "grasp",
                            verbose: bool = True) -> dict:
    """{"obs": [...], "act": [...]} -> 上で説明した注釈済み HDF5。

    素の h5py。並びは isaaclab_mimic のものなので、ファイルは中を見たり
    差分を取ったり、NVIDIA の道具に渡したりできる。落としたのは Isaac Sim
    への依存で、形式ではない。
    """
    import h5py

    obs_list, act_list = demos["obs"], demos["act"]
    n = len(obs_list) if max_demos is None else min(len(obs_list), int(max_demos))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    kept, rejected, lens, grasp_at = 0, 0, [], []
    with h5py.File(out_path, "w") as f:
        data = f.create_group("data")
        data.attrs["env_name"] = "arm_bcrl006_pickplace"
        data.attrs["action_scale"] = float(action_scale)
        for i in range(n):
            arrays = build_episode_arrays(np.asarray(obs_list[i]), np.asarray(act_list[i]),
                                          action_scale, eef_name, grasp_signal_name)
            if arrays is None:
                rejected += 1
                continue
            ep = data.create_group(f"demo_{kept}")
            ep.attrs["num_samples"] = int(arrays["actions"].shape[0])
            ep.attrs["success"] = True          # フェーズ1は成功だけ残す
            _write_group(ep, arrays)
            kept += 1
            lens.append(int(arrays["actions"].shape[0]))
            sig = arrays["obs"]["datagen_info"]["subtask_term_signals"][grasp_signal_name]
            grasp_at.append(int(np.argmax(sig > 0)))
        data.attrs["num_demos"] = kept

    stats = {
        "source_demos_seen": n,
        "annotated": kept,
        "rejected_no_clean_grasp": rejected,
        "mean_steps": float(np.mean(lens)) if lens else 0.0,
        "mean_grasp_step": float(np.mean(grasp_at)) if grasp_at else 0.0,
        "subtasks": ["grasp (ref: object)", "place (ref: place_target)"],
        "annotated_dataset": out_path,
    }
    if verbose:
        print("[mimicgen/annotate] " + json.dumps(stats, indent=2, ensure_ascii=False))
        if rejected:
            print(f"[mimicgen/annotate] NOTE: {rejected} demo(s) had no clean "
                  "open->closed->open gripper cycle and were dropped. A large "
                  "fraction here means the source demos are not actually "
                  "completing the task -- fix phase 1 before generating.")
    if kept == 0:
        raise RuntimeError(
            "no demo could be annotated: not one had a clean "
            "open->closed->open gripper cycle. Check phase 1's "
            "artifacts/data/collect_summary.json (source_success_rate).")
    return stats


def load_source_demos(path: str) -> dict:
    """フェーズ1の (.pt) データセットを読む。torch が必要なのはこの呼び出し
    だけで、だから他の部分はシミュレータ無しで試験できる。"""
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


# --------------------------------------------------------------------------
def _self_test() -> None:
    """numpy だけ。信号の形、境界の位置、姿勢の受け渡し、棄却の確認。"""
    T = 90
    grasp_at = 40
    obs = np.zeros((T, 31))
    act = np.zeros((T, 7))
    obj = np.array([0.20, -0.03, 0.015])
    tgt = obj + np.array([0.0, 0.05, 0.0])
    obs[:, OBS_OBJ_POS] = obj
    obs[:, OBS_PLACE_TARGET] = tgt
    obs[:, OBS_EEF_QUAT] = np.array([0.0, 1.0, 0.0, 0.0])       # ツールは下向き
    obs[:grasp_at, OBS_EEF_POS] = np.linspace([0.20, -0.03, 0.09],
                                              [0.20, -0.03, 0.023], grasp_at)
    obs[grasp_at:, OBS_EEF_POS] = np.linspace([0.20, -0.03, 0.023],
                                              [0.20, 0.02, 0.023], T - grasp_at)
    obs[:, OBS_ARM_Q] = np.array([0.25, -1.0, -0.1, -0.5, 0.0, 0.0])
    act[:, ACT_GRIPPER] = 1.0
    act[grasp_at:T - 8, ACT_GRIPPER] = -1.0                     # 閉じて、放した

    sig = grasp_signal(act[:, ACT_GRIPPER])
    assert sig is not None and sig.shape == (T, 1)
    edge = int(np.argmax(np.diff(sig[:, 0]) != 0)) + 1
    print(f"grasp signal 0->1 at step     : {edge}  (expected {grasp_at})")
    assert edge == grasp_at
    # Isaac Lab Mimic は境界を（遷移の添字 + 1）として解釈する。そこから
    # 導く2つのサブタスクは空でなく、デモ全体を覆う必要がある。
    end0 = edge + 1
    assert 0 < end0 < T, (end0, T)
    print(f"Mimic subtask boundaries      : [0, {end0}) and [{end0}, {T})")

    ep = build_episode_arrays(obs, act, action_scale=0.05)
    assert ep is not None
    dg = ep["obs"]["datagen_info"]
    assert dg["eef_pose"]["eef"].shape == (T, 4, 4)
    assert dg["target_eef_pose"]["eef"].shape == (T, 4, 4)
    assert np.allclose(dg["object_pose"]["object"][0, :3, 3], obj)
    assert np.allclose(dg["object_pose"]["place_target"][0, :3, 3], tgt)
    assert np.allclose(dg["eef_pose"]["eef"][0, :3, 3], [0.20, -0.03, 0.09])
    axis = dg["eef_pose"]["eef"][0, :3, :3] @ np.array([0, 0, 1.0])
    print(f"tool axis from recorded quat  : {np.round(axis, 3)}")
    assert axis[2] < -0.99, "tool axis must point down"

    # 目標姿勢。行動が0なら実測した関節のFKになり、グリッパの
    # チャンネルがアームの目標へ漏れてはならない
    tgt_pose = dg["target_eef_pose"]["eef"]
    print(f"target TCP at t=0             : {np.round(tgt_pose[0, :3, 3], 4)}  "
          "(home reference [0.2556, -0.0004, 0.1001])")
    assert np.abs(tgt_pose[0, :3, 3] - np.array([0.2556, -0.0004, 0.1001])).max() < 1e-3

    # 棄却の経路
    assert grasp_signal(np.ones(T)) is None                      # 一度も閉じない
    assert grasp_signal(-np.ones(T)) is None                     # 一度も開かない
    short = np.ones(T); short[1:] = -1.0
    assert grasp_signal(short) is None                           # 閉じるのが早すぎる
    print("rejection of malformed demos  : OK")

    print("\nannotate self-test OK")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        write_annotated_dataset(load_source_demos(sys.argv[1]), sys.argv[2],
                                action_scale=0.05)
    elif len(sys.argv) == 1:
        _self_test()
    else:
        print(__doc__)
        raise SystemExit(2)
