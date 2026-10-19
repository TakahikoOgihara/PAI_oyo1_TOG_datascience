# -*- coding: utf-8 -*-
"""MimicGen の2段目。生成。注釈済みデモを新しいシーンへ転写する。

アルゴリズム全体（Mandlekar et al., "MimicGen: A Data Generation System
for Scalable Robot Learning using Human Demonstrations", CoRL 2023）:

  1. 元デモをグリッパの遷移でサブタスクに割る（mimicgen/annotate.py）。
     各サブタスクは1つの基準物体についてのもの:
        サブタスク0  近づいて掴む    -> 基準 = ブロック
        サブタスク1  運んで置く      -> 基準 = 設置目標
  2. 環境をリセットする。新しいブロック姿勢と新しい目標になる。
  3. サブタスクごとに元デモを1本選ぶ（基準姿勢の最近傍。出力の多様性を
     保つため、近い順にk本の中から抽選する）。
  4. そのサブタスクの手先目標の軌跡を、基準物体の座標系で表し直し、
     その座標系を「いま物体がある場所」へ置く:

         T_eef_new = T_ref_new @ inv(T_ref_src) @ T_eef_src          (式1)

  5. アームの実際の位置から最初の転写姿勢まで補間する。そうすれば
     区間がつながって飛びが出ない。
  6. シミュレータで実行する（mujoco_sim/train_arm_mujoco.py --mode
     mimicgen）。スクリプトエキスパートと同じIK、同じ7次元の行動空間を
     通し、環境自身の成功判定が出たエピソードだけ残す。

6段目があることで、これはデータの補間ではなくデータの生成になる。
転写した軌跡は仮説で、審判は物理。歩留まりはだいたい30〜80%で、
落ちていく歩留まりは信号（元デモが新しい姿勢へ移せていない）。だから
``mimicgen.min_yield`` は、デモ30本で黙って学習するのではなく実行を
止める。

回転は転写の中では完全な SE(3) で運ぶが、実行側は位置のみのIK
（``mimicgen.ik_mode`` は ARM_BCRL005 でも "position" だった）。この
タスクではツールが常に下を向き、ブロックは回転体なので、転写の回転は
元デモの回転そのもので、追従させても何も増えない。姿勢の計算は一般の
場合で書いてある。非対称な物体が入った日に効くようにしてあり、誰かが
このファイルを書き直す日を待たなくてよい。

自己テスト（numpy だけ。シミュレータ不要）:
    python3 mimicgen/generation.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mimicgen.pose_np import mat_to_quat, quat_to_mat  # noqa: E402

GRASP_SIGNAL = "grasp"
EEF_NAME = "eef"
# サブタスクの添字 -> その軌跡を相対で表す物体
SUBTASK_REFERENCE = ("object", "place_target")


def _inv(T: np.ndarray) -> np.ndarray:
    """同次変換の逆（線形方程式を解かずに転置で作る）。"""
    out = np.eye(4)
    R = T[:3, :3]
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ T[:3, 3]
    return out


def _slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """最短弧のクォータニオン補間。wxyz、t: (K,) -> (K, 4)。"""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    if float(q0 @ q1) < 0.0:
        q1 = -q1                                # 近い方の弧を通る
    dot = float(np.clip(q0 @ q1, -1.0, 1.0))
    if dot > 0.9995:                            # ほぼ平行なら線形補間で厳密
        q = q0[None, :] + t[:, None] * (q1 - q0)[None, :]
    else:
        th = np.arccos(dot)
        s = np.sin(th)
        q = (np.sin((1 - t)[:, None] * th) * q0[None, :]
             + np.sin(t[:, None] * th) * q1[None, :]) / s
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def interpolate_poses(T_from: np.ndarray, T_to: np.ndarray, steps: int) -> np.ndarray:
    """(steps, 4, 4) の経路。位置は線形、回転は SLERP。``T_from`` を含まず
    ``T_to`` を含む。"""
    if steps <= 0:
        return T_to[None, ...].copy()
    t = np.linspace(0.0, 1.0, steps + 1)[1:]
    pos = T_from[:3, 3][None, :] + t[:, None] * (T_to[:3, 3] - T_from[:3, 3])[None, :]
    quat = _slerp(mat_to_quat(T_from[:3, :3]), mat_to_quat(T_to[:3, :3]), t)
    out = np.zeros((steps, 4, 4))
    out[:, :3, :3] = quat_to_mat(quat)
    out[:, :3, 3] = pos
    out[:, 3, 3] = 1.0
    return out


def warp(src_targets: np.ndarray, T_ref_src: np.ndarray,
         T_ref_new: np.ndarray) -> np.ndarray:
    """式1。``src_targets``: (T, 4, 4) -> 新しいシーンでの (T, 4, 4)。"""
    delta = T_ref_new @ _inv(T_ref_src)
    return delta[None, ...] @ src_targets


# ---------------------------------------------------------------------------
class DataGenInfoPool:
    """注釈済みの元デモを、サブタスクに割った形で持つ。

    プロセスごとに1つ作り、生成の試行ごとに引くだけ。だから高い処理
    （HDF5 の読み込み、区切り）は1回で済む。
    """

    def __init__(self, episodes: list):
        self.episodes = episodes
        self.num_subtasks = len(SUBTASK_REFERENCE)

    @classmethod
    def load(cls, path: str, max_demos: int = 0) -> "DataGenInfoPool":
        import h5py
        eps = []
        with h5py.File(path, "r") as f:
            data = f["data"]
            keys = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))
            if max_demos:
                keys = keys[:int(max_demos)]
            for k in keys:
                g = data[k]
                dg = g["obs"]["datagen_info"]
                sig = np.asarray(dg["subtask_term_signals"][GRASP_SIGNAL])[:, 0]
                # 0 -> 1 の遷移がサブタスクの境界。MimicGen の慣習では、
                # サブタスク0の終わりはその1ステップ後に置く
                boundary = int(np.argmax(sig > 0)) + 1
                T = int(g["actions"].shape[0])
                eps.append({
                    "actions": np.asarray(g["actions"]),
                    "target": np.asarray(dg["target_eef_pose"][EEF_NAME], dtype=np.float64),
                    "ref": {name: np.asarray(dg["object_pose"][name], dtype=np.float64)
                            for name in SUBTASK_REFERENCE},
                    "bounds": [(0, boundary), (boundary, T)],
                })
        if not eps:
            raise RuntimeError(f"{path} contains no annotated demonstrations")
        return cls(eps)

    def reference_positions(self, subtask: int) -> np.ndarray:
        """``subtask`` の開始時点での基準物体の位置 (n_demos, 3)。

        最近傍の元デモ選択が使う鍵がこれ。「このサブタスクを、いまの位置に
        いちばん近い物体位置で始めた記録はどれか」。
        """
        name = SUBTASK_REFERENCE[subtask]
        s0 = [ep["bounds"][subtask][0] for ep in self.episodes]
        return np.array([ep["ref"][name][s, :3, 3] for ep, s in zip(self.episodes, s0)])

    def segment(self, demo: int, subtask: int) -> dict:
        ep = self.episodes[demo]
        lo, hi = ep["bounds"][subtask]
        name = SUBTASK_REFERENCE[subtask]
        return {
            "target": ep["target"][lo:hi],
            "gripper": ep["actions"][lo:hi, 6],
            "T_ref_src": ep["ref"][name][lo],
        }


# ---------------------------------------------------------------------------
class TrajectoryGenerator:
    """生成の試行1回を計画する。新しいシーン -> 目標姿勢の列。"""

    def __init__(self, pool: DataGenInfoPool, selection: str = "nearest_neighbor_object",
                 nn_k: int = 3, interpolation_steps: int = 12,
                 action_noise: float = 0.0, seed: int = 0):
        self.pool = pool
        self.selection = str(selection)
        self.nn_k = max(1, int(nn_k))
        self.interp = max(0, int(interpolation_steps))
        self.action_noise = float(action_noise)
        self.rng = np.random.RandomState(int(seed))
        self._ref_pos = [pool.reference_positions(s) for s in range(pool.num_subtasks)]

    def select_source(self, subtask: int, ref_pos_new: np.ndarray) -> int:
        n = len(self.pool.episodes)
        if self.selection == "random" or n <= self.nn_k:
            return int(self.rng.randint(n))
        d = np.linalg.norm(self._ref_pos[subtask] - ref_pos_new[None, :], axis=1)
        # 最近傍1本ではなく、近い順にk本から一様に選ぶ。常に最小を取ると、
        # ある領域で生成したデモが全部同じ元軌跡の複製になり、データの量は
        # 増えても多様性は増えない。
        return int(self.rng.choice(np.argpartition(d, self.nn_k - 1)[:self.nn_k]))

    def plan_subtask(self, subtask: int, T_ref_new: np.ndarray,
                     T_eef_now: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """-> (目標姿勢 (K,4,4), グリッパ指令 (K,), 元デモの添字)。"""
        src = self.select_source(subtask, T_ref_new[:3, 3])
        seg = self.pool.segment(src, subtask)
        warped = warp(seg["target"], seg["T_ref_src"], T_ref_new)
        grip = seg["gripper"]
        if self.interp > 0 and len(warped) > 0:
            bridge = interpolate_poses(T_eef_now, warped[0], self.interp)
            warped = np.concatenate([bridge, warped[1:]], axis=0)
            # 橋渡しの間は、そのサブタスクの最初のグリッパ指令を保つ。橋渡しは
            # 位置を取り直す移動で、実演された操作の一部ではないので、
            # 途中でグリッパが変わってはいけない
            grip = np.concatenate([np.full(self.interp, grip[0]), grip[1:]])
        if self.action_noise > 0:
            warped = warped.copy()
            warped[:, :3, 3] += self.rng.randn(len(warped), 3) * self.action_noise
        return warped, grip, src


# ---------------------------------------------------------------------------
def _self_test() -> None:
    # 合成したサブタスクでの式1。ブロックへ真っ直ぐ降りる動き
    T_ref_src = np.eye(4)
    T_ref_src[:3, 3] = [0.20, -0.03, 0.015]
    src = np.zeros((10, 4, 4))
    src[:, :3, :3] = np.diag([1.0, -1.0, -1.0])          # ツールは下向き
    src[:, 3, 3] = 1.0
    src[:, :3, 3] = np.linspace([0.20, -0.03, 0.09], [0.20, -0.03, 0.023], 10)

    T_ref_new = np.eye(4)
    T_ref_new[:3, 3] = [0.17, 0.02, 0.015]
    out = warp(src, T_ref_src, T_ref_new)
    shift = T_ref_new[:3, 3] - T_ref_src[:3, 3]
    print(f"warp: pure translation? max err "
          f"{np.abs(out[:, :3, 3] - (src[:, :3, 3] + shift)).max():.2e}")
    assert np.abs(out[:, :3, 3] - (src[:, :3, 3] + shift)).max() < 1e-12
    assert np.abs(out[:, :3, :3] - src[:, :3, :3]).max() < 1e-12
    # 保たれるのは、デモが符号化している相対的な幾何。ツールは
    # 区間の終わりで、以前と正確に同じだけブロックの上にいる
    rel_src = src[-1, :3, 3] - T_ref_src[:3, 3]
    rel_new = out[-1, :3, 3] - T_ref_new[:3, 3]
    print(f"grasp offset preserved        : {np.round(rel_src, 5)} -> {np.round(rel_new, 5)}")
    assert np.allclose(rel_src, rel_new)

    # 基準を回したら、軌跡もその周りで回らないといけない（一般の SE(3)）
    a = 0.7
    T_rot = np.eye(4)
    T_rot[:3, :3] = [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
    T_rot[:3, 3] = T_ref_src[:3, 3]
    out_rot = warp(src, T_ref_src, T_rot)
    rel = out_rot[-1, :3, 3] - T_rot[:3, 3]
    assert np.allclose(np.linalg.norm(rel), np.linalg.norm(rel_src))
    print(f"rotated reference             : |offset| preserved ({np.linalg.norm(rel):.5f} m)")

    # 補間は飛びなくつながり、目標にちょうど着地する
    T_from = np.eye(4)
    T_from[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T_from[:3, 3] = [0.25, 0.0, 0.10]
    path = interpolate_poses(T_from, out[0], 8)
    assert path.shape == (8, 4, 4)
    assert np.allclose(path[-1], out[0], atol=1e-9)
    step = np.linalg.norm(np.diff(np.vstack([T_from[:3, 3][None], path[:, :3, 3]]), axis=0), axis=1)
    print(f"interpolation                 : {len(path)} steps, "
          f"max {step.max() * 1000:.1f} mm each, lands on target")
    assert step.std() < 1e-9                   # 等間隔

    # 実際の回転に対する SLERP は、単位ノルムと単調性を保つ
    q = _slerp(np.array([1.0, 0, 0, 0]), np.array([0.0, 1.0, 0, 0]), np.linspace(0, 1, 5))
    assert np.allclose(np.linalg.norm(q, axis=-1), 1.0)
    print(f"slerp norms                   : {np.round(np.linalg.norm(q, axis=-1), 6)}")

    # 元デモの選択は多様性を保つ
    pool = DataGenInfoPool([{
        "actions": np.zeros((20, 7)),
        "target": np.tile(np.eye(4), (20, 1, 1)),
        "ref": {"object": np.tile(np.eye(4), (20, 1, 1)) * 1.0,
                "place_target": np.tile(np.eye(4), (20, 1, 1))},
        "bounds": [(0, 10), (10, 20)],
    } for _ in range(8)])
    for i, ep in enumerate(pool.episodes):
        ep["ref"]["object"][:, :3, 3] = [0.16 + 0.01 * i, 0.0, 0.015]
    gen = TrajectoryGenerator(pool, nn_k=3, seed=0)
    picks = {gen.select_source(0, np.array([0.185, 0.0, 0.015])) for _ in range(40)}
    print(f"nearest-neighbour picks (k=3) : {sorted(picks)}")
    assert 1 < len(picks) <= 3, picks

    print("\ngeneration self-test OK")


if __name__ == "__main__":
    _self_test()
