# -*- coding: utf-8 -*-
"""スクリプトの pick&place エキスパート。ベクトル化した経由点の状態機械。

担当するのは段の論理だけ（純粋な torch。シミュレータにもIKにも依存
しない）。現在の手先（TCP）、ワーク、設置目標の位置（環境座標）を
受け取り、環境ごとに目標TCP位置とグリッパの開閉指令を出す。Isaac Lab 版
ではその目標を DifferentialIKController が関節位置の行動に変えていた。
ARM_BCRL001 の LiftPhaseMachine と分担は同一で、持ち上げるだけから
ARM_BCRL005 の「掴んで、持ち上げて、5cm動かして、置く」へ広げたもの。

これが遠隔操作を置き換える「自動デモ生成」の型（Isaac Lab Mimic や
MimicGen が産業化したのと同じ考え方）。学習そのものは robomimic /
ACT / Diffusion Policy が持つ。

段（環境ごとに独立に進む）:
  0 APPROACH  : ワークの approach_height 上で待つ。グリッパ開
  1 DESCEND   : 把持高さまで降りる。グリッパ開
  2 GRASP     : 位置を保ってグリッパを閉じ、grasp_steps 待つ
  3 LIFT      : 搬送高さまで上げる（xyは把持時で固定）。閉じたまま
  4 TRANSPORT : 設置目標の上の搬送高さへ移動。閉じたまま
  5 LOWER     : 目標の上の設置高さまで降りる。閉じたまま
  6 RELEASE   : 位置を保ってグリッパを開き、release_steps 待つ
  7 RETREAT   : 搬送高さまで戻る。グリッパ開
                （ワークが置かれた時点で、環境側が成功として終了する）

整定ゲート（許容内のステップが settle_steps 連続してはじめて経由点
到達と数える）と、把持・持ち上げの xy 固定は ARM_BCRL001 から変えて
いない。どちらも実機で効果を確認した修正。
"""

from __future__ import annotations
import torch

APPROACH, DESCEND, GRASP, LIFT, TRANSPORT, LOWER, RELEASE, RETREAT = range(8)


class PickPlacePhaseMachine:
    def __init__(self, num_envs: int, device: str, approach_height: float,
                 grasp_z_offset: float, pos_tol: float, grasp_steps: int,
                 release_steps: int, transport_height: float, place_z: float):
        self.n = num_envs
        self.device = device
        self.approach_height = float(approach_height)
        self.grasp_z_offset = float(grasp_z_offset)
        self.pos_tol = float(pos_tol)
        self.grasp_steps = int(grasp_steps)
        self.release_steps = int(release_steps)
        self.transport_height = float(transport_height)
        self.place_z = float(place_z)
        self.phase = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.timer = torch.zeros(num_envs, dtype=torch.long, device=device)
        # xy は把持時の値で固定する。そうしないと持ち上げの目標が、すでに
        # 指の中にあるワークを追いかけてしまう
        self.hold_xy = torch.zeros(num_envs, 2, device=device)
        # 整定ゲート（冒頭の説明を参照）
        self.settle = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.settle_steps = 5

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        self.phase[env_ids] = APPROACH
        self.timer[env_ids] = 0
        self.hold_xy[env_ids] = 0.0
        self.settle[env_ids] = 0

    def targets(self, eef_pos: torch.Tensor, obj_pos: torch.Tensor,
                place_pos: torch.Tensor):
        """eef_pos/obj_pos/place_pos はいずれも (n, 3) の環境座標。返すのは
        (target_pos (n,3), gripper_open (n,) bool) で、同時に段を進める
        in place."""
        n = self.n
        dev = self.device
        grasp_z = obj_pos[:, 2] + self.grasp_z_offset
        th = torch.full((n,), self.transport_height, device=dev)
        pz = torch.full((n,), self.place_z, device=dev)

        # 段ごとの目標
        approach = torch.stack([obj_pos[:, 0], obj_pos[:, 1],
                                grasp_z + self.approach_height], dim=-1)
        descend = torch.stack([obj_pos[:, 0], obj_pos[:, 1], grasp_z], dim=-1)
        grasp_hold = torch.stack([self.hold_xy[:, 0], self.hold_xy[:, 1], grasp_z], dim=-1)
        lift = torch.stack([self.hold_xy[:, 0], self.hold_xy[:, 1], th], dim=-1)
        transport = torch.stack([place_pos[:, 0], place_pos[:, 1], th], dim=-1)
        lower = torch.stack([place_pos[:, 0], place_pos[:, 1], pz], dim=-1)
        release_hold = torch.stack([place_pos[:, 0], place_pos[:, 1], pz], dim=-1)
        retreat = torch.stack([place_pos[:, 0], place_pos[:, 1], th], dim=-1)

        tgt = torch.zeros(n, 3, device=dev)
        for ph, t in ((APPROACH, approach), (DESCEND, descend), (GRASP, grasp_hold),
                      (LIFT, lift), (TRANSPORT, transport), (LOWER, lower),
                      (RELEASE, release_hold), (RETREAT, retreat)):
            tgt = torch.where((self.phase == ph).unsqueeze(1), t, tgt)
        gripper_open = (self.phase < GRASP) | (self.phase >= RELEASE)

        # 遷移。経由点の段では、手先が許容球の中に settle_steps 回連続で
        # 収まることを要求する
        reached = torch.norm(eef_pos - tgt, dim=-1) < self.pos_tol
        self.settle = torch.where(reached, self.settle + 1,
                                  torch.zeros_like(self.settle))
        stable = self.settle >= self.settle_steps

        adv_a = (self.phase == APPROACH) & stable
        adv_d = (self.phase == DESCEND) & stable
        adv_l = (self.phase == LIFT) & stable
        adv_t = (self.phase == TRANSPORT) & stable
        adv_w = (self.phase == LOWER) & stable
        self.phase[adv_a] = DESCEND
        self.phase[adv_d] = GRASP
        self.hold_xy[adv_d] = eef_pos[adv_d, :2]
        self.phase[adv_l] = TRANSPORT
        self.phase[adv_t] = LOWER
        self.phase[adv_w] = RELEASE
        self.settle[adv_a | adv_d | adv_l | adv_t | adv_w] = 0

        # 時間待ちの段（GRASP は閉じ、RELEASE は開く。どちらも静止したまま）
        in_grasp = self.phase == GRASP
        in_release = self.phase == RELEASE
        self.timer[in_grasp | in_release] += 1
        done_grasp = in_grasp & (self.timer >= self.grasp_steps)
        done_release = in_release & (self.timer >= self.release_steps)
        self.phase[done_grasp] = LIFT
        self.phase[done_release] = RETREAT
        self.timer[~(in_grasp | in_release)] = 0
        self.timer[done_grasp | done_release] = 0

        return tgt, gripper_open
