# -*- coding: utf-8 -*-
"""ドメインランダム化。ARM_BCRL005 の EventCfg に対応する MuJoCo 版。

Isaac Lab では DR を ``isaaclab.envs.mdp.events`` から取った
``EventManager`` の項として書いていた。MuJoCo にその管理者はいないので、
同じ軸を ``MjModel`` / ``MjData`` への明示的な書き込みとしてここに実装
する。軸の一覧、範囲、ON/OFFの既定値は
``configs/env/mycobot_pickplace.yaml`` から変えていない。移ったのは
仕組みだけで、対応は下の表のとおり。

| 軸                       | ARM_BCRL005 (Isaac Lab)               | ここでの実装                |
|--------------------------|---------------------------------------|-----------------------------|
| リセット時の関節ノイズ   | mdp.reset_joints_by_offset            | qpos への書き込み           |
| ワークの質量             | mdp.randomize_rigid_body_mass         | model.body_mass + inertia   |
| ワークの摩擦             | mdp.randomize_rigid_body_material     | model.geom_friction[:,0]    |
| ワークの反発             | ... 同じ項                            | model.geom_solref dampratio |
| アクチュエータのゲイン   | 自作項（kp/kv を連動）                | actuator_gainprm/biasprm    |
| 関節摩擦/アーマチュア    | mdp.randomize_joint_parameters        | dof_frictionloss/dof_armature |
| エピソード途中の外乱     | mdp.push_by_setting_velocity (interval) | タイマで qvel 書き込み    |
| 照明 / アルベド          | 自作項（USD）/ Replicator             | light_diffuse, geom_rgba    |

覚えておく性質が2つ。どちらも ARM_BCRL005 から引き継いでいる:

1. *ここでの操作はすべてリセット時のプロパティ書き換えで、描画も
   ステップも増えない。* だから DR が現実的な値段で済む。高い種類の DR
   （フレームごとのテクスチャや背景の再描画）は、既知の台に既知の物体
   1つという状況ではほとんど買えず、PDCA の周期を壊す。
2. *広い DR は無料ではない。* データを増やさずに分布を広げれば成功率は
   下がる。それを主張ではなく測るのが ``src/ablation.py`` の4条件比較。
   YAML の各範囲は「実物についての正直な不確かさ」であって、「できるだけ
   広く」ではない。

ARM_BCRL006 での変更。``domain_rand.enable=false`` のとき、邪魔物の
ブロックをランダムではなく固定の公称姿勢に置く。ARM_BCRL005 では DR の
スイッチが物理軸だけを支配していたので、邪魔物は常にランダムだった。
比較に意味を持たせるには「DR off」がシーンの摂動ゼロでなければならない。
ワーク自身の出現窓はどの条件でもランダムのまま。それは摂動ではなく
タスクだから。
"""

from __future__ import annotations

import numpy as np
import torch


class DomainRandomizer:
    def __init__(self, env):
        self.env = env
        self.model = env.model
        self.cfg = env.cfg
        self.dr = env.cfg.env.domain_rand
        self.enabled = bool(self.dr.enable)
        m = self.model
        # 公称値を一度だけ取っておく。どのランダマイザも「公称 x 係数」を
        # 書き、「前回 x 係数」にはしない。だからリセットを繰り返しても、
        # モデルが設定した動作点からランダムウォークで離れることはない。
        self.nom_body_mass = m.body_mass.copy()
        self.nom_body_inertia = m.body_inertia.copy()
        self.nom_geom_friction = m.geom_friction.copy()
        self.nom_geom_solref = m.geom_solref.copy()
        self.nom_geom_rgba = m.geom_rgba.copy()
        self.nom_gainprm = m.actuator_gainprm.copy()
        self.nom_biasprm = m.actuator_biasprm.copy()
        self.nom_frictionloss = m.dof_frictionloss.copy()
        self.nom_armature = m.dof_armature.copy()
        self.nom_light_diffuse = m.light_diffuse.copy()
        self._push_next = np.zeros(env.num_envs)
        # 質量・慣性・アーマチュアを書いたリセットで立てる旗。これらは
        # MuJoCo のコンパイル時の派生定数（body_invweight0、dof_invweight0、
        # dof_M0）に入っていて、拘束ソルバが接触のインピーダンスを
        # 正規化するのに使う。model.body_mass だけ書くとソルバは古い値で
        # 働き続け、把持が黙って保たなくなる。
        # これが真のとき env._reset_idx が mj_setConst() を呼ぶ。
        self.needs_setconst = False

    # ------------------------------------------------------------------
    def _rng_range(self, key, unit=None):
        rng = self.dr.get(key, None)
        if rng is None:
            return None
        lo, hi = (float(v) for v in rng)
        if unit is not None and lo == unit and hi == unit:
            return None                     # 恒等範囲 == その項はOFF
        return lo, hi

    def active_terms(self) -> list:
        """実際に生きているランダマイザ。起動時に表示する。

        ARM_BCRL005 ではこれが必要だった。PhysX の API が動くとランダマイザが
        黙って無効化され得たから。こちらで怖いのは逆（設定した範囲が恒等に
        なっていること）だが、どちらにしても「この実行は実際に何を
        ランダム化したのか」はログだけで答えられないといけない。
        """
        if not self.enabled:
            return []
        t = ["reset_arm_joint_noise"] if float(self.dr.joint_noise) > 0 else []
        if self._rng_range("object_mass") is not None:
            t.append("object_mass")
        if self._rng_range("object_friction") is not None:
            t.append("object_friction")
        if self._rng_range("object_restitution", 0.0) is not None:
            t.append("object_restitution")
        if self._rng_range("actuator_gain_scale", 1.0) is not None:
            t.append("actuator_gain_scale")
        if (self._rng_range("joint_friction_scale", 1.0) is not None
                or self._rng_range("joint_armature_scale", 1.0) is not None):
            t.append("joint_parameters")
        if self._rng_range("object_push_vel", 0.0) is not None:
            t.append("object_push(interval)")
        if float(self.dr.get("light_jitter", 0.0)) > 0 or self.dr.get("light_intensity"):
            t.append("appearance_light")
        if float(self.dr.get("desk_color_jitter", 0.0)) > 0 \
                or float(self.dr.get("object_color_jitter", 0.0)) > 0:
            t.append("appearance_color")
        if bool(self.dr.distractor):
            t.append("distractor_pose")
        return t

    def _u(self, lo, hi, n=None):
        """摂動側の乱数（DR）。"""
        return self.env.rng.uniform(lo, hi, size=n)

    def _task(self, lo, hi, n=None):
        """タスク側の乱数（ワークがどこに出るか）。別系列にしてあるので、
        同じシードなら DR のON/OFFに関係なく同じシーンになる。
        MuJoCoPickPlaceEnv.__init__ を参照。"""
        return self.env.task_rng.uniform(lo, hi, size=n)

    # ------------------------------------------------------------------
    def on_reset_round(self) -> None:
        """環境ごとではなくモデル全体にかかるランダマイザ。照明と、机と
        ブロックのアルベド。リセット1巡につき1回引き直す。ARM_BCRL005 の
        ``randomize_appearance`` と同じ扱い（Isaac Lab のドーム照明と USD
        マテリアルもモデル全体のものだった）。"""
        m = self.model
        m.light_diffuse[:] = self.nom_light_diffuse
        m.geom_rgba[:] = self.nom_geom_rgba
        if not self.enabled:
            return
        rng = self._rng_range("light_intensity")
        if rng is not None:
            lo, hi = rng
            base = float(self.cfg.env.lighting.intensity)
            scale = float(self._u(lo, hi)) / max(1e-6, base)
            m.light_diffuse[:] = np.clip(self.nom_light_diffuse * scale, 0.02, 1.0)
        for jit_key, geom_ids in (("desk_color_jitter", self.env.idx.desk_geomid),
                                  ("object_color_jitter", self.env.idx.obj_geomid)):
            jit = float(self.dr.get(jit_key, 0.0))
            if jit <= 0.0:
                continue
            d = self._u(-jit, jit, (len(geom_ids), 3))
            m.geom_rgba[geom_ids, :3] = np.clip(
                self.nom_geom_rgba[geom_ids, :3] + d, 0.0, 1.0)

    # ------------------------------------------------------------------
    def reset_envs(self, env_ids: np.ndarray) -> None:
        """環境ごとのリセット。アームの姿勢、ワークと目標、邪魔物、そして
        物理軸。``env._reset_idx`` から ``mj_forward`` の前に呼ばれる。"""
        e = self.cfg.env
        m, d = self.model, self.env.data
        idx = self.env.idx
        n = len(env_ids)

        # -- アームと指 ----------------------------------------------------
        noise = float(self.dr.joint_noise) if self.enabled else 0.0
        q = np.tile(self.env.home_q, (n, 1))
        if noise > 0:
            q = q + self._u(-noise, noise, (n, len(self.env.home_q)))
        d.qpos[idx.arm_qadr[env_ids]] = q
        d.qvel[idx.arm_dofadr[env_ids]] = 0.0
        d.qpos[idx.finger_qadr[env_ids]] = float(e.gripper_open)
        d.qvel[idx.finger_dofadr[env_ids]] = 0.0

        # -- ワークの出現と設置目標（タスクなので常にランダム） -----------
        x = self._task(float(e.object.spawn_x[0]), float(e.object.spawn_x[1]), n)
        y = self._task(float(e.object.spawn_y[0]), float(e.object.spawn_y[1]), n)
        z = float(e.object.height) / 2.0
        org = idx.env_origins[env_ids]
        qadr = idx.obj_qadr[env_ids]
        for k in range(n):
            a = qadr[k]
            d.qpos[a:a + 3] = (org[k, 0] + x[k], org[k, 1] + y[k], org[k, 2] + z)
            d.qpos[a + 3:a + 7] = (1.0, 0.0, 0.0, 0.0)
            dof = idx.obj_dofadr[env_ids][k]
            d.qvel[dof:dof + 6] = 0.0
        off = [float(v) for v in e.place_offset]
        ids = torch.as_tensor(env_ids, dtype=torch.long)
        self.env.place_target[ids, 0] = torch.as_tensor(x + off[0], dtype=torch.float32)
        self.env.place_target[ids, 1] = torch.as_tensor(y + off[1], dtype=torch.float32)
        self.env.place_target[ids, 2] = float(z + off[2])
        self.env.spawn_xy[ids, 0] = torch.as_tensor(x, dtype=torch.float32)
        self.env.spawn_xy[ids, 1] = torch.as_tensor(y, dtype=torch.float32)

        # -- distractor ----------------------------------------------------
        if idx.dist_qadr is not None:
            size = float(self.dr.distractor_size)
            if self.enabled:
                dx = self._u(float(self.dr.distractor_x[0]), float(self.dr.distractor_x[1]), n)
                dy = self._u(float(self.dr.distractor_y[0]), float(self.dr.distractor_y[1]), n)
                # ワークと目標の間の通り道を塞がない位置にする
                too_close = (np.abs(dx - x) < 0.07) & (np.abs(dy - y) < 0.12)
                dy = np.where(too_close, dy + 0.15 * np.sign(dy + 1e-6), dy)
            else:
                dx = np.full(n, 0.28)
                dy = np.full(n, 0.18)
            dq = idx.dist_qadr[env_ids]
            for k in range(n):
                a = dq[k]
                d.qpos[a:a + 3] = (org[k, 0] + dx[k], org[k, 1] + dy[k], org[k, 2] + size / 2)
                d.qpos[a + 3:a + 7] = (1.0, 0.0, 0.0, 0.0)

        if not self.enabled:
            self._restore_physics(env_ids)
            return

        # -- 物理軸 ---------------------------------------------------------
        bids = idx.obj_bodyid[env_ids]
        rng = self._rng_range("object_mass")
        if rng is not None:
            mass = self._u(rng[0], rng[1], n)
            scale = mass / np.maximum(self.nom_body_mass[bids], 1e-9)
            m.body_mass[bids] = mass
            m.body_inertia[bids] = self.nom_body_inertia[bids] * scale[:, None]
            self.needs_setconst = True

        gids = idx.obj_geomid[env_ids]
        rng = self._rng_range("object_friction")
        if rng is not None:
            # 静摩擦 == 動摩擦（MuJoCo のすべり摩擦係数は1つ）。ARM_BCRL005 でも
            # 結局そう設定していた。
            m.geom_friction[gids, 0] = self._u(rng[0], rng[1], n)
        rng = self._rng_range("object_restitution", 0.0)
        if rng is not None:
            # MuJoCo に反発係数は無い。跳ね返りは接触ソルバの基準の減衰比
            # solref[1] として表す（1.0 = 臨界減衰 = 跳ねない）。ここで写して
            # おけば、YAML 側はソルバのパラメータではなく物理量を書いたまま
            # でいられる。
            r = self._u(rng[0], rng[1], n)
            m.geom_solref[gids, 1] = np.clip(1.0 - 0.9 * r, 0.05, 1.0)

        rng = self._rng_range("actuator_gain_scale", 1.0)
        if rng is not None:
            # 剛性と減衰を同じ係数で動かす。Isaac Lab の
            # randomize_actuator_gains は独立に引くので、減衰比が動いて、
            # 振動する環境や這うような環境ができる。それは頑健性ではなく
            # ノイズ。連動させれば調整した比のまま動き、ゲインを
            # ランダム化する意味が保たれる。
            s = self._u(rng[0], rng[1], n)[:, None]
            aids = idx.arm_actid[env_ids]
            m.actuator_gainprm[aids, 0] = self.nom_gainprm[aids, 0] * s
            m.actuator_biasprm[aids, 1] = self.nom_biasprm[aids, 1] * s
            m.actuator_biasprm[aids, 2] = self.nom_biasprm[aids, 2] * s

        fr = self._rng_range("joint_friction_scale", 1.0)
        ar = self._rng_range("joint_armature_scale", 1.0)
        if fr is not None or ar is not None:
            dofs = idx.arm_dofadr[env_ids]
            if fr is not None:
                m.dof_frictionloss[dofs] = (self.nom_frictionloss[dofs]
                                            * self._u(fr[0], fr[1], (n, 1)))
            if ar is not None:
                m.dof_armature[dofs] = (self.nom_armature[dofs]
                                        * self._u(ar[0], ar[1], (n, 1)))
                self.needs_setconst = True

        pv = self._rng_range("object_push_vel", 0.0)
        if pv is not None:
            self._push_next[env_ids] = self._u(2.0, 6.0, n)

    def _restore_physics(self, env_ids: np.ndarray) -> None:
        """DR off のとき、ランダム化対象のプロパティを全部公称値へ戻す。

        必要なのは、データを集めた実行と方策を評価する実行とで
        ``domain_rand.enable`` が違い得るから。4条件比較はまさにそれを
        やっている。
        """
        m = self.model
        idx = self.env.idx
        bids, gids = idx.obj_bodyid[env_ids], idx.obj_geomid[env_ids]
        aids, dofs = idx.arm_actid[env_ids], idx.arm_dofadr[env_ids]
        if not np.array_equal(m.body_mass[bids], self.nom_body_mass[bids]):
            self.needs_setconst = True
        m.body_mass[bids] = self.nom_body_mass[bids]
        m.body_inertia[bids] = self.nom_body_inertia[bids]
        m.geom_friction[gids] = self.nom_geom_friction[gids]
        m.geom_solref[gids] = self.nom_geom_solref[gids]
        m.actuator_gainprm[aids] = self.nom_gainprm[aids]
        m.actuator_biasprm[aids] = self.nom_biasprm[aids]
        m.dof_frictionloss[dofs] = self.nom_frictionloss[dofs]
        if not np.array_equal(m.dof_armature[dofs], self.nom_armature[dofs]):
            self.needs_setconst = True
        m.dof_armature[dofs] = self.nom_armature[dofs]

    # ------------------------------------------------------------------
    def interval_push(self, dt: float) -> None:
        """エピソード途中の外乱。リセット時だけのランダマイザでは表現でき
        ない軸。既定はOFF。ONにした DR の中でもスクリプトエキスパートは
        成功し続けないといけない（その成功率がデータ品質の上限）から。
        RL専用の頑健性比較には役立つ。"""
        if not self.enabled:
            return
        pv = self._rng_range("object_push_vel", 0.0)
        if pv is None:
            return
        self.env.push_timer += dt
        due = (self.env.push_timer.numpy() >= self._push_next).nonzero()[0]
        if len(due) == 0:
            return
        d = self.env.data
        for i in due:
            dof = int(self.env.idx.obj_dofadr[i])
            d.qvel[dof:dof + 2] += self._u(pv[0], pv[1], 2) * self.env.rng.choice([-1, 1], 2)
            self.env.push_timer[int(i)] = 0.0
            self._push_next[i] = self._u(2.0, 6.0)
