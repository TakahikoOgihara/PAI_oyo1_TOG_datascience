# -*- coding: utf-8 -*-
"""MuJoCo 上の myCobot 280 Pi の pick&place。物理、観測、報酬、成功判定、
安全包絡の単一の正本。

デモ収集、MimicGen 生成、BC と視覚運動方策の評価、PPO 微調整、教師生徒
の蒸留、ビューアが、すべて同じものを使う。ARM_BCRL005 の
``mujoco_sim/train_arm_mujoco.py`` と同じ扱い。

ARM_BCRL005 から変えていないもの（意図的。これは下流の成果物すべてが
定義の土台にしている契約で、移植のついでに黙って動かしたら、この
プロジェクトの比較そのものが無効になる）:

  * 31次元の観測と、その順番
        q(8) | eef_pos(3) | eef_quat(4, wxyz) | obj(3) | obj-eef(3) |
        place_target(3) | prev_action(7)
  * 7次元の行動。関節位置の差分6 + グリッパの2値1チャンネル。有界な
    リード付きの目標積分、仮想床の二分探索、閉じ気味のグリッパ目標、
    2mm/ステップのレート制限付きの閉じ動作も含めて同じ
  * 報酬の全項と重み、成功判定、終了条件の集合
  * 仮想床。同じ平面、同じクリアランス、同じ順運動学
    （``deploy/raspi/mycobot_fk.py``）、Pi 側のガードと同じ二分探索
  * 20 Hz control

必然的に変わったもの。シミュレータ。Isaac Lab の管理クラス
（ObservationManager / RewardManager / EventManager / ActionTerm）は
Isaac Lab のAPIで、移植できる設計ではない。だから同じMDPをここでは
素のメソッドとして書き、EventManager は ``DomainRandomizer`` になった。
対応は1対1。

全部CPUで動く（MuJoCo はCPUのシミュレータ）。方策はGPUにいてよく、
テンソルは制御1ステップにつき1回だけその境界を渡る。(N, 31) の観測なら
その費用は無視できる。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from arm_common.kinematics import ToolKinematics  # noqa: E402
from mujoco_sim.scene import build_scene_xml, index_scene  # noqa: E402

NUM_ARM_JOINTS = 6
NUM_JOINTS = 8
ACT_DIM = 7
OBS_DIM = NUM_JOINTS + 3 + 4 + 3 + 3 + 3 + ACT_DIM      # 31

# 観測のスライス。凍結してある（mimicgen/annotate.py と
# arm_policy/dataset.py がまさにこの位置を参照する）
OBS_ARM_Q = slice(0, 6)
OBS_EEF_POS = slice(8, 11)
OBS_EEF_QUAT = slice(11, 15)
OBS_OBJ_POS = slice(15, 18)
OBS_PLACE_TARGET = slice(21, 24)


def _select_gl_backend() -> str:
    """この機械で実際に動くオフスクリーンGLバックエンドを選ぶ。

    順番は意図的。``osmesa`` は純粋なソフトウェアで、原理的にはいちばん
    遅い。しかしディスプレイサーバもドライバのEGLスタックにも依存しない
    唯一の選択肢で、WSL2 ではそこがオフスクリーン描画の壊れどころ。
    このプロジェクトの基準機では結局3つとも差は20%以内だった。律速が
    ラスタライズではなくフレームごとの画素の読み戻しだからで、だから
    ここでは正しさを優先する。
    MUJOCO_GL=egl|glfw|osmesa で上書きできる。
    """
    if os.environ.get("MUJOCO_GL"):
        return os.environ["MUJOCO_GL"]
    os.environ["MUJOCO_GL"] = "osmesa"
    return "osmesa"


class MuJoCoPickPlaceEnv:
    """1つの MuJoCo モデルの中に住む ``num_envs`` 個の環境をまとめて扱う。"""

    def __init__(self, cfg, num_envs: int, with_camera: bool = False,
                 seed: int = 42, device: str = "cpu"):
        _select_gl_backend()
        import mujoco

        self.mujoco = mujoco
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = device
        self.with_camera = bool(with_camera)
        self.rng = np.random.RandomState(int(seed))
        # 2つ目の乱数系列。摂動ではなくタスクそのものの乱数、つまり
        # ブロックがどこに出てどこへ運ぶかに使う。分けておけば、DR あり/なしの
        # 実行で同じシードなら同じ物体姿勢の列を見る。分けないと DR の抽選が
        # 共有系列をずらし、2つの条件が違うシーンで評価されることになる。
        # それはまさに src/ablation.py が避けるために存在する交絡。
        # だからここは系列を2本持つ。
        self.task_rng = np.random.RandomState(int(seed) + 991)

        e = cfg.env
        self.decimation = int(e.decimation)
        self.sim_dt = float(e.sim_dt)
        self.control_dt = self.decimation * self.sim_dt
        if abs(self.control_dt - float(e.control_dt)) > 1e-9:
            raise ValueError(
                f"env.control_dt ({float(e.control_dt)}) must equal decimation * sim_dt "
                f"({self.decimation} * {self.sim_dt} = {self.control_dt}). The 20 Hz "
                "control contract is shared with deploy/raspi/safety.py.")
        self.max_episode_length = int(e.episode_len_steps)
        self.action_scale = float(e.action_scale)

        xml, origins = build_scene_xml(cfg, self.num_envs, self.with_camera)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.idx = index_scene(self.model, self.num_envs, origins, self.with_camera)
        self.env_origins = torch.as_tensor(origins, dtype=torch.float32)

        # 安全面。Raspberry Pi 側が強制するものと同一。同じ順運動学、
        # 同じクリアランス、同じ二分探索（ARM_BCRL005 の実機での
        # 失敗記録の分類から引き継いだ）。
        self.tool_kin = ToolKinematics(device)
        self.tool_kin.assert_matches_reference()
        f = e.floor
        self.floor_min_z = float(f.z) + float(f.clearance)
        self.floor_band = float(f.penalty_band)
        self.floor_terminate_z = float(f.z) + float(f.terminate_below)

        self.home_q = np.array([float(v) for v in e.robot.default_joint_pos])
        self.gripper_open = float(e.gripper_open)
        self.close_width = 0.42 * 2.0 * float(e.object.radius)
        lim = self.model.jnt_range[
            [self.model.dof_jntid[a] for a in self.idx.arm_dofadr[0]]]
        self.q_lo = torch.as_tensor(lim[:, 0], dtype=torch.float32)
        self.q_hi = torch.as_tensor(lim[:, 1], dtype=torch.float32)

        from mujoco_sim.domain_rand import DomainRandomizer
        self.dr = DomainRandomizer(self)

        self._alloc_buffers()
        self._renderer = None
        self._setconst_scratch = None
        self._cam_h = int(e.camera.height)
        self._cam_w = int(e.camera.width)
        self._verified_fk = False

        n_dr = len(self.dr.active_terms())
        print(f"[env] MuJoCo: {self.num_envs} envs, obs {OBS_DIM}, action {ACT_DIM}, "
              f"control {1.0 / self.control_dt:.0f} Hz "
              f"(decimation {self.decimation} x {self.sim_dt * 1000:.2f} ms)")
        print(f"[env] domain randomisation terms ({n_dr}): {self.dr.active_terms()}")
        print(f"[env] model: nbody={self.model.nbody} nq={self.model.nq} "
              f"nu={self.model.nu} camera={'on' if self.with_camera else 'off'}")

    # ------------------------------------------------------------------
    def _alloc_buffers(self) -> None:
        n = self.num_envs
        nan = float("nan")
        z = lambda *s: torch.zeros(*s, dtype=torch.float32)   # noqa: E731
        self.place_target = z(n, 3)
        self.spawn_xy = z(n, 2)
        self.q_target = torch.as_tensor(
            np.tile(self.home_q, (n, 1)), dtype=torch.float32)
        self.grip_target = torch.full((n, 1), self.gripper_open)
        self.actions = z(n, ACT_DIM)
        self.tool_z = torch.full((n,), 1.0)
        self.floor_blocked = torch.zeros(n, dtype=torch.bool)
        self.episode_length_buf = torch.zeros(n, dtype=torch.long)
        self.max_obj_z = z(n)
        self.min_tool_z = torch.full((n,), 1.0)
        self.was_lifted = torch.zeros(n, dtype=torch.bool)
        self.final_max_obj_z = z(n)
        self.final_place_err = torch.full((n,), nan)
        self.final_min_tool_z = torch.full((n,), nan)
        self.push_timer = z(n)
        self.n_diverged = 0

    # -- 生の状態 -------------------------------------------------------
    def _qpos(self, adr: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(self.data.qpos[adr], dtype=torch.float32)

    def arm_q(self) -> torch.Tensor:
        return self._qpos(self.idx.arm_qadr)

    def finger_q(self) -> torch.Tensor:
        return self._qpos(self.idx.finger_qadr)

    def eef_pos_env(self) -> torch.Tensor:
        p = torch.as_tensor(self.data.site_xpos[self.idx.tcp_siteid], dtype=torch.float32)
        return p - self.env_origins

    def eef_quat_env(self) -> torch.Tensor:
        """TCP の姿勢、wxyz。MuJoCo の ``xquat`` は既に wxyz で、凍結した
        31次元の並び、``mimicgen/pose_np.py``、``mimicgen/import_real_demos.py``
        （実機。シミュレータ無し）がどれも使う規約と同じ。だから Isaac Lab 3.0
        と違って、この境界に変換は要らない。"""
        return torch.as_tensor(self.data.xquat[self.idx.flange_bodyid], dtype=torch.float32)

    def obj_pos_env(self) -> torch.Tensor:
        p = torch.as_tensor(self.data.xpos[self.idx.obj_bodyid], dtype=torch.float32)
        return p - self.env_origins

    def obj_lin_vel(self) -> torch.Tensor:
        v = self.data.qvel[self.idx.obj_dofadr[:, None] + np.arange(3)[None, :]]
        return torch.as_tensor(v, dtype=torch.float32)

    def qpos_policy(self) -> torch.Tensor:
        """配備できる自己受容感覚。アーム6関節 + 直前のグリッパ指令。
        実機が報告できるものと正確に同じ（pymycobot の ``get_radians`` と、
        ランタイム自身が最後に出したグリッパ指令）。"""
        return torch.cat([self.arm_q(), self.actions[:, 6:7]], dim=-1)

    # -- observation ---------------------------------------------------
    def get_observations(self) -> dict:
        eef = self.eef_pos_env()
        obj = self.obj_pos_env()
        obs = torch.cat([
            self.arm_q(), self.finger_q(),          # 8
            eef,                                    # 3
            self.eef_quat_env(),                    # 4
            obj,                                    # 3
            obj - eef,                              # 3
            self.place_target,                      # 3
            self.actions,                           # 7
        ], dim=-1)
        return {"policy": obs.to(self.device)}

    # -- 報酬（ステップごと。重みは configs/env/...yaml のまま） --------
    def _rewards(self) -> torch.Tensor:
        e = self.cfg.env
        w = e.reward
        eef, obj = self.eef_pos_env(), self.obj_pos_env()
        d = torch.norm(eef - obj, dim=-1)
        reach = 1.0 - torch.tanh(10.0 * d)
        lifted = (obj[:, 2] > float(e.lifted_thresh)).float()
        obj_half = float(e.object.height) / 2.0
        height = ((obj[:, 2] - obj_half)
                  / max(1e-6, float(e.lift_height) - obj_half)).clamp(0.0, 1.0)
        d_place = torch.norm(obj[:, :2] - self.place_target[:, :2], dim=-1)
        transport = self.was_lifted.float() * (1.0 - torch.tanh(10.0 * d_place))
        placed = (self.was_lifted
                  & (d_place < float(e.place_tol_xy))
                  & (obj[:, 2] < float(e.place_max_z))).float()
        succ = self.is_success().float()
        remaining = (self.max_episode_length - self.episode_length_buf).clamp(min=0).float()
        floor_enc = ((self.floor_min_z + self.floor_band - self.tool_z)
                     / max(1e-6, self.floor_band)).clamp(0.0, 1.0)
        act_l2 = torch.sum(self.actions ** 2, dim=-1)
        return (float(w.reach) * reach
                + float(w.lifted) * lifted
                + float(w.height) * height
                + float(w.transport) * transport
                + float(w.place) * placed
                + float(w.success) * succ
                + float(w.get("success_time_bonus", 0.0)) * remaining * succ
                - float(w.action_penalty) * act_l2
                - float(w.get("floor_penalty", 0.0)) * floor_enc)

    def is_success(self) -> torch.Tensor:
        e = self.cfg.env
        obj = self.obj_pos_env()
        near = torch.norm(obj[:, :2] - self.place_target[:, :2], dim=-1) < float(e.place_tol_xy)
        down = obj[:, 2] < float(e.place_max_z)
        slow = torch.norm(self.obj_lin_vel(), dim=-1) < 0.10
        released = self.actions[:, 6] > 0
        return self.was_lifted & near & down & slow & released

    # -- 行動項 ----------------------------------------------------------
    def _process_actions(self, action: torch.Tensor) -> None:
        """ARM_BCRL005 の ``MyCobotJointDeltaAction`` の契約をそのまま。
        有界なリード付きの目標積分（アンチワインドアップ）、仮想床の
        二分探索、閉じ気味のグリッパ目標、レート制限付きの閉じ動作。"""
        a = action.detach().to("cpu", torch.float32).clamp(-1.0, 1.0)
        self.actions = a
        q_now = self.arm_q()
        lead = 3.0 * self.action_scale
        self.q_target = (self.q_target + a[:, :NUM_ARM_JOINTS] * self.action_scale) \
            .clamp(q_now - lead, q_now + lead)
        q_want = self.q_target.clamp(self.q_lo, self.q_hi)
        self.q_target, self.floor_blocked = self.tool_kin.limit_to_floor(
            q_now, q_want, self.floor_min_z)
        self.tool_z = self.tool_kin.tool_min_z(q_now)
        self._cross_check_tool_fk()

        grip = torch.where(a[:, 6:7] > 0,
                           torch.full_like(a[:, 6:7], self.gripper_open),
                           torch.full_like(a[:, 6:7], self.close_width))
        self.grip_target = self.grip_target + (grip - self.grip_target).clamp(-0.002, 0.002)

    def _write_ctrl(self) -> None:
        self.data.ctrl[self.idx.arm_actid] = self.q_target.numpy()
        self.data.ctrl[self.idx.finger_actid] = self.grip_target.numpy()

    def _cross_check_tool_fk(self) -> None:
        """プロセスにつき1回、解析的なツールFKが MuJoCo 自身の TCP サイトと
        一致することを確認する。共有している安全包絡を信用できるのはこれが
        あるから。MJCF がガードの計算に使う定数と合わなくなったら、ロボットの
        上ではなくここで分かる。"""
        if self._verified_fk:
            return
        self._verified_fk = True
        err = float((self.tool_kin.tcp_pos(self.arm_q()) - self.eef_pos_env())
                    .norm(dim=-1).max())
        print(f"[env] tool FK vs MuJoCo TCP site: max error {err * 1000:.4f} mm")
        if err > 0.005:
            raise RuntimeError(
                f"tool FK disagrees with the simulated TCP by {err * 1000:.1f} mm. "
                "deploy/raspi/mycobot_fk.py's chain no longer matches the MJCF "
                "(did assets/build_mycobot_mjcf.py change?). The floor guard would "
                "be wrong on the real robot -- fix before training.")

    # -- stepping --------------------------------------------------------
    def step(self, action: torch.Tensor):
        self._process_actions(action)
        self._write_ctrl()
        for _ in range(self.decimation):
            self.mujoco.mj_step(self.model, self.data)
        # mj_step は qpos を最後に積分し、運動学を再計算しない。だから
        # data.site_xpos / data.xpos はまだ1サブステップ前の状態を表している。
        # これを入れずに観測を読むと、同じベクトルの中で eef_pos と obj_pos が
        # 関節角に対して4ms古くなる。実測で TCP 約0.5mmのずれ。設置許容の
        # 4分の1に当たり、記録する全デモにそれが焼き付く。
        # だからここで前方運動学を回し直す。
        self.mujoco.mj_forward(self.model, self.data)
        self.episode_length_buf += 1
        self._update_episode_stats()
        self.dr.interval_push(self.control_dt)

        rew = self._rewards()
        success = self.is_success()
        timeout = self.episode_length_buf >= self.max_episode_length
        # ガードがあるのにツールが面より下に押し込まれた場合は接触か、
        # そもそも面の下から始まった姿勢。非成功としてエピソードを終える。
        # そうすればその軌跡はデータセットに入らず、物理の中に隠れずに
        # 指標として出てくる。
        breach = self.tool_z < self.floor_terminate_z
        term = success
        trunc = (timeout | breach) & ~success

        # 発散ガード。MuJoCo はまれに接触を NaN/Inf まで吹き飛ばす。放って
        # おくと、発散した環境1つが記録デモに NaN を書き込み、被害は数時間後に
        # 「NaN の損失を出すが出どころが分からない学習器」として現れる。
        # 境界で捕まえる。その環境をリセットして数え、エピソードは非成功
        # として捨てる。
        # ここで止めれば、汚染はデータセットに届かない。
        diverged = ~torch.isfinite(
            torch.cat([self.arm_q(), self.obj_pos_env(), rew.unsqueeze(-1)], dim=-1)
        ).all(dim=-1)
        if bool(diverged.any()):
            self.n_diverged += int(diverged.sum())
            print(f"[env] WARNING: {int(diverged.sum())} environment(s) produced a "
                  f"non-finite state and were reset ({self.n_diverged} total). "
                  f"If this is not rare, lower env.sim_dt.", flush=True)
            rew = torch.where(diverged, torch.zeros_like(rew), rew)
            term = term & ~diverged
            trunc = trunc | diverged

        done_ids = (term | trunc).nonzero(as_tuple=False).flatten()
        if len(done_ids) > 0:
            self._reset_idx(done_ids.numpy())
        obs = self.get_observations()
        return obs, rew.to(self.device), term.to(self.device), trunc.to(self.device), {}

    def _update_episode_stats(self) -> None:
        obj_z = self.obj_pos_env()[:, 2]
        self.max_obj_z = torch.maximum(self.max_obj_z, obj_z)
        self.min_tool_z = torch.minimum(self.min_tool_z, self.tool_z)
        self.was_lifted |= obj_z > float(self.cfg.env.lifted_thresh)

    # -- reset ------------------------------------------------------------
    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.RandomState(int(seed))
            self.task_rng = np.random.RandomState(int(seed) + 991)
        self.mujoco.mj_resetData(self.model, self.data)
        self.dr.on_reset_round()
        self._reset_idx(np.arange(self.num_envs))
        self._update_episode_stats()
        return self.get_observations(), {}

    def _reset_idx(self, env_ids: np.ndarray) -> None:
        """順番に意味があり、ARM_BCRL005 と完全に同じ。リセットが物体を
        動かす前に終端時の指標を退避し（環境は ``step`` の中で自動リセット
        し、呼び出し側は退避した値を読む）、次にランダム化し、最後に行動の
        積分器をアームの実際の位置から再開する。"""
        if len(env_ids) == 0:
            return
        t = torch.as_tensor(env_ids, dtype=torch.long)
        self.final_max_obj_z[t] = self.max_obj_z[t]
        self.final_min_tool_z[t] = self.min_tool_z[t]
        self.final_place_err[t] = torch.norm(
            self.obj_pos_env()[t, :2] - self.place_target[t, :2], dim=-1)

        # 発散した環境は、リセットが書かない自由度（飛んでいった邪魔物、
        # 接触速度）に NaN を抱えていることがある。先にこの環境の速度状態を
        # 全部ゼロにして、リセットが発散を引き継がないようにする。
        bad = ~np.isfinite(self.data.qvel)
        if bad.any():
            self.data.qvel[bad] = 0.0
            self.data.qacc[~np.isfinite(self.data.qacc)] = 0.0
            self.data.qpos[~np.isfinite(self.data.qpos)] = 0.0
        self.dr.reset_envs(env_ids)
        if self.dr.needs_setconst:
            # 質量由来のソルバ定数を再計算する
            # （DomainRandomizer.needs_setconst 参照）。mj_setConst は渡した
            # MjData を作業領域として使い、qpos0 の状態で置き去りにするので、
            # 生きている状態を渡してはいけない。だから専用のバッファを使う。
            if self._setconst_scratch is None:
                self._setconst_scratch = self.mujoco.MjData(self.model)
            self.mujoco.mj_setConst(self.model, self._setconst_scratch)
            self.dr.needs_setconst = False
        self.mujoco.mj_forward(self.model, self.data)

        q_now = self.arm_q()
        self.q_target[t] = q_now[t]
        self.grip_target[t] = self.gripper_open
        self.actions[t] = 0.0
        self.tool_z[t] = 1.0
        self.floor_blocked[t] = False
        self.episode_length_buf[t] = 0
        self.max_obj_z[t] = 0.0
        self.min_tool_z[t] = 1.0
        self.was_lifted[t] = False
        self.push_timer[t] = 0.0
        self._write_ctrl()

    # -- 手首カメラ ---------------------------------------------------------
    def camera_rgb_uint8(self) -> torch.Tensor:
        """(N, H, W, 3) の uint8 手首カメラ画像。

        レンダラは1つで、環境ごとのカメラに向け直す。MuJoCo は1回の呼び出しで
        カメラ1つしか描かないので、制御1ステップあたりN回になる。だから
        ``collect``/``distill`` は num_envs を増やすのではなくプロセスで分割
        する（src/collect_data.py）。基準機では、仕事をどう並べても
        グラフィックスは毎秒約50フレームで飽和する。律速はそこにあり、
        物理側ではない。
        """
        if not self.with_camera:
            raise RuntimeError("camera frames requested in a camera-free mode")
        if self._renderer is None:
            self._renderer = self.mujoco.Renderer(self.model, self._cam_h, self._cam_w)
            # 方策に何を見せるかは、既定値ではなく製品上の判断:
            #  * サイトは消す。TCP のマーカーはグリッパの中に描かれるデバッグ
            #    用で、残すと実カメラが絶対に見せない緑の点を手がかりに
            #    学習してしまう。
            #  * 影は消す。この縮尺の MuJoCo の影マップは、台の上にギザギザの
            #    黒い塊を作る。方策が過適合できる模様で、実物の台には無い。
            #    しかも描画時間を食い、描画はこのパイプラインの律速なので、
            #    二重に損になる。
            #    どちらも意図して切っている。
            self._cam_opt = self.mujoco.MjvOption()
            self._cam_opt.sitegroup[:] = 0
            self._renderer.scene.flags[self.mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        out = np.empty((self.num_envs, self._cam_h, self._cam_w, 3), dtype=np.uint8)
        for i in range(self.num_envs):
            self._renderer.update_scene(self.data, camera=int(self.idx.wristcam_id[i]),
                                        scene_option=self._cam_opt)
            out[i] = self._renderer.render()
        return torch.from_numpy(out)

    def render_camera(self, cam_id: int, height: int, width: int) -> np.ndarray:
        """独立した描画（レポート動画やビューアのスクリーンショット用の
        シーンカメラ）。手首カメラの経路と分けてあるので、可視化のために
        方策の入力解像度が不意に変わることはない。"""
        r = self.mujoco.Renderer(self.model, height, width)
        try:
            r.update_scene(self.data, camera=cam_id)
            return r.render()
        finally:
            r.close()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
