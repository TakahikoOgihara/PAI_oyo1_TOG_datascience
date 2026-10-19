# -*- coding: utf-8 -*-
"""myCobot 280 Pi pick&place in MuJoCo -- the single source of truth for
physics, observation, reward, success judgment and the safety envelope.

Used identically by demo collection, MimicGen generation, BC/visuomotor
evaluation, PPO fine-tuning, teacher-student distillation and the viewer, just
as ``mujoco_sim/train_arm_mujoco.py`` was in ARM_BCRL005.

WHAT IS UNCHANGED FROM ARM_BCRL005 (deliberately -- these are the contracts
every downstream artifact is defined on, and a port that quietly moved them
would invalidate the comparison this project exists to make):

  * the 31-dim observation and its ORDER
        q(8) | eef_pos(3) | eef_quat(4, wxyz) | obj(3) | obj-eef(3) |
        place_target(3) | prev_action(7)
  * the 7-dim action: 6 joint-position deltas + 1 binary gripper channel,
    with bounded-lead target integration, the virtual-floor bisection, the
    under-closed gripper target and the 2 mm/step rate-limited close
  * every reward term and weight, the success test, the termination set
  * the virtual floor: same plane, same clearance, same forward kinematics
    (``deploy/raspi/mycobot_fk.py``), same bisection as the Raspberry Pi guard
  * 20 Hz control

WHAT NECESSARILY CHANGED: the simulator. Isaac Lab's manager classes
(ObservationManager / RewardManager / EventManager / ActionTerm) are Isaac Lab
API, not a portable design, so the same MDP is expressed as plain methods here
and the EventManager becomes ``DomainRandomizer``. The mapping is one-to-one
and is tabulated in change log §3.

Everything runs on CPU (MuJoCo is a CPU simulator); policies may live on the
GPU and the tensors cross that boundary once per control step, which for a
(N, 31) observation is free.
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

# observation slices, frozen (mimicgen/annotate.py and arm_policy/dataset.py
# index into exactly these)
OBS_ARM_Q = slice(0, 6)
OBS_EEF_POS = slice(8, 11)
OBS_EEF_QUAT = slice(11, 15)
OBS_OBJ_POS = slice(15, 18)
OBS_PLACE_TARGET = slice(21, 24)


def _select_gl_backend() -> str:
    """Pick an offscreen GL backend that actually works on this machine.

    Order is deliberate. ``osmesa`` is pure software: slowest in principle but
    the only one with no dependency on a display server or a driver's EGL
    stack, which under WSL2 is where offscreen rendering usually breaks. On
    this project's reference machine all three measure within 20 % of each
    other anyway (operations guide §5), because the bottleneck is the
    per-frame pixel read-back, not rasterisation -- so correctness wins.
    Override with MUJOCO_GL=egl|glfw|osmesa.
    """
    if os.environ.get("MUJOCO_GL"):
        return os.environ["MUJOCO_GL"]
    os.environ["MUJOCO_GL"] = "osmesa"
    return "osmesa"


class MuJoCoPickPlaceEnv:
    """Vectorised over ``num_envs`` environments living in one MuJoCo model."""

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
        # A SECOND stream, for the randomness that is the TASK rather than a
        # perturbation: where the block spawns and where it has to go.
        # Keeping it separate means a DR-on and a DR-off run with the same seed
        # see the SAME sequence of object poses -- otherwise the DR draws shift
        # the shared stream and the two conditions are evaluated on different
        # scenes, which is exactly the confound src/ablation.py exists to avoid.
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

        # SAFETY PLANE -- identical to the one the Raspberry Pi enforces: same
        # forward kinematics, same clearance, same bisection
        # (field-failure notes §4 of ARM_BCRL005, carried over).
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

    # -- raw state -----------------------------------------------------
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
        """TCP orientation, wxyz. MuJoCo's ``xquat`` is already wxyz, which is
        the convention the frozen 31-dim layout, ``mimicgen/pose_np.py`` and
        ``mimicgen/import_real_demos.py`` (real robot, no simulator) all use --
        so unlike Isaac Lab 3.0 there is no conversion at this boundary."""
        return torch.as_tensor(self.data.xquat[self.idx.flange_bodyid], dtype=torch.float32)

    def obj_pos_env(self) -> torch.Tensor:
        p = torch.as_tensor(self.data.xpos[self.idx.obj_bodyid], dtype=torch.float32)
        return p - self.env_origins

    def obj_lin_vel(self) -> torch.Tensor:
        v = self.data.qvel[self.idx.obj_dofadr[:, None] + np.arange(3)[None, :]]
        return torch.as_tensor(v, dtype=torch.float32)

    def qpos_policy(self) -> torch.Tensor:
        """The DEPLOYABLE proprioception: 6 arm joints + the previous gripper
        command -- exactly what the real robot can report (pymycobot
        ``get_radians`` plus the runtime's own last gripper command)."""
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

    # -- rewards (per step; weights are configs/env/...yaml verbatim) ---
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

    # -- action term ----------------------------------------------------
    def _process_actions(self, action: torch.Tensor) -> None:
        """Byte-for-byte ARM_BCRL005's ``MyCobotJointDeltaAction`` contract:
        bounded-lead target integration (anti-windup), the virtual-floor
        bisection, an under-closed gripper target and a rate-limited close."""
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
        """Once per process, assert the analytic tool FK agrees with MuJoCo's
        own TCP site. This is what makes the shared safety envelope
        trustworthy: if the MJCF ever stops matching the constants the guard is
        computed from, we find out here and not on the robot."""
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
        # mj_step integrates qpos LAST and does not re-run kinematics, so
        # data.site_xpos / data.xpos still describe the state one substep ago.
        # Reading the observation without this leaves eef_pos and obj_pos 4 ms
        # stale relative to the joint angles in the SAME vector -- measured at
        # ~0.5 mm of TCP disagreement, which is a quarter of the placement
        # tolerance and would be baked into every recorded demonstration.
        self.mujoco.mj_forward(self.model, self.data)
        self.episode_length_buf += 1
        self._update_episode_stats()
        self.dr.interval_push(self.control_dt)

        rew = self._rewards()
        success = self.is_success()
        timeout = self.episode_length_buf >= self.max_episode_length
        # tool pushed BELOW the plane despite the guard => contact, or a start
        # pose already under it. Ends the episode as a NON-success, so such
        # trajectories stay out of the dataset and show up in the metrics
        # instead of hiding in the physics.
        breach = self.tool_z < self.floor_terminate_z
        term = success
        trunc = (timeout | breach) & ~success

        # DIVERGENCE GUARD. MuJoCo can, rarely, blow a contact up into NaN/Inf.
        # Unattended, a single diverged environment writes NaNs into the
        # recorded demonstrations, and the damage surfaces hours later as a
        # trainer that produces a NaN loss with no indication of where it came
        # from. Catch it at the boundary: reset the environment, count it, and
        # let the episode be discarded as a non-success.
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
        """Order matters and mirrors ARM_BCRL005 exactly: stash the terminal
        metrics BEFORE the reset moves the object (the env auto-resets inside
        ``step``, and callers read the stashed values), then randomise, then
        restart the action integrator from where the arm actually is."""
        if len(env_ids) == 0:
            return
        t = torch.as_tensor(env_ids, dtype=torch.long)
        self.final_max_obj_z[t] = self.max_obj_z[t]
        self.final_min_tool_z[t] = self.min_tool_z[t]
        self.final_place_err[t] = torch.norm(
            self.obj_pos_env()[t, :2] - self.place_target[t, :2], dim=-1)

        # A diverged environment can carry NaN in dofs the reset does not write
        # (a flung distractor, a contact velocity). Zero this env's whole
        # velocity state first so the reset cannot inherit the divergence.
        bad = ~np.isfinite(self.data.qvel)
        if bad.any():
            self.data.qvel[bad] = 0.0
            self.data.qacc[~np.isfinite(self.data.qacc)] = 0.0
            self.data.qpos[~np.isfinite(self.data.qpos)] = 0.0
        self.dr.reset_envs(env_ids)
        if self.dr.needs_setconst:
            # Recompute the mass-derived solver constants (see
            # DomainRandomizer.needs_setconst). mj_setConst uses its MjData
            # argument as SCRATCH and leaves it at qpos0, so it must never be
            # handed the live state -- hence the dedicated buffer.
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

    # -- wrist camera -------------------------------------------------------
    def camera_rgb_uint8(self) -> torch.Tensor:
        """(N, H, W, 3) uint8 wrist-camera frames.

        One renderer, re-pointed at each environment's camera. MuJoCo renders
        a single camera per call, so this is N calls per control step -- which
        is why ``collect``/``distill`` shard across processes
        (src/collect_data.py) instead of raising num_envs: on this project's
        reference machine the graphics stack saturates at ~50 frames/s no
        matter how the work is arranged (operations guide §5).
        """
        if not self.with_camera:
            raise RuntimeError("camera frames requested in a camera-free mode")
        if self._renderer is None:
            self._renderer = self.mujoco.Renderer(self.model, self._cam_h, self._cam_w)
            # What the POLICY sees is a product decision, not a default:
            #  * sites off  -- the TCP marker is a debugging aid drawn inside
            #    the gripper; leaving it in would train the policy on a green
            #    dot the real camera will never show it.
            #  * shadows off -- MuJoCo's shadow map at this scale produces
            #    aliased black blobs across the bench, which is texture the
            #    policy can overfit and the real bench does not have. It also
            #    costs render time, and rendering is this pipeline's
            #    bottleneck (operations guide §5).
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
        """Free-standing render (scene camera for the report video / viewer
        screenshots). Kept separate from the wrist-camera path so the policy's
        input resolution is never accidentally changed by a visualisation."""
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
