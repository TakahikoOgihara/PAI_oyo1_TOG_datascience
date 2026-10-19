# -*- coding: utf-8 -*-
"""Domain randomisation -- the MuJoCo counterpart of ARM_BCRL005's EventCfg.

Isaac Lab expressed DR as ``EventManager`` terms drawn from
``isaaclab.envs.mdp.events``. MuJoCo has no such manager, so the same axes are
implemented here as explicit writes into ``MjModel`` / ``MjData``. The axis
list, the ranges and the on/off defaults are UNCHANGED from
``configs/env/mycobot_pickplace.yaml``; only the mechanism moved. The
correspondence is tabulated in change log §3-4.

| axis                     | ARM_BCRL005 (Isaac Lab)               | here                        |
|--------------------------|---------------------------------------|-----------------------------|
| arm joint noise at reset | mdp.reset_joints_by_offset            | qpos write                  |
| object mass              | mdp.randomize_rigid_body_mass         | model.body_mass + inertia   |
| object friction          | mdp.randomize_rigid_body_material     | model.geom_friction[:,0]    |
| object restitution       | ... same term                         | model.geom_solref dampratio |
| actuator gains           | project term (coupled kp/kv)          | actuator_gainprm/biasprm    |
| joint friction/armature  | mdp.randomize_joint_parameters        | dof_frictionloss/dof_armature |
| mid-episode push         | mdp.push_by_setting_velocity (interval) | qvel write on a timer     |
| light / albedo           | project term (USD) / Replicator       | light_diffuse, geom_rgba    |

TWO PROPERTIES WORTH KEEPING IN MIND, both carried over from ARM_BCRL005:

1. *Everything here is a property write at reset, not an extra render and not
   an extra step.* That is why DR is affordable at all; the expensive kinds
   (per-frame texture/background re-rendering) buy little for one known object
   on one known bench and would wreck the PDCA cycle time.
2. *Wide DR is not free.* Widening the distribution without adding data lowers
   success -- which is precisely what the 4-condition study in ``src/ablation.py``
   measures instead of asserting. Each range in the YAML is "the honest
   uncertainty about the real thing", not "as wide as possible".

ARM_BCRL006 CHANGE: with ``domain_rand.enable=false`` the distractor block is
placed at a FIXED nominal pose instead of a random one. ARM_BCRL005 kept
randomising it because its DR switch only governed the physical axes. For the
ablation to mean anything, "DR off" has to mean no scene perturbation at all;
the object's own spawn window stays randomised in every condition because that
is the TASK, not a perturbation.
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
        # nominal values captured once: every randomiser writes
        # nominal * factor, never previous * factor, so repeated resets cannot
        # random-walk the model away from its configured operating point.
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
        # Set when a reset writes a mass / inertia / armature. Those feed
        # MuJoCo's COMPILE-TIME derived constants (body_invweight0,
        # dof_invweight0, dof_M0), which the constraint solver uses to scale
        # contact impedance -- so writing model.body_mass alone leaves the
        # solver working from the old numbers and grasps silently stop holding.
        # env._reset_idx calls mj_setConst() when this is true.
        self.needs_setconst = False

    # ------------------------------------------------------------------
    def _rng_range(self, key, unit=None):
        rng = self.dr.get(key, None)
        if rng is None:
            return None
        lo, hi = (float(v) for v in rng)
        if unit is not None and lo == unit and hi == unit:
            return None                     # identity range == term switched off
        return lo, hi

    def active_terms(self) -> list:
        """The randomisers that are ACTUALLY live, printed at startup.

        ARM_BCRL005 needed this because a moved PhysX API could silently no-op
        a randomiser; here the risk is the opposite (a configured range that is
        the identity), and either way "what did this run actually randomise"
        must be answerable from the log alone.
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
        """Perturbation randomness (DR)."""
        return self.env.rng.uniform(lo, hi, size=n)

    def _task(self, lo, hi, n=None):
        """TASK randomness (where the block is). A separate stream, so the same
        seed gives the same scenes whether or not DR is on -- see
        MuJoCoPickPlaceEnv.__init__."""
        return self.env.task_rng.uniform(lo, hi, size=n)

    # ------------------------------------------------------------------
    def on_reset_round(self) -> None:
        """Randomisers that are global to the model rather than per-env: the
        light and the desk/block albedo. Re-sampled once per reset round,
        exactly as ARM_BCRL005's ``randomize_appearance`` was (Isaac Lab's dome
        light and the USD materials were also global)."""
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
        """Per-environment reset: arm pose, object/target, distractor, and the
        physical axes. Called from ``env._reset_idx`` before ``mj_forward``."""
        e = self.cfg.env
        m, d = self.model, self.env.data
        idx = self.env.idx
        n = len(env_ids)

        # -- arm + fingers ------------------------------------------------
        noise = float(self.dr.joint_noise) if self.enabled else 0.0
        q = np.tile(self.env.home_q, (n, 1))
        if noise > 0:
            q = q + self._u(-noise, noise, (n, len(self.env.home_q)))
        d.qpos[idx.arm_qadr[env_ids]] = q
        d.qvel[idx.arm_dofadr[env_ids]] = 0.0
        d.qpos[idx.finger_qadr[env_ids]] = float(e.gripper_open)
        d.qvel[idx.finger_dofadr[env_ids]] = 0.0

        # -- object spawn + place target (the TASK, randomised always) ----
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
                # keep clear of the working lane between the block and the target
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

        # -- physical axes --------------------------------------------------
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
            # static == dynamic (MuJoCo has one sliding-friction coefficient),
            # which is what ARM_BCRL005 configured anyway.
            m.geom_friction[gids, 0] = self._u(rng[0], rng[1], n)
        rng = self._rng_range("object_restitution", 0.0)
        if rng is not None:
            # MuJoCo has no restitution coefficient; bounciness lives in the
            # contact solver reference as the damping ratio solref[1]
            # (1.0 = critically damped = no bounce). Mapping it here keeps the
            # YAML describing the PHYSICAL quantity, not a solver parameter.
            r = self._u(rng[0], rng[1], n)
            m.geom_solref[gids, 1] = np.clip(1.0 - 0.9 * r, 0.05, 1.0)

        rng = self._rng_range("actuator_gain_scale", 1.0)
        if rng is not None:
            # stiffness AND damping scaled by the SAME factor. Isaac Lab's own
            # randomize_actuator_gains samples them independently, which moves
            # the damping ratio and produces environments that oscillate or
            # crawl: noise, not robustness. Coupling them keeps the ratio where
            # it was tuned, which is the entire point of randomising the gains.
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
        """DR off: put every randomisable property back to its nominal value.

        Necessary because ``domain_rand.enable`` can differ between the run
        that collected the data and the run that evaluates the policy -- which
        is exactly what the 4-condition ablation does.
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
        """Mid-episode disturbance -- the axis a reset-only randomiser cannot
        express at all. Ships OFF: the scripted expert has to keep succeeding
        through any DR that is on, because its success rate is the ceiling on
        data quality. Useful for an RL-only robustness ablation."""
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
