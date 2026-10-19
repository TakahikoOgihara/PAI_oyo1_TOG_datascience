# -*- coding: utf-8 -*-
"""Scripted pick-and-place expert -- the vectorized waypoint state machine.

Phase logic only (pure torch, no simulator/IK dependency): given the current
end-effector (TCP), object and place-target positions (env frame), it outputs
a target TCP position and a gripper open/close command per env. The Isaac Lab
script converts the target to joint-position actions through
DifferentialIKController -- identical division of labor to ARM_BCRL001's
LiftPhaseMachine, extended from lift-only to the full "pick, lift, move 5 cm,
put down" task of ARM_BCRL005.

This is the "automated demonstration generation" pattern that replaces
teleoperation (the same idea Isaac Lab Mimic / MimicGen industrialize).
robomimic / ACT / Diffusion Policy then own the learning.

Phases (per env, advancing independently):
  0 APPROACH  : hover approach_height above the object, gripper open
  1 DESCEND   : drop to grasp height, gripper open
  2 GRASP     : hold position, close gripper, wait grasp_steps
  3 LIFT      : raise to transport height (xy frozen at grasp), closed
  4 TRANSPORT : move to transport height above the place target, closed
  5 LOWER     : descend to place height above the target, closed
  6 RELEASE   : hold position, open gripper, wait release_steps
  7 RETREAT   : raise back to transport height, open
                (the env terminates on success once the object is placed)

The settle-gate (waypoint counts as reached only after settle_steps
consecutive in-tolerance steps) and the frozen-xy grasp/lift targets carry
over unchanged from ARM_BCRL001 -- both were live-verified fixes there.
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
        # xy frozen at grasp time so the lift target doesn't chase an object
        # that is already in the fingers
        self.hold_xy = torch.zeros(num_envs, 2, device=device)
        # settle gate (see module docstring)
        self.settle = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.settle_steps = 5

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        self.phase[env_ids] = APPROACH
        self.timer[env_ids] = 0
        self.hold_xy[env_ids] = 0.0
        self.settle[env_ids] = 0

    def targets(self, eef_pos: torch.Tensor, obj_pos: torch.Tensor,
                place_pos: torch.Tensor):
        """eef_pos/obj_pos/place_pos: (n, 3) env-frame. Returns
        (target_pos (n,3), gripper_open (n,) bool) and advances phases
        in place."""
        n = self.n
        dev = self.device
        grasp_z = obj_pos[:, 2] + self.grasp_z_offset
        th = torch.full((n,), self.transport_height, device=dev)
        pz = torch.full((n,), self.place_z, device=dev)

        # per-phase targets
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

        # transitions: waypoint phases require the eef to SETTLE inside the
        # tolerance ball for settle_steps consecutive calls
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

        # timer phases (GRASP closes, RELEASE opens, both while holding still)
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
