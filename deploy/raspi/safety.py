# -*- coding: utf-8 -*-
"""ARM_BCRL006 safety layer -- the last line of defence, ON the Raspberry Pi.

Design rule (design notes §3, field-failure notes §4): a learned
policy is never a safety mechanism. The guard below is plain, deterministic
control that runs BELOW the policy, on the machine that owns the servos, with
no dependency on the network, the Jetson, or the model being correct. If the
policy is perfect the guard is invisible; if the policy is wrong the guard is
what keeps the arm and the bench intact.

Three independent protections, in the order they act:

  1. RATE LIMIT   -- at most ``max_joint_step`` rad of commanded change per
                     20 Hz tick, i.e. the SAME joint velocity ceiling PhysX
                     enforced during training (env.robot.arm_velocity_limit *
                     control_dt). Without it the real servos execute a step
                     the policy only ever produced against a velocity-limited
                     simulator.
  2. VIRTUAL FLOOR-- forward kinematics (mycobot_fk.FloorGuard) rejects the
                     part of a command that would put ANY tool point below
                     ``floor_z + clearance``. Predictive, so the arm never
                     touches the desk in the first place. This is the
                     industrial "safety plane" pattern.
  3. IMPACT STOP  -- if the commanded target keeps running ahead of the
                     measured joints in the same direction (the signature of
                     "pushing against something"), latch a STOP: targets are
                     snapped back to the measured pose, an optional small
                     retreat is applied, and every reply reports the latch so
                     the Jetson pauses and the operator console shows why.
                     Servo current is used as a second, direct trigger when
                     the firmware exposes it.

Only 3 needs tuning; 1 and 2 are derived from values the training config
already fixes, and travel to the robot inside edge_config.json (no constant
is ever hand-copied between the training side and the robot).
"""

from __future__ import annotations
import time

from mycobot_fk import FloorGuard, NUM_ARM_JOINTS, tool_min_z

# reasons, surfaced verbatim in the bridge replies / telemetry / web console
OK = "ok"
FLOOR = "floor_plane"
IMPACT = "impact"
STALE = "stale_feedback"


class SafetyLayer:
    def __init__(self, *, control_dt: float = 0.05, max_joint_step: float = 0.10,
                 floor_z: float = 0.0, floor_clearance: float = 0.005,
                 stall_err: float = 0.10, stall_ticks: int = 8,
                 retreat_rad: float = 0.05, stale_s: float = 0.5,
                 current_limit_ma: int = 0):
        self.control_dt = float(control_dt)
        self.max_joint_step = float(max_joint_step)
        self.guard = FloorGuard(floor_z=floor_z, clearance=floor_clearance)
        self.stall_err = float(stall_err)
        self.stall_ticks = int(stall_ticks)
        self.retreat_rad = float(retreat_rad)
        self.stale_s = float(stale_s)
        self.current_limit_ma = int(current_limit_ma)

        self.latched = False
        self.reason = OK
        self.detail = ""
        self._stall_count = 0
        self.blocked_by_floor = False
        self.last_tool_z = float("nan")

    # -- state ----------------------------------------------------------
    def status(self) -> dict:
        return {"latched": bool(self.latched), "reason": self.reason,
                "detail": self.detail,
                "floor_blocked": bool(self.blocked_by_floor),
                "tool_z": round(float(self.last_tool_z), 4),
                "floor_min_z": round(self.guard.min_z, 4)}

    def latch(self, reason: str, detail: str = "") -> None:
        if not self.latched:
            print(f"[safety] LATCHED: {reason} {detail}")
        self.latched = True
        self.reason = reason
        self.detail = detail

    def reset(self) -> None:
        """Operator-acknowledged clear (web console 'Safety reset')."""
        self.latched = False
        self.reason = OK
        self.detail = ""
        self._stall_count = 0
        self.blocked_by_floor = False

    # -- protections ----------------------------------------------------
    def rate_limit(self, q_from, q_to):
        s = self.max_joint_step
        return [a + max(-s, min(s, b - a)) for a, b in zip(q_from, q_to)]

    def apply(self, q_meas, q_target_prev, q_cmd, meas_age_s: float):
        """Filter one control tick.

        q_meas          measured joint angles (rad)
        q_target_prev   the target sent on the previous tick
        q_cmd           the target the policy wants now
        meas_age_s      seconds since the last SUCCESSFUL encoder read

        -> (q_safe, allowed). ``allowed`` False means the tick was refused;
        the caller must hold position and stop streaming.
        """
        # feedback we cannot trust makes every other protection blind: the
        # rate limit and the floor guard both reason about q_meas.
        if meas_age_s > self.stale_s:
            self.latch(STALE, f"no encoder update for {meas_age_s:.2f}s")
        if self.latched:
            return list(q_meas), False

        q = self.rate_limit(q_target_prev, q_cmd)
        q, blocked = self.guard.limit(q_target_prev, q)
        self.blocked_by_floor = blocked
        self.last_tool_z = tool_min_z(q)

        # impact: target persistently ahead of measurement, same sign
        ahead = [t - m for t, m in zip(q, q_meas)]
        if max(abs(v) for v in ahead) > self.stall_err:
            self._stall_count += 1
        else:
            self._stall_count = 0
        if self._stall_count >= self.stall_ticks:
            self.latch(IMPACT, f"tracking error > {self.stall_err} rad for "
                               f"{self._stall_count} ticks (~"
                               f"{self._stall_count * self.control_dt:.1f}s)")
            return self.retreat_from(q_meas, ahead), False
        return q, True

    def retreat_from(self, q_meas, ahead):
        """After an impact, back off ``retreat_rad`` AGAINST the push."""
        out = []
        for m, a in zip(q_meas, ahead):
            step = 0.0
            if abs(a) > self.stall_err:
                step = -self.retreat_rad if a > 0 else self.retreat_rad
            out.append(m + step)
        # never retreat into the floor either
        safe, _ = self.guard.limit(list(q_meas), out)
        return safe

    def check_current(self, currents) -> bool:
        """Optional second impact trigger; ``currents`` may be None."""
        if not self.current_limit_ma or not currents:
            return True
        try:
            peak = max(abs(float(c)) for c in currents[:NUM_ARM_JOINTS])
        except (TypeError, ValueError):
            return True
        if peak > self.current_limit_ma:
            self.latch(IMPACT, f"servo current {peak:.0f} mA > "
                               f"{self.current_limit_ma} mA")
            return False
        return True


def from_edge_config(cfg: dict) -> SafetyLayer:
    """Build the layer from edge_config.json's ``safety`` block.

    The training side owns these numbers (configs/safety/default.yaml ->
    src/infer.py -> edge_bundle/edge_config.json), so the robot and the
    simulator can never drift apart by a hand-edited constant.
    """
    s = dict(cfg.get("safety", {}))
    return SafetyLayer(
        control_dt=float(cfg.get("control_dt", 0.05)),
        max_joint_step=float(s.get("max_joint_step", 0.10)),
        floor_z=float(s.get("floor_z", 0.0)),
        floor_clearance=float(s.get("floor_clearance", 0.005)),
        stall_err=float(s.get("stall_err", 0.10)),
        stall_ticks=int(s.get("stall_ticks", 8)),
        retreat_rad=float(s.get("retreat_rad", 0.05)),
        stale_s=float(s.get("stale_s", 0.5)),
        current_limit_ma=int(s.get("current_limit_ma", 0)),
    )


def _self_test() -> None:
    import mycobot_fk as fk
    s = SafetyLayer()
    home = list(fk.HOME_Q)
    print("status:", s.status())
    # 1) a huge command is rate limited to max_joint_step per joint
    big = [v + 1.0 for v in home]
    q, ok = s.apply(home, home, big, 0.0)
    assert ok and all(abs(a - b) <= s.max_joint_step + 1e-9 for a, b in zip(q, home))
    print("rate limit OK ->", [round(v, 3) for v in q])
    # 2) a SUSTAINED dive is cut at the floor plane (the rate limit acts
    #    first, so drive it for many ticks with the arm tracking perfectly)
    s.reset()
    dive = [0.25, -1.6, -1.0, 0.2, 0.0, 0.0]
    q = list(home)
    for _ in range(60):
        q, ok = s.apply(q, q, dive, 0.0)        # q_meas tracks the target
        assert ok, s.status()
        assert fk.tool_min_z(q) >= s.guard.min_z - 1e-6, fk.tool_min_z(q)
    print(f"floor guard OK -> after 60 ticks of a dive command, "
          f"tool_min_z={fk.tool_min_z(q) * 1000:.1f} mm "
          f"(plane {s.guard.min_z * 1000:.0f} mm), blocked={s.blocked_by_floor}")
    assert s.blocked_by_floor
    # 3) a persistent tracking error latches an impact stop
    s.reset()
    stuck = list(home)                          # measurement never moves
    target = list(home)
    for _ in range(s.stall_ticks + 2):
        cmd = [v - 0.5 for v in stuck]          # keep pushing, nothing moves
        target, ok = s.apply(stuck, target, cmd, 0.0)
    assert s.latched and s.reason == IMPACT, s.status()
    print("impact stop OK ->", s.status())
    # 4) stale feedback refuses the tick
    s.reset()
    _, ok = s.apply(home, home, home, 1.0)
    assert not ok and s.reason == STALE
    print("stale-feedback stop OK ->", s.status())
    print("\nsafety self-test OK")


if __name__ == "__main__":
    _self_test()
