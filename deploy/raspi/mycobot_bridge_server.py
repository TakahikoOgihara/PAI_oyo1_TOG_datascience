# -*- coding: utf-8 -*-
"""myCobot 280 Pi bridge server -- runs ON the arm's Raspberry Pi 4B.

The Jetson Orin Nano (policy side) talks to this server over a plain LAN
cable using newline-delimited JSON over TCP (default port 9000). This keeps
the Pi's job minimal and safe: it owns the serial link to the servos and
enforces the SAME control contract the policy was trained with in Isaac Lab:

  * joint-position DELTA commands, integrated into a target that may lead the
    MEASURED joints by at most 3 * action_scale (the anti-windup bound
    verified in simulation -- mujoco_sim/train_arm_mujoco.py _pre_physics_step)
  * hard joint-limit clamping (the official URDF's limits)
  * binary gripper open/close (pymycobot set_gripper_value), with the same
    settle time the simulator's GRASP/RELEASE phases hold still for
  * the ARM_BCRL006 SAFETY LAYER (safety.py): per-tick rate limit, virtual
    floor plane from forward kinematics, and an impact stop -- all three
    independent of the policy, all three enforced here because this is the
    machine that owns the servos (field-failure notes §4)

Every reply carries a "safety" block, so the Jetson runtime and the operator
console always see why the arm stopped.

Protocol (one JSON object per line; every reply carries "ok"):
  {"cmd":"state"}                          -> {"ok":1,"q":[6 rad],"grip":0|1,...}
  {"cmd":"deltas","d":[6 in -1..1],"grip":-1..1}
                                           -> apply one 20 Hz control tick
  {"cmd":"send_q","q":[6 rad],"speed":30}  -> absolute move (jog/home)
  {"cmd":"gripper","open":true|false}
  {"cmd":"home"}                           -> the training "ready" pose
  {"cmd":"stop"}                           -> hold current position
  {"cmd":"safety_reset"}                   -> clear a latched safety stop
  {"cmd":"release"}                        -> torque off (manual repositioning)

Run (on the Pi, after `pip3 install pymycobot`):
  python3 mycobot_bridge_server.py --port 9000 --edge-config ~/edge_bundle/edge_config.json
"""

from __future__ import annotations
import argparse
import json
import os
import socketserver
import threading
import time

from mycobot_fk import HOME_Q, Q_LIMITS, tcp_pos, tool_min_z
from safety import SafetyLayer, from_edge_config

# ---- control contract (defaults; edge_config.json overrides them) ----------
ACTION_SCALE = 0.05          # rad per control tick per unit action
LEAD = 3.0 * ACTION_SCALE    # anti-windup bound (rad)
CONTROL_DT = 0.05            # 20 Hz
GRIPPER_OPEN_VALUE = 100     # pymycobot set_gripper_value scale (0..100)
GRIPPER_CLOSE_VALUE = 20     # not 0: firm-but-finite squeeze (sim recipe)
GRIPPER_SPEED = 70
MOVE_SPEED = 40              # pymycobot speed for absolute moves
# The simulator holds the arm STILL while the gripper travels (expert
# grasp_steps=15 / release_steps=10 at 20 Hz). The real adaptive gripper needs
# a comparable amount of time, and the policy does not wait for it, so the
# bridge does: without this the arm keeps moving mid-close and knocks the
# block away (field-failure notes §2, cause F).
GRIPPER_SETTLE_S = 0.75
HOME_TOL = 0.05              # rad, per joint, for "is the arm actually homed?"


def _connect_robot(port: str, baud: int):
    """pymycobot's class name moved across versions -- try both."""
    try:
        from pymycobot import MyCobot280
        return MyCobot280(port, baud)
    except ImportError:
        from pymycobot.mycobot import MyCobot
        return MyCobot(port, baud)


class ArmState:
    """Single owner of the serial link. A reader refreshes the measured
    joint angles; command methods enforce the trained control contract AND
    the safety layer."""

    def __init__(self, port: str, baud: int, safety: SafetyLayer):
        self.mc = _connect_robot(port, baud)
        self.safety = safety
        self.lock = threading.Lock()
        self.q_meas = list(HOME_Q)
        self.meas_time = 0.0
        self.grip_open = 1
        self.grip_until = 0.0        # arm target frozen while the gripper moves
        self.last_cmd_time = time.time()
        self._refresh()
        # CRITICAL: the target starts at the MEASURED pose, never at HOME_Q.
        # ARM_BCRL002 seeded it with HOME_Q, so the very first streamed tick
        # commanded a jump from wherever the arm physically was
        # (field-failure notes §2, cause A).
        self.q_target = list(self.q_meas)

    def _refresh(self) -> None:
        with self.lock:
            try:
                rad = self.mc.get_radians()
            except Exception:
                rad = None
        if rad and len(rad) == 6 and all(isinstance(v, (int, float)) for v in rad):
            self.q_meas = [float(v) for v in rad]
            self.meas_time = time.time()

    def reader_loop(self) -> None:
        tick = 0
        while True:
            self._refresh()
            tick += 1
            # optional SECOND impact trigger, polled at 4 Hz so it never
            # competes with the 20 Hz command stream for the serial link.
            # Firmware without get_servo_currents simply never arms it.
            if self.safety.current_limit_ma and tick % 5 == 0:
                with self.lock:
                    try:
                        currents = self.mc.get_servo_currents()
                    except Exception:
                        currents = None
                self.safety.check_current(currents)
            time.sleep(CONTROL_DT)

    def meas_age(self) -> float:
        return time.time() - self.meas_time if self.meas_time else 1e9

    # ---- commands -----------------------------------------------------
    def state(self) -> dict:
        tcp = tcp_pos(self.q_meas)
        return {"ok": 1, "q": self.q_meas, "grip": self.grip_open,
                "q_target": self.q_target,
                "homed": self.is_homed(),
                "tcp": [round(v, 4) for v in tcp],
                "tool_z": round(tool_min_z(self.q_meas), 4),
                "meas_age": round(self.meas_age(), 3),
                "safety": self.safety.status()}

    def is_homed(self) -> bool:
        return all(abs(a - b) < HOME_TOL for a, b in zip(self.q_meas, HOME_Q))

    def deltas(self, d: list, grip: float) -> dict:
        """One 20 Hz control tick: policy deltas -> safe joint targets."""
        self.last_cmd_time = time.time()
        want_open = 1 if float(grip) > 0 else 0
        if want_open != self.grip_open:
            self._set_gripper(want_open)

        if time.time() < self.grip_until:
            # gripper in motion: hold the arm exactly where it is, like the
            # simulator's GRASP/RELEASE phases do
            self._send(self.q_target)
            return {"ok": 1, "q_target": self.q_target, "holding": "gripper",
                    "safety": self.safety.status()}

        # 1) the trained contract: bounded-lead delta integration
        cmd = []
        for i in range(6):
            t = self.q_target[i] + max(-1.0, min(1.0, float(d[i]))) * ACTION_SCALE
            t = max(self.q_meas[i] - LEAD, min(self.q_meas[i] + LEAD, t))
            lo, hi = Q_LIMITS[i]
            cmd.append(max(lo, min(hi, t)))

        # 2) the safety layer: rate limit -> virtual floor -> impact stop
        q_safe, allowed = self.safety.apply(self.q_meas, self.q_target, cmd,
                                            self.meas_age())
        self.q_target = list(q_safe)
        self._send(self.q_target)
        return {"ok": 1, "q_target": self.q_target, "allowed": bool(allowed),
                "safety": self.safety.status()}

    def _send(self, q: list, speed: int = MOVE_SPEED) -> None:
        with self.lock:
            # send_radians tracks the 20 Hz stream; the servo's own position
            # loop does the fine interpolation
            self.mc.send_radians(q, int(speed))

    def _set_gripper(self, open_: int) -> None:
        self.grip_open = open_
        self.grip_until = time.time() + GRIPPER_SETTLE_S
        with self.lock:
            try:
                self.mc.set_gripper_value(
                    GRIPPER_OPEN_VALUE if open_ else GRIPPER_CLOSE_VALUE, GRIPPER_SPEED)
            except Exception as e:
                print(f"[bridge] gripper command failed: {e}")

    def send_q(self, q: list, speed: int) -> dict:
        """Absolute move (jog / home). Still floor-guarded -- a mistyped jog
        must not be able to drive the tool through the desk either."""
        self.last_cmd_time = time.time()
        q = [max(lo, min(hi, float(v))) for v, (lo, hi) in zip(q, Q_LIMITS)]
        q, blocked = self.safety.guard.limit(self.q_meas, q)
        self.q_target = list(q)
        self._send(self.q_target, speed)
        return {"ok": 1, "floor_blocked": bool(blocked),
                "safety": self.safety.status()}

    def home(self) -> dict:
        """Move to the training 'ready' pose. Homing is what puts the robot
        back INSIDE the state distribution the policy was trained on, so the
        Jetson runtime refuses to start the policy until this succeeded."""
        self.safety.reset()
        rep = self.send_q(list(HOME_Q), MOVE_SPEED)
        rep["homed"] = self.is_homed()
        return rep

    def stop(self) -> dict:
        self.q_target = list(self.q_meas)
        with self.lock:
            try:
                self.mc.stop()
            except Exception:
                self.mc.send_radians(self.q_meas, MOVE_SPEED)
        return {"ok": 1, "safety": self.safety.status()}

    def safety_reset(self) -> dict:
        self.safety.reset()
        self.q_target = list(self.q_meas)
        return {"ok": 1, "safety": self.safety.status()}

    def release(self) -> dict:
        with self.lock:
            self.mc.release_all_servos()
        return {"ok": 1}


ARM: ArmState | None = None


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        peer = self.client_address[0]
        print(f"[bridge] client connected: {peer}")
        for line in self.rfile:
            try:
                msg = json.loads(line.decode("utf-8"))
                cmd = msg.get("cmd", "")
                if cmd == "state":
                    rep = ARM.state()
                elif cmd == "deltas":
                    rep = ARM.deltas(msg["d"], msg.get("grip", 1.0))
                elif cmd == "send_q":
                    rep = ARM.send_q(msg["q"], msg.get("speed", MOVE_SPEED))
                elif cmd == "gripper":
                    ARM._set_gripper(1 if msg.get("open", True) else 0)
                    rep = {"ok": 1, "safety": ARM.safety.status()}
                elif cmd == "home":
                    rep = ARM.home()
                elif cmd == "stop":
                    rep = ARM.stop()
                elif cmd == "safety_reset":
                    rep = ARM.safety_reset()
                elif cmd == "release":
                    rep = ARM.release()
                else:
                    rep = {"ok": 0, "error": f"unknown cmd {cmd!r}"}
            except Exception as e:
                rep = {"ok": 0, "error": str(e)}
            self.wfile.write((json.dumps(rep) + "\n").encode("utf-8"))
        print(f"[bridge] client disconnected: {peer}")


class ThreadingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    global ARM, ACTION_SCALE, LEAD, CONTROL_DT
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--serial", default="/dev/ttyAMA0",
                    help="myCobot 280 Pi built-in serial port")
    ap.add_argument("--baud", type=int, default=1000000)
    ap.add_argument("--edge-config", default=os.path.expanduser(
        "~/edge_bundle/edge_config.json"),
        help="the training side's control+safety contract (edge bundle). "
             "Falls back to this file's defaults when absent.")
    ap.add_argument("--floor-z", type=float, default=None,
                    help="override safety.floor_z (m): height of the DESK TOP "
                         "above the arm's base plane. 0 when the arm is bolted "
                         "flat on the desk; NEGATIVE if the base sits on a "
                         "riser (measure it -- the deployment notes)")
    args = ap.parse_args()

    cfg = {}
    try:
        with open(args.edge_config, encoding="utf-8") as f:
            cfg = json.load(f)
        print(f"[bridge] control+safety contract from {args.edge_config}")
    except OSError:
        print(f"[bridge] WARNING: {args.edge_config} not found -- using built-in "
              "defaults. Copy the edge bundle so the robot and the training "
              "config cannot drift apart.")
    safety = from_edge_config(cfg)
    if args.floor_z is not None:
        safety.guard.floor_z = float(args.floor_z)
    ACTION_SCALE = float(cfg.get("action_scale", ACTION_SCALE))
    LEAD = 3.0 * ACTION_SCALE
    CONTROL_DT = float(cfg.get("control_dt", CONTROL_DT))

    ARM = ArmState(args.serial, args.baud, safety)
    threading.Thread(target=ARM.reader_loop, daemon=True).start()
    print(f"[bridge] myCobot bridge on {args.host}:{args.port} "
          f"(serial {args.serial} @ {args.baud})")
    print(f"[bridge] q_meas   = {[round(v, 3) for v in ARM.q_meas]}")
    print(f"[bridge] TCP      = {[round(v, 3) for v in tcp_pos(ARM.q_meas)]} m "
          f"(home should be ~[0.256, 0.000, 0.100])")
    print(f"[bridge] safety   = floor plane z >= {safety.guard.min_z * 1000:.0f} mm, "
          f"max {safety.max_joint_step} rad/tick, impact stop at "
          f"{safety.stall_err} rad x {safety.stall_ticks} ticks")
    with ThreadingTCPServer((args.host, args.port), Handler) as srv:
        srv.serve_forever()


if __name__ == "__main__":
    main()
