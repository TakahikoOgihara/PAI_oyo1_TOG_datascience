# -*- coding: utf-8 -*-
"""Record REAL myCobot demonstrations by hand-guiding the arm.

This is the other half of the MimicGen story: phase 1's scripted expert is a
convenience, not a requirement. Ten demonstrations recorded here, converted by
mimicgen/import_real_demos.py, can replace it entirely -- and everything
downstream (phase 1.5 warping, phases 2-10) runs unchanged.

Kinesthetic teaching, using only the bridge protocol that already exists:

  1. ``release``  -> arm servos go limp, you move the arm by hand
  2. you guide the TCP onto the block and press ``o`` -- the object pose is
     captured by FORWARD KINEMATICS of the measured joints
     (mycobot_fk.tcp_pos). No camera, no marker, no calibration rig. This
     "touch-off" is the standard way to teach a point on a real cell.
  3. same for the place target with ``p``
  4. ``r`` starts recording joint angles at 20 Hz while you perform the task,
     toggling the gripper with ``g`` at the moment you would grasp / release
  5. ``d`` ends the episode and writes the JSON

The gripper edges are what MimicGen segments on (mimicgen/annotate.py), which
is exactly why they are recorded as an explicit operator action rather than
inferred.

Run (operator PC or the Pi itself; the bridge must be running):
    python3 record_demo.py --pi 192.168.10.10:9000 --out ~/real_demos
"""

from __future__ import annotations
import argparse
import json
import os
import socket
import sys
import threading
import time

from mycobot_fk import HOME_Q, tcp_pos, tool_min_z

CONTROL_DT = 0.05


class Bridge:
    def __init__(self, host: str, port: int):
        self.sock = socket.create_connection((host, port), timeout=5.0)
        self.rfile = self.sock.makefile("r")
        self.lock = threading.Lock()

    def call(self, msg: dict) -> dict:
        with self.lock:
            self.sock.sendall((json.dumps(msg) + "\n").encode())
            return json.loads(self.rfile.readline())


HELP = """
  o  capture OBJECT pose      (guide the TCP to the block centre first)
  p  capture PLACE target     (guide the TCP to where it should end up)
  g  toggle gripper open/close
  r  start / pause recording
  d  done -- save this episode and start the next one
  h  hold (torque ON, arm stops moving)   f  free (torque OFF, hand-guide)
  x  quit
"""


class Recorder:
    def __init__(self, bridge: Bridge, out_dir: str):
        self.b = bridge
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.reset_episode()
        self.episode = self._next_index()
        self.running = True

    def _next_index(self) -> int:
        n = 0
        while os.path.exists(os.path.join(self.out_dir, f"demo_{n:03d}.json")):
            n += 1
        return n

    def reset_episode(self):
        self.steps = []
        self.recording = False
        self.grip_open = 1
        self.object_pos = None
        self.place_pos = None

    # ---- operator commands -------------------------------------------
    def handle(self, key: str) -> None:
        st = self.b.call({"cmd": "state"})
        q = st["q"]
        if key == "o":
            self.object_pos = list(tcp_pos(q))
            print(f"  OBJECT  = {[round(v, 4) for v in self.object_pos]} m")
        elif key == "p":
            self.place_pos = list(tcp_pos(q))
            print(f"  PLACE   = {[round(v, 4) for v in self.place_pos]} m")
        elif key == "g":
            self.grip_open = 0 if self.grip_open else 1
            self.b.call({"cmd": "gripper", "open": bool(self.grip_open)})
            print(f"  gripper {'OPEN' if self.grip_open else 'CLOSED'}")
        elif key == "r":
            if self.object_pos is None or self.place_pos is None:
                print("  ! capture the object (o) and place target (p) first")
                return
            self.recording = not self.recording
            print(f"  recording {'ON' if self.recording else 'PAUSED'} "
                  f"({len(self.steps)} steps)")
        elif key == "f":
            self.b.call({"cmd": "release"})
            print("  torque OFF -- you can move the arm by hand")
        elif key == "h":
            self.b.call({"cmd": "stop"})
            print("  torque ON -- arm holds position")
        elif key == "d":
            self.save()
        elif key == "x":
            self.running = False
        else:
            print(HELP)

    # ---- recording ----------------------------------------------------
    def tick(self) -> None:
        if not self.recording:
            return
        st = self.b.call({"cmd": "state"})
        self.steps.append({"q": [float(v) for v in st["q"]],
                           "grip": int(self.grip_open)})

    def save(self) -> None:
        if len(self.steps) < 20:
            print(f"  ! only {len(self.steps)} steps -- too short, not saved")
            return
        grips = {s["grip"] for s in self.steps}
        if grips != {0, 1}:
            print("  ! the gripper never both closed AND re-opened during this "
                  "episode. MimicGen segments subtasks on those edges, so the "
                  "demo would be rejected by mimicgen/annotate.py. Not saved.")
            return
        path = os.path.join(self.out_dir, f"demo_{self.episode:03d}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"control_dt": CONTROL_DT,
                       "object_pos": self.object_pos,
                       "place_target_pos": self.place_pos,
                       "steps": self.steps}, f, indent=1)
        print(f"  saved {path}  ({len(self.steps)} steps, "
              f"{len(self.steps) * CONTROL_DT:.1f}s)")
        self.episode += 1
        self.reset_episode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi", default="192.168.10.10:9000")
    ap.add_argument("--out", default=os.path.expanduser("~/real_demos"))
    args = ap.parse_args()

    host, port = args.pi.split(":")
    b = Bridge(host, int(port))
    st = b.call({"cmd": "state"})
    print(f"[record_demo] connected. q = {[round(v, 3) for v in st['q']]}")
    print(f"[record_demo] TCP = {[round(v, 4) for v in tcp_pos(st['q'])]} m, "
          f"tool height {tool_min_z(st['q']) * 1000:.0f} mm")
    print(f"[record_demo] home pose reference = {list(HOME_Q)}")
    print(f"[record_demo] writing to {args.out}")
    print(HELP)

    rec = Recorder(b, args.out)
    stop = threading.Event()

    def _loop():
        nxt = time.perf_counter()
        while not stop.is_set():
            nxt += CONTROL_DT
            try:
                rec.tick()
            except Exception as e:
                print(f"  ! bridge error: {e}")
            d = nxt - time.perf_counter()
            if d > 0:
                time.sleep(d)
            else:
                nxt = time.perf_counter()

    threading.Thread(target=_loop, daemon=True).start()
    try:
        while rec.running:
            line = sys.stdin.readline()
            if not line:
                break
            for key in line.strip():
                rec.handle(key)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        try:
            b.call({"cmd": "stop"})
        except Exception:
            pass
    print("[record_demo] done. Convert with:\n"
          f"  python3 mimicgen/import_real_demos.py {args.out} "
          "artifacts/data/real_demos.pt")


if __name__ == "__main__":
    main()
