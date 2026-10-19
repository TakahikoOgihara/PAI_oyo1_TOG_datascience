# -*- coding: utf-8 -*-
"""実機の myCobot のデモを、手で誘導して記録する。

MimicGen の話のもう半分。フェーズ1のスクリプトエキスパートは便利な
手段であって、必須条件ではない。ここで録った10本を
mimicgen/import_real_demos.py で変換すれば、完全に置き換えられる。
下流（フェーズ1.5の転写、フェーズ2〜10）は何も変えずに動く。

やり方は直接教示。すでにあるブリッジのプロトコルだけを使う:

  1. ``release`` -> サーボのトルクが抜け、手でアームを動かせる
  2. TCP をブロックに当てて ``o`` を押す。物体の姿勢は、実測した関節の
     順運動学（mycobot_fk.tcp_pos）で取る。カメラもマーカーも校正治具も
     要らない。この「タッチオフ」は、実機のセルで点を教える標準的な
     やり方。
  3. 設置目標も ``p`` で同じように取る
  4. ``r`` で 20Hz の関節角の記録を始め、作業しながら、掴む／放す
     瞬間に ``g`` でグリッパを切り替える
  5. ``d`` でエピソードを終え、JSON を書く

MimicGen が区切りに使うのはグリッパの切り替わり（mimicgen/annotate.py）。
だからそれを、操作者の明示的な動作として記録している。
inferred.

実行（操作用PC か Pi 自身。ブリッジが動いていること）:
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

    # ---- 操作者からの指令 ---------------------------------------------
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
