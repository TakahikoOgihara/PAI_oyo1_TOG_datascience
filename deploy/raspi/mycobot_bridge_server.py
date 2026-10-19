# -*- coding: utf-8 -*-
"""myCobot 280 Pi のブリッジサーバ。アームの Raspberry Pi 4B の上で走る。

Jetson Orin Nano（方策側）は、素のLANケーブル越しに TCP の改行区切り
JSON（既定ポート9000）でこのサーバと話す。これで Pi の仕事は最小で
安全なものに保てる。Pi はサーボへのシリアルを持ち、方策が学習時に
使っていたのと同じ制御の契約を強制する:

  * 関節位置の差分指令。積分した目標が実測の関節より先行できるのは
    最大 3 * action_scale まで（シミュレータで検証したアンチワインド
    アップの境界。mujoco_sim/train_arm_mujoco.py の _pre_physics_step）
  * 関節の可動限界で固くクランプする（公式 URDF の限界値）
  * 2値のグリッパ開閉（pymycobot の set_gripper_value）。シミュレータの
    GRASP/RELEASE が静止している時間と同じ整定時間を取る
  * ARM_BCRL006 の安全層（safety.py）。刻みごとのレート制限、順運動学
    から作る仮想床、衝突停止。3つとも方策から独立で、3つともここで
    強制する。サーボを持っている機械がここだから
    （実機の失敗記録を参照）

どの応答にも "safety" のブロックが付くので、Jetson のランタイムと
操作コンソールは、アームが止まった理由を常に見られる。

プロトコル（1行に JSON 1つ。どの応答にも "ok" が付く）:
  {"cmd":"state"}                          -> {"ok":1,"q":[6 rad],"grip":0|1,...}
  {"cmd":"deltas","d":[6 in -1..1],"grip":-1..1}
                                           -> 20Hz の制御刻みを1回適用する
  {"cmd":"send_q","q":[6 rad],"speed":30}  -> 絶対位置で動かす（ジョグ/原点復帰）
  {"cmd":"gripper","open":true|false}
  {"cmd":"home"}                           -> 学習時の "ready" 姿勢へ
  {"cmd":"stop"}                           -> 現在位置で保持
  {"cmd":"safety_reset"}                   -> ラッチした安全停止を解除
  {"cmd":"release"}                        -> トルクを切る（手で動かす）

実行（Pi 上で `pip3 install pymycobot` のあと）:
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

# ---- 制御の契約（既定値。edge_config.json が上書きする） ------------------
ACTION_SCALE = 0.05          # 単位行動あたり、1制御刻みの rad
LEAD = 3.0 * ACTION_SCALE    # アンチワインドアップの境界（rad）
CONTROL_DT = 0.05            # 20 Hz
GRIPPER_OPEN_VALUE = 100     # pymycobot の set_gripper_value の尺度（0..100）
GRIPPER_CLOSE_VALUE = 20     # 0 にしない。しっかりだが有限の握り（シミュレータの処方）
GRIPPER_SPEED = 70
MOVE_SPEED = 40              # 絶対移動での pymycobot の速度
# シミュレータはグリッパが動く間アームを静止させる（エキスパートの
# grasp_steps=15 / release_steps=10、20Hz）。実物の適応グリッパにも
# 同じくらいの時間が必要で、方策はそれを待たない。だからブリッジが
# 待つ。これが無いと閉じ途中でアームが動き続け、ワークを弾き飛ばす
# （実機の失敗記録の原因F）。
GRIPPER_SETTLE_S = 0.75
HOME_TOL = 0.05              # rad。関節ごと。「アームは本当に原点復帰したか」の判定に使う


def _connect_robot(port: str, baud: int):
    """pymycobot のクラス名は版をまたいで変わった。両方試す。"""
    try:
        from pymycobot import MyCobot280
        return MyCobot280(port, baud)
    except ImportError:
        from pymycobot.mycobot import MyCobot
        return MyCobot(port, baud)


class ArmState:
    """シリアルを唯一所有する側。読み取りが実測の関節角を更新し、
    指令のメソッドが、学習した制御の契約と安全層の両方を
    強制する。"""

    def __init__(self, port: str, baud: int, safety: SafetyLayer):
        self.mc = _connect_robot(port, baud)
        self.safety = safety
        self.lock = threading.Lock()
        self.q_meas = list(HOME_Q)
        self.meas_time = 0.0
        self.grip_open = 1
        self.grip_until = 0.0        # グリッパが動いている間、アームの目標は凍結する
        self.last_cmd_time = time.time()
        self._refresh()
        # 重要。目標は実測の姿勢から始める。HOME_Q からは絶対に始めない。
        # ARM_BCRL002 は HOME_Q を初期値にしていたので、最初の1刻みで
        # アームが物理的にいた場所からの飛びを指令していた
        # （実機の失敗記録の原因A）。
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
            # 任意の2つ目の衝突トリガ。4Hz で見るので、20Hz の指令ストリームと
            # シリアルを取り合うことはない。get_servo_currents が無い
            # ファームウェアでは、単に一度も作動しない。
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
        """20Hz の制御刻み1回。方策の差分 -> 安全な関節目標。"""
        self.last_cmd_time = time.time()
        want_open = 1 if float(grip) > 0 else 0
        if want_open != self.grip_open:
            self._set_gripper(want_open)

        if time.time() < self.grip_until:
            # グリッパが動いている間は、シミュレータの GRASP/RELEASE と同じく
            # アームをその場に保持する
            self._send(self.q_target)
            return {"ok": 1, "q_target": self.q_target, "holding": "gripper",
                    "safety": self.safety.status()}

        # 1) 学習した契約。先行量を抑えた差分の積分
        cmd = []
        for i in range(6):
            t = self.q_target[i] + max(-1.0, min(1.0, float(d[i]))) * ACTION_SCALE
            t = max(self.q_meas[i] - LEAD, min(self.q_meas[i] + LEAD, t))
            lo, hi = Q_LIMITS[i]
            cmd.append(max(lo, min(hi, t)))

        # 2) 安全層。レート制限 -> 仮想床 -> 衝突停止
        q_safe, allowed = self.safety.apply(self.q_meas, self.q_target, cmd,
                                            self.meas_age())
        self.q_target = list(q_safe)
        self._send(self.q_target)
        return {"ok": 1, "q_target": self.q_target, "allowed": bool(allowed),
                "safety": self.safety.status()}

    def _send(self, q: list, speed: int = MOVE_SPEED) -> None:
        with self.lock:
            # send_radians が 20Hz のストリームに追従する。細かい補間は
            # サーボ自身の位置ループがやる
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
        """絶対移動（ジョグ／原点復帰）。これも床ガードを通す。打ち間違えた
        ジョグで、ツールが机を突き抜けられてはいけない。"""
        self.last_cmd_time = time.time()
        q = [max(lo, min(hi, float(v))) for v, (lo, hi) in zip(q, Q_LIMITS)]
        q, blocked = self.safety.guard.limit(self.q_meas, q)
        self.q_target = list(q)
        self._send(self.q_target, speed)
        return {"ok": 1, "floor_blocked": bool(blocked),
                "safety": self.safety.status()}

    def home(self) -> dict:
        """学習時の 'ready' 姿勢へ動かす。原点復帰は、ロボットを方策が
        学習した状態分布の中へ戻す操作。だから Jetson のランタイムは、
        これが成功するまで方策の起動を拒否する。"""
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
