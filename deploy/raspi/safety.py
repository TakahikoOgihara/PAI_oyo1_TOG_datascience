# -*- coding: utf-8 -*-
"""ARM_BCRL006 の安全層。最後の防衛線で、Raspberry Pi の上で走る。

設計の原則。学習した方策は安全機構ではない。下のガードは素朴で
決定的な制御で、方策より下、サーボを持っている機械の上で走り、
ネットワーク、Jetson、モデルの正しさに一切依存しない。方策が完璧なら
ガードは見えないままで、方策が間違っていればアームと台を無事に保つ
のがガードの仕事。学習側と安全側の分担をここで切っている。

独立した3つの保護を、効く順に:

  1. レート制限   20Hz の1刻みで指令が変われるのは最大
                     ``max_joint_step`` rad まで。学習中に PhysX が強制して
                     いたのと同じ関節速度の上限
                     （env.robot.arm_velocity_limit * control_dt）。これが
                     無いと、速度制限された相手にしか出したことのない
                     ステップを、実物のサーボが実行してしまう。
  2. 仮想床       順運動学（mycobot_fk.FloorGuard）が、ツールのどこか
                     1点でも ``floor_z + clearance`` より下に入る指令の、
                     その部分を切る。予測的なので、アームはそもそも机に
                     触らない。産業用の「安全平面」の型そのもので、
                     方策とは無関係に効く。
  3. 衝突停止     指令した目標が、実測の関節より同じ向きに走り続けたら
                     （「何かを押している」ときの特徴）、停止をラッチ
                     する。目標は実測の姿勢へ引き戻し、任意で少しだけ
                     退避し、どの応答もラッチを報告する。だから Jetson は
                     一時停止し、操作コンソールには理由が出る。
                     ファームウェアが出せるなら、サーボ電流を2つ目の
                     直接のトリガにも使う。それで押し付けを早く
                     止められる。

調整が要るのは3だけ。1と2は学習の設定が既に決めている値から導かれ、
edge_config.json に入ってロボットまで運ばれる（学習側とロボットの
間で、定数を手で写すことは一度も無い）。
"""

from __future__ import annotations
import time

from mycobot_fk import FloorGuard, NUM_ARM_JOINTS, tool_min_z

# 理由の文字列。ブリッジの応答／テレメトリ／Webコンソールにそのまま出る
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
        """操作者による解除（Webコンソールの 'Safety reset'）。"""
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
        """制御刻み1回を濾す。

        q_meas          実測の関節角（rad）
        q_target_prev   前の刻みで送った目標
        q_cmd           方策が今ほしい目標
        meas_age_s      最後にエンコーダを読めてからの秒数

        -> (q_safe, allowed)。``allowed`` が False なら、その刻みは拒否した。
        呼び出し側は位置を保持し、送信を止めること。
        """
        # 信用できない観測は、他の保護を全部盲目にする。レート制限も床ガードも
        # どちらも q_meas をもとに判断しているから。
        if meas_age_s > self.stale_s:
            self.latch(STALE, f"no encoder update for {meas_age_s:.2f}s")
        if self.latched:
            return list(q_meas), False

        q = self.rate_limit(q_target_prev, q_cmd)
        q, blocked = self.guard.limit(q_target_prev, q)
        self.blocked_by_floor = blocked
        self.last_tool_z = tool_min_z(q)

        # 衝突。目標が観測より、同じ符号で居続けている
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
        """衝突のあと、押された向きと逆に ``retreat_rad`` だけ退避する。"""
        out = []
        for m, a in zip(q_meas, ahead):
            step = 0.0
            if abs(a) > self.stall_err:
                step = -self.retreat_rad if a > 0 else self.retreat_rad
            out.append(m + step)
        # 退避先が床に入らないようにもする
        safe, _ = self.guard.limit(list(q_meas), out)
        return safe

    def check_current(self, currents) -> bool:
        """任意の2つ目の衝突トリガ。``currents`` は None でもよい。"""
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
    """edge_config.json の ``safety`` ブロックから安全層を組む。

    これらの数値は学習側が持つ（configs/safety/default.yaml ->
    src/infer.py -> edge_bundle/edge_config.json）。だからロボットと
    シミュレータが、手で書き換えた定数で食い違うことはない。
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
    # 1) 巨大な指令は、関節ごとに max_joint_step までレート制限される
    big = [v + 1.0 for v in home]
    q, ok = s.apply(home, home, big, 0.0)
    assert ok and all(abs(a - b) <= s.max_joint_step + 1e-9 for a, b in zip(q, home))
    print("rate limit OK ->", [round(v, 3) for v in q])
    # 2) 下へ潜り続ける指令は床の平面で切られる（レート制限が先に効くので、
    #    アームが完璧に追従する状態で何刻みも押し込む）
    s.reset()
    dive = [0.25, -1.6, -1.0, 0.2, 0.0, 0.0]
    q = list(home)
    for _ in range(60):
        q, ok = s.apply(q, q, dive, 0.0)        # q_meas が目標に追従する
        assert ok, s.status()
        assert fk.tool_min_z(q) >= s.guard.min_z - 1e-6, fk.tool_min_z(q)
    print(f"floor guard OK -> after 60 ticks of a dive command, "
          f"tool_min_z={fk.tool_min_z(q) * 1000:.1f} mm "
          f"(plane {s.guard.min_z * 1000:.0f} mm), blocked={s.blocked_by_floor}")
    assert s.blocked_by_floor
    # 3) 追従誤差が続くと衝突停止がラッチする
    s.reset()
    stuck = list(home)                          # 観測が一切動かない
    target = list(home)
    for _ in range(s.stall_ticks + 2):
        cmd = [v - 0.5 for v in stuck]          # 押し続けるが、何も動かない
        target, ok = s.apply(stuck, target, cmd, 0.0)
    assert s.latched and s.reason == IMPACT, s.status()
    print("impact stop OK ->", s.status())
    # 4) 観測が古いと、その刻みを拒否する
    s.reset()
    _, ok = s.apply(home, home, home, 1.0)
    assert not ok and s.reason == STALE
    print("stale-feedback stop OK ->", s.status())
    print("\nsafety self-test OK")


if __name__ == "__main__":
    _self_test()
