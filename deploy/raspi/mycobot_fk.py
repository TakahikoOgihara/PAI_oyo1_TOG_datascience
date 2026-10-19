# -*- coding: utf-8 -*-
"""myCobot 280 Pi の順運動学と、ツールの安全包絡。

依存ゼロ（標準ライブラリの ``math`` だけ）は意図的。このファイルが
「ツールはどこにいて、どこまで低く届くか」の正本で、Raspberry Pi 4B の
安全層の中で、numpy / torch / 方策より先に、かつそれらと無関係に
走らないといけない。``deploy/raspi/`` は自己完結したディレクトリとして
Pi に scp するので、ガードが肝心なときに依存を欠くことはない。

合わせるのはシミュレータ側で、逆ではない。
``arm_common/kinematics.py`` が同じ定数を取り込み、バッチ対応の torch
実装を足し、起動時に両者が 1e-6 m まで一致することを確認する。安全
包絡の正本が1つ。つまり方策が学ぶものが、そのまま実機が強制するもの。

定数は Elephant Robotics の公式 URDF
（mycobot_ros2 @ 3999e2c、mycobot_description/urdf/mycobot_280_pi）と、
assets/build_mycobot_mjcf.py が足すグリッパ／TCP の形状から来ている。

確認済み。q = HOME_Q = [0.25, -1.0, -0.1, -0.5, 0, 0] のとき
TCP = (0.2556, -0.0004, 0.1001) m で、ツール軸は真下を向く。
configs/env/mycobot_pickplace.yaml に書いた "ready" 姿勢と一致する。

自己テスト:  python3 mycobot_fk.py
"""

from __future__ import annotations
import math

# --- 運動学の連鎖。回転関節6つ。各自の +Z まわりに回る -----------------
# 関節 i の (origin_xyz, origin_rpy) を、関節 i-1 の座標系で表したもの。
# 順番 = ARM_JOINT_NAMES = joint2_to_joint1 .. joint6output_to_joint6。
CHAIN = (
    ((0.0, 0.0, 0.13956), (0.0, 0.0, 0.0)),
    ((0.0, 0.0, -0.001), (0.0, 1.5708, -1.5708)),
    ((-0.1104, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ((-0.096, 0.0, 0.06462), (0.0, 0.0, -1.5708)),
    ((0.0, -0.07318, -0.001), (1.5708, -1.5708, 0.0)),
    ((0.0, 0.0456, 0.0), (-1.5708, 0.0, 0.0)),
)
NUM_ARM_JOINTS = 6

# --- joint6_flange 座標でのツール形状（+Z がツール軸） -------------------
# assets/build_mycobot_mjcf.py と対応。gripper_base はフランジ+0.034
# （箱 56x26x40 mm、原点 +18 mm）、指の関節はフランジ+0.094（= TCP /
# 把持中心）、指パッドは 8x16x30 mm で原点 -5 mm。つまりパッドは
# フランジ+0.074 〜 +0.104 を占め、指先は TCP + 10 mm まで届く。
GRIPPER_MOUNT_Z = 0.034
FINGER_JOINT_Z = 0.060
TCP_Z = GRIPPER_MOUNT_Z + FINGER_JOINT_Z          # 0.094
FINGER_PAD_LEN = 0.030
FINGER_PAD_ORIGIN_Z = -0.005
FINGER_TIP_Z = TCP_Z + FINGER_PAD_ORIGIN_Z + FINGER_PAD_LEN / 2   # 0.104
FINGER_OPEN = 0.030

# 最小の高さが、机からのツールのクリアランスになるサンプル点。
# 指先は TCP より先まで届くので、TCP だけを見る床チェックは床チェックに
# ならない。ARM_BCRL002 の 65mm のパッドは TCP より 20mm 先まで届き、
# エキスパートの 4mm の把持オフセットと合わせて、記録した全デモで
# 指先を机より 1mm 下に入れていた（実機の失敗記録の原因C）。
TOOL_POINTS = (
    (0.0, 0.0, TCP_Z),                                   # 把持中心（TCP）
    (FINGER_OPEN + 0.004, 0.008, FINGER_TIP_Z),          # 左の指先の角
    (FINGER_OPEN + 0.004, -0.008, FINGER_TIP_Z),
    (-(FINGER_OPEN + 0.004), 0.008, FINGER_TIP_Z),       # 右の指先の角
    (-(FINGER_OPEN + 0.004), -0.008, FINGER_TIP_Z),
    (0.028, 0.013, GRIPPER_MOUNT_Z + 0.038),             # グリッパ本体の角
    (0.028, -0.013, GRIPPER_MOUNT_Z + 0.038),
    (-0.028, 0.013, GRIPPER_MOUNT_Z + 0.038),
    (-0.028, -0.013, GRIPPER_MOUNT_Z + 0.038),
)

# 公式 URDF の関節限界（rad）、J1〜J6
Q_LIMITS = ((-2.9321, 2.9321), (-2.4434, 2.4434), (-2.6179, 2.6179),
            (-2.6179, 2.6179), (-2.7052, 2.7925), (-3.14159, 3.14159))
# 学習時の "ready" 姿勢（configs/env/mycobot_pickplace.yaml の
# robot.default_joint_pos）。TCP ~ (0.256, 0.000, 0.100)、ツールは下向き
HOME_Q = (0.25, -1.0, -0.1, -0.5, 0.0, 0.0)


# --------------------------------------------------------------------------
# 最小限の剛体変換の代数。姿勢は (R, p) で、R は行優先の9要素
# --------------------------------------------------------------------------
def _rpy(r: float, p: float, y: float):
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    # Rz(y) @ Ry(p) @ Rx(r)。URDF の規約
    return (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
            sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
            -sp, cp * sr, cp * cr)


def _mul(a, b):
    """(R_a, p_a) o (R_b, p_b) -> 合成した姿勢。"""
    Ra, pa = a
    Rb, pb = b
    R = tuple(Ra[3 * i + 0] * Rb[0 + j] + Ra[3 * i + 1] * Rb[3 + j] + Ra[3 * i + 2] * Rb[6 + j]
              for i in range(3) for j in range(3))
    p = tuple(pa[i] + Ra[3 * i] * pb[0] + Ra[3 * i + 1] * pb[1] + Ra[3 * i + 2] * pb[2]
              for i in range(3))
    return R, p


def _rot_z(a: float):
    c, s = math.cos(a), math.sin(a)
    return ((c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0))


# 固定の関節原点の姿勢。q に依らないので一度だけ計算しておく
_ORIGINS = tuple((_rpy(*rpy), tuple(float(v) for v in xyz)) for xyz, rpy in CHAIN)


def flange_pose(q):
    """ロボット基準座標での joint6_flange リンクの (R, p)。q は rad。"""
    pose = ((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
    for origin, qi in zip(_ORIGINS, q):
        pose = _mul(_mul(pose, origin), _rot_z(float(qi)))
    return pose


def tcp_pos(q):
    """ロボット基準座標での TCP（把持中心）の位置。"""
    R, p = flange_pose(q)
    return tuple(p[i] + R[3 * i + 2] * TCP_Z for i in range(3))


def tool_points(q):
    """ロボット基準座標での、ツールの全安全サンプル点。"""
    R, p = flange_pose(q)
    out = []
    for lx, ly, lz in TOOL_POINTS:
        out.append(tuple(p[i] + R[3 * i] * lx + R[3 * i + 1] * ly + R[3 * i + 2] * lz
                         for i in range(3)))
    return out


def tool_min_z(q) -> float:
    """ロボット基準面から見た、ツール最下点の高さ。

    基準面は、シミュレータでも実機でも机の上面（机の上面 = z = 0 =
    ロボットの基準）。だから ``floor_z``（実測した台座のオフセット。
    普通は0）を引けば、この数値がそのまま机からのクリアランスになる。
    """
    R, p = flange_pose(q)
    return min(p[2] + R[6] * lx + R[7] * ly + R[8] * lz for lx, ly, lz in TOOL_POINTS)


def clamp_joint_limits(q):
    return [max(lo, min(hi, float(v))) for v, (lo, hi) in zip(q, Q_LIMITS)]


# --------------------------------------------------------------------------
# 床ガード本体
# --------------------------------------------------------------------------
class FloorGuard:
    """仮想床（産業用語の「安全平面」、作業範囲の柵）。

    ``limit(q_from, q_to)`` は、関節空間の直線 q_from -> q_to 上で、
    ツールが ``floor_z + clearance`` 以上に留まる最も遠い点を返す。
    机に食い込む指令は、単に拒否するのではなく短くする。だから
    アームは方策に接線方向では追従し続け、ただ平面より下へは
    降りられなくなる。

    二分探索（``iters`` 回。既定8でステップの約0.4%）は、20Hz の1刻み
    あたり順運動学10回ほど、つまり数十マイクロ秒で済む。
    on a Pi 4B.
    """

    def __init__(self, floor_z: float = 0.0, clearance: float = 0.005,
                 iters: int = 8):
        self.floor_z = float(floor_z)
        self.clearance = float(clearance)
        self.iters = int(iters)

    @property
    def min_z(self) -> float:
        return self.floor_z + self.clearance

    def is_safe(self, q) -> bool:
        return tool_min_z(q) >= self.min_z

    def limit(self, q_from, q_to):
        """-> (q_safe, blocked)。``blocked`` が True なら指令を切った。"""
        if self.is_safe(q_to):
            return list(q_to), False
        # すでに平面より下（校正ずれ、または押され中）。保持し、
        # 下へは絶対に進めない。
        if not self.is_safe(q_from):
            return list(q_from), True
        lo, hi = 0.0, 1.0                     # lo は安全と分かっている側、hi は危険側
        for _ in range(self.iters):
            mid = 0.5 * (lo + hi)
            q_mid = [a + (b - a) * mid for a, b in zip(q_from, q_to)]
            if self.is_safe(q_mid):
                lo = mid
            else:
                hi = mid
        return [a + (b - a) * lo for a, b in zip(q_from, q_to)], True


# --------------------------------------------------------------------------
def _self_test() -> None:
    tcp = tcp_pos(HOME_Q)
    R, _ = flange_pose(HOME_Q)
    axis_z = (R[2], R[5], R[8])          # 基準座標でのツールの +Z
    print(f"HOME_Q  TCP        = ({tcp[0]:.4f}, {tcp[1]:.4f}, {tcp[2]:.4f}) m")
    print(f"        tool axis  = ({axis_z[0]:+.3f}, {axis_z[1]:+.3f}, {axis_z[2]:+.3f})"
          "   (~(0,0,-1) = pointing straight down)")
    print(f"        tool_min_z = {tool_min_z(HOME_Q) * 1000:.1f} mm above the desk")
    assert abs(tcp[0] - 0.2556) < 1e-3 and abs(tcp[2] - 0.1001) < 1e-3, tcp
    assert axis_z[2] < -0.99, axis_z

    # 把持姿勢のクリアランスの連鎖。TCP = ワーク高さ/2 + expert.grasp_z_offset、
    # 指先 = TCP - (FINGER_TIP_Z - TCP_Z)
    tip_below_tcp = FINGER_TIP_Z - TCP_Z
    print(f"\nfingertips sit {tip_below_tcp * 1000:.0f} mm below the TCP "
          f"(ARM_BCRL002: 20 mm)")
    for name, off, tip_len in (("ARM_BCRL002 (65 mm pads, offset 4 mm)", 0.004, 0.020),
                               ("ARM_BCRL006 (30 mm pads, offset 8 mm)", 0.008, tip_below_tcp)):
        tcp_z = 0.015 + off                       # ワーク高さ 30mm / 2
        print(f"  {name}: TCP {tcp_z * 1000:4.0f} mm -> fingertips "
              f"{(tcp_z - tip_len) * 1000:+5.1f} mm")
    grasp_clear = (0.015 + 0.008) - tip_below_tcp
    assert grasp_clear > 0.005, ("the expert's grasp pose must clear the "
                                 f"5 mm floor plane, got {grasp_clear * 1000:.1f} mm")
    print(f"  => ARM_BCRL006 grasp clears the 5 mm floor plane by "
          f"{(grasp_clear - 0.005) * 1000:.0f} mm")
    guard = FloorGuard(floor_z=0.0, clearance=0.005)
    deep = [0.25, -1.6, -1.0, 0.2, 0.0, 0.0]          # 机を突き抜ける
    print(f"\ndive target tool_min_z = {tool_min_z(deep) * 1000:.1f} mm -> "
          f"safe={guard.is_safe(deep)}")
    q_safe, blocked = guard.limit(list(HOME_Q), deep)
    print(f"guard.limit -> blocked={blocked}, "
          f"tool_min_z={tool_min_z(q_safe) * 1000:.1f} mm "
          f"(must be >= {guard.min_z * 1000:.0f} mm)")
    assert blocked and tool_min_z(q_safe) >= guard.min_z - 1e-6
    print("\nself-test OK")


if __name__ == "__main__":
    _self_test()
