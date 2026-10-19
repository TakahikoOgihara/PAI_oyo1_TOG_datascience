# -*- coding: utf-8 -*-
"""myCobot 280 Pi forward kinematics + tool safety envelope.

ZERO DEPENDENCIES (stdlib ``math`` only) on purpose: this file is the
authoritative definition of "where is the tool, and how low does it reach",
and it must run on the Raspberry Pi 4B inside the safety layer -- before and
independently of numpy/torch/the policy. ``deploy/raspi/`` is scp'd to the Pi
as a self-contained directory (operations guide appendix A), so the guard can
never be missing a dependency at the moment it matters.

The simulator conforms to THIS file, not the other way round:
``arm_common/kinematics.py`` imports the same constants and adds a batched
torch implementation, with a start-up assert that both agree to 1e-6 m. One
source of truth for the safety envelope => what the policy learns is exactly
what the hardware enforces (field-failure notes §4).

Constants come from the official Elephant Robotics URDF
(mycobot_ros2 @ 3999e2c, mycobot_description/urdf/mycobot_280_pi) plus the
gripper/TCP geometry appended by assets/build_mycobot_mjcf.py.

Verified: q = HOME_Q = [0.25, -1.0, -0.1, -0.5, 0, 0] gives
TCP = (0.2556, -0.0004, 0.1001) m with the tool axis pointing straight down
-- matching configs/env/mycobot_pickplace.yaml's documented "ready" pose.

Self-test:  python3 mycobot_fk.py
"""

from __future__ import annotations
import math

# --- kinematic chain: 6 revolute joints, each rotating about its own +Z -----
# (origin_xyz, origin_rpy) of joint i, expressed in the frame of joint i-1.
# Order = ARM_JOINT_NAMES = joint2_to_joint1 .. joint6output_to_joint6.
CHAIN = (
    ((0.0, 0.0, 0.13956), (0.0, 0.0, 0.0)),
    ((0.0, 0.0, -0.001), (0.0, 1.5708, -1.5708)),
    ((-0.1104, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ((-0.096, 0.0, 0.06462), (0.0, 0.0, -1.5708)),
    ((0.0, -0.07318, -0.001), (1.5708, -1.5708, 0.0)),
    ((0.0, 0.0456, 0.0), (-1.5708, 0.0, 0.0)),
)
NUM_ARM_JOINTS = 6

# --- tool geometry in the joint6_flange frame (+Z = tool axis) -------------
# Mirrors assets/build_mycobot_mjcf.py: gripper_base at flange+0.034 (box
# 56x26x40 mm, origin +18 mm), finger joints at flange+0.094 (= TCP / grasp
# centre), finger pads 8x16x30 mm with origin -5 mm => pads span
# flange+0.074 .. +0.104, i.e. the fingertips reach TCP + 10 mm.
GRIPPER_MOUNT_Z = 0.034
FINGER_JOINT_Z = 0.060
TCP_Z = GRIPPER_MOUNT_Z + FINGER_JOINT_Z          # 0.094
FINGER_PAD_LEN = 0.030
FINGER_PAD_ORIGIN_Z = -0.005
FINGER_TIP_Z = TCP_Z + FINGER_PAD_ORIGIN_Z + FINGER_PAD_LEN / 2   # 0.104
FINGER_OPEN = 0.030

# Sample points whose minimum height is the tool's clearance over the desk.
# THE FINGERTIPS STILL REACH PAST THE TCP, so a TCP-only floor check is not a
# floor check. ARM_BCRL002's 65 mm pads reached 20 mm past it and, with the
# expert's 4 mm grasp offset, put the tips 1 mm BELOW the desk in every
# recorded demonstration (field-failure notes §2, cause C).
TOOL_POINTS = (
    (0.0, 0.0, TCP_Z),                                   # grasp centre (TCP)
    (FINGER_OPEN + 0.004, 0.008, FINGER_TIP_Z),          # L fingertip corners
    (FINGER_OPEN + 0.004, -0.008, FINGER_TIP_Z),
    (-(FINGER_OPEN + 0.004), 0.008, FINGER_TIP_Z),       # R fingertip corners
    (-(FINGER_OPEN + 0.004), -0.008, FINGER_TIP_Z),
    (0.028, 0.013, GRIPPER_MOUNT_Z + 0.038),             # gripper body corners
    (0.028, -0.013, GRIPPER_MOUNT_Z + 0.038),
    (-0.028, 0.013, GRIPPER_MOUNT_Z + 0.038),
    (-0.028, -0.013, GRIPPER_MOUNT_Z + 0.038),
)

# joint limits from the official URDF (rad), J1..J6
Q_LIMITS = ((-2.9321, 2.9321), (-2.4434, 2.4434), (-2.6179, 2.6179),
            (-2.6179, 2.6179), (-2.7052, 2.7925), (-3.14159, 3.14159))
# the training "ready" pose (configs/env/mycobot_pickplace.yaml
# robot.default_joint_pos) -- TCP ~ (0.256, 0.000, 0.100), tool pointing down
HOME_Q = (0.25, -1.0, -0.1, -0.5, 0.0, 0.0)


# --------------------------------------------------------------------------
# minimal rigid-transform algebra: a pose is (R, p) with R row-major 9-tuple
# --------------------------------------------------------------------------
def _rpy(r: float, p: float, y: float):
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    # Rz(y) @ Ry(p) @ Rx(r) -- the URDF convention
    return (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
            sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
            -sp, cp * sr, cp * cr)


def _mul(a, b):
    """(R_a, p_a) o (R_b, p_b) -> composed pose."""
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


# fixed joint-origin poses, precomputed once (they do not depend on q)
_ORIGINS = tuple((_rpy(*rpy), tuple(float(v) for v in xyz)) for xyz, rpy in CHAIN)


def flange_pose(q):
    """(R, p) of the joint6_flange link in the robot base frame, q in rad."""
    pose = ((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
    for origin, qi in zip(_ORIGINS, q):
        pose = _mul(_mul(pose, origin), _rot_z(float(qi)))
    return pose


def tcp_pos(q):
    """TCP (grasp-centre) position in the robot base frame."""
    R, p = flange_pose(q)
    return tuple(p[i] + R[3 * i + 2] * TCP_Z for i in range(3))


def tool_points(q):
    """Every safety sample point of the tool, in the robot base frame."""
    R, p = flange_pose(q)
    out = []
    for lx, ly, lz in TOOL_POINTS:
        out.append(tuple(p[i] + R[3 * i] * lx + R[3 * i + 1] * ly + R[3 * i + 2] * lz
                         for i in range(3)))
    return out


def tool_min_z(q) -> float:
    """Height of the LOWEST point of the tool above the robot base plane.

    The base plane is the desk top in both the simulator (desk top = z = 0 =
    robot base) and on the real bench, so this number IS the desk clearance
    once ``floor_z`` (a measured base-plate offset, normally 0) is subtracted.
    """
    R, p = flange_pose(q)
    return min(p[2] + R[6] * lx + R[7] * ly + R[8] * lz for lx, ly, lz in TOOL_POINTS)


def clamp_joint_limits(q):
    return [max(lo, min(hi, float(v))) for v, (lo, hi) in zip(q, Q_LIMITS)]


# --------------------------------------------------------------------------
# the floor guard itself
# --------------------------------------------------------------------------
class FloorGuard:
    """Virtual floor (industrial "safety plane" / workspace fence).

    ``limit(q_from, q_to)`` returns the furthest point along the straight
    joint-space segment q_from -> q_to whose tool stays at or above
    ``floor_z + clearance``. A command that would dig into the desk is
    therefore SHORTENED, not merely rejected: the arm keeps tracking the
    policy tangentially and simply cannot descend past the plane.

    Bisection (``iters`` steps, default 8 -> ~0.4 % of the step) costs ~10
    forward-kinematics evaluations per 20 Hz tick, i.e. tens of microseconds
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
        """-> (q_safe, blocked). ``blocked`` is True when the command was cut."""
        if self.is_safe(q_to):
            return list(q_to), False
        # Already below the plane (mis-calibrated start, or a push in
        # progress): hold, never continue downward.
        if not self.is_safe(q_from):
            return list(q_from), True
        lo, hi = 0.0, 1.0                     # lo is known-safe, hi unsafe
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
    axis_z = (R[2], R[5], R[8])          # tool +Z in base frame
    print(f"HOME_Q  TCP        = ({tcp[0]:.4f}, {tcp[1]:.4f}, {tcp[2]:.4f}) m")
    print(f"        tool axis  = ({axis_z[0]:+.3f}, {axis_z[1]:+.3f}, {axis_z[2]:+.3f})"
          "   (~(0,0,-1) = pointing straight down)")
    print(f"        tool_min_z = {tool_min_z(HOME_Q) * 1000:.1f} mm above the desk")
    assert abs(tcp[0] - 0.2556) < 1e-3 and abs(tcp[2] - 0.1001) < 1e-3, tcp
    assert axis_z[2] < -0.99, axis_z

    # grasp-pose clearance chain: TCP = object_height/2 + expert.grasp_z_offset,
    # fingertips = TCP - (FINGER_TIP_Z - TCP_Z)
    tip_below_tcp = FINGER_TIP_Z - TCP_Z
    print(f"\nfingertips sit {tip_below_tcp * 1000:.0f} mm below the TCP "
          f"(ARM_BCRL002: 20 mm)")
    for name, off, tip_len in (("ARM_BCRL002 (65 mm pads, offset 4 mm)", 0.004, 0.020),
                               ("ARM_BCRL006 (30 mm pads, offset 8 mm)", 0.008, tip_below_tcp)):
        tcp_z = 0.015 + off                       # object height 30 mm / 2
        print(f"  {name}: TCP {tcp_z * 1000:4.0f} mm -> fingertips "
              f"{(tcp_z - tip_len) * 1000:+5.1f} mm")
    grasp_clear = (0.015 + 0.008) - tip_below_tcp
    assert grasp_clear > 0.005, ("the expert's grasp pose must clear the "
                                 f"5 mm floor plane, got {grasp_clear * 1000:.1f} mm")
    print(f"  => ARM_BCRL006 grasp clears the 5 mm floor plane by "
          f"{(grasp_clear - 0.005) * 1000:.0f} mm")
    guard = FloorGuard(floor_z=0.0, clearance=0.005)
    deep = [0.25, -1.6, -1.0, 0.2, 0.0, 0.0]          # dives through the desk
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
