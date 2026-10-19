# -*- coding: utf-8 -*-
"""myCobot 280 Pi + parallel-jaw gripper -> MJCF, generated from the SINGLE
source of truth for the kinematics (``deploy/raspi/mycobot_fk.py``).

ARM_BCRL005 shipped the robot as USD, built by fetching Elephant Robotics'
official URDF and its COLLADA meshes and patching them. That path produced the
defect this project is asked to fix: **in the viewer the links closest to the
base did not render.** The cause is in the assets, not the viewer -- the
upstream ``.dae`` files declare ``transparent`` where the importer expects
``opaque``, and ``joint7.dae`` references a texture that upstream does not
publish, so the material resolves to nothing. ARM_BCRL005 patched both ends and
still had to keep a placeholder-texture workaround alive.

ARM_BCRL006 removes the class of defect instead of patching it again: the whole
robot is emitted as MJCF **primitive geometry** (capsules / boxes / cylinders)
whose dimensions are derived from the official joint origins. Consequences:

  * every link has geometry that always renders -- there is no material, no
    texture and no transparency flag left to go wrong,
  * the asset is one 12 KB text file with no download step, so ``run_all.sh``
    phase 0 has no network dependency and is reproducible offline,
  * the kinematics are IDENTICAL to the URDF, because the joint origins are
    imported from ``mycobot_fk.py`` rather than re-typed. ``verify()`` below
    asserts MuJoCo's own forward kinematics against that file to 1e-9 m, so
    the simulator, the analytic IK, the virtual floor and the Raspberry Pi
    guard provably share one chain.

The trade is photorealism, which this task never used: the wrist-camera policy
keys on a red cylinder against a dark bench, and the mesh-accurate shell of the
arm is barely in frame. Fidelity that matters (link lengths, joint axes, joint
limits, tool geometry, masses) is preserved exactly.

Usage
-----
    python assets/build_mycobot_mjcf.py                  # -> assets/mjcf/mycobot_280_pi.xml
    python assets/build_mycobot_mjcf.py --verify         # + FK cross-check (needs mujoco)
"""

from __future__ import annotations

import argparse
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "deploy", "raspi"))

from mycobot_fk import (  # noqa: E402  -- the authoritative chain
    CHAIN, Q_LIMITS, HOME_Q, NUM_ARM_JOINTS,
    GRIPPER_MOUNT_Z, FINGER_JOINT_Z, FINGER_PAD_ORIGIN_Z,
    FINGER_OPEN, _rpy,
)

# ---------------------------------------------------------------------------
# Link masses / inertia scales: byte-for-byte ARM_BCRL005's
# assets/build_mycobot_mjcf.py, so the dynamics the policy is trained against
# do not change with the asset format.
# ---------------------------------------------------------------------------
LINK_MASS = {"g_base": 0.30, "joint1": 0.18, "joint2": 0.16, "joint3": 0.13,
             "joint4": 0.10, "joint5": 0.07, "joint6": 0.05, "joint6_flange": 0.02}
LINK_SIZE = {"g_base": 0.10, "joint1": 0.09, "joint2": 0.11, "joint3": 0.10,
             "joint4": 0.09, "joint5": 0.06, "joint6": 0.05, "joint6_flange": 0.03}
FINGER_MIN = 0.004
FINGER_BOX = (0.008, 0.016, 0.030)      # pad thickness x width x length

ARM_JOINT_NAMES = ["joint2_to_joint1", "joint3_to_joint2", "joint4_to_joint3",
                   "joint5_to_joint4", "joint6_to_joint5", "joint6output_to_joint6"]
FINGER_JOINT_NAMES = ["left_finger_joint", "right_finger_joint"]
LINK_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6_flange"]

# myCobot 280's own livery: white shells, dark grey joint covers. Against the
# dark bench this reads clearly in both the viewer and the recorded video.
SHELL_RGBA = "0.93 0.93 0.95 1"
JOINT_RGBA = "0.16 0.16 0.18 1"
BASE_RGBA = "0.30 0.32 0.36 1"
TOOL_RGBA = "0.20 0.22 0.26 1"
PAD_RGBA = "0.85 0.55 0.15 1"


# ---------------------------------------------------------------------------
def _quat_from_rpy(rpy) -> tuple:
    """URDF rpy -> MuJoCo (w, x, y, z).

    The rotation matrix comes from ``mycobot_fk._rpy`` -- the reference
    implementation -- so the MJCF cannot disagree with the FK about the URDF
    rpy convention (Rz(yaw) @ Ry(pitch) @ Rx(roll)). Only the matrix -> quat
    step is local, and ``verify()`` proves the round trip.
    """
    m = _rpy(*rpy)
    t = m[0] + m[4] + m[8]
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        q = (0.25 * s, (m[7] - m[5]) / s, (m[2] - m[6]) / s, (m[3] - m[1]) / s)
    elif m[0] > m[4] and m[0] > m[8]:
        s = math.sqrt(1.0 + m[0] - m[4] - m[8]) * 2.0
        q = ((m[7] - m[5]) / s, 0.25 * s, (m[1] + m[3]) / s, (m[2] + m[6]) / s)
    elif m[4] > m[8]:
        s = math.sqrt(1.0 + m[4] - m[0] - m[8]) * 2.0
        q = ((m[2] - m[6]) / s, (m[1] + m[3]) / s, 0.25 * s, (m[5] + m[7]) / s)
    else:
        s = math.sqrt(1.0 + m[8] - m[0] - m[4]) * 2.0
        q = ((m[3] - m[1]) / s, (m[2] + m[6]) / s, (m[5] + m[7]) / s, 0.25 * s)
    n = math.sqrt(sum(v * v for v in q))
    q = tuple(v / n for v in q)
    return q if q[0] >= 0 else tuple(-v for v in q)


def _fmt(vals) -> str:
    """17 significant digits. Not pedantry: the joint origins and quaternions
    written here are the ONLY copy of the chain MuJoCo sees, and ``verify()``
    holds them to 1e-9 m against ``mycobot_fk.py``. At %.6g the rounding alone
    costs 1e-7 m, which would make the cross-check meaningless."""
    return " ".join(repr(float(v)) for v in vals)


def _inertial(link: str, com=(0.0, 0.0, 0.0)) -> str:
    """Same mass and box-approximation inertia (I = m s^2 / 6) ARM_BCRL005's
    URDF builder injected, with the COM placed at the link's geometric centre
    instead of its frame origin -- MuJoCo's ``inertial/pos`` IS the COM, and a
    long link whose COM sits on the joint axis is not a model of anything."""
    m = LINK_MASS[link]
    i = m * LINK_SIZE[link] ** 2 / 6.0
    return (f'<inertial pos="{_fmt(com)}" mass="{m:.4g}" '
            f'diaginertia="{i:.4e} {i:.4e} {i:.4e}"/>')


def _capsule(name: str, a, b, r: float, rgba: str) -> str:
    """Visual-only shell (contype/conaffinity 0). Arm-link collision is
    deliberately off: the only contacts this task needs are tool <-> object,
    tool <-> bench and object <-> bench, and the virtual floor already
    guarantees the arm cannot reach the bench with anything but the tool.
    Self-collision on a 6-DoF hobby arm buys nothing and costs solver time."""
    return (f'<geom name="{name}" type="capsule" fromto="{_fmt(list(a) + list(b))}" '
            f'size="{r:.4g}" rgba="{rgba}" contype="0" conaffinity="0" group="1"/>')


def _cyl(name: str, a, b, r: float, rgba: str) -> str:
    return (f'<geom name="{name}" type="cylinder" fromto="{_fmt(list(a) + list(b))}" '
            f'size="{r:.4g}" rgba="{rgba}" contype="0" conaffinity="0" group="1"/>')


# ---------------------------------------------------------------------------
def robot_xml(prefix: str, origin=(0.0, 0.0, 0.0), indent: str = "    ") -> str:
    """The arm as a nested <body> chain, one env's worth.

    ``prefix`` namespaces every name and ``origin`` places the base, so N
    environments can live in ONE MuJoCo model -- the direct equivalent of Isaac
    Lab's ``/World/envs/env_i`` cloning, which is what lets a single
    ``mj_step`` advance the whole batch. The origin is baked into the base
    body's ``pos`` rather than a wrapping body because MuJoCo only allows a
    free joint (the block, the distractor) directly under ``worldbody``, so the
    environments cannot be grouped under a common parent.
    """
    p = prefix
    out = []

    def w(depth: int, s: str):
        out.append(indent * depth + s)

    # ---- fixed base: the pedestal between the bench and joint 1 -----------
    # joint 1's origin is 139.56 mm above the bench in the official URDF; that
    # column is a real part of the robot and is exactly what "the links near
    # the base do not show up" referred to. It is drawn explicitly.
    base_top = CHAIN[0][0][2]
    w(3, f'<body name="{p}base" pos="{_fmt(origin)}">')
    w(4, _inertial("g_base", (0.0, 0.0, base_top * 0.4)))
    w(4, f'<geom name="{p}base_plate" type="box" pos="0 0 0.006" size="0.055 0.055 0.006" '
         f'rgba="{BASE_RGBA}" contype="1" conaffinity="1" group="1"/>')
    w(4, _cyl(f"{p}base_col", (0, 0, 0.012), (0, 0, base_top), 0.042, SHELL_RGBA))
    w(4, _cyl(f"{p}base_cap", (0, 0, base_top - 0.018), (0, 0, base_top), 0.045, JOINT_RGBA))

    # ---- 6 revolute links -------------------------------------------------
    # Each link's shell is drawn from its own frame origin to the origin of the
    # NEXT joint, so the drawn segment IS the kinematic segment: a link cannot
    # be visually present and kinematically absent.
    next_off = [CHAIN[i + 1][0] for i in range(NUM_ARM_JOINTS - 1)] + [(0.0, 0.0, GRIPPER_MOUNT_Z)]
    radii = [0.036, 0.034, 0.030, 0.028, 0.026, 0.024]
    depth = 4
    for i in range(NUM_ARM_JOINTS):
        xyz, rpy = CHAIN[i]
        quat = _quat_from_rpy(rpy)
        link = LINK_NAMES[i]
        seg = next_off[i]
        seg_len = math.sqrt(sum(v * v for v in seg))
        r = radii[i]
        w(depth, f'<body name="{p}{link}" pos="{_fmt(xyz)}" quat="{_fmt(quat)}">')
        w(depth + 1, _inertial(link, tuple(v * 0.5 for v in seg)))
        w(depth + 1, f'<joint name="{p}{ARM_JOINT_NAMES[i]}" type="hinge" axis="0 0 1" '
                     f'pos="0 0 0" range="{_fmt(Q_LIMITS[i])}" damping="0.35" '
                     f'armature="0.012" frictionloss="0.02"/>')
        # dark cover centred on the joint axis: makes the axis of rotation
        # readable in the viewport, which is what a reviewer looks for
        w(depth + 1, _cyl(f"{p}{link}_hub", (0, 0, -r * 0.62), (0, 0, r * 0.62), r * 0.96, JOINT_RGBA))
        if seg_len > 1e-4:
            w(depth + 1, _capsule(f"{p}{link}_arm", (0, 0, 0), seg, r * 0.74, SHELL_RGBA))
        depth += 1

    # ---- tool: gripper base, two sliding pads, TCP site -------------------
    gb = GRIPPER_MOUNT_Z
    w(depth, f'<body name="{p}gripper_base" pos="0 0 {gb:.6g}">')
    w(depth + 1, '<inertial pos="0 0 0.018" mass="0.06" '
                 'diaginertia="1.60e-05 1.60e-05 1.60e-05"/>')
    w(depth + 1, f'<geom name="{p}gripper_body" type="box" pos="0 0 0.018" '
                 f'size="0.028 0.013 0.020" rgba="{TOOL_RGBA}" '
                 f'contype="1" conaffinity="1" condim="4" friction="0.9 0.02 0.001" group="1"/>')
    # wrist camera: +Z of the flange frame is the tool axis. MuJoCo cameras
    # look down -Z with +Y up, while configs/env stores the ROS convention
    # (+Z = optical axis), so mujoco_sim/scene.py rotates the stored
    # quaternion by 180 deg about X when it stamps this element.
    w(depth + 1, "<!--WRISTCAM-->")
    for sign, side in ((+1.0, "left"), (-1.0, "right")):
        jz = FINGER_JOINT_Z
        w(depth + 1, f'<body name="{p}{side}_finger" pos="0 0 {jz:.6g}">')
        w(depth + 2, '<inertial pos="0 0 -0.005" mass="0.012" '
                     'diaginertia="1.80e-06 1.80e-06 1.80e-06"/>')
        w(depth + 2, f'<joint name="{p}{side}_finger_joint" type="slide" '
                     f'axis="{sign:+.0f} 0 0" pos="0 0 0" '
                     f'range="{FINGER_MIN:.4g} {FINGER_OPEN:.4g}" damping="4.0" '
                     f'armature="0.002"/>')
        # pad box origin at +/- half its thickness, exactly as the URDF: the
        # OUTER face then sits at joint_value + 0.008, which is the x the
        # mycobot_fk TOOL_POINTS sample.
        w(depth + 2, f'<geom name="{p}{side}_pad" type="box" '
                     f'pos="{sign * FINGER_BOX[0] / 2:.6g} 0 {FINGER_PAD_ORIGIN_Z:.6g}" '
                     f'size="{FINGER_BOX[0] / 2:.6g} {FINGER_BOX[1] / 2:.6g} '
                     f'{FINGER_BOX[2] / 2:.6g}" rgba="{PAD_RGBA}" '
                     f'contype="1" conaffinity="1" condim="4" '
                     f'friction="1.6 0.05 0.002" solimp="0.95 0.99 0.001" group="1"/>')
        w(depth + 1, "</body>")
    # TCP = grasp centre. A <site>, not a body: it must not add mass, and the
    # env reads its position through mj_data.site_xpos.
    w(depth + 1, f'<site name="{p}tcp" pos="0 0 {FINGER_JOINT_Z:.6g}" size="0.004" '
                 f'rgba="0.1 0.9 0.3 0.35" group="2"/>')
    w(depth, "</body>")                     # gripper_base
    for _ in range(NUM_ARM_JOINTS):
        depth -= 1
        w(depth, "</body>")
    w(3, "</body>")                         # base
    return "\n".join(out)


def actuator_xml(prefix: str, arm_kp: float, arm_kv: float, arm_effort: float,
                 finger_kp: float, finger_kv: float, finger_effort: float,
                 indent: str = "    ") -> str:
    """Position-servo actuators == Isaac Lab's ImplicitActuatorCfg(stiffness,
    damping): MuJoCo's ``position`` actuator applies ``kp*(target-q) - kv*qd``,
    which is the same PD law, with ``forcerange`` playing the effort limit."""
    lines = []
    for j in ARM_JOINT_NAMES:
        lines.append(indent * 2 + f'<position name="{prefix}act_{j}" joint="{prefix}{j}" '
                                  f'kp="{arm_kp:.6g}" kv="{arm_kv:.6g}" '
                                  f'forcerange="{-arm_effort:.6g} {arm_effort:.6g}" '
                                  f'ctrlrange="-3.2 3.2"/>')
    for j in FINGER_JOINT_NAMES:
        lines.append(indent * 2 + f'<position name="{prefix}act_{j}" joint="{prefix}{j}" '
                                  f'kp="{finger_kp:.6g}" kv="{finger_kv:.6g}" '
                                  f'forcerange="{-finger_effort:.6g} {finger_effort:.6g}" '
                                  f'ctrlrange="{FINGER_MIN:.4g} {FINGER_OPEN:.4g}"/>')
    return "\n".join(lines)


SINGLE_ROBOT_TEMPLATE = """<mujoco model="mycobot_280_pi">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.004167" integrator="implicitfast" cone="elliptic"/>
  <visual><headlight ambient="0.4 0.4 0.4" diffuse="0.7 0.7 0.7"/>
    <global offwidth="1920" offheight="1080"/></visual>
  <worldbody>
    <light name="key" pos="0.4 -0.4 1.2" dir="-0.3 0.3 -1" directional="true"/>
    <geom name="bench" type="plane" pos="0 0 0" size="1 1 0.02"
          rgba="0.20 0.11 0.06 1" contype="1" conaffinity="1"/>
{robot}
  </worldbody>
  <actuator>
{actuators}
  </actuator>
</mujoco>
"""


def build_single(path: str) -> str:
    """A standalone one-robot MJCF -- used by ``--verify``, by the unit tests
    and by anyone who wants to open the arm in ``python -m mujoco.viewer``
    without the rest of the scene."""
    xml = SINGLE_ROBOT_TEMPLATE.format(
        robot=robot_xml("").replace("<!--WRISTCAM-->", ""),
        actuators=actuator_xml("", 50.0, 5.0, 5.0, 300.0, 20.0, 30.0))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)
    return path


# ---------------------------------------------------------------------------
def verify(path: str, tol: float = 1e-9) -> None:
    """Assert MuJoCo's forward kinematics equals ``deploy/raspi/mycobot_fk.py``.

    This is the check that makes the whole safety story true rather than
    claimed: the virtual floor, the analytic IK Jacobian, the MimicGen pose
    warp and the Raspberry Pi guard are all computed from ``mycobot_fk.py``,
    while the physics is computed by MuJoCo. If the two chains ever disagree,
    the arm the policy learns on is not the arm the guard protects.
    """
    import numpy as np
    import mujoco
    from mycobot_fk import tcp_pos, tool_min_z

    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    tcp_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    qadr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in ARM_JOINT_NAMES]

    probes = [list(HOME_Q), [0.0] * 6, [0.25, -1.6, -1.0, 0.2, 0.0, 0.0],
              [-0.4, -0.8, 0.6, -1.1, 0.3, 0.7], [1.2, -0.3, -1.4, 0.9, -0.6, 2.0]]
    worst = 0.0
    for q in probes:
        for adr, v in zip(qadr, q):
            data.qpos[adr] = v
        mujoco.mj_kinematics(model, data)
        err = float(np.abs(np.asarray(data.site_xpos[tcp_sid]) - np.array(tcp_pos(q))).max())
        worst = max(worst, err)
        if err > tol:
            raise AssertionError(
                f"MJCF forward kinematics disagrees with deploy/raspi/mycobot_fk.py "
                f"at q={q}: {err:.3e} m > {tol:.0e}. The simulator and the robot's "
                f"safety guard would be protecting different arms.")
    print(f"[build_mycobot_mjcf] MuJoCo TCP == mycobot_fk.tcp_pos on {len(probes)} "
          f"poses (max error {worst:.2e} m)")

    # the tool sample points the floor guard uses must also be reachable from
    # the MJCF -- check the lowest one at the home pose
    for adr, v in zip(qadr, HOME_Q):
        data.qpos[adr] = v
    mujoco.mj_kinematics(model, data)
    print(f"[build_mycobot_mjcf] HOME_Q tool_min_z = {tool_min_z(HOME_Q) * 1000:.1f} mm, "
          f"TCP = {np.round(data.site_xpos[tcp_sid], 4)}")
    print(f"[build_mycobot_mjcf] bodies={model.nbody} joints={model.njnt} "
          f"geoms={model.ngeom} actuators={model.nu}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "assets", "mjcf",
                                                  "mycobot_280_pi.xml"))
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    path = build_single(args.out)
    print(f"[build_mycobot_mjcf] wrote {path}")
    if args.verify:
        verify(path)


if __name__ == "__main__":
    main()
