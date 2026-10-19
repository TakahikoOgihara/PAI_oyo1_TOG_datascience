# -*- coding: utf-8 -*-
"""Scene assembly: N pick-and-place environments inside ONE MuJoCo model.

This is the direct equivalent of Isaac Lab's ``InteractiveScene`` cloning that
ARM_BCRL005 relied on. Each environment is a ``<body>`` placed on a grid at
``env_origins[i]``, holding its own robot, bench board, block and distractor,
with every element name-prefixed ``e{i}_``. One ``mj_step`` then advances the
whole batch, so the vectorised MDP code in ``env.py`` reads exactly like the
Isaac Lab version did -- observations are computed in the ENV-LOCAL frame as
``world - env_origin``, which is the same contract the 31-dim observation, the
ONNX graphs and ``deploy/raspi/`` already assume.

Why one model instead of N separate ``MjData`` objects: the environments are
mechanically independent, so MuJoCo's factorisation is block-diagonal and the
cost is linear either way -- but a single model means a single step call, a
single set of camera ids, and per-environment domain randomisation is just an
indexed write into ``model.body_mass`` / ``model.geom_friction`` /
``model.actuator_gainprm``. That is what keeps ``DomainRandomizer`` readable.

The robot itself comes from ``assets/build_mycobot_mjcf.py``, which derives it
from ``deploy/raspi/mycobot_fk.py``. Nothing about the chain is re-typed here.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "deploy", "raspi"))

from assets.build_mycobot_mjcf import (  # noqa: E402
    ARM_JOINT_NAMES, FINGER_JOINT_NAMES, actuator_xml, robot_xml,
)

SCENE_TEMPLATE = """<mujoco model="arm_bcrl006_pickplace">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{sim_dt:.8f}" integrator="implicitfast" cone="elliptic"
          solver="Newton" iterations="30" ls_iterations="12"/>
  <size njmax="{njmax}" nconmax="{nconmax}"/>
  <visual>
    <headlight ambient="{amb:.3f} {amb:.3f} {amb:.3f}"
               diffuse="{hdif:.3f} {hdif:.3f} {hdif:.3f}" specular="0.1 0.1 0.1"/>
    <!-- the offscreen framebuffer must be at least as large as any
         camera we render: 224 for the wrist camera, 960x540 for the
         report video. MuJoCo defaults to 640x480 and raises otherwise. -->
    <global offwidth="1920" offheight="1080"/>
    <map znear="0.01" zfar="30"/>
    <quality shadowsize="2048" offsamples="4"/>
  </visual>
  <default>
    <geom solref="0.008 1" solimp="0.95 0.99 0.001"/>
  </default>
  <asset>
    <texture name="skybox" type="skybox" builtin="gradient" width="128" height="128"
             rgb1="0.10 0.12 0.16" rgb2="0.02 0.02 0.03"/>
  </asset>
  <worldbody>
    <light name="key" pos="{lx:.3f} {ly:.3f} 2.0" dir="-0.25 0.25 -1"
           directional="true" diffuse="{ldif:.3f} {ldif:.3f} {ldif:.3f}"
           specular="0.15 0.15 0.15" castshadow="true"/>
    <geom name="ground" type="plane" pos="0 0 {ground_z:.6f}" size="0 0 0.05"
          rgba="{ground_rgba}" contype="1" conaffinity="1"/>
    <!-- Framing for the report video and the viewer: 0.57 m from (0.21, 0, 0.08),
         which is the centre of the reachable workspace, so the arm fills the
         frame and the block, the target offset and the clutter are all in it.
         (These axes assume env 0, i.e. num_envs=1 -- which is what --mode play
         and src/record_video.py use.) -->
    <camera name="scenecam" pos="0.55 -0.36 0.36"
            xyaxes="0.7270 0.6866 0 -0.3379 0.3577 0.8703"/>
    <camera name="topcam" pos="0.21 0.0 0.55" xyaxes="0 1 0 -1 0 0"/>
{envs}
  </worldbody>
  <actuator>
{actuators}
  </actuator>
</mujoco>
"""


@dataclass
class SceneIndex:
    """Every id the env and the randomiser need, resolved once at build time.

    Resolving names to ids EAGERLY (instead of calling ``mj_name2id`` inside
    the step loop) is what keeps the 20 Hz control loop free of string work,
    and it means a renamed element fails at construction rather than silently
    addressing the wrong body 40 minutes into a run.
    """

    num_envs: int
    env_origins: np.ndarray                     # (N, 3)
    arm_qadr: np.ndarray = field(default=None)  # (N, 6) qpos addresses
    arm_dofadr: np.ndarray = field(default=None)
    finger_qadr: np.ndarray = field(default=None)   # (N, 2)
    finger_dofadr: np.ndarray = field(default=None)
    arm_actid: np.ndarray = field(default=None)     # (N, 6)
    finger_actid: np.ndarray = field(default=None)  # (N, 2)
    obj_bodyid: np.ndarray = field(default=None)    # (N,)
    obj_qadr: np.ndarray = field(default=None)      # (N,) freejoint qpos adr
    obj_dofadr: np.ndarray = field(default=None)
    obj_geomid: np.ndarray = field(default=None)
    dist_bodyid: np.ndarray = field(default=None)
    dist_qadr: np.ndarray = field(default=None)
    dist_geomid: np.ndarray = field(default=None)
    desk_geomid: np.ndarray = field(default=None)
    tcp_siteid: np.ndarray = field(default=None)
    flange_bodyid: np.ndarray = field(default=None)
    wristcam_id: np.ndarray = field(default=None)
    scenecam_id: int = -1
    topcam_id: int = -1


def _env_grid(n: int, spacing: float) -> np.ndarray:
    """Square-ish grid, same layout rule Isaac Lab uses for ``env_origins``."""
    cols = int(math.ceil(math.sqrt(n)))
    out = np.zeros((n, 3))
    for i in range(n):
        out[i, 0] = (i % cols) * spacing
        out[i, 1] = (i // cols) * spacing
    out[:, :2] -= out[:, :2].mean(axis=0)
    return out


def _wristcam_xml(prefix: str, cam_cfg) -> str:
    """Wrist camera inside the gripper_base body.

    ``configs/env`` stores the mount rotation in the ROS convention
    (+Z = optical axis), which is what the real camera's extrinsics are quoted
    in and what ARM_BCRL005 fed Isaac Lab. MuJoCo cameras look down -Z with +Y
    up, so the stored quaternion is post-multiplied by a 180 deg rotation about
    X. Converting HERE, at the single place the camera is created, keeps
    ``configs/env/mycobot_pickplace.yaml`` describing the physical camera
    rather than a simulator's internal convention.
    """
    w, x, y, z = (float(v) for v in cam_cfg.rot_wxyz)
    # q * (0, 1, 0, 0)
    q = (-x, w, z, -y)
    n = math.sqrt(sum(v * v for v in q))
    q = tuple(v / n for v in q)
    px, py, pz = (float(v) for v in cam_cfg.pos)
    # square sensor => vertical fov == horizontal fov; MuJoCo takes fovy in deg
    fovy = float(cam_cfg.hfov_deg) * float(cam_cfg.height) / float(cam_cfg.width)
    return (f'<camera name="{prefix}wristcam" pos="{px!r} {py!r} {pz!r}" '
            f'quat="{q[0]!r} {q[1]!r} {q[2]!r} {q[3]!r}" fovy="{fovy:.6g}"/>')


def build_scene_xml(cfg, num_envs: int, with_camera: bool,
                    env_spacing: float = 1.4) -> tuple[str, np.ndarray]:
    """-> (MJCF string, env_origins (N,3))."""
    e = cfg.env
    origins = _env_grid(num_envs, env_spacing)
    desk_sx, desk_sy, desk_sz = (float(v) for v in e.desk.size)
    # ARM_BCRL005's board is 1.2 m square because Isaac Lab spaced envs 2 m
    # apart. Here the boards must not overlap at the tighter spacing that keeps
    # the batch compact, so the board is clipped to the spacing.
    half_x = min(desk_sx, env_spacing * 0.98) / 2.0
    half_y = min(desk_sy, env_spacing * 0.98) / 2.0
    desk_rgba = " ".join(f"{float(c):.4f}" for c in e.desk.color) + " 1"
    obj_rgba = " ".join(f"{float(c):.4f}" for c in e.object.color) + " 1"
    ground_rgba = " ".join(f"{float(c):.4f}" for c in e.ground.color) + " 1"
    r = e.robot

    bodies = []
    for i in range(num_envs):
        p = f"e{i}_"
        ox, oy, oz = (float(v) for v in origins[i])
        b = []
        b.append(f'      <geom name="{p}desk" type="box" '
                 f'pos="{ox!r} {oy!r} {oz - desk_sz / 2:.6f}" '
                 f'size="{half_x:.6f} {half_y:.6f} {desk_sz / 2:.6f}" rgba="{desk_rgba}" '
                 f'contype="1" conaffinity="1" condim="4" friction="1.0 0.02 0.001"/>')
        rob = robot_xml(p, (ox, oy, oz))
        rob = rob.replace("<!--WRISTCAM-->",
                          _wristcam_xml(p, e.camera) if with_camera else "")
        b.append(rob)
        # the block: a red-painted wooden cylinder (assets/block_reference.jpg)
        spawn_x = (float(e.object.spawn_x[0]) + float(e.object.spawn_x[1])) / 2.0
        b.append(
            f'      <body name="{p}obj" '
            f'pos="{ox + spawn_x:.6f} {oy!r} {oz + float(e.object.height) / 2:.6f}">'
            f'<freejoint name="{p}obj_free"/>'
            f'<geom name="{p}obj_geom" type="cylinder" '
            f'size="{float(e.object.radius):.6f} {float(e.object.height) / 2:.6f}" '
            f'rgba="{obj_rgba}" mass="{float(e.object.mass):.6f}" condim="4" '
            f'friction="1.5 0.02 0.001" solimp="0.96 0.995 0.001" '
            f'contype="1" conaffinity="1"/></body>')
        if bool(e.domain_rand.distractor):
            ds = float(e.domain_rand.distractor_size)
            b.append(
                f'      <body name="{p}distractor" '
                f'pos="{ox + 0.28:.6f} {oy + 0.16:.6f} {oz + ds / 2:.6f}">'
                f'<freejoint name="{p}distractor_free"/>'
                f'<geom name="{p}distractor_geom" type="box" '
                f'size="{ds / 2:.6f} {ds / 2:.6f} {ds / 2:.6f}" '
                f'rgba="0.45 0.45 0.50 1" mass="0.05" condim="4" '
                f'friction="1.0 0.02 0.001" contype="1" conaffinity="1"/></body>')
        bodies.append("\n".join(b))

    actuators = "\n".join(
        actuator_xml(f"e{i}_", float(r.arm_stiffness), float(r.arm_damping),
                     float(r.arm_effort_limit), float(r.finger_stiffness),
                     float(r.finger_damping), float(r.finger_effort_limit))
        for i in range(num_envs))

    # Isaac Lab drove a dome light in lux; MuJoCo's light is a unit-less
    # diffuse colour. env.lighting.intensity keeps its role as THE exposure
    # knob the phase-2 image gate reacts to, mapped onto [0,1] around the
    # 900 lux the ARM_BCRL005 scene was validated at.
    ldif = min(1.0, max(0.05, float(e.lighting.intensity) / 1400.0))
    xml = SCENE_TEMPLATE.format(
        sim_dt=float(e.sim_dt),
        njmax=max(2000, 200 * num_envs), nconmax=max(1000, 100 * num_envs),
        amb=0.34 * ldif + 0.16, hdif=0.50 * ldif + 0.12, ldif=ldif,
        lx=float(origins[:, 0].mean()) + 0.4, ly=float(origins[:, 1].mean()) - 0.4,
        ground_z=-desk_sz - 0.001, ground_rgba=ground_rgba,
        envs="\n".join(bodies), actuators=actuators)
    return xml, origins


def index_scene(model, num_envs: int, origins: np.ndarray, with_camera: bool) -> SceneIndex:
    import mujoco

    def jid(n):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)

    def aid(n):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)

    def bid(n):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)

    def gid(n):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)

    def sid(n):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)

    def cid(n):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, n)

    idx = SceneIndex(num_envs=num_envs, env_origins=origins.copy())
    n = num_envs
    idx.arm_qadr = np.zeros((n, 6), dtype=np.int32)
    idx.arm_dofadr = np.zeros((n, 6), dtype=np.int32)
    idx.finger_qadr = np.zeros((n, 2), dtype=np.int32)
    idx.finger_dofadr = np.zeros((n, 2), dtype=np.int32)
    idx.arm_actid = np.zeros((n, 6), dtype=np.int32)
    idx.finger_actid = np.zeros((n, 2), dtype=np.int32)
    idx.obj_bodyid = np.zeros(n, dtype=np.int32)
    idx.obj_qadr = np.zeros(n, dtype=np.int32)
    idx.obj_dofadr = np.zeros(n, dtype=np.int32)
    idx.obj_geomid = np.zeros(n, dtype=np.int32)
    idx.desk_geomid = np.zeros(n, dtype=np.int32)
    idx.tcp_siteid = np.zeros(n, dtype=np.int32)
    idx.flange_bodyid = np.zeros(n, dtype=np.int32)
    has_dist = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "e0_distractor") >= 0
    if has_dist:
        idx.dist_bodyid = np.zeros(n, dtype=np.int32)
        idx.dist_qadr = np.zeros(n, dtype=np.int32)
        idx.dist_geomid = np.zeros(n, dtype=np.int32)
    if with_camera:
        idx.wristcam_id = np.zeros(n, dtype=np.int32)

    for i in range(n):
        p = f"e{i}_"
        for j, name in enumerate(ARM_JOINT_NAMES):
            k = jid(p + name)
            idx.arm_qadr[i, j] = model.jnt_qposadr[k]
            idx.arm_dofadr[i, j] = model.jnt_dofadr[k]
            idx.arm_actid[i, j] = aid(f"{p}act_{name}")
        for j, name in enumerate(FINGER_JOINT_NAMES):
            k = jid(p + name)
            idx.finger_qadr[i, j] = model.jnt_qposadr[k]
            idx.finger_dofadr[i, j] = model.jnt_dofadr[k]
            idx.finger_actid[i, j] = aid(f"{p}act_{name}")
        idx.obj_bodyid[i] = bid(p + "obj")
        k = jid(p + "obj_free")
        idx.obj_qadr[i] = model.jnt_qposadr[k]
        idx.obj_dofadr[i] = model.jnt_dofadr[k]
        idx.obj_geomid[i] = gid(p + "obj_geom")
        idx.desk_geomid[i] = gid(p + "desk")
        idx.tcp_siteid[i] = sid(p + "tcp")
        idx.flange_bodyid[i] = bid(p + "joint6_flange")
        if has_dist:
            idx.dist_bodyid[i] = bid(p + "distractor")
            idx.dist_qadr[i] = model.jnt_qposadr[jid(p + "distractor_free")]
            idx.dist_geomid[i] = gid(p + "distractor_geom")
        if with_camera:
            idx.wristcam_id[i] = cid(p + "wristcam")
    idx.scenecam_id = cid("scenecam")
    idx.topcam_id = cid("topcam")
    return idx
