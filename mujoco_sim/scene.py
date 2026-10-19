# -*- coding: utf-8 -*-
"""シーンの組み立て。1つの MuJoCo モデルの中に N 個の環境を作る。

ARM_BCRL005 が頼っていた Isaac Lab の ``InteractiveScene`` のクローンに
直接対応するもの。各環境は ``env_origins[i]`` の格子上に置いた
``<body>`` で、自分のロボット、台の板、ブロック、邪魔物を持ち、要素名は
すべて ``e{i}_`` を前置する。``mj_step`` 1回でバッチ全体が進むので、
``env.py`` のベクトル化された MDP のコードは Isaac Lab 版とそのまま同じ
読み方になる。観測は環境ローカル座標、つまり ``world - env_origin`` で
計算する。31次元の観測、ONNX グラフ、``deploy/raspi/`` が前提にしている
契約と同じ。

N個の ``MjData`` ではなく1つのモデルにする理由。環境は力学的に独立
なので MuJoCo の分解はブロック対角で、コストはどちらでも線形。ただし
1モデルならステップ呼び出しが1回、カメラIDの集合も1つで済み、環境
ごとのドメインランダム化は ``model.body_mass`` / ``model.geom_friction``
/ ``model.actuator_gainprm`` への添字付き書き込みだけになる。それが
``DomainRandomizer`` を読めるものに保っている。

ロボット自体は ``assets/build_mycobot_mjcf.py`` が
``deploy/raspi/mycobot_fk.py`` から導く。連鎖をここで打ち直すことはない。
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
    """環境とランダマイザが必要とするIDを、構築時に一度だけ解決しておく。

    名前からIDを先に引いておく（ステップの中で ``mj_name2id`` を呼ばない）
    ことが、20Hzの制御ループから文字列処理を追い出している。要素名を
    変えたときに、40分走ってから静かに別のボディを触るのではなく、
    構築時に失敗するようにもなる。
    """

    num_envs: int
    env_origins: np.ndarray                     # (N, 3)
    arm_qadr: np.ndarray = field(default=None)  # (N, 6) の qpos アドレス
    arm_dofadr: np.ndarray = field(default=None)
    finger_qadr: np.ndarray = field(default=None)   # (N, 2)
    finger_dofadr: np.ndarray = field(default=None)
    arm_actid: np.ndarray = field(default=None)     # (N, 6)
    finger_actid: np.ndarray = field(default=None)  # (N, 2)
    obj_bodyid: np.ndarray = field(default=None)    # (N,)
    obj_qadr: np.ndarray = field(default=None)      # (N,) の free joint の qpos アドレス
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
    """ほぼ正方の格子。Isaac Lab が ``env_origins`` に使う配置規則と同じ。"""
    cols = int(math.ceil(math.sqrt(n)))
    out = np.zeros((n, 3))
    for i in range(n):
        out[i, 0] = (i % cols) * spacing
        out[i, 1] = (i // cols) * spacing
    out[:, :2] -= out[:, :2].mean(axis=0)
    return out


def _wristcam_xml(prefix: str, cam_cfg) -> str:
    """gripper_base ボディの中に付ける手首カメラ。

    ``configs/env`` は取り付けの回転を ROS 規約（+Z が光軸）で持っている。
    実カメラの外部パラメータもその規約で表記され、ARM_BCRL005 が Isaac Lab
    に渡していたのもそれ。MuJoCo のカメラは -Z 方向を見て +Y が上なので、
    保存されたクォータニオンに X 軸まわり180度の回転を後ろから掛ける。
    カメラを作る唯一の場所であるここで変換しておけば、
    ``configs/env/mycobot_pickplace.yaml`` はシミュレータ内部の規約ではなく
    物理的なカメラを書いたままでいられる。
    """
    w, x, y, z = (float(v) for v in cam_cfg.rot_wxyz)
    # q * (0, 1, 0, 0)
    q = (-x, w, z, -y)
    n = math.sqrt(sum(v * v for v in q))
    q = tuple(v / n for v in q)
    px, py, pz = (float(v) for v in cam_cfg.pos)
    # 正方センサなので縦画角 == 横画角。MuJoCo は fovy を度で取る
    fovy = float(cam_cfg.hfov_deg) * float(cam_cfg.height) / float(cam_cfg.width)
    return (f'<camera name="{prefix}wristcam" pos="{px!r} {py!r} {pz!r}" '
            f'quat="{q[0]!r} {q[1]!r} {q[2]!r} {q[3]!r}" fovy="{fovy:.6g}"/>')


def build_scene_xml(cfg, num_envs: int, with_camera: bool,
                    env_spacing: float = 1.4) -> tuple[str, np.ndarray]:
    """-> (MJCF の文字列, env_origins (N,3))。"""
    e = cfg.env
    origins = _env_grid(num_envs, env_spacing)
    desk_sx, desk_sy, desk_sz = (float(v) for v in e.desk.size)
    # ARM_BCRL005 の板は 1.2m 四方。Isaac Lab が環境を2m間隔で置いていた
    # から。ここではバッチを詰めるために間隔を狭めているので、板が重な
    # らないよう、板は間隔に合わせて切る。
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
        # ブロック。赤く塗った木の円柱（assets/block_reference.jpg）
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

    # Isaac Lab はドーム照明を lux で駆動していたが、MuJoCo の照明は
    # 無次元の拡散色。env.lighting.intensity は、フェーズ2の画質ゲートが
    # 反応する露出のつまみという役割を保ったまま、ARM_BCRL005 のシーンを
    # 検証した900 lux を中心に [0,1] へ写す。
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
