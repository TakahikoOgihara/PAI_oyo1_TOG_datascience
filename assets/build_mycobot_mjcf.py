# -*- coding: utf-8 -*-
"""myCobot 280 Pi + 平行2指グリッパ -> MJCF。運動学の正本
（``deploy/raspi/mycobot_fk.py``）から生成する。

ARM_BCRL005 ではロボットを USD で出荷していた。Elephant Robotics の公式
URDF と COLLADA メッシュを取ってきて、継ぎを当てて作るやり方。その経路が、
このプロジェクトが直すよう言われた不具合を生んだ。ビューアで、土台に近い
リンクが描画されない。原因はビューアではなく資産側で、上流の ``.dae`` が、
インポータが ``opaque`` を期待する場所に ``transparent`` と書いていて、
しかも ``joint7.dae`` が上流で公開されていないテクスチャを参照している。
だからマテリアルが何にも解決しない。ARM_BCRL005 は両端に継ぎを当て、
それでも仮テクスチャの回避策を抱え続けた。

ARM_BCRL006 は、また継ぎを当てるのではなく不具合の種類ごと消した。ロボット
全体を MJCF のプリミティブ形状（カプセル／箱／円柱）として出し、寸法は
公式の関節原点から導く。結果:

  * どのリンクも必ず描画される形状を持つ。壊れ得るマテリアルも、
    テクスチャも、透明フラグも残っていない
  * 資産は12KBのテキスト1枚で、ダウンロードの段が無い。だから
    ``run_all.sh`` のフェーズ0はネットワークに依存せず、オフラインで再現できる
  * 運動学は URDF と同一。関節原点を打ち直すのではなく
    ``mycobot_fk.py`` から取り込んでいるから。下の ``verify()`` が MuJoCo
    自身の順運動学をそのファイルと 1e-9 m まで突き合わせるので、
    シミュレータ、解析IK、仮想床、Raspberry Pi のガードが1本の連鎖を
    共有していることを証明できる

代償は写実性だが、このタスクでは一度も使っていない。手首カメラの方策は
濃い台の上の赤い円柱を手がかりにするし、メッシュ精度のアームの外殻は
ほとんど画角に入らない。意味のある忠実さ（リンク長、関節軸、関節限界、
ツール形状、質量）は厳密に保っている。

Usage
-----
    python assets/build_mycobot_mjcf.py                  # -> assets/mjcf/mycobot_280_pi.xml
    python assets/build_mycobot_mjcf.py --verify         # + FKの突き合わせ（mujoco が必要）
"""

from __future__ import annotations

import argparse
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "deploy", "raspi"))

from mycobot_fk import (  # noqa: E402  -- 正本の連鎖
    CHAIN, Q_LIMITS, HOME_Q, NUM_ARM_JOINTS,
    GRIPPER_MOUNT_Z, FINGER_JOINT_Z, FINGER_PAD_ORIGIN_Z,
    FINGER_OPEN, _rpy,
)

# ---------------------------------------------------------------------------
# リンクの質量と慣性の尺度は ARM_BCRL005 の
# assets/build_mycobot_mjcf.py と同一。だから方策が学習する相手の力学は、
# 資産の形式を変えても変わらない。
# ---------------------------------------------------------------------------
LINK_MASS = {"g_base": 0.30, "joint1": 0.18, "joint2": 0.16, "joint3": 0.13,
             "joint4": 0.10, "joint5": 0.07, "joint6": 0.05, "joint6_flange": 0.02}
LINK_SIZE = {"g_base": 0.10, "joint1": 0.09, "joint2": 0.11, "joint3": 0.10,
             "joint4": 0.09, "joint5": 0.06, "joint6": 0.05, "joint6_flange": 0.03}
FINGER_MIN = 0.004
FINGER_BOX = (0.008, 0.016, 0.030)      # パッドの 厚み x 幅 x 長さ

ARM_JOINT_NAMES = ["joint2_to_joint1", "joint3_to_joint2", "joint4_to_joint3",
                   "joint5_to_joint4", "joint6_to_joint5", "joint6output_to_joint6"]
FINGER_JOINT_NAMES = ["left_finger_joint", "right_finger_joint"]
LINK_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6_flange"]

# myCobot 280 自身の塗り分け。白い外殻に、濃い灰色の関節カバー。濃い台を
# 背景にすると、ビューアでも録画した動画でもはっきり読める。
SHELL_RGBA = "0.93 0.93 0.95 1"
JOINT_RGBA = "0.16 0.16 0.18 1"
BASE_RGBA = "0.30 0.32 0.36 1"
TOOL_RGBA = "0.20 0.22 0.26 1"
PAD_RGBA = "0.85 0.55 0.15 1"


# ---------------------------------------------------------------------------
def _quat_from_rpy(rpy) -> tuple:
    """URDF の rpy -> MuJoCo の (w, x, y, z)。

    回転行列は ``mycobot_fk._rpy``、つまり正本の実装から持ってくる。だから
    MJCF が URDF の rpy の規約（Rz(yaw) @ Ry(pitch) @ Rx(roll)）について
    FK と食い違うことはない。ローカルなのは行列 -> クォータニオンの段だけで、
    その往復は ``verify()`` が証明する。
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
    """有効桁17桁。細かすぎるのではない。ここに書く関節原点とクォータニオン
    が、MuJoCo が見る連鎖の唯一の写しで、``verify()`` がそれを
    ``mycobot_fk.py`` に対して 1e-9 m まで縛る。%.6g だと丸めだけで
    1e-7 m を食い、突き合わせが無意味になる。"""
    return " ".join(repr(float(v)) for v in vals)


def _inertial(link: str, com=(0.0, 0.0, 0.0)) -> str:
    """ARM_BCRL005 の URDF 生成器が入れていたのと同じ質量、同じ箱近似の
    慣性（I = m s^2 / 6）。ただし重心はリンクの幾何中心に置く。フレームの
    原点ではない。MuJoCo の ``inertial/pos`` は重心そのもので、重心が
    関節軸の上に乗った長いリンクは、何のモデルにもなっていない。"""
    m = LINK_MASS[link]
    i = m * LINK_SIZE[link] ** 2 / 6.0
    return (f'<inertial pos="{_fmt(com)}" mass="{m:.4g}" '
            f'diaginertia="{i:.4e} {i:.4e} {i:.4e}"/>')


def _capsule(name: str, a, b, r: float, rgba: str) -> str:
    """見た目だけの外殻（contype/conaffinity 0）。アームのリンクの衝突は
    意図的に切ってある。このタスクに必要な接触はツール<->ワーク、
    ツール<->台、ワーク<->台だけで、しかも仮想床が、アームがツール以外で
    台に届かないことを既に保証している。6自由度のホビーアームの自己衝突は
    何も買わず、ソルバの時間だけ食う。"""
    return (f'<geom name="{name}" type="capsule" fromto="{_fmt(list(a) + list(b))}" '
            f'size="{r:.4g}" rgba="{rgba}" contype="0" conaffinity="0" group="1"/>')


def _cyl(name: str, a, b, r: float, rgba: str) -> str:
    return (f'<geom name="{name}" type="cylinder" fromto="{_fmt(list(a) + list(b))}" '
            f'size="{r:.4g}" rgba="{rgba}" contype="0" conaffinity="0" group="1"/>')


# ---------------------------------------------------------------------------
def robot_xml(prefix: str, origin=(0.0, 0.0, 0.0), indent: str = "    ") -> str:
    """アームを入れ子の <body> の連鎖として作る。1環境ぶん。

    ``prefix`` が全ての名前に名前空間を付け、``origin`` が土台を置くので、
    N個の環境が1つの MuJoCo モデルに入る。Isaac Lab の
    ``/World/envs/env_i`` のクローンに直接対応し、これが1回の
    ``mj_step`` でバッチ全体を進められる理由。原点は包むボディではなく
    土台ボディの ``pos`` に焼き込む。MuJoCo は free joint（ブロック、
    邪魔物）を ``worldbody`` の直下にしか許さないので、環境を共通の親の
    下にまとめられないから。
    """
    p = prefix
    out = []

    def w(depth: int, s: str):
        out.append(indent * depth + s)

    # ---- 固定の土台。台と関節1の間の支柱 ---------------------------------
    # 公式 URDF では関節1の原点は台から 139.56 mm 上にある。その支柱は
    # ロボットの実在する部品で、「土台に近いリンクが出てこない」が指して
    # いたのは正にこれ。だから明示的に描く。
    base_top = CHAIN[0][0][2]
    w(3, f'<body name="{p}base" pos="{_fmt(origin)}">')
    w(4, _inertial("g_base", (0.0, 0.0, base_top * 0.4)))
    w(4, f'<geom name="{p}base_plate" type="box" pos="0 0 0.006" size="0.055 0.055 0.006" '
         f'rgba="{BASE_RGBA}" contype="1" conaffinity="1" group="1"/>')
    w(4, _cyl(f"{p}base_col", (0, 0, 0.012), (0, 0, base_top), 0.042, SHELL_RGBA))
    w(4, _cyl(f"{p}base_cap", (0, 0, base_top - 0.018), (0, 0, base_top), 0.045, JOINT_RGBA))

    # ---- 回転6リンク -----------------------------------------------------
    # 各リンクの外殻は、自分のフレーム原点から次の関節の原点まで描く。
    # つまり描いた線分が運動学の線分そのもの。リンクが見た目には在るのに
    # 運動学上は無い、という状態になり得ない。
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
        # 関節軸を中心に置く濃いカバー。回転軸がビューポートで読めるように
        # する。査読者が見るのはそこだから
        w(depth + 1, _cyl(f"{p}{link}_hub", (0, 0, -r * 0.62), (0, 0, r * 0.62), r * 0.96, JOINT_RGBA))
        if seg_len > 1e-4:
            w(depth + 1, _capsule(f"{p}{link}_arm", (0, 0, 0), seg, r * 0.74, SHELL_RGBA))
        depth += 1

    # ---- ツール。グリッパ土台、滑る2枚のパッド、TCP のサイト ------------
    gb = GRIPPER_MOUNT_Z
    w(depth, f'<body name="{p}gripper_base" pos="0 0 {gb:.6g}">')
    w(depth + 1, '<inertial pos="0 0 0.018" mass="0.06" '
                 'diaginertia="1.60e-05 1.60e-05 1.60e-05"/>')
    w(depth + 1, f'<geom name="{p}gripper_body" type="box" pos="0 0 0.018" '
                 f'size="0.028 0.013 0.020" rgba="{TOOL_RGBA}" '
                 f'contype="1" conaffinity="1" condim="4" friction="0.9 0.02 0.001" group="1"/>')
    # 手首カメラ。フランジ座標の +Z がツール軸。MuJoCo のカメラは -Z 方向を
    # 見て +Y が上で、一方 configs/env は ROS 規約（+Z が光軸）で持っている。
    # だから mujoco_sim/scene.py が、この要素を書き出すときに保存された
    # クォータニオンを X 軸まわり180度回す。
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
        # パッドの箱の原点は厚みの半分ずつずらす。URDF と同じ。すると外側の
        # 面が joint_value + 0.008 に来て、それが mycobot_fk の TOOL_POINTS が
        # 見る x になる。
        w(depth + 2, f'<geom name="{p}{side}_pad" type="box" '
                     f'pos="{sign * FINGER_BOX[0] / 2:.6g} 0 {FINGER_PAD_ORIGIN_Z:.6g}" '
                     f'size="{FINGER_BOX[0] / 2:.6g} {FINGER_BOX[1] / 2:.6g} '
                     f'{FINGER_BOX[2] / 2:.6g}" rgba="{PAD_RGBA}" '
                     f'contype="1" conaffinity="1" condim="4" '
                     f'friction="1.6 0.05 0.002" solimp="0.95 0.99 0.001" group="1"/>')
        w(depth + 1, "</body>")
    # TCP = 把持中心。ボディではなく <site> にする。質量を足してはならず、
    # 環境は mj_data.site_xpos で位置を読む。
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
    """位置サーボのアクチュエータ == Isaac Lab の ImplicitActuatorCfg
    (stiffness, damping)。MuJoCo の ``position`` は ``kp*(target-q) - kv*qd``
    を与える同じPD則で、``forcerange`` が出力制限の役を果たす。"""
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
    """ロボット1体だけの独立した MJCF。``--verify``、単体テスト、そして
    シーン抜きで ``python -m mujoco.viewer`` にアームを出したい人のために
    使う。"""
    xml = SINGLE_ROBOT_TEMPLATE.format(
        robot=robot_xml("").replace("<!--WRISTCAM-->", ""),
        actuators=actuator_xml("", 50.0, 5.0, 5.0, 300.0, 20.0, 30.0))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)
    return path


# ---------------------------------------------------------------------------
def verify(path: str, tol: float = 1e-9) -> None:
    """MuJoCo の順運動学が ``deploy/raspi/mycobot_fk.py`` と等しいことを確認する。

    安全の話を、主張ではなく事実にしている検査がこれ。仮想床、解析IKの
    ヤコビアン、MimicGen の姿勢転写、Raspberry Pi のガードは全部
    ``mycobot_fk.py`` から計算され、物理は MuJoCo が計算する。2つの連鎖が
    食い違えば、方策が学習しているアームは、ガードが守っているアームでは
    ない。
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

    # 床ガードが使うツールのサンプル点も、MJCF から到達できないといけない。
    # home 姿勢で最も低い点を確認する
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
