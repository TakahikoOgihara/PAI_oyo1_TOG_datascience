# -*- coding: utf-8 -*-
"""ARM_BCRL006 phase 5 (optional) -- export the demonstrations to the
LeRobot dataset format.

LeRobot (HuggingFace) is the emerging industry-standard interchange format
for robot-learning demonstrations (ALOHA/SO-10x ecosystems, ACT / Diffusion
Policy trainers all consume it). This exporter converts the SAME processed
demos the robomimic pipeline trains on into a LeRobotDataset, so the project
demonstrates fluency in BOTH of the two formats a physical-AI consultant
actually encounters: robomimic HDF5 (NVIDIA/Isaac ecosystem) and LeRobot
(HuggingFace ecosystem).

Honest scoping note: a PUBLIC LeRobot dataset records someone else's robot
and scene, so it cannot be closed-loop evaluated in this MuJoCo
environment (embodiment mismatch). The portable, verifiable direction is the
one implemented here: OUR sim demos -> LeRobot format (usable by any LeRobot
trainer, shareable on the Hub).

This phase is OPTIONAL and self-skipping: lerobot is not part of the
isaac-lab-train container's verified base install; ``pip install lerobot``
enables it. run_all.sh treats a failure here as a warning, never a pipeline
stop. The LeRobot dataset API is still evolving, so the create/add_frame
calls are defensively wrapped with clear version-mismatch messages.

Usage:  python lerobot_export/export_lerobot.py [--fps 20] [--repo-id arm_bcrl006/pickplace]
"""

from __future__ import annotations
import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK_TEXT = "Pick up the red cylinder block, move it 5 cm and put it down (myCobot 280 Pi)."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=20)  # matches env control_dt=0.05
    ap.add_argument("--repo-id", type=str, default="arm_bcrl006/pickplace")
    ap.add_argument("--demos", type=str,
                    default=os.path.join(REPO_ROOT, "artifacts", "data", "bc_processed_demos.pt"))
    ap.add_argument("--out", type=str,
                    default=os.path.join(REPO_ROOT, "artifacts", "lerobot"))
    args = ap.parse_args()

    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as e:
        print(f"[lerobot_export] SKIP: lerobot is not installed ({e}). "
              "Enable this phase with: pip install lerobot")
        return 0

    ds = torch.load(args.demos, weights_only=False)
    obs_list, act_list = ds["obs"], ds["act"]
    rgb_list = ds.get("rgb", None)
    flat_dim = int(obs_list[0].shape[-1])
    act_dim = int(act_list[0].shape[-1])
    # obs_list[0]は、0番目のエピソードの観測時系列です。
    # shape[-1]は、最後の軸のサイズを取得します。

    features = {
        "observation.state": {"dtype": "float32", "shape": (flat_dim,), "names": None},
        "action": {"dtype": "float32", "shape": (act_dim,), "names": None},
    }
    if rgb_list is not None:
        h, w, c = (int(v) for v in rgb_list[0].shape[1:])
        # shapeは、テンソルやNumPy配列の「各軸のサイズ」を表します。
        # rgb_list[0]は0番目のエピソードの画像時系列全体です。
        # rgb_list[0].shape[1:]は、shapeの1番目以降を全部取り出すという意味なので、h, w, c 。
        features["observation.images.front"] = {"dtype": "video", "shape": (h, w, c), "names": None}

    root = os.path.join(args.out, args.repo_id.replace("/", "_"))
    if os.path.exists(root):
        shutil.rmtree(root)  # deterministic re-export (artifacts dir only)
        # rootで指定されたディレクトリを、中にあるファイルやサブディレクトリごと全部削除する処理です。
        # shutilはPython標準ライブラリで、ファイルやディレクトリをまとめて操作します。
    os.makedirs(os.path.dirname(root), exist_ok=True)

    try:
        dataset = LeRobotDataset.create(repo_id=args.repo_id, fps=args.fps,
                                        features=features, root=root)
        # 「これからLeRobot形式のデータセットを書き込むための空のデータセット領域を、仕様付きで新規作成する」処理
    except TypeError as e:
        print(f"[lerobot_export] SKIP: LeRobotDataset.create signature mismatch ({e}); "
              "installed lerobot version differs from the v2.x API this exporter targets")
        return 0

    n_frames = 0
    for i, (o, a) in enumerate(zip(obs_list, act_list)):
        T = o.shape[0]
        for t in range(T):
            frame = {
                "observation.state": o[t].numpy().astype(np.float32),
                "action": a[t].numpy().astype(np.float32),
            }
            if rgb_list is not None:
                frame["observation.images.front"] = rgb_list[i][t].numpy()
            try:
                dataset.add_frame(frame, task=TASK_TEXT)
                # 作成した1フレームを、現在構築中のエピソードに追加します。
            except TypeError:
                # older API: task passed inside the frame dict
                frame["task"] = TASK_TEXT
                dataset.add_frame(frame)
            n_frames += 1
        dataset.save_episode()
        # add_frame()で追加していたフレーム群を、1つのエピソードとして確定・保存
        if (i + 1) % 25 == 0:
            print(f"[lerobot_export] {i + 1}/{len(obs_list)} episodes")

    summary = {"episodes": len(obs_list), "frames": n_frames, "root": root,
              "repo_id": args.repo_id, "fps": args.fps,
              "has_video": rgb_list is not None}
    with open(os.path.join(args.out, "lerobot_export_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("[lerobot_export]", json.dumps(summary, indent=2))
    print("[lerobot_export] load it back with: "
          f"LeRobotDataset('{args.repo_id}', root='{root}')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
