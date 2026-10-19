# -*- coding: utf-8 -*-
"""ARM_BCRL006 フェーズ5（任意）。デモを LeRobot のデータセット形式へ
書き出す。

LeRobot（HuggingFace）は、ロボット学習のデモ交換形式として事実上の
標準になりつつある（ALOHA/SO-10x 系、ACT や Diffusion Policy の学習器が
どれもこれを食う）。この書き出しは、robomimic のパイプラインが学習に
使うのと同じ処理済みデモを LeRobotDataset に変換する。だから、物理AIの
仕事で実際に出会う2つの形式、robomimic HDF5（NVIDIA / Isaac 系）と
LeRobot（HuggingFace 系）の両方を扱えることを示せる。
(HuggingFace ecosystem).

範囲についての正直な注記。公開されている LeRobot のデータセットは
他人のロボットとシーンの記録なので、この MuJoCo 環境では閉ループ評価
できない（身体が違う）。移植できて検証もできる向きは、ここで実装した
方向、つまり自分のシミュレータのデモ -> LeRobot 形式（どの LeRobot の
学習器でも使えて、Hub でも共有できる）。

このフェーズは任意で、自分でスキップする。lerobot は環境構築の検証済み
の土台には入っていないので、``pip install lerobot`` で有効になる。
run_all.sh はここでの失敗を警告として扱い、パイプラインは止めない。
LeRobot のデータセットAPIはまだ動いているので、create/add_frame の
呼び出しは、版違いが分かるメッセージ付きで防御的に包んである。

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
    ap.add_argument("--fps", type=int, default=20)  # 環境の control_dt=0.05 に合わせる
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
        shutil.rmtree(root)  # 決定的に書き出し直す（artifacts ディレクトリのみ）
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
                # 古いAPI。task はフレームの辞書の中で渡す
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
