# -*- coding: utf-8 -*-
"""カメラ仕様の確認。シミュレータでデータを集める前に Jetson 上で走らせる。
シミュレータのカメラは、実カメラの解像度・縦横比・画角と合っていないと
いけない。合っていなければ、方策は実機で二度と出会わない幾何で学習する
ことになる。

1) 列挙して撮る:
     python3 check_camera.py                 # モードを列挙し、probe.jpg を保存
2) 定規で水平画角を測る:
     定規をカメラに正面から向けて置き、距離 D (m) を測る。
     画像の左右端に写る幅 W (m) を読み、
     python3 check_camera.py --distance 0.30 --width 0.35
     -> configs/env/mycobot_pickplace.yaml に貼る hfov_deg を表示する

ランタイムは実画像を正方形に中央切り出しして方策の入力へ縮小する
（edge_config.json）。だから合わせるべきは、実カメラの垂直画角と
シミュレータの正方画角。表示される「square-crop hfov」はそれを
織り込んである（横長センサなら、切り出しても高さは全部残る）。
sensor).
"""

from __future__ import annotations
import argparse
import math
import shutil
import subprocess


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--distance", type=float, default=None, help="ruler distance from lens (m)")
    ap.add_argument("--width", type=float, default=None, help="visible width at that distance (m)")
    args = ap.parse_args()

    if shutil.which("v4l2-ctl"):
        print("=== v4l2 devices ===")
        subprocess.run(["v4l2-ctl", "--list-devices"], check=False)
        print(f"=== formats of /dev/video{args.camera} ===")
        subprocess.run(["v4l2-ctl", "-d", f"/dev/video{args.camera}",
                        "--list-formats-ext"], check=False)
    else:
        print("(v4l2-ctl not found -- `sudo apt install v4l-utils` for the mode list)")

    import cv2
    cap = cv2.VideoCapture(args.camera)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"could not read from camera {args.camera}")
    h, w = frame.shape[:2]
    print(f"\ncaptured {w}x{h} -> probe.jpg")
    cv2.imwrite("probe.jpg", frame)

    if args.distance and args.width:
        hfov = math.degrees(2 * math.atan(args.width / 2 / args.distance))
        # ランタイムの正方の中央切り出しは、横長センサの高さを全部残す。
        # つまり正方の画角は垂直画角に等しい
        vis_w = args.width * min(h, w) / w
        sq = math.degrees(2 * math.atan(vis_w / 2 / args.distance))
        print(f"full-frame hfov      = {hfov:.1f} deg")
        print(f"square-crop hfov     = {sq:.1f} deg   <-- set env.camera.hfov_deg to this")
        print("then set env.camera.width/height to the policy input size (default 224).")
    else:
        print("\nFOV measurement: rerun with --distance <m> --width <m> (see docstring)")


if __name__ == "__main__":
    main()
