# -*- coding: utf-8 -*-
"""Camera spec check -- run this on the Jetson BEFORE any sim data collection
(operations guide §2-1): the sim camera must match the REAL camera's
resolution/aspect and field of view, or the policy trains on a geometry it
never sees on the robot.

1) enumerate + capture:
     python3 check_camera.py                 # lists modes, saves probe.jpg
2) measure the horizontal FOV with a ruler:
     put a ruler flat, facing the camera, at a measured distance D (m);
     read the visible width W (m) at the image's horizontal edges, then
     python3 check_camera.py --distance 0.30 --width 0.35
     -> prints hfov_deg to copy into configs/env/mycobot_pickplace.yaml

The runtime center-crops the real frame to a square and resizes to the
policy input (edge_config.json), so what must match is the VERTICAL fov of
the real camera vs the sim's square fov -- the printed "square-crop hfov"
already accounts for that (crop keeps the full image height on a landscape
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
        # the runtime's square center-crop keeps the full height of a
        # landscape sensor: the square fov equals the VERTICAL fov
        vis_w = args.width * min(h, w) / w
        sq = math.degrees(2 * math.atan(vis_w / 2 / args.distance))
        print(f"full-frame hfov      = {hfov:.1f} deg")
        print(f"square-crop hfov     = {sq:.1f} deg   <-- set env.camera.hfov_deg to this")
        print("then set env.camera.width/height to the policy input size (default 224).")
    else:
        print("\nFOV measurement: rerun with --distance <m> --width <m> (see docstring)")


if __name__ == "__main__":
    main()
