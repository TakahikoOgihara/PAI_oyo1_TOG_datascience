# -*- coding: utf-8 -*-
"""ARM_BCRL006 phase 2 -- preprocessing, dataset analysis & robomimic HDF5.

Same hygiene as ARM_BCRL001: per-row outlier clipping on the flat channels
(global z-score), the SAME row mask applied to every modality (flat / rgb /
actions stay timestep-aligned), analysis artifacts, then conversion to the
robomimic HDF5 schema (bc_hdf5/to_robomimic_hdf5.py).
"""

from __future__ import annotations
import json
import os
import sys

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


from mlops import plotting
from bc_hdf5.to_robomimic_hdf5 import convert

# flat = [q(8), eef_pos(3), eef_quat(4), obj_pos(3), obj_minus_eef(3),
#         place_target(3), prev_action(7)]
FLAT_DIM = 31


def image_quality_report(rgb_list, cfg_iq, max_frames: int = 2000) -> dict:
    """Measure whether the wrist-camera frames actually CONTAIN the task.

    ARM_BCRL002 relied on a human glancing at camera_samples.png. The frames
    that passed that review are almost uniformly white -- an 0.78-albedo bench under a bright light saturates the sensor -- so the deployed
    visuomotor policy was trained on images with next to no information about
    where the block is. It then behaved, on the real robot, exactly like a
    policy that cannot see: it ran the average descend-and-close motion and
    drove into the desk (field-failure notes §2, cause B).

    Four cheap statistics turn that review into a gate:
      mean_pixel        overall exposure
      saturated_frac    fraction of pixels pinned at >= 250 (clipped detail)
      dynamic_range     p95 - p5 of the luminance histogram (contrast)
      red_frame_frac    fraction of frames where a red-cylinder-like pixel
                        cluster is detectable at all
    """
    import numpy as _np
    frames = []
    for traj in rgb_list:
        n = traj.shape[0]
        step = max(1, n // 8)
        frames.append(traj[::step].numpy())
        if sum(f.shape[0] for f in frames) >= max_frames:
            break
    x = _np.concatenate(frames, axis=0)[:max_frames].astype(_np.float32)
    lum = x @ _np.array([0.299, 0.587, 0.114], dtype=_np.float32)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    # "reddish and clearly not grey": the block is (0.75, 0.08, 0.08)
    red_mask = (r > g + 30) & (r > b + 30) & (r > 60)
    red_px = red_mask.reshape(red_mask.shape[0], -1).mean(axis=1)
    report = {
        "frames_checked": int(x.shape[0]),
        "mean_pixel": float(x.mean()),
        "saturated_frac": float((x >= 250).mean()),
        "dynamic_range": float(_np.percentile(lum, 95) - _np.percentile(lum, 5)),
        "red_frame_frac": float((red_px > 0.002).mean()),   # >0.2 % of pixels
    }
    fails = []
    if report["mean_pixel"] > float(cfg_iq.max_mean):
        fails.append(f"mean_pixel {report['mean_pixel']:.0f} > {float(cfg_iq.max_mean):.0f} "
                     "(OVEREXPOSED: lower env.lighting.intensity / "
                     "env.domain_rand.light_intensity, or darken env.desk.color)")
    if report["mean_pixel"] < float(cfg_iq.min_mean):
        fails.append(f"mean_pixel {report['mean_pixel']:.0f} < {float(cfg_iq.min_mean):.0f} "
                     "(UNDEREXPOSED: raise env.lighting.intensity)")
    if report["saturated_frac"] > float(cfg_iq.max_saturated):
        fails.append(f"saturated_frac {report['saturated_frac']:.3f} > "
                     f"{float(cfg_iq.max_saturated):.3f} (clipped highlights: "
                     "same fix as overexposure)")
    if report["dynamic_range"] < float(cfg_iq.min_dynamic_range):
        fails.append(f"dynamic_range {report['dynamic_range']:.0f} < "
                     f"{float(cfg_iq.min_dynamic_range):.0f} (flat image: check the "
                     "light AND that the desk/block colours differ)")
    if report["red_frame_frac"] < float(cfg_iq.min_red_frames):
        fails.append(f"red_frame_frac {report['red_frame_frac']:.2f} < "
                     f"{float(cfg_iq.min_red_frames):.2f} (the RED BLOCK IS NOT "
                     "VISIBLE in most frames: check env.camera.pos/rot_wxyz and "
                     "hfov_deg against artifacts/data/camera_samples.png)")
    report["passed"] = not fails
    report["failures"] = fails
    return report


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    data_dir = os.path.join(REPO_ROOT, "artifacts", "data")
    torch.manual_seed(cfg.seed)

    # ARM_BCRL006: phase 1.5 (MimicGen) writes an AUGMENTED dataset. Prefer it
    # when it exists so phases 2-10 need no other change; fall back to phase 1's
    # output when MimicGen is disabled or has not run. Non-destructive on
    # purpose -- bc_demos.pt always remains the untouched source, which is what
    # makes phase 1.5 re-runnable and what MimicGen re-reads as its source.
    demo_path = os.path.join(data_dir, "bc_demos.pt")
    mg_path = os.path.join(data_dir, str(cfg.mimicgen.output_file))
    if bool(cfg.mimicgen.enable) and os.path.exists(mg_path):
        demo_path = mg_path
        print(f"[preprocess] using the MimicGen-augmented dataset: {mg_path}")
    else:
        print(f"[preprocess] using the phase-1 dataset: {demo_path}")
    ds = torch.load(demo_path, weights_only=False)
    obs_list, act_list = ds["obs"], ds["act"]
    rgb_list = ds.get("rgb", None)
    n0 = sum(o.shape[0] for o in obs_list)

    all_obs = torch.cat(obs_list).float()
    mean, std = all_obs.mean(0), all_obs.std(0).clamp_min(1e-6)

    filt_obs, filt_act, filt_rgb = [], [], []
    for k, (o, a) in enumerate(zip(obs_list, act_list)):
        z = (o.float() - mean).abs() / std
        keep = (z <= cfg.bc.outlier_z).all(dim=1)
        if keep.sum() > 0:
            filt_obs.append(o[keep])
            filt_act.append(a[keep])
            if rgb_list is not None:
                filt_rgb.append(rgb_list[k][keep])
    n1 = sum(o.shape[0] for o in filt_obs)

    processed_path = os.path.join(data_dir, "bc_processed_demos.pt")
    processed = {"obs": filt_obs, "act": filt_act}
    if rgb_list is not None:
        processed["rgb"] = filt_rgb
    torch.save(processed, processed_path)

    summary = {
        "demo_source": os.path.basename(demo_path),
        "raw_transitions": int(n0),
        "after_outlier_clip": int(n1),
        "removed_outliers": int(n0 - n1),
        "trajectories": len(filt_obs),
    }
    with open(os.path.join(data_dir, "preprocess_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("[preprocess]", json.dumps(summary, indent=2))

    # analysis: the review-relevant scalars, not 31 raw histograms
    flat = torch.cat(filt_obs).numpy()
    act = torch.cat(filt_act).numpy()
    eef = flat[:, 8:11]
    obj = flat[:, 15:18]
    target = flat[:, 21:24]
    arrays = {
        "eef_obj_dist": np.linalg.norm(obj - eef, axis=1),
        "obj_z": obj[:, 2],
        "obj_target_dist_xy": np.linalg.norm(obj[:, :2] - target[:, :2], axis=1),
        "eef_z": eef[:, 2],
        "gripper_action": act[:, 6],
        "arm_action_norm": np.linalg.norm(act[:, :6], axis=1),
    }
    for j in range(6):
        arrays[f"act:joint{j + 1}"] = act[:, j]
    plotting.hist_grid(arrays, os.path.join(data_dir, "dataset_analysis.png"),
                       title="ARM_BCRL006 demonstration distribution (MuJoCo)")

    # ---- wrist-camera frames -------------------------------------------
    # The gate is about the CAMERA's configuration, so it reads phase 1's
    # frames -- not whatever dataset happens to be used for training. That
    # matters because phase 1.5 ships with the camera off (mimicgen.camera):
    # its generated demos feed the state policy, and rendering ~3.5 GB of
    # pixels nothing reads would be the largest single cost in the pipeline.
    cam_source = os.path.join(data_dir, "bc_demos.pt")
    cam_rgb = filt_rgb if (rgb_list is not None and len(filt_rgb) > 0) else None
    if cam_rgb is None and os.path.exists(cam_source):
        # mmap: phase 1's archive is a few GB of uint8 frames and the gate only
        # samples a few hundred of them. Reading it into RAM to look at 1 % of
        # it is how a pipeline that ran fine on a server starts swapping on a
        # laptop. Falls back to a normal load on any torch that refuses.
        try:
            cam_ds = torch.load(cam_source, weights_only=False, mmap=True,
                                map_location="cpu")
        except (TypeError, RuntimeError):
            cam_ds = torch.load(cam_source, weights_only=False)
        cam_rgb = cam_ds.get("rgb", None)
        if cam_rgb:
            print(f"[preprocess] image gate reads the phase-1 frames "
                  f"({len(cam_rgb)} demos from {os.path.basename(cam_source)})")

    if cam_rgb:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        k = min(16, len(cam_rgb))
        fig, axes = plt.subplots(4, 4, figsize=(8, 8), squeeze=False)
        for j in range(16):
            ax = axes[j // 4][j % 4]
            ax.axis("off")
            if j < k:
                ax.imshow(cam_rgb[j][0].numpy())
                ax.set_title(f"demo {j} (T={cam_rgb[j].shape[0]})", fontsize=8)
        fig.suptitle("ARM_BCRL006 wrist-camera frames (first frame of each demo)\n"
                     "check view + FOV against the real camera (camera spec check)")
        fig.tight_layout()
        fig.savefig(os.path.join(data_dir, "camera_samples.png"), dpi=120)
        plt.close(fig)

    # ---- wrist-camera data-quality GATE (ARM_BCRL006) --------------------
    iq = cfg.bc.get("image_quality", None)
    if cam_rgb and iq is not None and bool(iq.enable):
        report = image_quality_report(cam_rgb, iq)
        summary["image_quality"] = report
        with open(os.path.join(data_dir, "preprocess_summary.json"), "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print("[preprocess] image quality:", json.dumps(report, indent=2))
        if not report["passed"]:
            print("\n[preprocess] ERROR: the wrist-camera dataset FAILED the "
                  "quality gate. Training on it wastes GPU hours and ships a "
                  "policy that cannot see:", file=sys.stderr)
            for f_ in report["failures"]:
                print(f"  - {f_}", file=sys.stderr)
            print("  Inspect artifacts/data/camera_samples.png, fix the scene, "
                  "re-run phase 1 (collect). To override deliberately: "
                  "bc.image_quality.enable=false", file=sys.stderr)
            raise SystemExit(1)
        print("[preprocess] image-quality gate PASSED")
    elif iq is not None and bool(iq.enable):
        print("[preprocess] WARNING: no wrist-camera frames anywhere (phase 1 ran "
              "with collect.camera=false) -- the image-quality gate did NOT run. "
              "That is fine for a state-policy study; it is NOT fine before "
              "training a visuomotor policy.")

    hdf5_path = os.path.join(data_dir, "arm_bcrl006_demos.hdf5")
    convert(processed_path, hdf5_path, val_split=float(cfg.bc.val_split), seed=int(cfg.seed))
    print("[preprocess] wrote", data_dir)


if __name__ == "__main__":
    main()
