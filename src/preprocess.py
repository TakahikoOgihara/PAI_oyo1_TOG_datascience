# -*- coding: utf-8 -*-
"""ARM_BCRL006 フェーズ2。前処理、データセット分析、robomimic HDF5 化。

衛生面は ARM_BCRL001 と同じ。平坦チャンネルの行ごとの外れ値クリップ
（全体のzスコア）、同じ行マスクを全モダリティに適用（flat / rgb /
actions の時刻を揃える）、分析用の成果物、そして robomimic の HDF5
スキーマへの変換（bc_hdf5/to_robomimic_hdf5.py）。
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
    """手首カメラの画像に、本当にタスクが写っているかを測る。

    ARM_BCRL002 は camera_samples.png を人が一瞥する運用に頼っていた。その
    確認を通った画像はほぼ一様に白い。反射率0.78の台を明るい照明で照らすとセンサが飽和する。結果として配備した
    視覚運動方策は、ワークの位置の情報がほとんど無い画像で学習されていた。
    そして実機では、見えていない方策がするとおりに振る舞った。平均的な
    「降りて閉じる」動作をなぞって、机に突っ込んだ（実機の失敗記録に
    原因として残してある）。

    安い統計4つで、その一瞥をゲートに変える:
      mean_pixel        全体の露出
      saturated_frac    250以上に張り付いた画素の割合（階調の飛び）
      dynamic_range     輝度ヒストグラムの p95 - p5（コントラスト）
      red_frame_frac    赤い円柱らしい画素の塊が、そもそも検出できた
                        フレームの割合
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
    # 「赤っぽく、灰色ではない」の判定。ワークの色は (0.75, 0.08, 0.08)
    red_mask = (r > g + 30) & (r > b + 30) & (r > 60)
    red_px = red_mask.reshape(red_mask.shape[0], -1).mean(axis=1)
    report = {
        "frames_checked": int(x.shape[0]),
        "mean_pixel": float(x.mean()),
        "saturated_frac": float((x >= 250).mean()),
        "dynamic_range": float(_np.percentile(lum, 95) - _np.percentile(lum, 5)),
        "red_frame_frac": float((red_px > 0.002).mean()),   # 画素の0.2%超
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

    # ARM_BCRL006 では、フェーズ1.5（MimicGen）が増補データセットを書く。
    # それがあればそちらを優先するので、フェーズ2から10は他に何も変えなくて
    # よい。MimicGen が無効、または未実行ならフェーズ1の出力に戻る。
    # 非破壊なのは意図的で、bc_demos.pt は常に手を入れない元データとして
    # 残る。だからフェーズ1.5は何度でも回せるし、MimicGen も毎回そこを読む。
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

    # 分析は査読に関係するスカラーだけ出す。生のヒストグラム31本は出さない
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

    # ---- 手首カメラの画像 ------------------------------------------------
    # このゲートが見るのはカメラの設定なので、読むのはフェーズ1の画像で、
    # 学習に使うデータセットの方ではない。これが効くのは、フェーズ1.5が
    # カメラOFFで出荷されているから（mimicgen.camera）。生成デモは状態方策
    # 用で、誰も読まない3.5GBの画素を描けば、パイプライン最大の単一コストに
    # なってしまう。
    cam_source = os.path.join(data_dir, "bc_demos.pt")
    cam_rgb = filt_rgb if (rgb_list is not None and len(filt_rgb) > 0) else None
    if cam_rgb is None and os.path.exists(cam_source):
        # mmap で読む。フェーズ1のアーカイブは数GBのuint8画像で、ゲートが
        # 見るのはそのうち数百枚だけ。1%を見るために全部をRAMに載せるのが、
        # 「サーバーでは快調だったパイプラインがノートPCでスワップし始める」
        # 原因。拒否する torch なら普通の load に落ちる。
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

    # ---- 手首カメラの画質ゲート（ARM_BCRL006） --------------------------
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
