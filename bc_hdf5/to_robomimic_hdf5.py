# -*- coding: utf-8 -*-
"""ARM_BCRL006 が集めた軌跡ごとの {obs, act, rgb} テンソルを、robomimic の
標準データセット形式（``robomimic.utils.dataset.SequenceDataset`` と
``robomimic.scripts.train.train`` が期待する形）に変換する。仕組みは
ARM_BCRL001 と同一で、変わったのは環境のメタデータだけ:

  /data                               (属性: env_args = JSON の環境メタデータ)
  /data/demo_0/obs/flat               [T_0, 31] float32
  /data/demo_0/obs/rgb                [T_0, H, W, 3] uint8（画像を集めた場合のみ）
  /data/demo_0/actions                [T_0, 7] float32
  /data/demo_0/dones                  [T_0]    int（最終ステップで1）
  /data/demo_0/rewards                [T_0]    float32（全0。BC に RL の目的は無い）
  /data/demo_0.attrs['num_samples']   T_0
  /mask/train, /mask/valid            デモキーの文字列配列（学習/検証の分割）

画像は robomimic 自身のデータセット処理と同じく uint8 の HWC で保存する。
軌跡は可変長（成功で早く終わる）なので、デモの境界は入力のリストから
直接取る。固定長で切ることは決してしない。
reshape.
"""

from __future__ import annotations
import json
import os

import h5py
import numpy as np
import torch


def convert(demos_path: str, out_path: str, val_split: float, seed: int = 42) -> dict:
    """``demos_path`` が指すのは {"obs": [T_i,F]..., "act": [T_i,A]...、
    任意で "rgb": [T_i,H,W,3] uint8...} という、軌跡ごとのテンソルの
    リストの辞書。mujoco_sim/train_arm_mujoco.py --mode collect が作り、
    src/preprocess.py が絞り込んだもの。"""
    # torch 2.6 以降は weights_only=True が既定で、テンソルのリストを
    # 持つ辞書を拒否する。このアーカイブは同じパイプラインの1つ前の段で、
    # このリポジトリ自身が作っている。
    ds = torch.load(demos_path, weights_only=False)
    obs_list, act_list = ds["obs"], ds["act"]
    rgb_list = ds.get("rgb", None)
    n_demos = len(obs_list)
    if rgb_list is not None and len(rgb_list) != n_demos:
        raise ValueError(f"rgb list length {len(rgb_list)} != obs list length {n_demos} -- "
                         "preprocess must filter all modalities together")

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_demos)
    n_val = max(1, int(n_demos * val_split))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    obs_dim = obs_list[0].shape[-1]
    act_dim = act_list[0].shape[-1]
    rgb_shape = list(rgb_list[0].shape[1:]) if rgb_list is not None else None
    with h5py.File(out_path, "w") as f:
        grp = f.create_group("data")
        grp.attrs["env_args"] = json.dumps({
            "env_name": "MyCobotPickPlace-ARM_BCRL006", "type": 2,
            "env_kwargs": {"obs_dim": int(obs_dim), "act_dim": int(act_dim),
                           "rgb_shape": rgb_shape},
        })
        demo_names = []
        for i in range(n_demos):
            obs = obs_list[i].numpy().astype(np.float32)
            act = act_list[i].numpy().astype(np.float32)
            t = obs.shape[0]
            demo_grp = grp.create_group(f"demo_{i}")
            demo_grp.attrs["num_samples"] = t
            obs_grp = demo_grp.create_group("obs")
            obs_grp.create_dataset("flat", data=obs)
            if rgb_list is not None:
                rgb = rgb_list[i].numpy().astype(np.uint8)
                if rgb.shape[0] != t:
                    raise ValueError(f"demo_{i}: rgb has {rgb.shape[0]} frames but flat has {t} rows")
                obs_grp.create_dataset("rgb", data=rgb, compression="gzip", compression_opts=4)
            demo_grp.create_dataset("actions", data=act)
            demo_grp.create_dataset("rewards", data=np.zeros(t, dtype=np.float32))
            dones = np.zeros(t, dtype=np.int64)
            dones[-1] = 1
            demo_grp.create_dataset("dones", data=dones)
            demo_names.append(f"demo_{i}")

        mask = f.create_group("mask")
        mask.create_dataset("train", data=np.array([demo_names[i] for i in train_idx], dtype="S"))
        mask.create_dataset("valid", data=np.array([demo_names[i] for i in val_idx], dtype="S"))

    summary = {"n_demos": n_demos, "n_train": len(train_idx), "n_valid": len(val_idx),
               "obs_dim": int(obs_dim), "act_dim": int(act_dim),
               "rgb_shape": rgb_shape,
               "total_transitions": int(sum(o.shape[0] for o in obs_list))}
    print("[to_robomimic_hdf5]", json.dumps(summary, indent=2))
    return summary
