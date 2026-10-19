# -*- coding: utf-8 -*-
"""ARM_BCRL006 phase 3 -- behavioral-cloning training with robomimic (NVIDIA's own
imitation-learning library -- also the algorithm backend Isaac Lab Mimic's
data-generation workflows target).

The earlier projects in this series called robomimic's real training entry point
(``robomimic.scripts.train.train(config, device)``) on a state-only (8-dim)
dataset. BC004 keeps that exact structure and upgrades the observation to the
multimodal contract the collected demos now carry:

  * obs/flat : the frozen 31-dim state vector
               (low_dim modality -- robomimic MLP branch)
  * obs/rgb  : 224x224x3 uint8 wrist-camera frames
               (rgb modality -- robomimic's stock VisualCore encoder:
                ResNet18Conv backbone + SpatialSoftmax pooling. This is
                robomimic's canonical visuomotor-BC configuration, selected
                purely by LISTING the key under observation.modalities.obs.rgb
                -- no custom network code.)

Two deliberate config changes vs the AMR BC projects, both consequences of going visuomotor:

  * ``train.hdf5_normalize_obs = False`` -- robomimic's obs-normalization
    path would compute mean/std over the image tensor too, which is neither
    meaningful nor what any robomimic image config does; images are handled
    by the library's own /255 + HWC->CHW processing instead. Dropping
    normalization ALSO removes the library's documented
    "no obs-normalization with validation" limitation that forced the AMR BC projects to
    disable validation...
  * ...so ``experiment.validate = True`` here: robomimic now keeps
    "_best_validation_" checkpoints (the /mask/valid split written by
    bc_hdf5/to_robomimic_hdf5.py), and _find_best_checkpoint picks those --
    the standard early-model-selection robomimic workflow.

The exported ONNX bakes the image processing (/255, HWC->CHW) into the graph
so the ROS2/eval side feeds raw frames -- see _MultiModalExportAdapter.
Config is Hydra's (configs/config.yaml + configs/{env,bc,mlops}/*.yaml) with
Optuna multirun support; no simulator is needed here at all.

ARM_BCRL006 additions:
  * robomimic's own TensorBoard curves are replayed into W&B after training, so
    the BC loss sits on the same project page as the PPO and ACT curves.
    robomimic owns its training loop and exposes no per-epoch hook, so a
    post-run import is the honest way to get them there.
  * robomimic is installed with --no-deps (its egl_probe dependency is a CMake
    extension that only serves robosuite's offscreen renderer, which this
    project never uses). See change log §7.
"""

from __future__ import annotations
import glob
import json
import os
import sys
import time

import h5py
import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.config import config_factory
from robomimic.scripts.train import train as robomimic_train

from mlops import export as onnx_export
from mlops import registry
from mlops.tracking import (import_tensorboard_curves_to_wandb, mlflow_run,
                            set_seed, wandb_env, wandb_run)

ACT_DIM = 7


def _dataset_rows(hdf5_path: str, mask: str) -> int:
    with h5py.File(hdf5_path, "r") as f:
        keys = [k.decode() if isinstance(k, bytes) else str(k) for k in f["mask"][mask][:]]
        return int(sum(f["data"][k].attrs["num_samples"] for k in keys))


def _dataset_shapes(hdf5_path: str) -> dict:
    """
    HDF5データセットを実際に開いて、flat 観測の次元数と rgb 画像のshapeを読む関数 です。
    目的は、観測次元をコードにハードコードせず、データセット自体を正としてrobomimic設定を作ることです。
    （flatは、複数の数値観測を1本の一次元ベクトルへ連結したもの。
    　例えばLiDARが32方向なら、6個の状態値　+ 2個の前回アクション　+ 32個のLiDAR距離　= 40次元）
    """
    with h5py.File(hdf5_path, "r") as f:
        demo = f["data"][sorted(f["data"].keys())[0]]
        flat_dim = int(demo["obs/flat"].shape[-1])
        rgb_shape = tuple(int(v) for v in demo["obs/rgb"].shape[1:]) if "rgb" in demo["obs"] else None
    return {"flat_dim": flat_dim, "rgb_shape": rgb_shape}


def build_robomimic_config(cfg: DictConfig, hdf5_path: str, exp_name: str, use_rgb: bool,
                           batch_size: int):
    config = config_factory(algo_name="bc")
    """
    robomimicのBC用デフォルトconfigを作る関数呼び出し
    robomimicの Behavioral Cloning用config object を作っています。
    BCアルゴリズム用のrobomimic標準設定のひな型を作って、この後に必要に応じて上書きする。
    """
    with config.values_unlocked():
        config.experiment.name = exp_name
        # Validation is ON in BC004 (see module docstring: possible because
        # obs normalization is off), so "_best_validation_" checkpoints exist.
        config.experiment.validate = True
        config.experiment.logging.log_tb = True
        # robomimic's OWN wandb logger hard-asserts on robomimic.macros.WANDB_ENTITY
        # being pre-configured -- disabled in favor of mlops.tracking.wandb_run(),
        # exactly as in the AMR BC projects.
        config.experiment.logging.log_wandb = False
        config.experiment.logging.wandb_proj_name = cfg.mlops.wandb.project
        config.experiment.rollout.enabled = False  # closed-loop eval is src/evaluate.py's job (Isaac Lab)
        config.experiment.save.enabled = True
        config.experiment.save.every_n_epochs = int(cfg.bc.num_epochs)  # always keep a final checkpoint
        config.experiment.save.on_best_validation = True

        config.train.data = hdf5_path
        config.train.output_dir = os.path.join(REPO_ROOT, "artifacts", "runs")
        config.train.hdf5_filter_key = "train"
        config.train.hdf5_validation_filter_key = "valid"
        config.train.hdf5_normalize_obs = False
        # Caching every 84x84x3 frame in RAM ("all") is what robomimic's own
        # image configs avoid; "low_dim" keeps vectors cached and streams
        # images from HDF5 -- the library's recommended image-dataset mode.
        config.train.hdf5_cache_mode = "low_dim" if use_rgb else "all"
        config.train.seq_length = 1
        config.train.batch_size = int(batch_size)
        config.train.num_epochs = int(cfg.bc.num_epochs)
        # ARM_BCRL006: one EPOCH = one full pass over the split, not a fixed
        # number of gradient steps. robomimic defaults to
        # epoch_every_n_steps=100 / validation_epoch_every_n_steps=10 and then
        # calls next() that many times on a loader that may have fewer batches
        # -- which raises StopIteration and kills the run on any dataset
        # smaller than those defaults assume (it does, on the ablation's
        # smaller datasets). None means "iterate the loader", which is both
        # safe and the more honest meaning of bc.num_epochs.
        config.experiment.epoch_every_n_steps = None
        config.experiment.validation_epoch_every_n_steps = None
        config.train.seed = int(cfg.seed)
        config.train.dataset_keys = ["actions", "dones"]
        config.observation.modalities.obs.low_dim = ["flat"]
        # THE image-BC switch: listing "rgb" here makes robomimic build its
        # default VisualCore (ResNet18Conv + SpatialSoftmax) encoder for it.
        config.observation.modalities.obs.rgb = ["rgb"] if use_rgb else []

        config.algo.actor_layer_dims = list(cfg.bc.hidden)
        config.algo.optim_params.policy.learning_rate.initial = float(cfg.bc.lr)
        config.algo.optim_params.policy.regularization.L2 = float(cfg.bc.weight_decay)
    config.lock()
    return config


class _MultiModalExportAdapter(nn.Module):
    """Tensor-in/tensor-out wrapper around robomimic's ActorNetwork for ONNX.

    Inputs (positional, matching mlops.export.export_onnx_multi's specs):
      flat : (B, F) float32 -- fed raw (BC004 trains without obs normalization)
      rgb  : (B, H, W, 3) float32 in [0, 255] -- the raw camera frame; the
             /255 scaling and HWC->CHW transpose robomimic's ObsUtils applies
             at train time are BAKED IN here, so the deployment side never
             re-implements preprocessing.
    """

    def __init__(self, policy_net: nn.Module, use_rgb: bool,
                 obs_mean: torch.Tensor | None = None, obs_std: torch.Tensor | None = None):
        super().__init__()
        self.policy_net = policy_net
        self.use_rgb = use_rgb
        self.has_norm = obs_mean is not None
        if self.has_norm:
            self.register_buffer("obs_mean", obs_mean)
            self.register_buffer("obs_std", obs_std)

    def forward(self, flat: torch.Tensor, rgb: torch.Tensor | None = None) -> torch.Tensor:
        if self.has_norm:
            flat = (flat - self.obs_mean) / self.obs_std
        obs = {"flat": flat}
        if self.use_rgb:
            obs["rgb"] = rgb.permute(0, 3, 1, 2) / 255.0
        return self.policy_net(obs)


def _find_best_checkpoint(models_dir: str) -> str:
    ckpts = glob.glob(os.path.join(models_dir, "**", "*.pth"), recursive=True)
    if not ckpts:
        raise FileNotFoundError(f"no robomimic checkpoints found under {models_dir}")
    best = [c for c in ckpts if "_best_validation_" in c]
    candidates = best if best else ckpts
    return max(candidates, key=os.path.getmtime)
"""
_best_validation_ という名前を含むcheckpointを優先して候補にしているので、
結果として「validation best checkpointの中から最新のもの」を選んでいます。
"""


def export_checkpoint_to_onnx(ckpt_path: str, onnx_path: str, device) -> tuple[dict, dict]:
    """Load a robomimic ``.pth`` checkpoint and export/validate/benchmark it as
    ONNX. Handles both the multimodal (flat+rgb) and flat-only cases, and both
    with/without baked normalization stats -- everything is read back from the
    checkpoint itself (shape metadata + normalization stats), so this function
    needs no external knowledge of how the model was configured. Reused by
    src/infer.py against the MLflow-registry champion."""
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=False)
    """
    checkpointからpolicyと辞書を読む
    これはrobomimicのユーティリティで、ckpt_pathの.pthを開き。
    robomimicのpolicy objectを復元する + checkpointの中身辞書 ckpt_dict も返す

    policy は推論に使えるrobomimic policy wrapperです。
    ckpt_dict は、checkpoint内のmetadataやnormalization statsなどを含む辞書です。

    ここで自分で torch.load(...) していない理由は、
    robomimic checkpointは単なる state_dict ではなく、
    configやshape metadataも含むrobomimic独自構造だからです。
    """

    shape_meta = ckpt_dict.get("shape_metadata", {})
    all_shapes = shape_meta.get("all_shapes", {})
    flat_dim = int(all_shapes["flat"][0]) if "flat" in all_shapes else None
    use_rgb = "rgb" in all_shapes
    if use_rgb:
        # robomimic stores processed shapes (C, H, W); the adapter takes raw (H, W, C)
        c, h, w = (int(v) for v in all_shapes["rgb"])
        rgb_spec = ("rgb", (h, w, c))

    obs_mean = obs_std = None
    stats = ckpt_dict.get("obs_normalization_stats", None)
    if stats is not None and "flat" in stats:
        obs_mean = torch.as_tensor(np.asarray(stats["flat"]["mean"])[0], dtype=torch.float32)
        obs_std = torch.as_tensor(np.asarray(stats["flat"]["std"])[0], dtype=torch.float32)
        if flat_dim is None:
            flat_dim = obs_mean.shape[0]
    if flat_dim is None:
        raise RuntimeError(f"could not determine flat obs dim from checkpoint {ckpt_path}")

    policy_net = policy.policy.nets["policy"]
    """
    FileUtils.policy_from_checkpoint(...) で返ってきた policy は、robomimicの高レベルwrapperです。
    ONNX exportしたいのは、その中の実際のPyTorch nn.Module、つまりactor network本体です。
    それが、policy.policy.nets["policy"]
    """
    adapter = _MultiModalExportAdapter(policy_net, use_rgb, obs_mean, obs_std)
    """
    データの形（辞書/テンソル、rgbをHWC→CHWなど）の変換、標準化

    _MultiModalExportAdapter は、このファイル内で定義された nn.Module wrapperです。
    目的は、robomimic policyをONNXに出しやすい形にすることです。
    robomimicのpolicyは本来、辞書入力を受けます。

    しかしONNX exportでは、辞書よりも、明示的なtensor入力にしたい。
    tensor入力をrobomimic形式の辞書に変換します。
    さらにnormalization statsがある場合は、
    flat = (flat - self.obs_mean) / self.obs_std もadapter内部で行います。

    """

    input_specs = [("flat", (flat_dim,))] + ([rgb_spec] if use_rgb else [])
    onnx_export.export_onnx_multi(adapter, input_specs, onnx_path)

    # ONNX一致検証結果。
    val = onnx_export.validate_onnx_multi(adapter, input_specs, onnx_path)
    # latency測定結果。
    lat = onnx_export.benchmark_latency_multi(adapter, input_specs, onnx_path)
    return val, lat

"""
@hydra.main(...)デコレータ
def main(cfg: DictConfig) -> float:

このデコレータとomegaconfのDictConfigにより、
mainは単なるPython関数から、
Hydraによって設定・実験を管理されるアプリケーションのエントリーポイントになる。

config.yamlを探し、その固定の設定を読み取ったうえで、
その設定をもとに複数の設定の組み合わせパターンを生成して、
その各設定パターンでmainを実行して、その結果もhydraで管理できる。

通常は、設定どおりに1回実行するだけ。
複数パターンを実行するのは、--multirunまたは-mを明示した場合。例えば、
python train.py --multirun \
  bc.lr=0.0001,0.0003 \
  bc.batch_size=64,128
（--multirunという名前と機能はHydra自身が標準で定義しているもの）

@hydra.main(...)：Hydraに設定の読込・合成・CLI上書き・実行管理をさせる
DictConfig：Hydraが合成した設定を、cfg.bc.lrのように扱うOmegaConfの設定オブジェクト型
-> float：戻り値がfloatであることを示す型ヒント。
"""

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> float:
    set_seed(int(cfg.seed))
    wandb_env(cfg)
    data_dir = os.path.join(REPO_ROOT, "artifacts", "data")
    hdf5_path = os.path.join(data_dir, "arm_bcrl006_demos.hdf5")
    deploy_dir = os.path.join(REPO_ROOT, "artifacts", "deploy")
    os.makedirs(deploy_dir, exist_ok=True)

    shapes = _dataset_shapes(hdf5_path)
    use_rgb = bool(cfg.bc.use_rgb) and shapes["rgb_shape"] is not None
    if bool(cfg.bc.use_rgb) and shapes["rgb_shape"] is None:
        print("[train_bc] WARNING: bc.use_rgb=true but the dataset has no obs/rgb -- "
              "training state-only (re-collect with the camera to get frames)")
    print(f"[train_bc] dataset: flat_dim={shapes['flat_dim']} rgb_shape={shapes['rgb_shape']} use_rgb={use_rgb}")

    # robomimic's loaders use drop_last=True, so a batch larger than a quarter
    # of the split leaves too few batches to train (or validate) on. Clamping
    # here keeps the configured batch size for real runs and keeps short runs
    # -- the ablation, a smoke test -- from silently doing nothing.
    n_valid = _dataset_rows(hdf5_path, "valid")
    batch_size = max(16, min(int(cfg.bc.batch_size), n_valid // 2,
                             _dataset_rows(hdf5_path, "train") // 4))
    if batch_size != int(cfg.bc.batch_size):
        print(f"[train_bc] batch_size {int(cfg.bc.batch_size)} -> {batch_size} "
              f"(train rows {_dataset_rows(hdf5_path, 'train')}, valid rows {n_valid})")

    exp_name = f"{cfg.experiment_name}_{int(time.time() * 1000)}"
    config = build_robomimic_config(cfg, hdf5_path, exp_name, use_rgb, batch_size)
    device = TorchUtils.get_torch_device(try_to_use_cuda=(cfg.device == "cuda"))

    with mlflow_run(cfg, run_name=cfg.experiment_name), \
            wandb_run(cfg, run_name=cfg.experiment_name) as _wb:
        """
        with mlflow_run(...), wandb_run(...):
        MLflowとWeights & Biasesの実験記録を開始し、
        その実験記録が有効な範囲内で、学習・ONNX変換・モデル登録を実行するという意味。

        mlflow_runは、tracking.pyで、下記のように定義されている。
        @contextmanager  # 普通の関数をwithで使える「開始処理＋終了処理」に変えるデコレータ
        def mlflow_run(cfg, run_name):
        import mlflow
        mlflow.set_tracking_uri(cfg.mlops.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.mlops.mlflow.experiment_name)
        with mlflow.start_run(run_name=run_name) as run:
        　mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
        　mlflow.set_tags({...})
        yield run  # with内の処理が終わったら関数へ戻って終了処理
        """
        robomimic_train(config, device=device)
        """
        このrobomimic_trainの入力や出力などは、configの中でbuild_robomimic_configで設定している。
        学習して.pthチェックポイント等を保存　⇒　ONNXへ変換　⇒　MLflow/W&Bへ結果を記録
        """

        base_output_dir = os.path.join(config.train.output_dir, config.experiment.name)
        ckpt_path = _find_best_checkpoint(base_output_dir)
        print("[train_bc] best checkpoint:", ckpt_path)

        onnx_path = os.path.join(deploy_dir, "arm_bcrl006_bc_policy.onnx")
        # ONNXへの変換、検証（ ONNXの出力一致と推論速度を検証）
        val, lat = export_checkpoint_to_onnx(ckpt_path, onnx_path, device)
        print("[train_bc] onnx:", onnx_path, "validation:", val, "latency:", lat)

        best_val_loss = None
        if "_best_validation_" in ckpt_path:
            try:
                best_val_loss = float(os.path.basename(ckpt_path).split("_best_validation_")[1].rsplit(".pth", 1)[0])
            except ValueError:
                best_val_loss = None

        import mlflow
        final_metrics = {k: v for k, v in {
            "best_val_loss": best_val_loss,
            "onnx_max_abs_diff": val.get("max_abs_diff"),
            "onnx_latency_ms": lat.get("onnx_ms"),
        }.items() if v is not None}
        mlflow.log_metrics(final_metrics)
        mlflow.set_tag("checkpoint_path", ckpt_path)
        mlflow.set_tag("use_rgb", str(use_rgb))
        if _wb is not None:
            _wb.log(final_metrics)

        metric_key, mode = ("best_val_loss", "min") if best_val_loss is not None else ("onnx_max_abs_diff", "min")
        metrics_for_registry = {"best_val_loss": best_val_loss if best_val_loss is not None else 0.0,
                                "onnx_max_abs_diff": val.get("max_abs_diff", 0.0),
                                "onnx_latency_ms": lat.get("onnx_ms", -1.0)}
        version = registry.register_onnx_model(onnx_path, "arm_bcrl006_bc_policy", metrics_for_registry)
        is_champion = registry.promote_if_best("arm_bcrl006_bc_policy", version.version, metric_key,
                                               metrics_for_registry, mode=mode)
        print(f"[train_bc] registered arm_bcrl006_bc_policy v{version.version}; champion={is_champion}")
        mlflow.log_artifact(onnx_path, artifact_path="deploy")

    # robomimic's TensorBoard curves -> W&B (see the module docstring).
    # It nests the run under a timestamp directory of its own making, so the
    # path is discovered rather than assumed.
    tb_dirs = sorted(glob.glob(os.path.join(base_output_dir, "*", "logs", "tb")),
                     key=os.path.getmtime)
    tb_dir = tb_dirs[-1] if tb_dirs else os.path.join(base_output_dir, "logs", "tb")
    n = import_tensorboard_curves_to_wandb(cfg, tb_dir, f"{cfg.experiment_name}_bc_curves",
                                           prefix="bc/")
    print(f"[train_bc] tensorboard: {tb_dir} ({n} steps mirrored to W&B)")

    result = {"best_val_loss": best_val_loss, "onnx_max_abs_diff": val.get("max_abs_diff")}
    with open(os.path.join(REPO_ROOT, "artifacts", "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return float(best_val_loss) if best_val_loss is not None else float(val.get("max_abs_diff", 1.0))


if __name__ == "__main__":
    main()
