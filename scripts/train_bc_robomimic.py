# -*- coding: utf-8 -*-
"""ARM_BCRL006 フェーズ3。robomimic（NVIDIA 自身の模倣学習ライブラリで、
Isaac Lab Mimic のデータ生成が狙うアルゴリズムの土台でもある）による
行動クローンの学習。

この系列の前のプロジェクトでは、robomimic の本当の学習入口
（``robomimic.scripts.train.train(config, device)``）を、状態だけの
8次元データセットに対して呼んでいた。BC004 はその構造をそのまま保ち、
観測を、集めたデモが持つマルチモーダルの契約へ上げた:

  * obs/flat : 凍結した31次元の状態ベクトル
               （low_dim モダリティ。robomimic の MLP 枝）
  * obs/rgb  : 224x224x3 の uint8 手首カメラ画像
               （rgb モダリティ。robomimic 標準の VisualCore
                エンコーダ = ResNet18Conv + SpatialSoftmax。
                これが robomimic の正統な視覚運動BCの構成で、
                選び方は observation.modalities.obs.rgb に鍵を
                並べるだけ。自作のネットワークは書かない。）

視覚運動にしたことの帰結として、AMR の BC から意図的に変えた設定が2つ:

  * ``train.hdf5_normalize_obs = False``。robomimic の観測正規化の経路は、
    画像テンソルに対しても平均と標準偏差を計算してしまう。それは意味が
    無いし、robomimic のどの画像設定もそうしていない。画像はライブラリ
    自身の /255 と HWC->CHW で扱う。正規化を外すと、AMR の BC で検証を
    切らせていた「観測正規化と検証は併用できない」という文書化された
    制限も一緒に消える。
    だから外してある。
  * ...そこで ``experiment.validate = True`` にしてある。robomimic は
    「_best_validation_」のチェックポイントを残すようになり
    （/mask/valid の分割は bc_hdf5/to_robomimic_hdf5.py が書く）、
    _find_best_checkpoint がそれを選ぶ。標準的な早期選択の流れ。

書き出す ONNX は画像の前処理（/255、HWC->CHW）をグラフに焼き込むので、
評価側は生のフレームを渡せる。_MultiModalExportAdapter を参照。
設定は Hydra（configs/config.yaml と configs/{env,bc,mlops}/*.yaml）で、
Optuna の multirun にも対応。ここではシミュレータは一切要らない。

ARM_BCRL006 での追加:
  * robomimic 自身の TensorBoard の曲線を、学習後に W&B へ流し込む。だから
    BC の損失が、PPO と ACT の曲線と同じプロジェクトのページに並ぶ。
    robomimic は学習ループを自分で持ち、エポックごとのフックを出さない
    ので、終わった後に取り込むのが正直なやり方。
  * robomimic は --no-deps で入れる（依存の egl_probe は CMake の拡張で、
    robosuite のオフスクリーン描画のためだけに存在し、このプロジェクトは
    使わない）。
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
        # 検証は BC004 ではON（冒頭参照。観測正規化を切ったので可能）。
        # だから「_best_validation_」のチェックポイントが存在する。
        config.experiment.validate = True
        config.experiment.logging.log_tb = True
        # robomimic 自身の wandb ロガーは robomimic.macros.WANDB_ENTITY の
        # 事前設定を強く要求する。AMR の BC と同じく、
        # mlops.tracking.wandb_run() を使うのでこちらは切る。
        config.experiment.logging.log_wandb = False
        config.experiment.logging.wandb_proj_name = cfg.mlops.wandb.project
        config.experiment.rollout.enabled = False  # 閉ループ評価は src/evaluate.py の仕事
        config.experiment.save.enabled = True
        config.experiment.save.every_n_epochs = int(cfg.bc.num_epochs)  # 最後のチェックポイントは常に残す
        config.experiment.save.on_best_validation = True

        config.train.data = hdf5_path
        config.train.output_dir = os.path.join(REPO_ROOT, "artifacts", "runs")
        config.train.hdf5_filter_key = "train"
        config.train.hdf5_validation_filter_key = "valid"
        config.train.hdf5_normalize_obs = False
        # 84x84x3 のフレームを全部RAMに持つ（"all"）のは、robomimic 自身の
        # 画像設定も避けている。"low_dim" ならベクトルはキャッシュし、画像は
        # HDF5 から流す。ライブラリが画像データセットに推奨するやり方。
        config.train.hdf5_cache_mode = "low_dim" if use_rgb else "all"
        config.train.seq_length = 1
        config.train.batch_size = int(batch_size)
        config.train.num_epochs = int(cfg.bc.num_epochs)
        # ARM_BCRL006 では、1エポック = 分割を1周で、固定回数の勾配ステップ
        # ではない。robomimic の既定は epoch_every_n_steps=100 /
        # validation_epoch_every_n_steps=10 で、バッチ数がそれより少ない
        # ローダに対してもその回数 next() を呼ぶ。だから既定が想定するより
        # 小さいデータセットでは StopIteration で実行が死ぬ（比較実験の
        # 小さいデータセットで実際に死ぬ）。None は「ローダを1周する」の意味で、
        # 安全で、しかも bc.num_epochs の意味としても正直。
        # だから None にしてある。
        config.experiment.epoch_every_n_steps = None
        config.experiment.validation_epoch_every_n_steps = None
        config.train.seed = int(cfg.seed)
        config.train.dataset_keys = ["actions", "dones"]
        config.observation.modalities.obs.low_dim = ["flat"]
        # 画像BCの切り替えはここ。"rgb" を並べると robomimic が既定の
        # VisualCore（ResNet18Conv + SpatialSoftmax）を作る。
        config.observation.modalities.obs.rgb = ["rgb"] if use_rgb else []

        config.algo.actor_layer_dims = list(cfg.bc.hidden)
        config.algo.optim_params.policy.learning_rate.initial = float(cfg.bc.lr)
        config.algo.optim_params.policy.regularization.L2 = float(cfg.bc.weight_decay)
    config.lock()
    return config


class _MultiModalExportAdapter(nn.Module):
    """robomimic の ActorNetwork を、ONNX 用にテンソル入出力で包む。

    入力（位置引数。mlops.export.export_onnx_multi の仕様に対応）:
      flat : (B, F) float32。そのまま渡す（BC004 は観測正規化なしで学習）
      rgb  : (B, H, W, 3) float32 の [0, 255]。生のカメラ画像。robomimic の
             ObsUtils が学習時に当てる /255 と HWC->CHW は、ここに焼き込んで
             ある。だから配備側が前処理を書き直すことは無い。
             形はそのままで渡せる。
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
    """robomimic の ``.pth`` を読み、ONNX として書き出し・検証・計測する。
    マルチモーダル（flat+rgb）と flat だけの場合、正規化統計を焼き込む
    場合と焼き込まない場合の、どれも扱える。必要なもの（形のメタデータと
    正規化統計）は全部チェックポイントから読み直すので、この関数は
    モデルがどう設定されたかを外から知る必要がない。src/infer.py が
    MLflow レジストリの優勝モデルに対して再利用する。"""
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
        # robomimic は処理後の形 (C, H, W) を保存する。アダプタは生の (H, W, C) を取る
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

    # robomimic のローダは drop_last=True なので、分割の4分の1より大きい
    # バッチにすると、学習（や検証）に足るバッチ数が残らない。ここで
    # 抑えておけば、本番の実行では設定したバッチ幅のままで、短い実行
    # （比較実験、スモークテスト）が黙って何もしないことも防げる。
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

    # robomimic の TensorBoard の曲線 -> W&B（冒頭参照）。
    # robomimic は自分で作ったタイムスタンプのディレクトリの下に run を
    # 入れるので、パスは決め打ちではなく探して見つける。
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
