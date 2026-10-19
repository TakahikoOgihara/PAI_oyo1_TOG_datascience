# -*- coding: utf-8 -*-
"""ARM_BCRL006 phase 8 -- visuomotor policy training (ACT default, Diffusion
Policy via visuomotor.type=diffusion).

Input dataset: artifacts/data/visuomotor_demos.pt -- the RL-champion teacher
rollouts recorded with the wrist camera by phase 7 (--mode distill). Fallback
(visuomotor.source=expert, or when no distill file exists): the phase-1
scripted-expert demos. Either way each frame pairs (wrist RGB, qpos) with the
teacher's action, and training-time augmentation (color jitter + random
shift) supplies the light domain randomization.

Outputs (artifacts/deploy/):
  * ACT      : arm_bcrl006_act_policy.onnx        (rgb+qpos -> action chunk)
  * Diffusion: arm_bcrl006_dp_policy.onnx         (encoder: rgb+qpos -> emb)
               arm_bcrl006_dp_policy_denoiser.onnx (DDIM loop body)
ONNX metadata carries policy_type/chunk/ddim_steps/... so every consumer
(sim eval's OnnxPolicy, deploy/jetson/policy_runtime.py) self-configures from
the file. Registered in the MLflow Model Registry with a champion alias,
exactly like the BC/RL models. No Isaac Sim / AppLauncher needed here.
"""

from __future__ import annotations
import csv
import json
import os
import sys

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from arm_policy.act import ACTPolicy, ACTInference
from arm_policy.diffusion import DiffusionPolicy
from arm_policy.dataset import ChunkDataset, train_val_split
from mlops import export as onnx_export
from mlops import plotting, registry
from mlops.tracking import (mlflow_run, set_seed, tensorboard_writer,
                            wandb_env, wandb_run)

ACT_DIM = 7


def _set_onnx_metadata(path: str, meta: dict) -> None:
    import onnx
    model = onnx.load(path)
    del model.metadata_props[:]
    for k, v in meta.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = str(k), str(v)
    onnx.save(model, path)


def _load_demos(cfg: DictConfig) -> tuple[dict, str]:
    data_dir = os.path.join(REPO_ROOT, "artifacts", "data")
    distill = os.path.join(data_dir, "visuomotor_demos.pt")
    expert = os.path.join(data_dir, "bc_demos.pt")
    if str(cfg.visuomotor.source) == "rl_teacher" and os.path.exists(distill):
        return torch.load(distill, weights_only=False), "rl_teacher"
    if str(cfg.visuomotor.source) == "rl_teacher":
        print("[train_visuomotor] WARNING: no visuomotor_demos.pt (run phase 7 "
              "distill first) -- falling back to the scripted-expert demos")
    return torch.load(expert, weights_only=False), "expert"


def _build_model(cfg: DictConfig, device) -> torch.nn.Module:
    v = cfg.visuomotor
    req_bb = bool(v.train.get("require_pretrained_backbone", True))
    if str(v.type) == "diffusion":
        model = DiffusionPolicy(
            qpos_dim=int(v.qpos_dim), act_dim=ACT_DIM, 
            chunk=int(v.chunk), # 一度の推論で予測する将来行動のステップ数で
            obs_embed_dim=int(v.diffusion.obs_embed_dim), # 画像とqposから作る、観測全体の圧縮表現の次元数
            down_dims=tuple(int(d) for d in v.diffusion.down_dims), # Diffusion Policyの1D Temporal U-Net内部のチャンネル数
            diffusion_steps=int(v.diffusion.diffusion_steps), # 学習時の拡散過程を何段階に分けるか
            ddim_steps=int(v.diffusion.ddim_steps), # 推論時に実際に何回ノイズ除去ネットワークを実行するか
            require_pretrained_backbone=req_bb)
    else:
        model = ACTPolicy(
            qpos_dim=int(v.qpos_dim), act_dim=ACT_DIM, chunk=int(v.chunk),
            dim_model=int(v.act.dim_model), 
            # Transformer内部で使う各トークンの特徴次元です。一般にd_modelとも呼ばれます。
            # このACTでは、次のものをすべてdim_model次元へ揃えます。
            # 各画像トークン, qposトークン, CVAE潜在トークン, 行動生成用query
            # コードでは画像のResNet特徴512チャンネルを1×1 Convでdim_modelへ変換します。
            # self.input_proj = nn.Conv2d(512,dim_model,kernel_size=1,)
            # qposもLinearでdim_modelへ変換します。
            # self.qpos_proj = nn.Linear(qpos_dim,dim_model,)
            n_heads=int(v.act.n_heads),
            # Multi-Head Attentionのヘッド数です。
            # dim_modelは通常n_headsで割り切れる必要があります。
            dim_feedforward=int(v.act.dim_feedforward),
            # Transformer各層の中にあるFeed-Forward Networkの中間次元です。
            enc_layers=int(v.act.enc_layers), 
            # 観測を処理するTransformer Encoderの層数です。
            dec_layers=int(v.act.dec_layers),
            # 行動チャンクを生成するTransformer Decoderの層数です。
            latent_dim=int(v.act.latent_dim),
            # ACTのCVAEが使う潜在変数zの次元です。
            kl_weight=float(v.act.kl_weight),
            # CVAEのKL損失を、全体損失の中でどれだけ重視するかです。
            dropout=float(v.act.dropout), 
            # Transformer内部で一部の要素をランダムに無効化する正則化率です。
            image_size=int(v.image_size),
            require_pretrained_backbone=req_bb)
            # 入力画像の高さ・幅です。このコードは正方形画像を前提にしています。
    return model.to(device)


@torch.no_grad()
def _val_metrics(model, loader, device, ptype: str) -> tuple[float, float]:
    """-> (training-objective val loss, DEPLOYMENT-condition action L1).

    ARM_BCRL006 change: ARM_BCRL002 selected its best checkpoint on the
    training objective alone. For ACT that objective is L1 + kl_weight * KL
    evaluated with a SAMPLED CVAE latent -- but the exported ONNX always
    runs with z = 0 (the CVAE mean). The number the checkpoint was chosen by
    was therefore not the number the shipped graph produces.

    ``val_l1_deploy`` is the mean absolute action error under exactly the
    deployed condition, so it is directly interpretable: multiply by
    env.action_scale (0.05 rad) to read it as radians of per-joint, per-step
    command error, and by the chunk length to bound the open-loop drift over
    one receding-horizon window. That is what makes it usable as an
    acceptance criterion (operations guide §4).
    """
    model.eval()
    tot, l1_tot, n = 0.0, 0.0, 0
    for batch in loader:
        rgb = batch["rgb"].to(device)
        qpos = batch["qpos"].to(device)
        chunk = batch["chunk"].to(device)
        is_pad = batch["is_pad"].to(device)
        b = rgb.shape[0]
        loss, _ = model.loss(rgb, qpos, chunk, is_pad)
        tot += float(loss) * b
        if ptype == "diffusion":
            pred = model.sample(rgb, qpos)
        else:
            pred, _, _ = model(rgb, qpos, actions=None)   # z = 0, as exported
            pred = pred.clamp(-1.0, 1.0)
        err = (pred - chunk).abs() * (~is_pad).unsqueeze(-1)
        l1_tot += float(err.sum() / ((~is_pad).sum() * pred.shape[-1] + 1e-6)) * b
        n += b
    model.train()
    return tot / max(1, n), l1_tot / max(1, n)


def _export_act(model: ACTPolicy, v, onnx_path: str) -> tuple[dict, dict]:
    wrapper = ACTInference(model).eval().cpu()
    h = w = int(v.image_size)
    specs = [("rgb", (h, w, 3)), ("qpos", (int(v.qpos_dim),))]
    onnx_export.export_onnx_multi(wrapper, specs, onnx_path)
    val = onnx_export.validate_onnx_multi(wrapper, specs, onnx_path, n=4)
    lat = onnx_export.benchmark_latency_multi(wrapper, specs, onnx_path, iters=30)
    _set_onnx_metadata(onnx_path, {
        "policy_type": "act", "chunk": int(v.chunk), "act_dim": ACT_DIM,
        "image_size": int(v.image_size), "qpos_dim": int(v.qpos_dim),
        "execute_horizon": int(v.execute_horizon)})
    return val, lat


def _export_diffusion(model: DiffusionPolicy, v, onnx_path: str) -> tuple[dict, dict]:
    h = w = int(v.image_size)
    enc = model.encoder.eval().cpu()
    specs = [("rgb", (h, w, 3)), ("qpos", (int(v.qpos_dim),))]
    onnx_export.export_onnx_multi(enc, specs, onnx_path)
    val_enc = onnx_export.validate_onnx_multi(enc, specs, onnx_path, n=4)
    lat = onnx_export.benchmark_latency_multi(enc, specs, onnx_path, iters=30)

    # denoiser graph (int64 timestep input -> custom export, not the helper)
    den = model.denoiser.eval().cpu()
    den_path = onnx_path.replace(".onnx", "_denoiser.onnx")
    K, emb = int(v.chunk), int(v.diffusion.obs_embed_dim)
    dummy = (torch.randn(1, K, ACT_DIM), torch.zeros(1, dtype=torch.long),
             torch.randn(1, emb))
    den_kwargs = dict(
        input_names=["noisy_chunk", "timestep", "obs_emb"], output_names=["eps"],
        dynamic_axes={"noisy_chunk": {0: "batch"}, "timestep": {0: "batch"},
                      "obs_emb": {0: "batch"}, "eps": {0: "batch"}},
        opset_version=17)
    try:  # legacy exporter: torch>=2.9 dynamo default can bake batch=1 (see mlops/export.py)
        torch.onnx.export(den, dummy, den_path, dynamo=False, **den_kwargs)
    except TypeError:
        torch.onnx.export(den, dummy, den_path, **den_kwargs)
    # validate the denoiser graph against torch
    import onnxruntime as ort
    sess = ort.InferenceSession(den_path, providers=["CPUExecutionProvider"])
    x = torch.randn(4, K, ACT_DIM)
    t = torch.randint(0, model.diffusion_steps, (4,))
    e = torch.randn(4, emb)
    with torch.no_grad():
        y_t = den(x, t, e).numpy()
    y_o = sess.run(None, {"noisy_chunk": x.numpy(), "timestep": t.numpy(),
                          "obs_emb": e.numpy()})[0]
    val = {"encoder_max_abs_diff": val_enc.get("max_abs_diff", -1.0),
           "denoiser_max_abs_diff": float(np.abs(y_t - y_o).max()),
           "max_abs_diff": max(val_enc.get("max_abs_diff", 0.0),
                               float(np.abs(y_t - y_o).max()))}
    meta = {"policy_type": "diffusion", "chunk": K, "act_dim": ACT_DIM,
            "image_size": int(v.image_size), "qpos_dim": int(v.qpos_dim),
            "execute_horizon": int(v.execute_horizon),
            "diffusion_steps": int(v.diffusion.diffusion_steps),
            "ddim_steps": int(v.diffusion.ddim_steps)}
    _set_onnx_metadata(onnx_path, meta)
    _set_onnx_metadata(den_path, meta)
    return val, lat


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> float:
    set_seed(int(cfg.seed))
    wandb_env(cfg)
    v = cfg.visuomotor
    ptype = str(v.type)
    device = torch.device("cuda" if (cfg.device == "cuda" and torch.cuda.is_available())
                          else "cpu")
    runs_dir = os.path.join(REPO_ROOT, "artifacts", "runs", "visuomotor")
    deploy_dir = os.path.join(REPO_ROOT, "artifacts", "deploy")
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(deploy_dir, exist_ok=True)

    demos, source = _load_demos(cfg)
    if len(demos.get("obs", [])) < 4:
        raise SystemExit(
            f"[train_visuomotor] ERROR: the dataset has {len(demos.get('obs', []))} "
            "trajectories. ACT needs a few hundred. The cause is upstream, not here: "
            "check artifacts/data/distill_summary.json (kept_trajectories) and, if the "
            "teacher is the problem, artifacts/eval/eval_report_rl_finetune.json.")
    train_idx, val_idx = train_val_split(demos, float(v.train.val_split), int(cfg.seed))
    train_ds = ChunkDataset(demos, int(v.chunk), aug_cfg=v.train.aug, indices=train_idx)
    val_ds = ChunkDataset(demos, int(v.chunk), aug_cfg=None, indices=val_idx)
    print(f"[train_visuomotor] type={ptype} source={source} "
          f"train={len(train_ds)} val={len(val_ds)} frames")
    nw = int(v.train.get("num_workers", 4))
    train_loader = DataLoader(train_ds, batch_size=int(v.train.batch_size), shuffle=True,
                              num_workers=nw, pin_memory=True, drop_last=True,
                              persistent_workers=nw > 0)
    val_loader = DataLoader(val_ds, batch_size=int(v.train.batch_size), shuffle=False,
                            num_workers=max(0, nw // 2), pin_memory=True)

    model = _build_model(cfg, device)
    optim = torch.optim.AdamW(model.parameters(), lr=float(v.train.lr),
                              weight_decay=float(v.train.weight_decay))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=int(v.train.epochs))

    metrics_csv = os.path.join(runs_dir, f"{ptype}_metrics.csv")
    best_val, best_state = float("inf"), None
    model_name = f"arm_bcrl006_{'dp' if ptype == 'diffusion' else 'act'}_policy"

    with mlflow_run(cfg, run_name=f"{cfg.experiment_name}_{ptype}"), \
            wandb_run(cfg, run_name=f"{cfg.experiment_name}_{ptype}") as _wb, \
            tensorboard_writer(cfg, f"visuomotor_{ptype}") as _tb, \
            open(metrics_csv, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=["step", "train_loss", "val_loss",
                                                  "val_l1_deploy"])
        writer.writeheader()
        model.train()
        for epoch in range(int(v.train.epochs)):
            tot, n = 0.0, 0
            for batch in train_loader:
                loss, parts = model.loss(
                    batch["rgb"].to(device, non_blocking=True),
                    batch["qpos"].to(device, non_blocking=True),
                    batch["chunk"].to(device, non_blocking=True),
                    batch["is_pad"].to(device, non_blocking=True))
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                tot += float(loss) * batch["rgb"].shape[0]
                n += batch["rgb"].shape[0]
            sched.step()
            train_loss = tot / max(1, n)
            val_loss, val_l1 = _val_metrics(model, val_loader, device, ptype)
            writer.writerow({"step": epoch, "train_loss": train_loss,
                             "val_loss": val_loss, "val_l1_deploy": val_l1})
            fcsv.flush()
            import mlflow
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss,
                                "val_l1_deploy": val_l1}, step=epoch)
            if _wb is not None:
                _wb.log({"train_loss": train_loss, "val_loss": val_loss,
                         "val_l1_deploy": val_l1})
            if _tb is not None:
                # NOTE: not `k, v` -- `v` is the config alias (cfg.visuomotor)
                # used by the loop bound and the ONNX export below.
                for _k, _val in (("train_loss", train_loss), ("val_loss", val_loss),
                                 ("val_l1_deploy", val_l1)):
                    _tb.add_scalar(f"{ptype}/{_k}", _val, epoch)
            # select on the DEPLOYED condition, not the training objective
            if val_l1 < best_val:
                best_val = val_l1
                best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            print(f"[train_visuomotor] epoch {epoch + 1}/{int(v.train.epochs)} "
                  f"train={train_loss:.4f} val={val_loss:.4f} "
                  f"val_l1_deploy={val_l1:.4f} best={best_val:.4f} "
                  f"(~{best_val * float(cfg.env.action_scale) * 1000:.1f} mrad/joint/step)")

        model.load_state_dict(best_state)
        ckpt_path = os.path.join(runs_dir, f"{ptype}_best.pt")
        torch.save({"state_dict": best_state, "best_val_l1_deploy": best_val,
                    "type": ptype, "source": source}, ckpt_path)

        model = model.cpu().eval()
        onnx_path = os.path.join(deploy_dir, f"{model_name}.onnx")
        if ptype == "diffusion":
            val, lat = _export_diffusion(model, v, onnx_path)
        else:
            val, lat = _export_act(model, v, onnx_path)
        print(f"[train_visuomotor] onnx: {onnx_path} validation: {val} latency: {lat}")

        plotting.plot_training_curves(metrics_csv,
                                      os.path.join(runs_dir, f"{ptype}_loss.png"),
                                      x="step",
                                      ys=["train_loss", "val_loss", "val_l1_deploy"],
                                      title=f"ARM_BCRL006 {ptype} training")

        import mlflow
        mlflow.log_metrics({"best_val_l1_deploy": best_val,
                            "onnx_max_abs_diff": val.get("max_abs_diff", -1.0),
                            "onnx_latency_ms": lat.get("onnx_ms", -1.0)})
        mlflow.set_tag("checkpoint_path", ckpt_path)
        mlflow.set_tag("dataset_source", source)
        metrics_for_registry = {"best_val_l1_deploy": best_val,
                                "onnx_max_abs_diff": val.get("max_abs_diff", 0.0),
                                "onnx_latency_ms": lat.get("onnx_ms", -1.0)}
        version = registry.register_onnx_model(onnx_path, model_name, metrics_for_registry)
        is_champion = registry.promote_if_best(model_name, version.version,
                                               "best_val_l1_deploy",
                                               metrics_for_registry, mode="min")
        print(f"[train_visuomotor] registered {model_name} v{version.version}; "
              f"champion={is_champion}")

        report = {"onnx_path": onnx_path, "checkpoint": ckpt_path, "type": ptype,
                  "dataset_source": source, "best_val_l1_deploy": best_val,
                  "validation": val, "latency": lat}
        with open(os.path.join(deploy_dir, f"deploy_report_{ptype}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    return float(best_val)


if __name__ == "__main__":
    main()
