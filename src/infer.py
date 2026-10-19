# -*- coding: utf-8 -*-
"""ARM_BCRL006 フェーズ10。配備。MLflow レジストリの優勝モデルを再書き出し
して確認し、エッジ配布物を組み立てる。

ARM_BCRL001 と同じ配備時の手順（「レジストリが優勝と言っているものを
出せ、再書き出しして検証し、配備する成果物の閉ループ指標を確認する」）を、
視覚運動方策とエッジ配布物の分だけ広げたもの:
bundle:

  * arm_bcrl006_bc_policy   robomimic のチェックポイント -> ONNX 再書き出しと確認
  * arm_bcrl006_rl_policy   配備する ONNX の閉ループ確認
  * arm_bcrl006_act_policy  （または _dp_policy）配備する ONNX の閉ループ
                            確認。これが実際に出荷する成果物
  * artifacts/deploy/edge_bundle/   Jetson Orin Nano に scp する正確な
    ファイル一式（視覚運動方策の ONNX（＋デノイザ）、配備メタデータ、
    環境のカメラ仕様）。配備手順ではこのディレクトリをそのまま複製する。
"""

from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys

import hydra
import robomimic.utils.torch_utils as TorchUtils
from mlflow import MlflowClient
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from scripts.train_bc_robomimic import export_checkpoint_to_onnx

SIM = os.path.join(REPO_ROOT, "mujoco_sim", "train_arm_mujoco.py")
# 評価1本につきシミュレータを1プロセス。起動にはこのインタプリタ
# （sys.executable）を使うので、子プロセスも同じ仮想環境を継ぐ。
# ARM_BCRL005 では Isaac Sim 付属の python.sh が必要で、isaaclab.sh が
# 子の終了コードを飲むことまで注記する必要があった。ローカルでは
# どちらも起きず、check=True が実際にパイプラインを止めてくれる。
def _sim_env() -> dict:
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "osmesa")
    return env


def _champion_checkpoint(cfg: DictConfig, name: str) -> str | None:
    import mlflow
    mlflow.set_tracking_uri(cfg.mlops.mlflow.tracking_uri)
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(name, "champion")
        run = client.get_run(mv.run_id)
        return run.data.tags["checkpoint_path"]
    except Exception as e:
        print(f"[infer] no registry champion for {name}: {e}")
        return None


def _confirm_onnx(cfg: DictConfig, onnx_path: str, tag: str) -> dict:
    import onnxruntime as _ort
    wants_rgb = any(i.name == "rgb" for i in _ort.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"]).get_inputs())
    n_envs = int(cfg.eval.get("visuomotor_num_envs", cfg.eval.num_envs)) if wants_rgb \
        else int(cfg.eval.num_envs)
    n_eps = int(cfg.eval.get("visuomotor_episodes", cfg.eval.episodes)) if wants_rgb \
        else int(cfg.eval.episodes)
    cmd = [sys.executable, SIM, "--num_envs", str(n_envs),
           "--mode", "eval_only", "--baseline", "checkpoint", "--onnx_path", onnx_path,
           "--eval_out", tag,
           "--set", f"eval.episodes={n_eps}", f"eval.num_envs={n_envs}",
           f"eval.max_steps={cfg.eval.max_steps}"]
    env = _sim_env()
    print("[infer]", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
    with open(os.path.join(REPO_ROOT, "artifacts", "eval", f"eval_report_{tag}.json"),
              encoding="utf-8") as f:
        return json.load(f)


def _release_gate(cfg: DictConfig, metrics: dict, tag: str) -> bool:
    """この成果物をロボットに載せてよいかを判定する。

    ARM_BCRL002 にはこの判定が無く、フェーズ10はフェーズ8が作ったものを
    何でも梱包した。梱包されたものはシミュレータ上でも成功率55%、平均
    設置誤差64mm、平均最高到達高さ69mm。つまりワークを持ち上げられて
    いない回が多かった。実機はシミュレータの数値から引き算するだけなので、
    それを出荷すれば実機 0/10 は確定だった（原因の分類は実機の失敗記録に
    残してある）。

    リリースゲートはパイプライン全体でいちばん安い管理策で、「MLOps の
    仕組みがある」を「うちの MLOps が出荷を決めている」に変える唯一の
    部品。しきい値は configs/config.yaml の deploy。
    """
    d = cfg.deploy
    checks = [
        ("success_rate", metrics.get("success_rate", 0.0),
         float(d.min_success_rate), "min"),
        ("mean_place_err", metrics.get("mean_place_err", 1e9),
         float(d.max_place_err), "max"),
        ("mean_max_lift", metrics.get("mean_max_lift", 0.0),
         float(d.min_lift), "min"),
    ]
    failures = []
    print(f"\n[infer] ===== RELEASE GATE ({tag}) =====")
    for name, value, threshold, mode in checks:
        ok = (value >= threshold) if mode == "min" else (value <= threshold)
        comp = ">=" if mode == "min" else "<="
        print(f"[infer]   {'PASS' if ok else 'FAIL'}  {name} = {value:.4f} "
              f"{comp} {threshold}")
        if not ok:
            failures.append(f"{name}={value:.4f} (need {comp} {threshold})")
    # 安全系のテレメトリは報告するが、合否には使わない。机の近くで方策が
    # どう振る舞ったかを説明する診断情報で、合否判定ではない
    if "mean_min_tool_z" in metrics:
        print(f"[infer]   INFO  mean_min_tool_z = "
              f"{metrics['mean_min_tool_z'] * 1000:.1f} mm, worst = "
              f"{metrics.get('worst_min_tool_z', float('nan')) * 1000:.1f} mm "
              f"(floor plane is at "
              f"{(float(cfg.env.floor.z) + float(cfg.env.floor.clearance)) * 1000:.0f} mm)")
    if not failures:
        print("[infer] release gate PASSED -- building the edge bundle\n")
        return True
    if bool(d.get("force", False)):
        print("[infer] release gate FAILED but deploy.force=true -- bundling "
              "anyway. Record why in the run notes.\n")
        return True
    print("[infer] release gate FAILED: " + "; ".join(failures))
    print("[infer] NO edge bundle was built. This policy is not fit to run on "
          "the robot.\n[infer] Levers, in order of effect (operations guide "
          "§1-4):\n"
          "  1. teacher quality  -- rl.total_iterations / rl.init_noise_std\n"
          "  2. distillation volume -- visuomotor.distill.episodes\n"
          "  3. student capacity -- visuomotor.train.epochs\n"
          "  4. perception       -- env.camera.* + the phase-2 image-quality "
          "report\n[infer] To ship it anyway on purpose: deploy.force=true\n")
    return False


def _copy_into(src: str, dst_dir: str) -> None:
    # shutil.copy2 は使わない。/workspace_host はバインドマウントで、古い
    # 成果物が別のuid所有になっていることがある。内容の書き込みはACLで
    # 通るが、copystat の utime/chmod は所有権が要るので EPERM で死ぬ。
    dst = os.path.join(dst_dir, os.path.basename(src))
    if os.path.exists(dst):
        os.remove(dst)
    shutil.copyfile(src, dst)


def _build_edge_bundle(cfg: DictConfig, vm_onnx: str, report: dict) -> str:
    bundle = os.path.join(REPO_ROOT, "artifacts", "deploy", "edge_bundle")
    os.makedirs(bundle, exist_ok=True)
    _copy_into(vm_onnx, bundle)
    denoiser = vm_onnx.replace(".onnx", "_denoiser.onnx")
    if os.path.exists(denoiser):
        _copy_into(denoiser, bundle)
    for side in (vm_onnx + ".data", denoiser + ".data"):
        if os.path.exists(side):  # ONNX の外部重みファイル
            _copy_into(side, bundle)
    # ONNX のメタデータを外に出す。TensorRT の .engine（ONNXのメタデータを
    # 持たない）を読んだときでも、Jetson 側が自分で構成できるようにするため
    import onnxruntime as ort
    meta = ort.InferenceSession(
        vm_onnx, providers=["CPUExecutionProvider"]).get_modelmeta().custom_metadata_map or {}
    # 環境設定のうち実行時に関係する部分だけ。制御の契約とカメラ仕様
    edge_cfg = {
        "onnx_metadata": dict(meta),
        "control_dt": float(cfg.env.control_dt),
        "action_scale": float(cfg.env.action_scale),
        "gripper_open": float(cfg.env.gripper_open),
        "default_joint_pos": [float(v) for v in cfg.env.robot.default_joint_pos],
        "camera": OmegaConf.to_container(cfg.env.camera, resolve=True),
        "execute_horizon": int(cfg.visuomotor.execute_horizon),
        "onnx": os.path.basename(vm_onnx),
        # Raspberry Pi 側が強制する安全包絡
        # （configs/safety/default.yaml -> deploy/raspi/safety.py）。モデルと
        # 一緒に出荷するのはカメラ仕様と同じ考え方で、定数を手作業でロボットに
        # 写さないようにするため。そうすればガードとシミュレータが食い違わない。
        # ここは設計上の約束として、必ず自動で運ぶ。
        "safety": OmegaConf.to_container(cfg.safety, resolve=True),
    }
    with open(os.path.join(bundle, "edge_config.json"), "w", encoding="utf-8") as f:
        json.dump(edge_cfg, f, indent=2, ensure_ascii=False)
    with open(os.path.join(bundle, "deploy_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return bundle


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    実機へ最終的に持っていくものだけを考えるなら、蒸留後のACT／Diffusion Policyだけで足ります。
    infer.pyがBC・BC+RL・最終視覚方策の3段階を扱っているのは、最終推論に全部必要だからではなく、主に次のためです。

    1. 各段階のモデルが本当にデプロイ可能か確認する
    2. 学習パイプライン全体の成果物と性能を記録する
    3. BC → RL → 視覚方策というモデルの来歴を残す
    4. 最終的なJetson配布物をまとめる
    """
    deploy_dir = os.path.join(REPO_ROOT, "artifacts", "deploy")
    os.makedirs(deploy_dir, exist_ok=True)
    report = {}

    # --- BC 優勝モデル: robomimic のチェックポイントから再書き出し --------
    bc_ckpt = _champion_checkpoint(cfg, "arm_bcrl006_bc_policy")
    if bc_ckpt and os.path.exists(bc_ckpt):
        onnx_path = os.path.join(deploy_dir, "arm_bcrl006_bc_policy.onnx")
        device = TorchUtils.get_torch_device(try_to_use_cuda=(cfg.device == "cuda"))
        val, lat = export_checkpoint_to_onnx(bc_ckpt, onnx_path, device)
        metrics = _confirm_onnx(cfg, onnx_path, "bc_deployed")
        report["bc"] = {"checkpoint": bc_ckpt, "onnx_path": onnx_path,
                        "onnx_validation": val, "latency": lat, "deployed_metrics": metrics}

    # --- RL 優勝モデル: 配備する ONNX 成果物を確認 ------------------------
    rl_onnx = os.path.join(deploy_dir, "arm_bcrl006_rl_policy.onnx")
    if os.path.exists(rl_onnx):
        metrics = _confirm_onnx(cfg, rl_onnx, "rl_deployed")
        report["rl"] = {"onnx_path": rl_onnx, "deployed_metrics": metrics}

    # --- 視覚運動の優勝モデル（エッジ成果物）: 確認して梱包 --------------
    short = "dp" if str(cfg.visuomotor.type) == "diffusion" else "act"
    vm_onnx = os.path.join(deploy_dir, f"arm_bcrl006_{short}_policy.onnx")
    if os.path.exists(vm_onnx):
        metrics = _confirm_onnx(cfg, vm_onnx, f"{short}_deployed")
        report[short] = {"onnx_path": vm_onnx, "deployed_metrics": metrics}
        passed = _release_gate(cfg, metrics, short.upper())
        report["release_gate"] = {
            "passed": bool(passed),
            "forced": bool(cfg.deploy.get("force", False)),
            "thresholds": OmegaConf.to_container(cfg.deploy, resolve=True),
            "metrics": metrics,
        }
        if passed:
            bundle = _build_edge_bundle(cfg, vm_onnx, report)
            report["edge_bundle"] = bundle
            print(f"[infer] edge bundle ready: {bundle} (scp to the Jetson -- the deployment notes)")

    # 配備確認は誰かが最後に行う測定で、関係者が実際に聞くのはこれ
    # （「出荷するものは動くのか」）。学習曲線と同じ W&B のページに
    # 載せておく。
    from mlops.tracking import wandb_log_eval
    deployed = {k.upper(): v["deployed_metrics"] for k, v in report.items()
                if isinstance(v, dict) and "deployed_metrics" in v}
    if deployed:
        wandb_log_eval(cfg, f"{cfg.experiment_name}_deployed", deployed)

    with open(os.path.join(deploy_dir, "deploy_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("[infer] deploy report:", json.dumps(report, indent=2)[:1600])
    # 「正常終了したのに配備できるものが何も無い」が、このフェーズが
    # 防ぐために存在する失敗の形。
    if report.get("release_gate") and not report["release_gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
