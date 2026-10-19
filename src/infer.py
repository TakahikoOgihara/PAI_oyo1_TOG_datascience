# -*- coding: utf-8 -*-
"""ARM_BCRL006 phase 10 -- deployment: re-export/confirm the MLflow registry
champions and assemble the EDGE BUNDLE.

Same deployment-time step as ARM_BCRL001 ("give me whatever the registry says
is champion, re-export/validate it, confirm the deployed artifact's
closed-loop metrics"), extended with the visuomotor policy and the edge
bundle:

  * arm_bcrl006_bc_policy   robomimic checkpoint -> ONNX re-export + confirm
  * arm_bcrl006_rl_policy   deployed ONNX closed-loop confirmation
  * arm_bcrl006_act_policy  (or _dp_policy) deployed ONNX closed-loop
                            confirmation -- THE artifact that ships
  * artifacts/deploy/edge_bundle/   the exact file set to scp to the Jetson
    Orin Nano (visuomotor ONNX(+denoiser), deploy metadata, env camera spec)
    -- operations guide appendix A copies this directory verbatim.
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
# One simulator process per evaluation, launched with THIS interpreter
# (sys.executable), so the child inherits the same virtualenv. ARM_BCRL005
# needed Isaac Sim's own python.sh here and had to document that isaaclab.sh
# swallows the child's exit code; neither problem exists locally, and
# check=True now actually stops the pipeline when the simulator fails.
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
    """Decide whether this artifact is allowed onto the robot.

    ARM_BCRL002 had no such decision: phase 10 packaged whatever phase 8
    produced. What it packaged succeeded 55 % of the time IN SIMULATION with a
    64 mm mean placement error and a mean peak lift of 69 mm -- i.e. it often
    never picked the block up at all. Hardware only subtracts from a
    simulation number, so shipping that guaranteed the observed 0-out-of-10
    (field-failure notes §2, cause D).

    A release gate is the cheapest control in the whole pipeline, and it is
    the one thing that turns "we have an MLOps stack" into "our MLOps stack
    decides what ships". Thresholds: configs/config.yaml -> deploy.
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
    # safety telemetry is reported but never blocks: it describes HOW the
    # policy behaved near the desk, which is diagnostic, not pass/fail
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
    # NOT shutil.copy2: /workspace_host is a bind mount where stale artifacts
    # may be owned by another uid -- content writes pass via ACL but copystat's
    # utime/chmod need ownership and die with EPERM. Recreate, copy bytes only.
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
        if os.path.exists(side):  # ONNX external-weights companion file
            _copy_into(side, bundle)
    # ONNX metadata copied out so the Jetson runtime self-configures even
    # when it loads a TensorRT .engine (which carries no ONNX metadata)
    import onnxruntime as ort
    meta = ort.InferenceSession(
        vm_onnx, providers=["CPUExecutionProvider"]).get_modelmeta().custom_metadata_map or {}
    # the runtime-relevant slice of the env config: control contract + camera
    edge_cfg = {
        "onnx_metadata": dict(meta),
        "control_dt": float(cfg.env.control_dt),
        "action_scale": float(cfg.env.action_scale),
        "gripper_open": float(cfg.env.gripper_open),
        "default_joint_pos": [float(v) for v in cfg.env.robot.default_joint_pos],
        "camera": OmegaConf.to_container(cfg.env.camera, resolve=True),
        "execute_horizon": int(cfg.visuomotor.execute_horizon),
        "onnx": os.path.basename(vm_onnx),
        # The SAFETY ENVELOPE the Raspberry Pi will enforce
        # (configs/safety/default.yaml -> deploy/raspi/safety.py). Shipping it
        # with the model is the same principle as shipping the camera spec:
        # no constant is ever hand-copied onto the robot, so the guard and the
        # simulator cannot drift apart (design notes §3).
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

    # --- BC champion: re-export from the robomimic checkpoint -------------
    bc_ckpt = _champion_checkpoint(cfg, "arm_bcrl006_bc_policy")
    if bc_ckpt and os.path.exists(bc_ckpt):
        onnx_path = os.path.join(deploy_dir, "arm_bcrl006_bc_policy.onnx")
        device = TorchUtils.get_torch_device(try_to_use_cuda=(cfg.device == "cuda"))
        val, lat = export_checkpoint_to_onnx(bc_ckpt, onnx_path, device)
        metrics = _confirm_onnx(cfg, onnx_path, "bc_deployed")
        report["bc"] = {"checkpoint": bc_ckpt, "onnx_path": onnx_path,
                        "onnx_validation": val, "latency": lat, "deployed_metrics": metrics}

    # --- RL champion: confirm the deployed ONNX artifact ------------------
    rl_onnx = os.path.join(deploy_dir, "arm_bcrl006_rl_policy.onnx")
    if os.path.exists(rl_onnx):
        metrics = _confirm_onnx(cfg, rl_onnx, "rl_deployed")
        report["rl"] = {"onnx_path": rl_onnx, "deployed_metrics": metrics}

    # --- visuomotor champion (the edge artifact): confirm + bundle --------
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

    # The deployment confirmation is the LAST measurement anyone makes, and it
    # is the one a stakeholder actually asks about ("does the thing we are
    # shipping work?"). Put it on the same W&B page as the training curves.
    from mlops.tracking import wandb_log_eval
    deployed = {k.upper(): v["deployed_metrics"] for k, v in report.items()
                if isinstance(v, dict) and "deployed_metrics" in v}
    if deployed:
        wandb_log_eval(cfg, f"{cfg.experiment_name}_deployed", deployed)

    with open(os.path.join(deploy_dir, "deploy_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("[infer] deploy report:", json.dumps(report, indent=2)[:1600])
    # A pipeline that "finished successfully" while producing nothing
    # deployable is the failure mode this whole phase exists to prevent.
    if report.get("release_gate") and not report["release_gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
