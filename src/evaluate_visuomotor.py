# -*- coding: utf-8 -*-
"""ARM_BCRL006 phase 9 -- closed-loop evaluation of the deployed visuomotor
policy (ACT by default, Diffusion Policy when visuomotor.type=diffusion).

Runs the exported ONNX in the SAME PhysX env (camera on, receding-horizon
chunk execution handled by the sim script's OnnxPolicy), then assembles the
full story: scripted expert vs BC vs BC+RL vs the deployed image policy --
the numbers that show what each pipeline stage bought.
"""

from __future__ import annotations
import json
import os
import sys
import subprocess

import hydra
import mlflow
from omegaconf import DictConfig

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from mlops import plotting
from mlops.tracking import mlflow_run, wandb_log_eval

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


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    eval_dir = os.path.join(REPO_ROOT, "artifacts", "eval")
    os.makedirs(eval_dir, exist_ok=True)
    ptype = str(cfg.visuomotor.type)
    short = "dp" if ptype == "diffusion" else "act"
    onnx_path = os.path.join(REPO_ROOT, "artifacts", "deploy",
                             f"arm_bcrl006_{short}_policy.onnx")

    # An image policy renders a frame per environment per step, so its battery
    # is sized by the graphics budget, not by statistics: 16 envs x 4 episodes
    # = 64 episodes ~ 4 min, against 256 episodes ~ 20 min for the state
    # policies. eval.visuomotor_* keeps that trade explicit instead of hidden
    # in a magic number (operations guide §5).
    n_envs = int(cfg.eval.get("visuomotor_num_envs", cfg.eval.num_envs))
    n_eps = int(cfg.eval.get("visuomotor_episodes", cfg.eval.episodes))
    cmd = [sys.executable, SIM, "--num_envs", str(n_envs),
           "--mode", "eval_only", "--baseline", "checkpoint",
           "--onnx_path", onnx_path, "--eval_out", short,
           "--set", f"eval.episodes={n_eps}", f"eval.num_envs={n_envs}",
           f"eval.max_steps={cfg.eval.max_steps}",
           f"visuomotor.execute_horizon={cfg.visuomotor.execute_horizon}"]
    env = _sim_env()
    print("[eval_visuomotor]", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)

    with open(os.path.join(eval_dir, f"eval_report_{short}.json"), encoding="utf-8") as f:
        vm_metrics = json.load(f)

    results = {short.upper(): vm_metrics}
    for name, fname in (("scripted_expert", "eval_report_scripted_expert.json"),
                        ("BC", "eval_report_bc.json"),
                        ("BC_RL_finetune", "eval_report_rl_finetune.json")):
        p = os.path.join(eval_dir, fname)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                results[name] = json.load(f)

    improvement = {}
    for base in ("BC", "BC_RL_finetune"):
        if base in results:
            b = results[base]
            improvement[f"vs_{base}"] = {
                "success_rate_delta": vm_metrics["success_rate"] - b["success_rate"],
                "mean_max_lift_delta": vm_metrics["mean_max_lift"] - b["mean_max_lift"],
            }

    report = {"onnx_path": onnx_path, "results": results, "improvement": improvement}
    with open(os.path.join(eval_dir, "eval_report_visuomotor.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    plotting.bar_compare(results, ["success_rate", "mean_max_lift", "mean_reward"],
                         os.path.join(eval_dir, "visuomotor_comparison.png"),
                         title=f"ARM_BCRL006: pipeline stages vs deployed {short.upper()}")

    wandb_log_eval(cfg, f"{cfg.experiment_name}_eval_{short}", results)

    with mlflow_run(cfg, run_name=f"{cfg.experiment_name}_eval_{short}"):
        for name, m in results.items():
            mlflow.log_metrics({f"{name}_{k}": v for k, v in m.items()
                                if isinstance(v, (int, float)) and v == v})
        mlflow.log_artifacts(eval_dir, artifact_path="eval")

    print("[eval_visuomotor] improvement:", json.dumps(improvement, indent=2))


if __name__ == "__main__":
    main()
