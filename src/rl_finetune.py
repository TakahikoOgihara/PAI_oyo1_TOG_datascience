# -*- coding: utf-8 -*-
"""ARM_BCRL006 phase 6 -- RSL-RL PPO fine-tuning FROM the BC policy.

This is the demo-bootstrapped RL pattern in one place:
  1. drive mujoco_sim/train_arm_mujoco.py --mode rl (which loads the newest
     robomimic BC checkpoint into the PPO actor before training -- rl.bc_init),
  2. read back the BC eval report (src/evaluate.py) and the fresh RL eval
     report, and
  3. write the BC -> BC+RL improvement measurement + comparison chart.

Why this is the mainstream shape: BC alone is capped by its demonstrator
(the scripted IK expert), and PPO-from-scratch on a sparse-ish manipulation
task burns hours rediscovering what the demos already knew. Initialising PPO
from the BC weights (identical MLP, identical activation, identical tanh head,
no obs normalisation on either side -- enforced by configs/rl/ppo.yaml) gets
the best of both, and the delta is measured, not asserted. Run with
``rl.bc_init=false`` to quantify exactly what the initialisation buys.
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
from mlops.tracking import mlflow_run

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

    cmd = [sys.executable, SIM, "--num_envs", str(cfg.rl.num_envs), "--mode", "rl",
           "--set", f"rl.total_iterations={cfg.rl.total_iterations}",
           f"rl.bc_init={cfg.rl.bc_init}", f"rl.lr={cfg.rl.lr}",
           f"eval.episodes={cfg.eval.episodes}", f"eval.max_steps={cfg.eval.max_steps}"]
    # eval.episodesと、eval.max_stepsは、
    # 評価する試行回数と、1試行を最大何ステップ動かすかの違い

    env = _sim_env()
    print("[rl_finetune]", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)

    with open(os.path.join(eval_dir, "eval_report_rl_finetune.json"), encoding="utf-8") as f:
        rl_metrics = json.load(f)

    results = {"BC_RL_finetune": rl_metrics}
    bc_report_path = os.path.join(eval_dir, "eval_report_bc.json")
    improvement = {}
    if os.path.exists(bc_report_path):
        with open(bc_report_path, encoding="utf-8") as f:
            bc_metrics = json.load(f)
        results["BC"] = bc_metrics
        improvement["vs_BC"] = {
            "success_rate_delta": rl_metrics["success_rate"] - bc_metrics["success_rate"],
            "mean_max_lift_delta": rl_metrics["mean_max_lift"] - bc_metrics["mean_max_lift"],
            "reward_delta": rl_metrics["mean_reward"] - bc_metrics["mean_reward"],
        }
    else:
        print("[rl_finetune] no BC eval report found (run src/evaluate.py first) -- "
              "skipping the BC-vs-RL improvement measurement")

    report = {"results": results, "improvement": improvement}
    with open(os.path.join(eval_dir, "eval_report_bc_vs_rl.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    if "BC" in results:
        plotting.bar_compare(results, ["success_rate", "mean_max_lift", "mean_reward"],
                             os.path.join(eval_dir, "bc_vs_rl_comparison.png"),
                             title="ARM_BCRL006: BC vs BC+RL fine-tune")

    with mlflow_run(cfg, run_name=f"{cfg.experiment_name}_bc_vs_rl"):
        for name, m in results.items():
            mlflow.log_metrics({f"{name}_{k}": v for k, v in m.items()
                               if isinstance(v, (int, float)) and v == v})
        mlflow.log_artifacts(eval_dir, artifact_path="eval")
        # artifact_path="eval"とは:
        # MLflowのartifact領域内で、どのフォルダ名の下に保存するかを指定

    print("[rl_finetune] improvement:", json.dumps(improvement, indent=2))


if __name__ == "__main__":
    main()
