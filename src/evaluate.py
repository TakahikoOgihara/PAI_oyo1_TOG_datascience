# -*- coding: utf-8 -*-
"""ARM_BCRL006 phase 4 -- closed-loop evaluation of the BC policy.

Compares the robomimic-trained BC policy against random control and the
scripted IK expert on the same randomised-episode battery, in the SAME MuJoCo
environment used for demo collection (one simulator subprocess per method, so
a crash in one baseline cannot corrupt another's numbers). The RL fine-tuning
phase (src/rl_finetune.py) reuses these reports for the BC-vs-BC+RL
improvement measurement, and src/ablation.py reuses this whole script's shape
for its 4x2 grid.

RANDOM AND THE SCRIPTED EXPERT ARE NOT DECORATION. Random is the floor: a
success rate that does not clearly beat it means the policy learned nothing.
The expert is the ceiling: BC cannot exceed its demonstrator, so the gap
between them is the only honest measure of how much of the demonstration the
clone actually captured.
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


def _run_baseline(baseline: str, num_envs: int, episodes: int, max_steps: int,
                  onnx_path: str | None, tag: str | None = None) -> dict:
    out = os.path.join(REPO_ROOT, "artifacts", "eval")
    os.makedirs(out, exist_ok=True)
    tag = tag or baseline
    cmd = [sys.executable, SIM, "--num_envs", str(num_envs), "--mode", "eval_only",
           "--baseline", baseline, "--eval_out", tag,
           "--set", f"eval.episodes={episodes}", f"eval.max_steps={max_steps}"]
    if baseline == "checkpoint" and onnx_path:
        cmd += ["--onnx_path", onnx_path]
    env = _sim_env()
    print("[eval]", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
    with open(os.path.join(out, f"eval_report_{tag}.json"), encoding="utf-8") as f:
        return json.load(f)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    out = os.path.join(REPO_ROOT, "artifacts", "eval")
    os.makedirs(out, exist_ok=True)

    onnx_path = cfg.get("onnx_path", None) or os.path.join(
        REPO_ROOT, "artifacts", "deploy", "arm_bcrl006_bc_policy.onnx")
    print("[eval] BC onnx checkpoint:", onnx_path)

    results = {
        "random": _run_baseline("random", cfg.eval.num_envs, cfg.eval.episodes, cfg.eval.max_steps, None),
        "scripted_expert": _run_baseline("scripted_expert", cfg.eval.num_envs, cfg.eval.episodes,
                                         cfg.eval.max_steps, None),
        "BC": _run_baseline("checkpoint", cfg.eval.num_envs, cfg.eval.episodes, cfg.eval.max_steps,
                            onnx_path, tag="bc"),
    }
    for name, m in results.items():
        print(f"[eval] {name}: {json.dumps(m)}")

    bc = results["BC"]
    improvement = {}
    for base in ("random", "scripted_expert"):
        b = results[base]
        improvement[f"vs_{base}"] = {
            "success_rate_delta": bc["success_rate"] - b["success_rate"],
            "mean_max_lift_delta": bc["mean_max_lift"] - b["mean_max_lift"],
            "reward_delta": bc["mean_reward"] - b["mean_reward"],
        }

    report = {"onnx_path": onnx_path, "results": results, "improvement": improvement}
    with open(os.path.join(out, "eval_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    plotting.bar_compare(results, ["success_rate", "mean_max_lift", "mean_reward"],
                         os.path.join(out, "eval_comparison.png"),
                         title="ARM_BCRL006 BC vs baselines")

    # the closed-loop numbers go to W&B too: a training curve cannot tell a
    # stakeholder whether the robot does the task, and these are the only
    # numbers that can (mlops/tracking.py::wandb_log_eval)
    wandb_log_eval(cfg, f"{cfg.experiment_name}_eval_bc", results)

    with mlflow_run(cfg, run_name=f"{cfg.experiment_name}_eval"):
        for name, m in results.items():
            mlflow.log_metrics({f"{name}_{k}": v for k, v in m.items()
                               if isinstance(v, (int, float)) and v == v})
        mlflow.log_artifacts(out, artifact_path="eval")

    print("[eval] improvement:", json.dumps(improvement, indent=2))


if __name__ == "__main__":
    main()
