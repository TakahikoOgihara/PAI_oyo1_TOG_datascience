# -*- coding: utf-8 -*-
"""ARM_BCRL006 フェーズ4。BC方策の閉ループ評価。

robomimic で学習した BC 方策を、ランダム制御とスクリプトIKエキスパートと
同じランダム化エピソード群で比べる。環境はデモ収集に使ったのと同じ
MuJoCo（手法ごとに1プロセス立てるので、片方が落ちても他方の数値は
壊れない）。RL微調整のフェーズ（src/rl_finetune.py）は BC と BC+RL の
改善量を測るのにこの結果を読み直し、src/ablation.py は 4x2 格子のために
このスクリプトの形をそのまま使う。評価の入口は1つに保ってある。


ランダムとスクリプトエキスパートは飾りではない。ランダムは下限で、
それを明確に上回らない成功率は、方策が何も学んでいないという意味。
エキスパートは上限で、BC は教師役を超えられない。両者の差が、複製が
デモのどれだけを実際に捉えたかの、唯一正直な尺度になる。だから3つを
必ず同じ条件で並べて出す。
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
# 評価1本につきシミュレータを1プロセス。起動にはこのインタプリタ
# （sys.executable）を使うので、子プロセスも同じ仮想環境を継ぐ。
# ARM_BCRL005 では Isaac Sim 付属の python.sh が必要で、isaaclab.sh が
# 子の終了コードを飲むことまで注記する必要があった。ローカルでは
# どちらも起きず、check=True が実際にパイプラインを止めてくれる。
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

    # 閉ループの数値は W&B にも送る。学習曲線は「ロボットが仕事をするか」を
    # 関係者に伝えられず、それを伝えられるのはこの数値だけ
    # （mlops/tracking.py::wandb_log_eval）
    wandb_log_eval(cfg, f"{cfg.experiment_name}_eval_bc", results)

    with mlflow_run(cfg, run_name=f"{cfg.experiment_name}_eval"):
        for name, m in results.items():
            mlflow.log_metrics({f"{name}_{k}": v for k, v in m.items()
                               if isinstance(v, (int, float)) and v == v})
        mlflow.log_artifacts(out, artifact_path="eval")

    print("[eval] improvement:", json.dumps(improvement, indent=2))


if __name__ == "__main__":
    main()
