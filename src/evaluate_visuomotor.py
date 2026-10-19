# -*- coding: utf-8 -*-
"""ARM_BCRL006 フェーズ9。配備する視覚運動方策の閉ループ評価。
既定は ACT、visuomotor.type=diffusion なら Diffusion Policy。

書き出した ONNX を同じ環境で走らせる（カメラON、後退ホライズンの
チャンク実行はシミュレータ側の OnnxPolicy が面倒を見る）。そのうえで
スクリプトエキスパート vs BC vs BC+RL vs 配備する画像方策を並べ、
パイプラインの各段が何を買ったのかが見える形にする。
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


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    eval_dir = os.path.join(REPO_ROOT, "artifacts", "eval")
    os.makedirs(eval_dir, exist_ok=True)
    ptype = str(cfg.visuomotor.type)
    short = "dp" if ptype == "diffusion" else "act"
    onnx_path = os.path.join(REPO_ROOT, "artifacts", "deploy",
                             f"arm_bcrl006_{short}_policy.onnx")

    # 画像方策は1環境1ステップごとに1枚描くので、評価の規模は統計ではなく
    # 描画の予算で決まる。16環境 x 4エピソード = 64エピソードで約4分、
    # 状態方策の256エピソードは約20分。eval.visuomotor_* はその取引を
    # マジックナンバーに隠さず明示するためにある。数を変えたいときは
    # ここを触る。
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
