# -*- coding: utf-8 -*-
"""ARM_BCRL006 フェーズ6。BC方策から始める PPO 微調整。

デモで初期化する強化学習の型を、1箇所にまとめたもの:
  1. mujoco_sim/train_arm_mujoco.py --mode rl を叩く（学習前に最新の
     robomimic BC チェックポイントを PPO のアクターに読み込む。rl.bc_init）、
  2. BC の評価結果（src/evaluate.py）と、今回の RL の評価結果を
     読み直し、
  3. BC -> BC+RL の改善量と比較図を書く。

これが定番の形である理由。BC だけでは教師役（スクリプトIKエキスパート）
が上限になり、報酬が薄めの操作タスクで PPO をゼロから回すと、デモが
既に知っていたことを再発見するのに何時間も燃やす。PPO を BC の重みから
初期化すれば（同じMLP、同じ活性化、同じtanhヘッド、どちらも観測正規化
なし。configs/rl/ppo.yaml が強制する）両方の良いところが取れる。
そして差は主張ではなく実測する。``rl.bc_init=false`` で回せば、初期化が
何を買っているかを正確に数値化できる。
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
