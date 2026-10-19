# -*- coding: utf-8 -*-
"""ARM_BCRL006 フェーズ7。蒸留。RLの優勝方策でカメラ付きデモを再収集する。

教師は状態方策で、シミュレータしか知らないワークの姿勢を読んでいる。
生徒は視覚運動方策で、手首カメラと自分の関節角しか見ない。実機にあるのは
それだけ。フェーズ7はその橋渡しで、教師を走らせて
(rgb, qpos) -> 教師の行動 を記録し、フェーズ8に渡す。

特権教師／センサ生徒の定番の型そのもので、配備する成果物が実行時に
ワークの姿勢を要らない理由がこれ。
"""

from __future__ import annotations

import json
import os
import sys

import hydra
from omegaconf import DictConfig

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src._shard import (auto_workers, merge_shards,      # noqa: E402
                        merge_summaries, run_sharded, DATA_DIR)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    d = cfg.visuomotor.distill
    workers = auto_workers(int(d.get("workers", 0)))
    extra = ["--num_envs", str(int(d.num_envs)), "--seed", str(int(cfg.seed)),
             "--set", f"visuomotor.distill.episodes={d.episodes}",
             f"visuomotor.distill.num_envs={d.num_envs}",
             f"visuomotor.distill.min_success={d.min_success}",
             f"seed={cfg.seed}"]
    print(f"[distill] {workers} worker(s) x {int(d.num_envs)} envs x {int(d.episodes)} "
          f"batches ~ {workers * int(d.num_envs) * int(d.episodes)} teacher rollouts")
    paths = run_sharded("distill", workers, extra, "visuomotor_demos.pt")
    stats = merge_shards(paths, os.path.join(DATA_DIR, "visuomotor_demos.pt"))
    summary = merge_summaries("visuomotor_demos", workers, "distill_summary.json")
    summary.update(stats)
    with open(os.path.join(DATA_DIR, "distill_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("[distill]", json.dumps(summary, indent=2))
    if summary.get("kept_trajectories", 0) < 32:
        raise SystemExit(
            f"[distill] ERROR: only {summary.get('kept_trajectories', 0)} successful "
            "teacher rollouts. ACT needs a few hundred; either the teacher is weak "
            "(check artifacts/eval/eval_report_rl_finetune.json) or "
            "visuomotor.distill.episodes is too small.")


if __name__ == "__main__":
    main()
