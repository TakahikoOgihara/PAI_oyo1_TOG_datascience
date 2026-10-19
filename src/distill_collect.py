# -*- coding: utf-8 -*-
"""ARM_BCRL006 phase 7 -- DISTILL: the RL champion re-collects camera demos.

The teacher is a STATE policy: it reads the block pose, which only the
simulator knows. The student is a VISUOMOTOR policy: it sees a wrist camera and
its own joint angles, which is all the real robot has. Phase 7 is the bridge --
run the teacher, record (rgb, qpos) -> teacher action, and hand that to phase 8.

This is the standard privileged-teacher / sensor-student pattern, and it is the
reason the deployed artifact needs no object pose at runtime.
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
