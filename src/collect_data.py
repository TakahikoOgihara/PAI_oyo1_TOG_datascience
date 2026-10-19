# -*- coding: utf-8 -*-
"""ARM_BCRL006 phase 1 -- scripted-expert demonstration collection.

No teleop rig, no manual demos: the demonstrations come from the IK state
machine inside mujoco_sim/ (the automated demo-generation pattern MimicGen
industrialises), with DART-style noise, a novice-operator mix and failure
filtering. Wrist-camera frames are recorded alongside, because phase 2's
image-quality gate and phase 8's visuomotor fallback both need them.

ARM_BCRL006 change vs ARM_BCRL005: the run is SHARDED across processes. Demo
collection is render-bound, not physics-bound, and rendering is embarrassingly
parallel -- see src/_shard.py for the measurement that motivates it.
"""

from __future__ import annotations

import json
import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from mlops.tracking import set_seed                      # noqa: E402
from src._shard import (auto_workers, merge_shards,      # noqa: E402
                        merge_summaries, run_sharded, DATA_DIR)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(DATA_DIR, "resolved_config.yaml"))
    set_seed(int(cfg.seed))

    workers = auto_workers(int(cfg.collect.get("workers", 0)))
    # the per-shard env count is the configured count divided among workers, so
    # `collect.num_envs` keeps meaning "how many environments in total"
    per = max(1, int(cfg.collect.num_envs) // workers)
    extra = ["--num_envs", str(per), "--seed", str(int(cfg.seed)),
             "--set", f"collect.episodes={cfg.collect.episodes}",
             f"collect.num_envs={per}",
             f"collect.action_noise={cfg.collect.action_noise}",
             f"collect.novice_ratio={cfg.collect.novice_ratio}",
             f"collect.keep_failed={cfg.collect.keep_failed}",
             f"collect.camera={cfg.collect.get('camera', True)}",
             f"env.domain_rand.enable={cfg.env.domain_rand.enable}",
             f"seed={cfg.seed}"]
    print(f"[collect] {workers} worker(s) x {per} envs x {cfg.collect.episodes} batches "
          f"~ {workers * per * int(cfg.collect.episodes)} trajectories, "
          f"domain_rand={bool(cfg.env.domain_rand.enable)}")
    paths = run_sharded("collect", workers, extra, "bc_demos.pt")
    stats = merge_shards(paths, os.path.join(DATA_DIR, "bc_demos.pt"))
    summary = merge_summaries("bc_demos", workers, "collect_summary.json")
    summary.update(stats)
    with open(os.path.join(DATA_DIR, "collect_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("[collect]", json.dumps(summary, indent=2))

    # The expert's own success rate is the CEILING on data quality: a policy
    # cloned from demonstrations cannot be better than the demonstrations. Stop
    # here rather than spending the rest of the pipeline on a bad dataset.
    floor = float(cfg.mimicgen.get("min_source_success", 0.7))
    if summary.get("source_success_rate", 1.0) < floor:
        raise SystemExit(
            f"[collect] ERROR: the scripted expert succeeded only "
            f"{summary['source_success_rate']:.2f} of the time (need >= {floor}). "
            "Fix the expert or the scene before training: check "
            "configs/env/mycobot_pickplace.yaml expert.* and any domain-"
            "randomisation axis you switched on.")


if __name__ == "__main__":
    main()
