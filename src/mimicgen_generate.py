# -*- coding: utf-8 -*-
"""ARM_BCRL006 phase 1.5 -- MimicGen demonstration generation.

Three steps, one command:

  1. ANNOTATE  mimicgen/annotate.py   phase-1 .pt -> object-centric HDF5
                                      (subtasks segmented on the gripper signal)
  2. GENERATE  --mode mimicgen        warp each subtask onto a freshly
                                      randomised scene, replay it, keep the
                                      episodes the env's own success term
                                      accepts  (sharded across processes)
  3. MERGE     source demos + generated demos -> the dataset phase 2 reads

WHAT THIS BUYS, concretely:
  * the demo SOURCE becomes swappable. ``mimicgen.source_file=real_demos.pt``
    points step 1 at a converted hand-guided recording of the REAL arm
    (mimicgen/import_real_demos.py) and the rest of the pipeline does not
    change. ``mimicgen.max_source_demos=10`` rehearses that regime in
    simulation BEFORE anyone touches hardware.
  * it is one arm of the 2x2 study (src/ablation.py): does generating more
    data from the same demonstrations actually raise closed-loop success?

The run STOPS if the yield is below ``mimicgen.min_yield``. A low yield is not
a nuisance, it is the measurement: it says the warped trajectories are not
executable from the new object poses, and training on the handful that survived
would hide that behind a slightly worse final number.
"""

from __future__ import annotations

import json
import os
import sys

import hydra
import torch
from omegaconf import DictConfig

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from mimicgen.annotate import load_source_demos, write_annotated_dataset  # noqa: E402
from src._shard import (auto_workers, merge_shards,       # noqa: E402
                        merge_summaries, run_sharded, DATA_DIR)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    mg = cfg.mimicgen
    out_path = os.path.join(DATA_DIR, str(mg.output_file))
    if not bool(mg.enable):
        # Leave no stale augmented dataset behind: phase 2 prefers it when it
        # exists, so a disabled phase 1.5 that left one lying around would
        # silently keep training on last run's augmentation.
        if os.path.exists(out_path):
            os.remove(out_path)
        print("[mimicgen] mimicgen.enable=false -- skipping (phase 2 will use "
              "the phase-1 dataset)")
        return

    source = os.path.join(DATA_DIR, str(mg.source_file))
    annotated = os.path.join(DATA_DIR, str(mg.annotated_file))
    print(f"[mimicgen] 1/3 annotate: {source} -> {annotated}")
    stats = write_annotated_dataset(
        load_source_demos(source), annotated, action_scale=float(cfg.env.action_scale),
        max_demos=int(mg.max_source_demos) or None)

    workers = auto_workers(int(mg.get("workers", 0)))
    per = max(1, int(mg.num_envs) // workers)
    extra = ["--num_envs", str(per), "--seed", str(int(cfg.seed)),
             "--set", f"mimicgen.num_envs={per}", f"mimicgen.episodes={mg.episodes}",
             f"mimicgen.annotated_file={mg.annotated_file}",
             f"mimicgen.max_source_demos={mg.max_source_demos}",
             f"mimicgen.selection={mg.selection}", f"mimicgen.nn_k={mg.nn_k}",
             f"mimicgen.interpolation_steps={mg.interpolation_steps}",
             f"mimicgen.action_noise={mg.action_noise}",
             f"env.domain_rand.enable={cfg.env.domain_rand.enable}",
             f"collect.camera={mg.get('camera', False)}",
             f"seed={cfg.seed}"]
    print(f"[mimicgen] 2/3 generate: {workers} worker(s) x {per} envs x "
          f"{int(mg.episodes)} batches ~ {workers * per * int(mg.episodes)} attempts")
    paths = run_sharded("mimicgen", workers, extra, "bc_demos_mimicgen_raw.pt")
    raw = os.path.join(DATA_DIR, "bc_demos_mimicgen_raw.pt")
    merge_shards(paths, raw)
    gen_stats = merge_summaries("bc_demos_mimicgen_raw", workers,
                                "mimicgen_generate_summary.json")

    yield_ = float(gen_stats.get("yield", 0.0))
    if yield_ < float(mg.min_yield):
        raise SystemExit(
            f"[mimicgen] ERROR: yield {yield_:.2f} < mimicgen.min_yield "
            f"{float(mg.min_yield)}. The warped trajectories are not executable from "
            "the new object poses. Likely causes, in order:\n"
            "  1. the source demos do not actually solve the task -- check "
            "artifacts/data/collect_summary.json source_success_rate\n"
            "  2. the new object poses are outside the region the source demos "
            "cover -- widen collect coverage or narrow env.object.spawn_*\n"
            "  3. mimicgen.interpolation_steps too small: the bridge between "
            "subtasks is too fast for the arm to track\n"
            "  4. a domain-randomisation axis the replay cannot survive (e.g. "
            "object_push_vel).\n"
            "To generate anyway and inspect the result: mimicgen.min_yield=0")

    print("[mimicgen] 3/3 merge")
    generated = torch.load(raw, map_location="cpu", weights_only=False)
    merged = generated
    if bool(mg.keep_source):
        src = torch.load(source, map_location="cpu", weights_only=False)
        keys = [k for k in ("obs", "act", "rgb") if k in src and k in generated]
        merged = {k: list(src[k]) + list(generated[k]) for k in keys}
    torch.save(merged, out_path + ".tmp")
    os.replace(out_path + ".tmp", out_path)
    os.remove(raw)

    summary = {
        "annotate": stats,
        "generate": gen_stats,
        "keep_source": bool(mg.keep_source),
        "source_demos": len(load_source_demos(source)["obs"]),
        "generated_demos": len(generated["obs"]),
        "output_demos": len(merged["obs"]),
        "output_transitions": int(sum(o.shape[0] for o in merged["obs"])),
        "output_file": out_path,
    }
    with open(os.path.join(DATA_DIR, "mimicgen_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("[mimicgen]", json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
