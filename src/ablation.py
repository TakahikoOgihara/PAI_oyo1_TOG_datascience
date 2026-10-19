# -*- coding: utf-8 -*-
"""THE STUDY -- does MimicGen help? does domain randomisation help? do they
compose?

Four TRAINING conditions (the 2x2 over the two data-side interventions):

    A  baseline      mimicgen off   DR off
    B  mimicgen      mimicgen on    DR off
    C  dr            mimicgen off   DR on
    D  mimicgen_dr   mimicgen on    DR on

Each trained policy is evaluated on BOTH test environments:

    nominal      DR off -- "the bench exactly as modelled"
    randomised   DR on  -- "the bench with honest uncertainty in mass, friction,
                           servo gains, start pose, lighting and clutter"

EVALUATING ON BOTH IS THE POINT, and it is the part most single-number
comparisons get wrong. One test environment cannot separate "this intervention
made the policy better" from "this intervention made the test easier or
harder". Measured only on a nominal bench, a DR-trained policy typically looks
WORSE than a baseline that quietly overfitted to it -- and a team that stops
there ships the overfitted one. The 4x2 grid is what turns four numbers into a
conclusion you can act on:

    * the nominal column          how much capacity each intervention spent
    * the randomised column       how much robustness it bought
    * randomised - nominal        the sim-to-real-shaped gap, per condition
    * D vs B and D vs C           whether the two interventions compose or
                                  compete for the same fixed model capacity

Each cell is repeated over ``ablation.seeds``. A success rate measured once on
256 episodes already carries a +-3 pp standard error before seed variance is
counted at all; reporting mean and spread is the difference between a result
and an anecdote.

WHY THE COMPARISON RUNS ON THE BC STAGE (phases 1 -> 1.5 -> 2 -> 3 -> 4). Both
interventions act on the DATASET, and BC is the one stage whose output is a
pure function of the dataset. Inserting PPO and the ACT distillation between
cause and effect would add two stochastic stages to every cell and multiply the
cost by the whole pipeline instead of by seven minutes -- for a result that
would be harder, not easier, to attribute.

This script is an ORCHESTRATOR: it drives the same phase scripts ``run_all.sh``
drives, with Hydra overrides. Nothing about the pipeline is duplicated here, so
the numbers it reports are produced by the code that ships.

    python src/ablation.py                       # the full 4 x 3-seed grid (~80 min)
    python src/ablation.py ablation.seeds=[0]    # one seed (~27 min)
    python src/ablation.py ablation.match_data=true
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

import hydra
import numpy as np
from omegaconf import DictConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM = os.path.join(REPO_ROOT, "mujoco_sim", "train_arm_mujoco.py")
DATA_DIR = os.path.join(REPO_ROOT, "artifacts", "data")
EVAL_DIR = os.path.join(REPO_ROOT, "artifacts", "eval")
DEPLOY_DIR = os.path.join(REPO_ROOT, "artifacts", "deploy")

# name -> (mimicgen, DR, report label, extra Hydra overrides)
#
# The last four are a DECOMPOSITION of the DR axis, not part of the 2x2. They
# are off by default (ablation.enable_conditions) and exist because "DR helps"
# is not an actionable finding: the next question a client asks is WHICH part
# of it helped, and the answer decides what has to be measured on the real
# bench. `bash run_ablation.sh ablation.enable_conditions=[dr_pose,dr_physics]
# ablation.out_dir=artifacts/ablation_dr` runs it into its own directory.
_DR_PHYSICS_OFF = ["env.domain_rand.object_mass=[0.020,0.020]",
                   "env.domain_rand.object_friction=[1.5,1.5]",
                   "env.domain_rand.actuator_gain_scale=[1.0,1.0]",
                   "env.domain_rand.desk_color_jitter=0.0",
                   "env.domain_rand.object_color_jitter=0.0",
                   "env.domain_rand.light_intensity=[900.0,900.0]"]
# (mimicgen, DR, report label, extra Hydra overrides, collect-episode multiplier)
CONDITIONS = {
    "baseline":     (False, False, "A: neither", [], 1.0),
    "mimicgen":     (True, False, "B: MimicGen only", [], 1.0),
    "dr":           (False, True, "C: DR only", [], 1.0),
    "mimicgen_dr":  (True, True, "D: MimicGen + DR", [], 1.0),

    # --- opt-in: DATA-MATCHED controls -------------------------------------
    # MimicGen changes two things at once: how MANY rows the dataset has, and
    # how DIVERSE the object poses in it are. These two conditions collect
    # proportionally more REAL expert demonstrations instead, so the row count
    # matches the MimicGen conditions and only the diversity differs.
    #
    # The practical reading is the one that matters: in simulation, collecting
    # more expert demos is nearly free, so if A+ matches B then MimicGen bought
    # nothing HERE. On a real robot the same rows cost a person an afternoon
    # each -- which is the entire reason MimicGen exists. The comparison
    # measures the mechanism; the deployment context decides its value.
    "baseline_xl":  (False, False, "A+: more real demos (data-matched)", [], 2.8),
    "dr_xl":        (False, True, "C+: DR, more real demos (data-matched)", [], 2.8),

    # --- opt-in: DR decomposition ------------------------------------------
    # "DR helps" (or does not) is not actionable -- the next question is WHICH
    # part, and the answer decides what has to be measured on the real bench.
    "dr_pose":      (False, True, "C1: DR, start pose only", _DR_PHYSICS_OFF, 1.0),
    "dr_physics":   (False, True, "C2: DR, physics only",
                     ["env.domain_rand.joint_noise=0.0"], 1.0),
    "mg_dr_pose":   (True, True, "D1: MimicGen + pose DR", _DR_PHYSICS_OFF, 1.0),
    "mg_dr_physics": (True, True, "D2: MimicGen + physics DR",
                      ["env.domain_rand.joint_noise=0.0"], 1.0),
}
METRICS = ("success_rate", "mean_max_lift", "mean_place_err", "mean_reward")


def _run(cmd: list, tag: str) -> None:
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "osmesa")
    env.setdefault("PWD", REPO_ROOT)
    r = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-4000:])
        print(r.stderr[-4000:], file=sys.stderr)
        raise RuntimeError(f"{tag} failed (exit {r.returncode})")


def _py(script: str, *over: str) -> list:
    return [sys.executable, os.path.join(REPO_ROOT, script), *over]


def _subsample(path: str, max_transitions: int, seed: int) -> int:
    """Trim a dataset to at most ``max_transitions`` rows, whole demos only.

    ``ablation.match_data=true`` uses this to give every condition the same
    number of training rows. MimicGen adds data VOLUME as well as data
    DIVERSITY, and with matching off the B/D columns enjoy both; the gap
    between the two runs is exactly "how much of MimicGen's gain is just more
    rows". Trimming by whole trajectories, not by frames, keeps every episode
    intact -- BC trains on transitions but the dataset's unit of independence
    is the demonstration.
    """
    import torch
    d = torch.load(path, map_location="cpu", weights_only=False)
    order = np.random.RandomState(seed).permutation(len(d["obs"]))
    keep, total = [], 0
    for i in order:
        n = d["obs"][i].shape[0]
        if total + n > max_transitions:
            continue
        keep.append(int(i))
        total += n
    torch.save({k: [v[i] for i in keep] for k, v in d.items()}, path)
    return total


def run_cell(cfg: DictConfig, cond: str, seed: int, out_dir: str,
             cap_transitions: int | None) -> dict:
    mimic, dr, _label, extra, ep_scale = CONDITIONS[cond]
    a = cfg.ablation
    os.makedirs(out_dir, exist_ok=True)
    common = [f"seed={seed}", f"env.domain_rand.enable={dr}"] + list(extra)
    t0 = time.perf_counter()

    # -- phase 1: collect. Camera OFF: the study measures a STATE policy, and
    # rendering would cost 20x the wall clock for frames nothing here reads.
    episodes = max(1, round(int(a.collect_episodes) * float(ep_scale)))
    _run(_py("src/collect_data.py", *common,
             f"collect.episodes={episodes}", "collect.camera=false"),
         f"{cond}/collect")
    # -- phase 1.5: MimicGen (the first axis)
    _run(_py("src/mimicgen_generate.py", *common, f"mimicgen.enable={mimic}",
             "collect.camera=false"), f"{cond}/mimicgen")

    src_file = os.path.join(DATA_DIR, "bc_demos_mimicgen.pt" if mimic else "bc_demos.pt")
    if cap_transitions:
        kept = _subsample(src_file, cap_transitions, seed)
        print(f"    [{cond}] data-matched to {kept} transitions")
    # -- phase 2 + 3
    _run(_py("src/preprocess.py", *common, f"mimicgen.enable={mimic}"), f"{cond}/preprocess")
    _run(_py("scripts/train_bc_robomimic.py", *common,
             f"bc.num_epochs={int(a.bc_epochs)}"), f"{cond}/bc")

    # -- phase 4, on BOTH test environments. Same ONNX, same episode count,
    # same seed offset: only env.domain_rand.enable differs.
    onnx = os.path.join(DEPLOY_DIR, "arm_bcrl006_bc_policy.onnx")
    cell = {"condition": cond, "seed": seed, "mimicgen": mimic, "domain_rand": dr}
    for regime in list(a.eval_regimes):
        tag = f"abl_{cond}_s{seed}_{regime}"
        _run([sys.executable, SIM, "--mode", "eval_only", "--baseline", "checkpoint",
              "--onnx_path", onnx, "--eval_out", tag,
              "--num_envs", str(int(cfg.eval.num_envs)), "--seed", str(1000 + seed),
              "--eval_dr", "on" if regime == "randomised" else "off",
              # the TEST bench is the same for every condition: the decomposition
              # overrides shape the TRAINING data, never the battery
              "--set", f"eval.episodes={int(a.eval_episodes)}",
              f"eval.num_envs={int(cfg.eval.num_envs)}",
              f"eval.max_steps={int(cfg.eval.max_steps)}"], f"{cond}/eval_{regime}")
        with open(os.path.join(EVAL_DIR, f"eval_report_{tag}.json"), encoding="utf-8") as f:
            cell[regime] = json.load(f)

    with open(os.path.join(DATA_DIR, "preprocess_summary.json"), encoding="utf-8") as f:
        pre = json.load(f)
    cell["dataset"] = {"trajectories": pre["trajectories"],
                       "transitions": pre["after_outlier_clip"],
                       "source": pre["demo_source"]}
    cell["wall_clock_s"] = round(time.perf_counter() - t0, 1)

    # keep the per-cell evidence: the dataset summary, the BC ONNX and both
    # eval reports, so a reviewer can re-check any single number
    for name in ("collect_summary.json", "preprocess_summary.json", "mimicgen_summary.json"):
        p = os.path.join(DATA_DIR, name)
        if os.path.exists(p):
            shutil.copyfile(p, os.path.join(out_dir, name))
    shutil.copyfile(onnx, os.path.join(out_dir, "bc_policy.onnx"))
    with open(os.path.join(out_dir, "cell.json"), "w", encoding="utf-8") as f:
        json.dump(cell, f, indent=2, ensure_ascii=False)
    return cell


# ---------------------------------------------------------------------------
def aggregate(cells: list, regimes: list) -> dict:
    out = {}
    for cond in CONDITIONS:
        rows = [c for c in cells if c["condition"] == cond]
        if not rows:
            continue
        entry = {"label": CONDITIONS[cond][2],
                 "mimicgen": CONDITIONS[cond][0], "domain_rand": CONDITIONS[cond][1],
                 "seeds": [c["seed"] for c in rows],
                 "transitions": float(np.mean([c["dataset"]["transitions"] for c in rows])),
                 "trajectories": float(np.mean([c["dataset"]["trajectories"] for c in rows]))}
        for regime in regimes:
            for m in METRICS:
                v = np.array([c[regime][m] for c in rows], dtype=float)
                v = v[np.isfinite(v)]
                entry[f"{regime}/{m}"] = float(v.mean()) if len(v) else float("nan")
                entry[f"{regime}/{m}_std"] = float(v.std()) if len(v) else float("nan")
        if len(regimes) == 2:
            entry["robustness_gap"] = (entry["randomised/success_rate"]
                                       - entry["nominal/success_rate"])
        out[cond] = entry
    return out


def write_markdown(agg: dict, regimes: list, path: str, meta: dict,
                   cells: list | None = None) -> str:
    L = ["# ARM_BCRL006 -- MimicGen x Domain Randomisation ablation", "",
         f"- seeds per cell: {meta['seeds']}  (mean +- std over seeds)",
         f"- episodes per cell per regime: {meta['episodes']}",
         f"- BC epochs: {meta['bc_epochs']}   data-matched: {meta['match_data']}",
         "- policy under test: robomimic BC (low_dim), closed loop in MuJoCo", "",
         "| condition | MimicGen | DR | train demos | train rows | "
         + " | ".join(f"success ({r})" for r in regimes) + " | gap |",
         "|---|:--:|:--:|--:|--:|" + "--:|" * (len(regimes) + 1)]
    for cond, e in agg.items():
        cols = " | ".join(f"{e[f'{r}/success_rate']:.3f} ± {e[f'{r}/success_rate_std']:.3f}"
                          for r in regimes)
        gap = f"{e.get('robustness_gap', float('nan')):+.3f}"
        L.append(f"| {e['label']} | {'yes' if e['mimicgen'] else 'no'} | "
                 f"{'yes' if e['domain_rand'] else 'no'} | {e['trajectories']:.0f} | "
                 f"{e['transitions']:.0f} | {cols} | {gap} |")
    L += ["", "Secondary metrics (randomised test environment):", "",
          "| condition | place error [mm] | peak lift [mm] | mean return |",
          "|---|--:|--:|--:|"]
    last = regimes[-1]
    for cond, e in agg.items():
        L.append(f"| {e['label']} | {e[f'{last}/mean_place_err'] * 1000:.1f} | "
                 f"{e[f'{last}/mean_max_lift'] * 1000:.1f} | "
                 f"{e[f'{last}/mean_reward']:.0f} |")
    if cells:
        # Per-seed numbers, not just the mean. At the edge of feasibility the
        # seed-to-seed spread can exceed the effect being measured, and a
        # reader has to be able to see that rather than be told it.
        L += ["", "Per-seed success rate (every cell, no aggregation):", "",
              "| condition | seed | train rows | "
              + " | ".join(regimes) + " |",
              "|---|--:|--:|" + "--:|" * len(regimes)]
        for c in cells:
            L.append(f"| {CONDITIONS[c['condition']][2]} | {c['seed']} | "
                     f"{c['dataset']['transitions']} | "
                     + " | ".join(f"{c[r]['success_rate']:.3f}" for r in regimes) + " |")
    text = "\n".join(L) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def plot_grid(agg: dict, regimes: list, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conds = list(agg)
    x = np.arange(len(conds))
    w = 0.8 / max(1, len(regimes))
    fig, ax = plt.subplots(figsize=(1.9 * len(conds) + 3, 4.2))
    for j, r in enumerate(regimes):
        mean = [agg[c][f"{r}/success_rate"] for c in conds]
        std = [agg[c][f"{r}/success_rate_std"] for c in conds]
        b = ax.bar(x + j * w - 0.4 + w / 2, mean, w, yerr=std, capsize=4,
                   label=f"test: {r}", color=["#3182ce", "#c05621"][j % 2])
        for rect, v in zip(b, mean):
            ax.text(rect.get_x() + rect.get_width() / 2, v, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([agg[c]["label"] for c in conds], fontsize=9)
    ax.set_ylabel("closed-loop success rate")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    ax.set_title("MimicGen x domain randomisation: BC success rate\n"
                 "(mean ± std over seeds; same policy on both test environments)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    import mlflow
    from mlops.tracking import mlflow_run, wandb_log_eval

    a = cfg.ablation
    out_root = os.path.join(REPO_ROOT, str(a.out_dir))
    os.makedirs(out_root, exist_ok=True)
    conds = [c for c in list(a.enable_conditions) if c in CONDITIONS]
    regimes = list(a.eval_regimes)
    seeds = [int(s) for s in a.seeds]
    print(f"[ablation] {len(conds)} conditions x {len(seeds)} seeds x {len(regimes)} "
          f"test regimes = {len(conds) * len(seeds) * len(regimes)} closed-loop batteries")

    cap = None
    if bool(a.match_data):
        # the smallest condition is always the one with no MimicGen; measure it
        # once and cap everything to it
        print("[ablation] match_data=true -- probing the baseline dataset size first")
        probe = run_cell(cfg, "baseline", seeds[0],
                         os.path.join(out_root, "_probe"), None)
        cap = int(probe["dataset"]["transitions"])
        print(f"[ablation] capping every condition to {cap} transitions")

    cells = []
    t0 = time.perf_counter()
    for cond in conds:
        for seed in seeds:
            print(f"\n[ablation] === {CONDITIONS[cond][2]}  seed {seed} ===", flush=True)
            cell = run_cell(cfg, cond, seed, os.path.join(out_root, f"{cond}_s{seed}"), cap)
            cells.append(cell)
            line = "  ".join(f"{r}={cell[r]['success_rate']:.3f}" for r in regimes)
            print(f"[ablation] {cond} seed {seed}: {line}  "
                  f"({cell['dataset']['transitions']} rows, {cell['wall_clock_s']:.0f} s)",
                  flush=True)

    agg = aggregate(cells, regimes)
    meta = {"seeds": seeds, "episodes": int(a.eval_episodes) * int(cfg.eval.num_envs),
            "bc_epochs": int(a.bc_epochs), "match_data": bool(a.match_data),
            "collect_episodes": int(a.collect_episodes),
            "wall_clock_s": round(time.perf_counter() - t0, 1)}
    with open(os.path.join(out_root, "ablation_results.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "aggregate": agg, "cells": cells}, f,
                  indent=2, ensure_ascii=False)
    md = write_markdown(agg, regimes, os.path.join(out_root, "ablation.md"), meta, cells)
    plot_grid(agg, regimes, os.path.join(out_root, "ablation.png"))
    print("\n" + md)

    # both trackers: MLflow is the ledger a reviewer audits, W&B is the page a
    # stakeholder looks at
    with mlflow_run(cfg, run_name=f"{cfg.experiment_name}_ablation"):
        mlflow.log_params({f"ablation.{k}": v for k, v in meta.items()})
        for cond, e in agg.items():
            mlflow.log_metrics({f"{cond}.{k}": v for k, v in e.items()
                                if isinstance(v, (int, float)) and v == v})
        mlflow.log_artifacts(out_root, artifact_path="ablation")
    wandb_log_eval(cfg, f"{cfg.experiment_name}_ablation",
                   {f"{e['label']} [{r}]": {m: e[f"{r}/{m}"] for m in METRICS}
                    for r in regimes for e in agg.values()})
    print(f"[ablation] artifacts: {out_root} "
          f"(ablation.md, ablation.png, ablation_results.json)")


if __name__ == "__main__":
    main()
