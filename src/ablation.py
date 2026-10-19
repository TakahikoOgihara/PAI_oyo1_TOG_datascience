# -*- coding: utf-8 -*-
"""比較実験。MimicGen は効くのか、ドメインランダム化は効くのか、
その2つは組み合わさるのか。

学習側の4条件（データ側の2つの介入の 2x2）:

    A  baseline      mimicgen なし   DR なし
    B  mimicgen      mimicgen あり   DR なし
    C  dr            mimicgen なし   DR あり
    D  mimicgen_dr   mimicgen あり   DR あり

学習した方策は、2つの試験環境の両方で評価する:

    nominal      DR なし。「モデル化したとおりのベンチ」
    randomised   DR あり。「質量、摩擦、サーボゲイン、初期姿勢、照明、
                           置き物に、正直な不確かさを入れたベンチ」

両方で測ることが要点で、単一の数値で比べる議論がいちばん外すところ。
試験環境が1つだと「介入で方策が良くなった」と「介入で試験が易しく
（難しく）なった」を区別できない。公称ベンチだけで測ると、DRで学習した
方策は、そこに静かに過適合した素のベースラインより悪く見えるのが普通で、
そこで止めたチームは過適合した方を出荷する。4x2 の格子は、4つの数字を
行動に移せる結論に変えるためにある。読み方は次のとおり:


    * nominal 列            各介入がモデル容量をどれだけ使ったか
    * randomised 列         それでどれだけ頑健性を買えたか
    * randomised - nominal  条件ごとの、実機移行で問題になる形の落差
    * D vs B、D vs C        2つの介入が足し合うのか、同じ有限の容量を
                                  取り合うのか

各セルは ``ablation.seeds`` の分だけ繰り返す。256エピソードで1回測った
成功率には、シード間のばらつきを数える前から±3ポイントの標準誤差が
ある。平均とばらつきを併記するかどうかが、結果と逸話の違いになる。


比較を BC 段（フェーズ 1 -> 1.5 -> 2 -> 3 -> 4）で回す理由。2つの介入は
どちらもデータセットに作用し、BC は出力がデータセットの純粋な関数に
なる唯一の段だから。原因と結果の間に PPO と ACT の蒸留を挟むと、各セルに
確率的な段が2つ増え、コストが7分ではなくパイプライン全体の倍率で乗る。
しかも結果の帰属は易しくなるどころか難しくなる。だからここは BC 段に
絞っている。

このスクリプトは指揮役で、``run_all.sh`` が叩くのと同じフェーズ
スクリプトを Hydra 上書き付きで叩くだけ。パイプラインは複製していない
ので、報告する数値は出荷しているコードが出した数値そのもの。

    python src/ablation.py                       # 4条件 x 3シードの全格子（約80分）
    python src/ablation.py ablation.seeds=[0]    # 1シード（約27分）
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

# 名前 -> (mimicgen, DR, 表示ラベル, 追加の Hydra 上書き)
#
# 後ろの4つは DR 軸の分解で、2x2 の一部ではない。既定ではOFF
# （ablation.enable_conditions）。「DRは効く」だけでは行動に移せない
# からで、次に聞かれるのは「DRのどの部分が効いたのか」であり、その
# 答えが実機ベンチで何を測るべきかを決める。
# bench. `bash run_ablation.sh ablation.enable_conditions=[dr_pose,dr_physics]
# ablation.out_dir=artifacts/ablation_dr` runs it into its own directory.
_DR_PHYSICS_OFF = ["env.domain_rand.object_mass=[0.020,0.020]",
                   "env.domain_rand.object_friction=[1.5,1.5]",
                   "env.domain_rand.actuator_gain_scale=[1.0,1.0]",
                   "env.domain_rand.desk_color_jitter=0.0",
                   "env.domain_rand.object_color_jitter=0.0",
                   "env.domain_rand.light_intensity=[900.0,900.0]"]
# (mimicgen, DR, 表示ラベル, 追加の Hydra 上書き, 収集エピソード倍率)
CONDITIONS = {
    "baseline":     (False, False, "A: neither", [], 1.0),
    "mimicgen":     (True, False, "B: MimicGen only", [], 1.0),
    "dr":           (False, True, "C: DR only", [], 1.0),
    "mimicgen_dr":  (True, True, "D: MimicGen + DR", [], 1.0),

    # --- 任意: データ量を揃えた対照条件 ------------------------------------
    # MimicGen は2つのことを同時に変える。データセットの行数と、その中の
    # 物体姿勢の多様性。この2条件はその代わりに本物のエキスパートデモを
    # 比例して多く集めるので、行数は MimicGen 条件と揃い、違うのは
    # 多様性だけになる。
    #
    # 実務上の読み方はこう。シミュレータではエキスパートデモの追加収集が
    # ほぼ無料なので、A+ が B に並ぶなら、ここでは MimicGen は何も買って
    # いない。実機では同じ行数が1本あたり人の半日を食う。それが MimicGen の
    # 存在理由そのもの。比較が測るのは仕組みで、その価値を決めるのは
    # 使う場面。
    "baseline_xl":  (False, False, "A+: more real demos (data-matched)", [], 2.8),
    "dr_xl":        (False, True, "C+: DR, more real demos (data-matched)", [], 2.8),

    # --- 任意: DR の分解 ----------------------------------------------------
    # 「DRは効く（効かない）」では行動に移せない。次の問いは「どの部分が」で、
    # その答えが実機ベンチで何を測るべきかを決める。
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
    """データセットを最大 ``max_transitions`` 行に切る。デモ単位でしか切らない。

    ``ablation.match_data=true`` がこれを使って、全条件の学習行数を揃える。
    MimicGen はデータの多様性だけでなく量も増やすので、揃えないと B/D 列は
    両方の恩恵を受ける。揃えた実行との差が、そのまま「MimicGen の利得の
    うち、どれだけが単に行数のおかげか」になる。フレーム単位ではなく
    軌跡単位で切るのは、エピソードを壊さないため。BC が学習するのは
    遷移だが、データセットの独立の単位はデモ1本なので、切るのは必ず
    デモの切れ目にする。
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

    # -- フェーズ1: 収集。カメラはOFF。この実験が測るのは状態方策で、
    # ここで誰も読まない画像を描くと実時間が20倍になる。
    episodes = max(1, round(int(a.collect_episodes) * float(ep_scale)))
    _run(_py("src/collect_data.py", *common,
             f"collect.episodes={episodes}", "collect.camera=false"),
         f"{cond}/collect")
    # -- フェーズ1.5: MimicGen（1つ目の軸）
    _run(_py("src/mimicgen_generate.py", *common, f"mimicgen.enable={mimic}",
             "collect.camera=false"), f"{cond}/mimicgen")

    src_file = os.path.join(DATA_DIR, "bc_demos_mimicgen.pt" if mimic else "bc_demos.pt")
    if cap_transitions:
        kept = _subsample(src_file, cap_transitions, seed)
        print(f"    [{cond}] data-matched to {kept} transitions")
    # -- フェーズ2 + 3
    _run(_py("src/preprocess.py", *common, f"mimicgen.enable={mimic}"), f"{cond}/preprocess")
    _run(_py("scripts/train_bc_robomimic.py", *common,
             f"bc.num_epochs={int(a.bc_epochs)}"), f"{cond}/bc")

    # -- フェーズ4。2つの試験環境の両方で。同じONNX、同じエピソード数、
    # 同じシードずらしで、違うのは env.domain_rand.enable だけ。
    onnx = os.path.join(DEPLOY_DIR, "arm_bcrl006_bc_policy.onnx")
    cell = {"condition": cond, "seed": seed, "mimicgen": mimic, "domain_rand": dr}
    for regime in list(a.eval_regimes):
        tag = f"abl_{cond}_s{seed}_{regime}"
        _run([sys.executable, SIM, "--mode", "eval_only", "--baseline", "checkpoint",
              "--onnx_path", onnx, "--eval_out", tag,
              "--num_envs", str(int(cfg.eval.num_envs)), "--seed", str(1000 + seed),
              "--eval_dr", "on" if regime == "randomised" else "off",
              # 試験ベンチは全条件で同じ。分解用の上書きが変えるのは学習データで、
              # 評価側は変えない
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

    # セルごとの証跡を残す。データセット要約、BCのONNX、両方の評価結果。
    # 査読者が任意の数値を1つずつ再確認できるようにするため
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
        # 平均だけでなくシードごとの数値も出す。実現可能性の境界では、シード間
        # のばらつきが測ろうとしている効果を超えることがある。それは読む側が
        # 自分で見られるようにすべきで、言葉で言い添えるものではない。
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
        # いちばん小さい条件は必ず MimicGen 無しの条件。それを1回測って、
        # 全条件をその値に揃える
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

    # トラッカーは両方使う。MLflow は査読者が監査する台帳、W&B は
    # 関係者が眺めるページ。
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
