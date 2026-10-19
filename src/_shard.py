# -*- coding: utf-8 -*-
"""シミュレータをN個のプロセスで並列に回し、分割結果を結合する。

存在理由。基準機（WSL2、RTX 5050 Laptop）で実測すると、MuJoCo の物理は
64環境で毎秒約4000環境ステップ進むのに、グラフィックスは 224x224 の
手首画像1枚に約20ms かかる。しかも EGL、GLFW、OSMesa のどれに替えても、
プロセスを増やしても目立って速くならない。だからデモ収集は2桁の差で
描画律速で、効く手は独立したシミュレータを同時に何本も走らせることだけ。
各ワーカーが自分の MjModel、MjData、GLコンテキストを持つので、共有する
ものは何も無く、競合するのは GPU/CPU のラスタライザ本体だけになる。
読み戻しを分散できるのが、この構成の効き目。

形としては定番の並列化そのもので、stable-baselines3 の ``SubprocVecEnv``、
DataLoader の ``num_workers``、キューの後ろに並ぶN個のPodと同じ。正しさに
関わるのはシードで、各シャードがシードをずらす
（``train_arm_mujoco.py::_shard_seed``）。だからシャードの和集合は、
同じエピソードのN本の複製ではなく、1つの独立同分布の標本になる。
"""

from __future__ import annotations

import os
import subprocess
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM = os.path.join(REPO_ROOT, "mujoco_sim", "train_arm_mujoco.py")
DATA_DIR = os.path.join(REPO_ROOT, "artifacts", "data")


def auto_workers(requested: int, cap: int = 6) -> int:
    """0 を指定すると自動。上限を設けているのは、律速が1つのラスタライザの
    共有だから。プロセスを6個より増やしても合計のフレームレートは伸びず、
    メモリだけ増える。"""
    if int(requested) > 0:
        return int(requested)
    return max(1, min(cap, (os.cpu_count() or 4) // 3))


def run_sharded(mode: str, workers: int, extra: list, out_stem: str,
                env_overrides: dict | None = None) -> list:
    """``workers`` 個のシミュレータプロセスを起動して待つ。-> シャードのパス一覧。"""
    procs = []
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "osmesa")
    env.update(env_overrides or {})
    for i in range(workers):
        cmd = [sys.executable, SIM, "--mode", mode,
               "--shard", str(i), "--num_shards", str(workers),
               "--out_name", out_stem] + list(extra)
        if workers == 1:
            print("[shard]", " ".join(cmd), flush=True)
        procs.append(subprocess.Popen(cmd, cwd=REPO_ROOT, env=env))
    if workers > 1:
        print(f"[shard] {workers} x {mode} running in parallel "
              f"(stem {out_stem})", flush=True)
    codes = [p.wait() for p in procs]
    bad = [i for i, c in enumerate(codes) if c != 0]
    if bad:
        raise RuntimeError(f"{mode} shard(s) {bad} failed with codes "
                           f"{[codes[i] for i in bad]}")
    stem = out_stem[:-3] if out_stem.endswith(".pt") else out_stem
    if workers == 1:
        return [os.path.join(DATA_DIR, f"{stem}.pt")]
    return [os.path.join(DATA_DIR, f"{stem}.shard{i}.pt") for i in range(workers)]


def merge_shards(paths: list, out_path: str, delete: bool = True) -> dict:
    """シャードごとの軌跡リストを1つのデータセットに連結する。"""
    merged: dict = {}
    for p in paths:
        d = torch.load(p, map_location="cpu", weights_only=False)
        for k, v in d.items():
            merged.setdefault(k, []).extend(v)
    if not merged:
        raise RuntimeError(f"no data in {paths}")
    if len(paths) == 1 and os.path.abspath(paths[0]) == os.path.abspath(out_path):
        return {"trajectories": len(merged["obs"]),
                "transitions": int(sum(o.shape[0] for o in merged["obs"]))}
    torch.save(merged, out_path + ".tmp")
    os.replace(out_path + ".tmp", out_path)
    if delete:
        for p in paths:
            if os.path.abspath(p) != os.path.abspath(out_path):
                os.remove(p)
    return {"trajectories": len(merged["obs"]),
            "transitions": int(sum(o.shape[0] for o in merged["obs"]))}


def merge_summaries(stem: str, workers: int, out_name: str) -> dict:
    """シャードごとのカウンタを合算して1つの要約にする。これで受け入れ判定は、
    何本に分割して走らせても1つのファイルを読めば済む。"""
    import json
    total = {"trajectories": 0, "kept_trajectories": 0, "dropped_trajectories": 0,
             "kept_transitions": 0, "attempts": 0, "generated": 0,
             "generated_transitions": 0}
    successes = 0
    found = 0
    for i in range(workers):
        name = f"{stem}_summary.json" if workers == 1 else f"{stem}_summary.shard{i}.json"
        p = os.path.join(DATA_DIR, name)
        if not os.path.exists(p):
            continue
        found += 1
        with open(p, encoding="utf-8") as f:
            s = json.load(f)
        for k in list(total):
            total[k] += int(s.get(k, 0) or 0)
        successes += float(s.get("source_success_rate", 0.0)) * int(s.get("trajectories", 0) or 0)
        if workers > 1:
            os.remove(p)
    total = {k: v for k, v in total.items() if v}
    if total.get("trajectories"):
        total["source_success_rate"] = successes / total["trajectories"]
    if total.get("attempts"):
        total["yield"] = total["generated"] / total["attempts"]
    total["shards"] = found
    with open(os.path.join(DATA_DIR, out_name), "w", encoding="utf-8") as f:
        json.dump(total, f, indent=2, ensure_ascii=False)
    return total
