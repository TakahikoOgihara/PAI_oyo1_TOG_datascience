# -*- coding: utf-8 -*-
"""ARM_BCRL006 フェーズ1。スクリプトエキスパートによるデモ収集。

遠隔操作の装置も手作業のデモも使わない。デモは mujoco_sim/ の中のIK状態
機械が作る（MimicGen が産業化した自動デモ生成のやり方）。DART流のノイズ、
初心者オペレータの混入、失敗の除去付き。手首カメラの画像も一緒に記録
する。フェーズ2の画質ゲートとフェーズ8の視覚運動方策が、どちらもそれを
必要とするから。

ARM_BCRL005 からの変更点。実行をプロセス分割した。デモ収集は物理ではなく
描画が律速で、描画は素直に並列化できる。その実測と分割の詳細は
src/_shard.py にある。
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
    # シャードごとの環境数は、設定値をワーカーで割った数。これで
    # `collect.num_envs` の意味が「合計で何環境か」のままになる
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

    # エキスパート自身の成功率が、データ品質の上限になる。デモから複製した
    # 方策がデモより上手くなることはない。悪いデータセットに残りの
    # パイプラインを費やす前に、ここで止める。
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
