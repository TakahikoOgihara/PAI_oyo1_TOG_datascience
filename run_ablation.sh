#!/bin/bash
# =====================================================================
# 比較実験: {MimicGen off/on} x {ドメインランダム化 off/on} の 2x2。
# 学習した各方策を、公称ベンチとランダム化ベンチの両方で評価する。
# 両方で測る理由と実験設計の意図は src/ablation.py の docstring に
# まとめてある。
#
# 実行するのは run_all.sh と同じフェーズスクリプトで、変えるのは Hydra の
# 上書きだけ。パイプラインを複製していないので、出てくる数値は出荷して
# いるコードが出した数値そのもの。
#
#   bash run_ablation.sh                        # 4条件 x 3シード（約80分）
#   bash run_ablation.sh ablation.seeds=[0]     # 1シード（約27分）
#   bash run_ablation.sh ablation.match_data=true
#                                               # 条件間でデータセットの件数を
#                                               # 揃え、MimicGen の多様性と
#                                               # 量を切り分ける
#
# 出力先 -> artifacts/ablation/
#   ablation.md            レポートに載せる表
#   ablation.png           シード間の平均±標準偏差の棒グラフ
#   ablation_results.json  全セル・全指標・全シードの生データ
#   <condition>_s<seed>/   セルごとの証跡（データセット要約と ONNX）
# さらに MLflow の run 1件と W&B の評価ページ1件。
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")"
export PWD="$(pwd)"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
PY="${PY:-$(command -v python)}"

echo "############ ASSETS ############"
$PY assets/build_mycobot_mjcf.py --verify
echo "############ 2x2 ABLATION ############"
$PY src/ablation.py "$@"
echo
echo "artifacts/ablation/ablation.md:"
cat artifacts/ablation/ablation.md
