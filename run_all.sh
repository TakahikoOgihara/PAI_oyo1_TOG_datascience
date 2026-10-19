#!/bin/bash
# =====================================================================
# ARM_BCRL006 の一気通貫パイプライン。ローカルPC 1台で動く。コンテナ不要、
# GPUサーバー不要、クラウド不要。物理は MuJoCo（CPU）で、GPUを使うのは学習
# フェーズだけ。
#
#   0) assets      myCobot 280 Pi の MJCF を deploy/raspi/mycobot_fk.py の
#                  関節原点から生成し、MuJoCo 自身の順運動学と突き合わせて検証
#                  （ダウンロードもメッシュも不要）
#   1) collect     スクリプトIKエキスパートでデモ収集。手首カメラも記録し、
#                  プロセス分割で並列に回す
#  1.5) MIMICGEN   既定ではOFF。このフェーズはスキップ行を出して終わる。
#                  mimicgen.enable=true にすると、サブタスクを注釈して
#                  物体中心の各区間を新しくランダム化したシーンへ転写し、
#                  再生して環境自身の成功判定が通ったものだけ残す。
#                  デモの出どころは差し替え可能で、実機デモ10本程度でも
#                  そのまま動く（configs/mimicgen/default.yaml 参照）。
#                  MimicGen が本来効くのはシミュレータではなくそちらの状況。
#                  ここでは生成1本の価値が収集1本の約0.6と実測されたので、
#                  既定は収集を増やす側にした（数値は artifacts/ablation/）。
#   2) preprocess  外れ値クリップ + robomimic HDF5 + 画質ゲート
#                  -> 手首カメラ画像が使えないならここで止める
#   3) BC          robomimic の BC（low_dim）-> ONNX + MLflow 登録
#   4) eval        閉ループ評価: ランダム / スクリプトエキスパート / BC
#   5) LeRobot     データセット書き出し（任意。未インストールなら自動スキップ）
#   6) RL          BCの重みから初期化した stable-baselines3 の PPO。
#                  4つの安定化策 + 閉ループでのチェックポイント選択 +
#                  BCに対する性能劣化ガード付き
#   7) distill     RLの優勝方策で手首カメラ付きデモを再収集
#   8) visuomotor  ACT（既定）/ Diffusion Policy -> ONNX + 登録
#   9) eval        配備する画像方策の閉ループ評価
#  10) deploy      リリースゲート -> Jetson/Pi 向けエッジ配布物
#
# ゲートは2つ（フェーズ2とフェーズ10）。途中で止まるのが正しい挙動で、
# 動かない成果物をロボットに届けないために置いてある。
#
# 比較実験は別コマンド。パイプラインには含めない:
#   bash run_ablation.sh          # MimicGen x ドメインランダム化の 2x2
#
# 設定は Hydra（configs/config.yaml と configs/*/）。何でも上書きできる:
#   bash run_all.sh bc.num_epochs=100 rl.total_iterations=200
#   bash run_all.sh visuomotor.type=diffusion       # Diffusion Policy 版
#   bash run_all.sh rl.bc_init=false                # PPOをゼロから回す比較
#   bash run_all.sh mimicgen.enable=true            # フェーズ1.5をON（既定OFF）
#   bash run_all.sh mimicgen.max_source_demos=10 mimicgen.selection=random
#                                                   # 実機デモが少ない状況を模す
#   bash run_all.sh env.domain_rand.joint_friction_scale=[0.8,1.3]
#                                                   # 既定OFFのDR軸を有効化
#   bash run_all.sh collect.workers=8               # レンダリング分割を増やす
#
# スモークテスト（約6分。配線の確認であって、方策の性能は何も示さない）:
#   bash run_all.sh collect.episodes=1 mimicgen.episodes=1 bc.num_epochs=5 \
#        rl.total_iterations=6 rl.critic_warmup_iters=2 rl.bc_anchor_decay_iters=4 \
#        rl.save_interval=2 visuomotor.distill.episodes=1 visuomotor.train.epochs=2 \
#        visuomotor.train.batch_size=8 visuomotor.train.val_split=0.3 \
#        eval.episodes=1 eval.num_envs=8 eval.visuomotor_num_envs=4 \
#        mimicgen.min_yield=0 deploy.force=true
#
# 使い方:  bash run_all.sh [追加の hydra 上書き...]
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")"
export PWD="$(pwd)"
# MuJoCo のオフスクリーン描画バックエンド。osmesa はディスプレイサーバも
# ドライバのEGLスタックも要らないので、WSL2、コンテナ、SSH越しで同じ結果に
# なる。egl / glfw も動くし差は20%以内。律速はフレームごとの読み戻しで、
# ラスタライズではない。
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
PY="${PY:-$(command -v python)}"
if [ -z "$PY" ]; then echo "no python on PATH -- activate the venv (see setup_env.sh)"; exit 1; fi
EXTRA_ARGS=("$@")

echo "############ 0/10  ASSETS (myCobot 280 Pi -> MJCF, FK cross-checked) ############"
$PY assets/build_mycobot_mjcf.py --verify

echo "############ 1/10  DEMONSTRATION COLLECTION (scripted IK expert, wrist camera) ############"
echo "[run_all] >>> phase 1 is the MimicGen SOURCE. To use real-robot demos"
echo "[run_all] >>> instead, see configs/mimicgen/default.yaml and set"
echo "[run_all] >>> mimicgen.source_file=<converted dataset>"
$PY src/collect_data.py "${EXTRA_ARGS[@]}"

echo "############ 1.5/10  MIMICGEN DEMONSTRATION GENERATION ############"
$PY src/mimicgen_generate.py "${EXTRA_ARGS[@]}"

echo "############ 2/10  PREPROCESS + ROBOMIMIC HDF5 + IMAGE-QUALITY GATE ############"
$PY src/preprocess.py "${EXTRA_ARGS[@]}"
echo "[run_all] >>> gate results: artifacts/data/preprocess_summary.json"
echo "[run_all] >>> and review artifacts/data/camera_samples.png against the real"
echo "[run_all] >>> camera (camera spec check)"

echo "############ 3/10  ROBOMIMIC BC TRAINING + ONNX + MLflow REGISTRY ############"
$PY scripts/train_bc_robomimic.py "${EXTRA_ARGS[@]}"

echo "############ 4/10  BC EVALUATION vs BASELINES ############"
$PY src/evaluate.py "${EXTRA_ARGS[@]}"

echo "############ 5/10  LEROBOT DATASET EXPORT (optional) ############"
$PY lerobot_export/export_lerobot.py \
  || echo "[run_all] WARNING: LeRobot export failed/skipped (pip install lerobot to enable)"

echo "############ 6/10  PPO FINE-TUNE FROM BC + IMPROVEMENT REPORT ############"
$PY src/rl_finetune.py "${EXTRA_ARGS[@]}"

echo "############ 7/10  DISTILL: RL-CHAMPION CAMERA DEMOS (teacher-student) ############"
$PY src/distill_collect.py "${EXTRA_ARGS[@]}"

echo "############ 8/10  VISUOMOTOR TRAINING (ACT / Diffusion Policy) + ONNX ############"
$PY scripts/train_visuomotor.py "${EXTRA_ARGS[@]}"

echo "############ 9/10  VISUOMOTOR CLOSED-LOOP EVALUATION + COMPARISON ############"
$PY src/evaluate_visuomotor.py "${EXTRA_ARGS[@]}"

echo "############ 10/10  DEPLOY: RELEASE GATE + REGISTRY CHAMPIONS + EDGE BUNDLE ############"
$PY src/infer.py "${EXTRA_ARGS[@]}"

echo "############ DONE. artifacts/ ############"
find artifacts -maxdepth 2 \( -name "*.json" -o -name "*.png" -o -name "*.onnx" \) | sort

cat <<'EOF'

Next:
  mlflow ui / tensorboard --logdir artifacts/tb   read the runs and decide
  bash run_ablation.sh    the 2x2 MimicGen x domain-randomisation study
  python src/record_video.py            the submission video
  python mujoco_sim/train_arm_mujoco.py --mode play      watch it live
Real robot: scp artifacts/deploy/edge_bundle to the Jetson AND deploy/raspi/
to the Raspberry Pi. edge_config.json carries the SAFETY ENVELOPE (floor
plane, rate limit, impact stop) -- the Pi reads it; never hand-copy those
numbers.
EOF
