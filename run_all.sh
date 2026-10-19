#!/bin/bash
# =====================================================================
# ARM_BCRL006 end-to-end pipeline -- runs on ONE local PC, no container,
# no GPU server, no cloud. Physics is MuJoCo (CPU); only the learning
# phases touch the GPU.
#
#   0) assets      myCobot 280 Pi MJCF, generated from the joint origins in
#                  deploy/raspi/mycobot_fk.py and CROSS-CHECKED against
#                  MuJoCo's own forward kinematics (no download, no meshes)
#   1) collect     scripted IK expert demos, wrist camera recorded, sharded
#                  across processes
#  1.5) MIMICGEN   OFF BY DEFAULT -- this phase prints a skip line and exits.
#                  With mimicgen.enable=true: annotate -> warp each
#                  object-centric subtask onto a freshly randomised scene ->
#                  replay -> keep what the env's own success term accepts.
#                  The demo SOURCE is swappable: ~10 real-robot demos work here
#                  unchanged (see configs/mimicgen/default.yaml) -- that regime, not this
#                  simulator, is what MimicGen is for. The 2x2 study measured
#                  generated rows as worth ~0.6 of a collected row HERE, so the
#                  default collects more instead (artifacts/ablation/ has the numbers).
#   2) preprocess  outlier clip + robomimic HDF5 + IMAGE-QUALITY GATE
#                  -> stops the pipeline if the wrist frames are unusable
#   3) BC          robomimic BC (low_dim) -> ONNX + MLflow Registry
#   4) eval        closed loop: random / scripted expert / BC
#   5) LeRobot     dataset export (OPTIONAL -- self-skips if not installed)
#   6) RL          stable-baselines3 PPO initialised FROM the BC weights,
#                  with the four stabilisers + closed-loop checkpoint
#                  selection + a regression guard against BC
#   7) distill     the RL champion re-collects wrist-camera demos
#   8) visuomotor  ACT (default) / Diffusion Policy -> ONNX + Registry
#   9) eval        closed-loop evaluation of the deployed image policy
#  10) deploy      RELEASE GATE -> edge bundle for the Jetson/Pi stack
#
# TWO GATES CAN STOP THE RUN (phase 2 and phase 10). That is the point: they
# exist to stop an artifact that cannot work from reaching a robot.
#
# THE STUDY IS A SEPARATE COMMAND -- it is not part of the pipeline:
#   bash run_ablation.sh          # 2x2 MimicGen x domain randomisation
#
# Config is Hydra's (configs/config.yaml + configs/*/). Override anything:
#   bash run_all.sh bc.num_epochs=100 rl.total_iterations=200
#   bash run_all.sh visuomotor.type=diffusion       # Diffusion Policy variant
#   bash run_all.sh rl.bc_init=false                # PPO-from-scratch ablation
#   bash run_all.sh mimicgen.enable=true            # turn phase 1.5 ON (default off)
#   bash run_all.sh mimicgen.max_source_demos=10 mimicgen.selection=random
#                                                   # rehearse the few-real-demos regime
#   bash run_all.sh env.domain_rand.joint_friction_scale=[0.8,1.3]
#                                                   # switch on a DR axis that ships off
#   bash run_all.sh collect.workers=8               # more render shards
#
# Smoke test (~6 min, proves the plumbing, proves nothing about the policy):
#   bash run_all.sh collect.episodes=1 mimicgen.episodes=1 bc.num_epochs=5 \
#        rl.total_iterations=6 rl.critic_warmup_iters=2 rl.bc_anchor_decay_iters=4 \
#        rl.save_interval=2 visuomotor.distill.episodes=1 visuomotor.train.epochs=2 \
#        visuomotor.train.batch_size=8 visuomotor.train.val_split=0.3 \
#        eval.episodes=1 eval.num_envs=8 eval.visuomotor_num_envs=4 \
#        mimicgen.min_yield=0 deploy.force=true
#
# Usage:  bash run_all.sh [extra hydra overrides...]
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")"
export PWD="$(pwd)"
# MuJoCo offscreen backend. osmesa needs no display server and no driver EGL
# stack, which is what makes this reproducible under WSL2, in a container and
# over SSH. egl / glfw also work here and measure within 20 % -- the bottleneck
# is the per-frame read-back, not rasterisation (operations guide §5).
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
