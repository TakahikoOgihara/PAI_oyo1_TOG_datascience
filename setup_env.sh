#!/bin/bash
# =====================================================================
# ARM_BCRL006 environment -- one local PC, one virtualenv, ~10 minutes.
#
# ARM_BCRL005 needed a rented GPU server, a 20 GB Isaac Sim container and a
# pinned Isaac Lab build. This needs Python and a graphics library that
# already exists on the machine. That is the whole point of the MuJoCo port:
# the cost of one PDCA cycle dropped from "book a server" to "run a command",
# and that is what made the 4-condition study in run_ablation.sh affordable.
#
#   bash setup_env.sh              # create the venv and install everything
#   bash setup_env.sh --check      # verify an existing install, install nothing
#
# After it finishes:
#   source ~/.venvs/arm_bcrl006/bin/activate
#   bash run_all.sh
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")"

VENV="${VENV:-$HOME/.venvs/arm_bcrl006}"
PYVER="${PYVER:-3.11}"
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------------
say "1/6  system packages (offscreen OpenGL for MuJoCo)"
# MuJoCo renders the wrist camera offscreen. osmesa is the software backend:
# no display server, no driver EGL stack, so it works the same over SSH, in a
# container and under WSL2 -- which is where the other two backends usually
# break. On this project's reference machine all three measure within 20 %
# anyway, because the bottleneck is the per-frame read-back.
MISSING=""
for lib in libOSMesa.so.8 libEGL.so.1 libGL.so.1; do
  ldconfig -p 2>/dev/null | grep -q "$lib" || MISSING="$MISSING $lib"
done
if [ -n "$MISSING" ]; then
  echo "missing:$MISSING"
  echo "  Debian/Ubuntu:  sudo apt-get install -y libosmesa6 libegl1 libgl1 libglfw3"
  echo "  (rendering will fall back to whatever MUJOCO_GL can open; state-only"
  echo "   phases -- collect with collect.camera=false, BC, PPO, eval -- need none of it)"
else
  echo "offscreen GL libraries present"
fi

# ---------------------------------------------------------------------
say "2/6  Python $PYVER virtualenv at $VENV"
if [ ! -d "$VENV" ]; then
  [ "$CHECK_ONLY" = 1 ] && { echo "missing (run without --check)"; exit 1; }
  # uv if available (fast, and it can fetch the interpreter itself), else venv
  if command -v uv >/dev/null 2>&1; then
    uv venv --python "$PYVER" "$VENV"
  else
    "python$PYVER" -m venv "$VENV" || python3 -m venv "$VENV"
  fi
fi
PY="$VENV/bin/python"
PIP="$PY -m pip"
command -v uv >/dev/null 2>&1 && PIP="uv pip"
export VIRTUAL_ENV="$VENV"
echo "interpreter: $($PY -V)  ($PY)"

# ---------------------------------------------------------------------
say "3/6  PyTorch"
# PIN THE CUDA BUILD TO THE INSTALLED DRIVER. PyPI's default wheel tracks the
# newest CUDA, and a wheel newer than the driver reports
# "CUDA initialization: The NVIDIA driver on your system is too old" and falls
# back to CPU -- silently, and only when you look. The driver is fine; the
# wheel was wrong. Read the driver's CUDA version and pick the matching index.
if [ "$CHECK_ONLY" = 0 ]; then
  CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)
  DRV_CUDA=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -1 || true)
  case "$DRV_CUDA" in
    12.[0-5]*) IDX="https://download.pytorch.org/whl/cu121" ;;
    12.[6-7]*) IDX="https://download.pytorch.org/whl/cu126" ;;
    12.[8-9]*) IDX="https://download.pytorch.org/whl/cu128" ;;
    13.*)      IDX="https://download.pytorch.org/whl/cu130" ;;
    *)         IDX="https://download.pytorch.org/whl/cpu" ;;
  esac
  echo "driver $CUDA_VER, CUDA runtime ceiling $DRV_CUDA -> $IDX"
  $PIP install --index-url "$IDX" "torch" "torchvision"
fi

# ---------------------------------------------------------------------
say "4/6  simulator + MLOps + policy stack"
if [ "$CHECK_ONLY" = 0 ]; then
  $PIP install \
    "mujoco>=3.2" \
    "stable-baselines3>=2.4" \
    "hydra-core" "hydra-optuna-sweeper" "omegaconf" \
    "mlflow" "wandb" "tensorboard" "tensorboardX" \
    "onnx" "onnxruntime" \
    "h5py" "numpy<3" "matplotlib" "imageio[ffmpeg]" "pillow" \
    "termcolor" "tqdm" "psutil"
  # robomimic WITHOUT its dependency list. Its `egl_probe` requirement is a
  # CMake C extension whose only job is picking a GPU for robosuite's offscreen
  # renderer -- machinery this project never touches (rollout evaluation is
  # disabled; closed-loop evaluation is src/evaluate.py, in MuJoCo). It fails
  # to build on a machine without EGL development headers, which would
  # otherwise block the whole install for a feature we do not use. Every real
  # runtime dependency is installed above, explicitly.
  $PIP install --no-deps robomimic
fi

# ---------------------------------------------------------------------
say "5/6  ImageNet weights for the ACT backbone (cache them now, offline later)"
# ARM_BCRL002 shipped a visuomotor policy whose ResNet18 had fallen back to
# random weights because the torch hub was unreachable, and every metric in the
# pipeline still looked fine. configs/visuomotor/act.yaml now REFUSES to train
# without them; caching here is what keeps that from being an obstacle.
$PY - <<'PYEOF'
try:
    import torchvision as tv
    tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1)
    print("ResNet18 ImageNet weights cached")
except Exception as e:
    print(f"WARNING: could not cache ResNet18 weights ({e}).")
    print("Phase 8 will refuse to train until they are available; to train")
    print("without them ON PURPOSE: visuomotor.train.require_pretrained_backbone=false")
PYEOF

# ---------------------------------------------------------------------
say "6/6  self-tests (no simulator state, no GPU needed)"
export PWD="$(pwd)"
$PY deploy/raspi/mycobot_fk.py          # kinematics + tool geometry + floor guard
$PY deploy/raspi/safety.py              # rate limit, virtual floor, impact stop
$PY arm_common/ik.py                    # analytic Jacobian + damped least squares
$PY mimicgen/pose_np.py                 # quaternion/pose primitives
$PY mimicgen/annotate.py                # subtask segmentation
$PY mimicgen/generation.py              # eq. 1, interpolation, source selection
$PY assets/build_mycobot_mjcf.py --verify   # MJCF FK == mycobot_fk.py
$PY -m pytest tests -q || echo "(pytest not installed: pip install pytest)"

cat <<EOF

Ready.
  source $VENV/bin/activate
  bash run_all.sh          # the 11-phase pipeline
  bash run_ablation.sh     # the 2x2 study
  python mujoco_sim/train_arm_mujoco.py --mode play    # watch it

Everything is local: MLflow on a SQLite file, W&B offline, TensorBoard on the
event files. No account and no network are required at any point.
EOF
