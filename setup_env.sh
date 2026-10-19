#!/bin/bash
# =====================================================================
# ARM_BCRL006 の環境構築。ローカルPC 1台、venv 1つ、約10分。
#
# ARM_BCRL005 はレンタルGPUサーバーと20GBの Isaac Sim コンテナ、版を固定した
# Isaac Lab が必要だった。こちらは Python と、マシンに既にあるグラフィックス
# ライブラリだけで済む。MuJoCo へ移した狙いはそこで、PDCA 1周のコストが
# 「サーバーを予約する」から「コマンドを1つ流す」に下がった。それで
# run_ablation.sh の4条件比較が現実的な値段になった。
#
#   bash setup_env.sh              # venv を作って全部入れる
#   bash setup_env.sh --check      # 既存環境の検証だけ。何もインストールしない
#
# 終わったら:
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
# MuJoCo は手首カメラをオフスクリーンで描画する。osmesa はソフトウェア
# バックエンドで、ディスプレイサーバもドライバのEGLスタックも要らない。
# SSH越し、コンテナ内、WSL2 でも同じに動く。他の2つはそこで壊れやすい。
# 基準機では3つとも差は20%以内だった。律速がフレームごとの読み戻し
# だからで、描画そのものではない。
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
  # uv があれば使う（速いし、インタプリタ自体も取ってこれる）。無ければ venv
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
# CUDAビルドを、入っているドライバに合わせて固定する。PyPI の既定ホイールは
# 最新CUDAに追従するので、ドライバより新しいホイールを入れると
# 「CUDA initialization: The NVIDIA driver on your system is too old」と出て
# CPUに落ちる。黙って落ちるので、見に行くまで気づかない。悪いのはドライバ
# ではなくホイール。ドライバのCUDA版を読んで、合う索引を選ぶ。
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
  # robomimic を依存リスト抜きで入れる。依存に入っている `egl_probe` は
  # CMake の C 拡張で、仕事は robosuite のオフスクリーン描画用に GPU を
  # 選ぶことだけ。このプロジェクトでは一切使わない（ロールアウト評価は
  # 無効、閉ループ評価は src/evaluate.py が MuJoCo 上で行う）。しかも EGL の
  # 開発ヘッダが無いマシンではビルドに失敗するので、使わない機能のせいで
  # 環境構築全体が止まる。実行時に本当に必要な依存は、上で明示的に
  # 入れてある。
  $PIP install --no-deps robomimic
fi

# ---------------------------------------------------------------------
say "5/6  ImageNet weights for the ACT backbone (cache them now, offline later)"
# ARM_BCRL002 では torch hub に届かず ResNet18 がランダム初期化のままの
# 視覚運動方策を出荷してしまい、それでもパイプラインの指標は全部正常に見えた。
# 今は configs/visuomotor/act.yaml が重み無しでの学習を拒否する。ここで
# キャッシュしておけば、その拒否が作業の邪魔にならない。
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
$PY deploy/raspi/mycobot_fk.py          # 運動学 + ツール形状 + 床ガード
$PY deploy/raspi/safety.py              # レート制限、仮想床、衝突停止
$PY arm_common/ik.py                    # 解析ヤコビアン + 減衰最小二乗
$PY mimicgen/pose_np.py                 # クォータニオンと姿勢の基本演算
$PY mimicgen/annotate.py                # サブタスク分割
$PY mimicgen/generation.py              # 式1、補間、ソース選択
$PY assets/build_mycobot_mjcf.py --verify   # MJCF の FK == mycobot_fk.py
$PY -m pytest tests -q || echo "(pytest not installed: pip install pytest)"

cat <<EOF

Ready.
  source $VENV/bin/activate
  bash run_all.sh          # 11フェーズのパイプライン
  bash run_ablation.sh     # 2x2 の比較実験
  python mujoco_sim/train_arm_mujoco.py --mode play    # 動きを見る

Everything is local: MLflow on a SQLite file, W&B offline, TensorBoard on the
event files. No account and no network are required at any point.
EOF
