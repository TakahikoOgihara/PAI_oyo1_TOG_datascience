#!/bin/bash
# =====================================================================
# THE STUDY: 2x2 over {MimicGen off/on} x {domain randomisation off/on},
# each trained policy evaluated on BOTH a nominal and a randomised test
# bench. See src/ablation.py's docstring for why both test benches are
# required, and design notes §5 for the experimental design.
#
# Runs the same phase scripts run_all.sh runs, with Hydra overrides --
# nothing about the pipeline is duplicated, so the reported numbers come
# from the code that ships.
#
#   bash run_ablation.sh                        # 4 conditions x 3 seeds (~80 min)
#   bash run_ablation.sh ablation.seeds=[0]     # 1 seed (~27 min)
#   bash run_ablation.sh ablation.match_data=true
#                                               # equalise dataset SIZE across
#                                               # conditions, isolating MimicGen's
#                                               # diversity from its volume
#
# Outputs -> artifacts/ablation/
#   ablation.md            the table that goes into the report
#   ablation.png           grouped bar chart, mean +- std over seeds
#   ablation_results.json  every cell, every metric, every seed
#   <condition>_s<seed>/   per-cell evidence (dataset summaries + the ONNX)
# and one MLflow run + one W&B evaluation page.
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
