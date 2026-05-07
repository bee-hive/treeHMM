#!/usr/bin/env bash
# ============================================================
# run_pipeline.sh
#
# Runs the full CVAT ground-truth crop analysis pipeline:
#   1. calculate_emissions.py   (AnalysisEnv)
#   2. fit_arhmm.py             (treeHMM_env)
#   3. create_overlay_videos.py (AnalysisEnv)
#
# Usage:
#   bash run_pipeline.sh
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo " CVAT Ground-Truth Crop Analysis Pipeline"
echo "============================================================"
echo ""

# ------------------------------------------------------------------
# Step 1: Calculate emissions  (AnalysisEnv)
# ------------------------------------------------------------------
echo ">>> Step 1/3: Calculating emissions (AnalysisEnv)"
echo "------------------------------------------------------------"
conda run --no-capture-output -n AnalysisEnv python "${SCRIPT_DIR}/calculate_emissions.py"
echo ""
echo ">>> Step 1/3 complete."
echo ""

# ------------------------------------------------------------------
# Step 2: Fit AR-HMM          (treeHMM_env)
# ------------------------------------------------------------------
echo ">>> Step 2/3: Fitting AR-HMM (treeHMM_env)"
echo "------------------------------------------------------------"
conda run --no-capture-output -n treeHMM_env python "${SCRIPT_DIR}/fit_arhmm.py"
echo ""
echo ">>> Step 2/3 complete."
echo ""

# ------------------------------------------------------------------
# Step 3: Create overlay videos (AnalysisEnv)
# ------------------------------------------------------------------
echo ">>> Step 3/3: Creating overlay videos (AnalysisEnv)"
echo "------------------------------------------------------------"
conda run --no-capture-output -n AnalysisEnv python "${SCRIPT_DIR}/create_overlay_videos.py"
echo ""
echo ">>> Step 3/3 complete."
echo ""

echo "============================================================"
echo " Pipeline finished successfully!"
echo "============================================================"
