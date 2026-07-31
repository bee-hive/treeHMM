#!/usr/bin/env bash
# ============================================================
# run_pipeline.sh
#
# Runs the full CVAT ground-truth crop analysis pipeline:
#   1. calculate_emissions.py     (occident)
#   2. compute_dino_embeddings.py (cs229Dino)   [skipped if use_dino: false]
#   3. reduce_dino_pca.py         (cs229Dino)   [skipped if use_dino: false]
#   4. fit_arhmm.py               (treeHMM_env)
#   5. create_overlay_videos.py   (occident)
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

# Read use_dino from config to decide whether to run the DINOv2 steps.
USE_DINO=$(python -c "import yaml;print(yaml.safe_load(open('${SCRIPT_DIR}/config.yml')).get('use_dino', False))")

# ------------------------------------------------------------------
# Step 1: Calculate emissions  (occident)
# ------------------------------------------------------------------
echo ">>> Step 1/5: Calculating emissions (occident)"
echo "------------------------------------------------------------"
conda run --no-capture-output -n OccidentAnalysis python "${SCRIPT_DIR}/calculate_emissions.py"
echo ""
echo ">>> Step 1/5 complete."
echo ""

# ------------------------------------------------------------------
# Steps 2-3: DINOv2 embeddings + PCA  (cs229Dino)
# ------------------------------------------------------------------
if [ "$USE_DINO" = "True" ]; then
  echo ">>> Step 2/5: Computing DINOv2 embeddings (cs229Dino)"
  echo "------------------------------------------------------------"
  conda run --no-capture-output -n cs229Dino python "${SCRIPT_DIR}/compute_dino_embeddings.py"
  echo ""
  echo ">>> Step 2/5 complete."
  echo ""

  echo ">>> Step 3/5: PCA-reducing DINOv2 embeddings (cs229Dino)"
  echo "------------------------------------------------------------"
  conda run --no-capture-output -n cs229Dino python "${SCRIPT_DIR}/reduce_dino_pca.py"
  echo ""
  echo ">>> Step 3/5 complete."
  echo ""
else
  echo ">>> use_dino is False; skipping DINOv2 steps 2-3."
  echo ""
fi

# ------------------------------------------------------------------
# Step 4: Fit AR-HMM          (treeHMM_env)
# ------------------------------------------------------------------
echo ">>> Step 4/5: Fitting AR-HMM (treeHMM_env)"
echo "------------------------------------------------------------"
conda run --no-capture-output -n treeHMM_env python "${SCRIPT_DIR}/fit_arhmm.py"
echo ""
echo ">>> Step 4/5 complete."
echo ""

# ------------------------------------------------------------------
# Step 5: Create overlay videos (occident)
# ------------------------------------------------------------------
echo ">>> Step 5/5: Creating overlay videos (occident)"
echo "------------------------------------------------------------"
conda run --no-capture-output -n OccidentAnalysis python "${SCRIPT_DIR}/create_overlay_videos.py"
echo ""
echo ">>> Step 5/5 complete."
echo ""

echo "============================================================"
echo " Pipeline finished successfully!"
echo "============================================================"
