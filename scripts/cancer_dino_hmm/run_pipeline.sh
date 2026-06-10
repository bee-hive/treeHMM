#!/usr/bin/env bash
# ============================================================
# Cancer-cell DINOv2 tree-AR-HMM pipeline
# ============================================================
# Runs all steps in order, each in its correct conda env.
# All shared parameters live in config.yml.
#
#   Step 0  regen_type_sep_tracks.py     OccidentAnalysis  (MarsonImagingPipeline XMLutils)
#   Step 1  calculate_cancer_emissions.py OccidentAnalysis (StatUtils features)
#   Step 2  compute_dino_embeddings.py   cs229Dino         (self-contained, DINOv2)
#   Step 3  reduce_dino_pca.py           cs229Dino         (PCA)
#   Step 4  fit_cancer_arhmm.py          treeHMM_env       (JAX / model fit)
#   Step 5  create_cancer_overlay_videos.py OccidentAnalysis (video render)
#
# Usage:
#   cd scripts/cancer_dino_hmm
#   bash run_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")"

OCCIDENT_ENV=OccidentAnalysis   # env with MarsonImagingPipeline deps
DINO_ENV=cs229Dino              # env with torch/transformers/dinov2-base + sklearn
FIT_ENV=treeHMM_env             # env with JAX / dynamax / the model

run() { echo; echo ">>> $*"; conda run --no-capture-output -n "$@"; }

# Read use_dino from config (default true)
USE_DINO=$(python -c "import yaml;print(yaml.safe_load(open('config.yml')).get('use_dino',True))")

run "$OCCIDENT_ENV" python regen_type_sep_tracks.py       # Step 0
run "$OCCIDENT_ENV" python calculate_cancer_emissions.py  # Step 1

if [ "$USE_DINO" = "True" ]; then
  run "$DINO_ENV" python compute_dino_embeddings.py        # Step 2
  run "$DINO_ENV" python reduce_dino_pca.py                # Step 3
else
  echo ">>> use_dino is False; skipping DINOv2 steps 2-3."
fi

run "$FIT_ENV" python fit_cancer_arhmm.py                 # Step 4
run "$OCCIDENT_ENV" python create_cancer_overlay_videos.py # Step 5

echo
echo "Pipeline complete."
