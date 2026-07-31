#!/usr/bin/env bash
# ============================================================
# DINOv2 HMM over Caliban cancer nuclei (Finetuned_Analysis_250t_600xy)
# ============================================================
# k=2 latent states, intended to separate ISOLATED cancer cells from cells in
# AGGREGATES, driven purely by the appearance of a 30x30 px patch centered on
# each nucleus.  All shared parameters live in config.yml.
#
#   Step 1  calculate_nuclei_emissions.py   OccidentAnalysis (parquet -> centroids + diagnostics)
#   Step 2  compute_dino_embeddings.py      cs229Dino        (DINOv2, ~983k patches)
#   Step 3  reduce_dino_pca.py              cs229Dino        (whitened PCA)
#   Step 4  fit_nuclei_hmm.py               treeHMM_env      (JAX tree-HMM fit + validation)
#   Step 5  create_nuclei_overlay_videos.py OccidentAnalysis (state overlay mp4)
#
# Usage:
#   cd scripts/marson_nuclei_dino
#   bash run_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")"

OCCIDENT_ENV=OccidentAnalysis
DINO_ENV=cs229Dino
FIT_ENV=treeHMM_env

run() { echo; echo ">>> $*"; conda run --no-capture-output -n "$@"; }

USE_DINO=$(python -c "import yaml;print(yaml.safe_load(open('config.yml')).get('use_dino',True))")

run "$OCCIDENT_ENV" python calculate_nuclei_emissions.py    # Step 1

if [ "$USE_DINO" = "True" ]; then
  run "$DINO_ENV" python compute_dino_embeddings.py         # Step 2
  run "$DINO_ENV" python reduce_dino_pca.py                 # Step 3
else
  echo ">>> use_dino is False; skipping DINOv2 steps 2-3."
fi

run "$FIT_ENV" python fit_nuclei_hmm.py                     # Step 4
run "$OCCIDENT_ENV" python create_nuclei_overlay_videos.py  # Step 5

echo
echo "Pipeline complete."
