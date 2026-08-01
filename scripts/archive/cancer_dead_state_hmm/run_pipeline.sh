#!/usr/bin/env bash
# ============================================================
# Cancer-cell DEAD-STATE HMM experiments
# ============================================================
#   Step 1  calculate_dead_state_features.py   OccidentAnalysis
#   Step 2  fit_dead_state_hmm.py              treeHMM_env
#   Step 3  evaluate_dead_states.py            OccidentAnalysis
#   Step 4  create_state_overlay_videos.py     OccidentAnalysis
#   Step 5  write_fit_readmes.py               OccidentAnalysis
#
# All shared parameters live in config.yml.  Type-separated track generation
# is not repeated here -- scripts/cancer_dino_hmm/type_sep_tracks is reused.
#
# Usage:
#   bash run_pipeline.sh            # everything
#   bash run_pipeline.sh --no-video # skip the (slow) video render
set -euo pipefail
cd "$(dirname "$0")"

OCCIDENT_ENV=OccidentAnalysis   # env with MarsonImagingPipeline deps
FIT_ENV=treeHMM_env             # env with JAX / dynamax / the model

NO_VIDEO=0
[ "${1:-}" = "--no-video" ] && NO_VIDEO=1

run() { echo; echo ">>> $*"; conda run --no-capture-output -n "$@"; }

run "$OCCIDENT_ENV" python calculate_dead_state_features.py   # Step 1
run "$FIT_ENV"      python fit_dead_state_hmm.py              # Step 2
run "$OCCIDENT_ENV" python evaluate_dead_states.py            # Step 3

if [ "$NO_VIDEO" -eq 0 ]; then
  run "$OCCIDENT_ENV" python create_state_overlay_videos.py   # Step 4
else
  echo ">>> Skipping video render."
fi

# Last, so the per-fit READMEs pick up the evaluation results.
run "$OCCIDENT_ENV" python write_fit_readmes.py               # Step 5

echo
echo "Pipeline complete."
