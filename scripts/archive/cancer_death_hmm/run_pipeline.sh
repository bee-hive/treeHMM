#!/usr/bin/env bash
# ============================================================
# Cancer-cell DEATH-state HMM experiments
# ============================================================
#   Step 1  calculate_cancer_phase_emissions.py  OccidentAnalysis
#   Step 2  compute_dino_embeddings.py           cs229Dino
#   Step 3  reduce_dino_pca.py                   cs229Dino
#   Step 4  fit_death_hmm.py                     treeHMM_env
#   Step 5  evaluate_death_states.py             OccidentAnalysis
#
# Steps 2-3 are the expensive ones and are only needed by the DINO arms.
# To run just the cheap arms:
#     bash run_pipeline.sh --no-dino
#
# All shared parameters live in config.yml.  Step 0 (type-separated track
# regeneration) is not repeated here -- scripts/cancer_dino_hmm/type_sep_tracks
# is reused as-is.
set -euo pipefail
cd "$(dirname "$0")"

OCCIDENT_ENV=OccidentAnalysis   # env with MarsonImagingPipeline deps
DINO_ENV=cs229Dino              # env with torch/transformers/dinov2-base + sklearn
FIT_ENV=treeHMM_env             # env with JAX / dynamax / the model

NO_DINO=0
[ "${1:-}" = "--no-dino" ] && NO_DINO=1

run() { echo; echo ">>> $*"; conda run --no-capture-output -n "$@"; }

USE_DINO=$(python -c "import yaml;print(yaml.safe_load(open('config.yml')).get('use_dino',True))")

run "$OCCIDENT_ENV" python calculate_cancer_phase_emissions.py   # Step 1

if [ "$USE_DINO" = "True" ] && [ "$NO_DINO" -eq 0 ]; then
  run "$DINO_ENV" python compute_dino_embeddings.py              # Step 2
  run "$DINO_ENV" python reduce_dino_pca.py                      # Step 3
  run "$OCCIDENT_ENV" python check_dino_confounds.py             # Step 3b
  run "$FIT_ENV" python fit_death_hmm.py                         # Step 4 (all arms)
else
  echo ">>> Skipping DINOv2 steps; fitting the non-DINO arms only."
  NON_DINO_ARMS=$(python -c "import yaml;print(' '.join(a['name'] for a in yaml.safe_load(open('config.yml'))['arms'] if not a.get('use_dino')))")
  run "$FIT_ENV" python fit_death_hmm.py $NON_DINO_ARMS           # Step 4 (subset)
fi

run "$OCCIDENT_ENV" python evaluate_death_states.py               # Step 5
run "$OCCIDENT_ENV" python create_state_overlay_videos.py         # Step 6

echo
echo "Pipeline complete."
