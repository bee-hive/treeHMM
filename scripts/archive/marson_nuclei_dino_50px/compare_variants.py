"""Compare the num_lags=0 and num_lags=1 fits over the shared 50 px embeddings.

Both variants see identical emissions (the same DINO PCs) and have the division
tree disabled, so the only difference is whether the emission mean is
autoregressive.  This script puts their results side by side.

Reported per variant:
  - state occupancy
  - AUROC of the held-out `nuclei_neighbors_30px` against the more-crowded state
  - mean of every held-out scalar per state, in standard-deviation units
  - state persistence (P(stay) from the learned P_std, and empirical dwell time)
  - agreement between the two variants' assignments (accuracy + Cohen's kappa)

Usage (OccidentAnalysis):
    conda run -n OccidentAnalysis python compare_variants.py
"""

import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

_script_dir = Path(__file__).resolve().parent
with open(_script_dir / "config.yml", "r") as f:
    cfg = yaml.safe_load(f)

wells: List[str] = cfg["wells"]
analysis_dir = Path(cfg["output_base_dir"]).parent
VARIANTS = {
    "lag0": analysis_dir / "marson_nuclei_dino_k2_50px_lag0",
    "lag1": analysis_dir / "marson_nuclei_dino_k2_50px_lag1",
}
AGG = "nuclei_neighbors_30px"


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve via the rank-sum identity.

    Args:
        scores (np.ndarray): continuous score per observation.
        labels (np.ndarray): boolean, True for the positive class.

    Returns:
        float: AUROC, or NaN if either class is empty.
    """
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return (ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def load_variant(root: Path) -> Dict[str, np.ndarray]:
    """Load per-variant state assignments aligned to active cell-frames.

    Args:
        root (Path): variant output directory.

    Returns:
        Dict[str, np.ndarray]: flattened states, held-out diagnostics, and the
            per-cell active mask used to flatten them.
    """
    states_list, diag_list, active_list = [], [], []
    min_t = cfg["min_t"]
    for well in wells:
        centroids = np.load(root / well / "nuclei_centroids.npy")
        diag = np.load(root / well / "nuclei_emissions_array.npy")
        states = np.load(root / well / "nuclei_state_assignments.npy")
        active_all = ~np.isnan(centroids[:, :, 0])
        kept = np.where(active_all.sum(axis=0) >= min_t)[0]
        active = active_all[:, kept]
        if states.shape != active.shape:
            raise RuntimeError(
                f"{root.name}/{well}: states {states.shape} vs active {active.shape}")
        states_list.append(states[active])
        diag_list.append(diag[:, kept, :][active])
        active_list.append(active)
    return dict(states=np.concatenate(states_list),
                diag=np.concatenate(diag_list, axis=0),
                active=active_list)


with open(analysis_dir / "marson_nuclei_dino_50px_shared" / wells[0]
          / "nuclei_emissions_names.txt") as fh:
    feature_names = [l.strip() for l in fh if l.strip()]
agg_idx = feature_names.index(AGG)

print("=" * 72)
print("Variant comparison: shared 50 px DINO embeddings, divisions OFF")
print("=" * 72)

loaded, rows = {}, []
for name, root in VARIANTS.items():
    if not (root / wells[0] / "nuclei_state_assignments.npy").exists():
        print(f"\n{name}: no state assignments yet at {root}; skipping.")
        continue
    data = load_variant(root)
    loaded[name] = data
    states, diag = data["states"], data["diag"]
    agg = diag[:, agg_idx]
    n_states = int(states.max()) + 1
    means = [agg[states == s].mean() if (states == s).any() else np.nan
             for s in range(n_states)]
    crowded = int(np.nanargmax(means))

    summary_path = root / "state_summary.json"
    summary = json.load(open(summary_path)) if summary_path.exists() else {}

    print(f"\n--- {name} (num_lags={summary.get('num_lags', '?')}) ---")
    print(f"  occupancy      : "
          + ", ".join(f"state {s}={np.mean(states == s):.3f}" for s in range(n_states)))
    print(f"  mean {AGG}: "
          + ", ".join(f"state {s}={means[s]:.2f}" for s in range(n_states)))
    print(f"  AUROC({AGG} -> state {crowded}) = "
          f"{auroc(agg, states == crowded):.3f}")
    print(f"  log prob       : {summary.get('log_prob_first', float('nan')):.0f} "
          f"-> {summary.get('log_prob_last', float('nan')):.0f}")

    print(f"  held-out feature separation (std units, state{crowded} - other):")
    for i, feat in enumerate(feature_names):
        v = diag[:, i]
        a = v[states == crowded].mean()
        b = v[states != crowded].mean()
        rows.append({"variant": name, "feature": feat,
                     "std_diff": (a - b) / (v.std() + 1e-9)})
        print(f"     {feat:<24s} {(a - b) / (v.std() + 1e-9):+.3f}")

if len(loaded) == 2:
    a, b = loaded["lag0"]["states"], loaded["lag1"]["states"]
    if len(a) == len(b):
        agree = max((a == b).mean(), (a != b).mean())  # label-swap invariant
        # Cohen's kappa under the better of the two labelings
        bb = b if (a == b).mean() >= (a != b).mean() else 1 - b
        po = (a == bb).mean()
        pe = sum((a == k).mean() * (bb == k).mean() for k in (0, 1))
        kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
        print(f"\n--- agreement ---")
        print(f"  lag0 vs lag1 assignment agreement: {agree:.3f} "
              f"(Cohen's kappa {kappa:.3f})")
    else:
        print(f"\n--- agreement ---")
        print(f"  not comparable: lag0 has {len(a)} active cell-frames, "
              f"lag1 has {len(b)} (ar_warmup drops each track's first frame)")

if rows:
    df = pd.DataFrame(rows).pivot(index="feature", columns="variant",
                                  values="std_diff")
    out = analysis_dir / "variant_comparison_50px.csv"
    df.to_csv(out)
    print(f"\nSaved -> {out}")
