"""Step 5: Score every fitted arm against the ground-truth death events.

Four independent checks, three of which need no manual labels at all:

1. RECALL against `death_annotations.yml`.  For a frame-level event, does
   the cell ENTER the death state within `match_tolerance_frames` of the
   annotated window?  For the already-dead cell and the one whose timing is
   unresolved, the question is cell-level: does the cell ever occupy the
   death state (and, for the already-dead cell, for most of its track)?

2. DIVISION SPECIFICITY (label-free).  Mitotic rounding has essentially the
   death signature -- area down, circularity up -- so the 11 annotated
   cancer division events are known non-death shape changes.  Any death
   state entry landing on a division frame is a confirmed false positive.

3. T-CELL ENRICHMENT (label-free).  Deaths should be enriched for nearby
   T cells.  Reported as an AUROC of `dilated_t_cell_neighbors` separating
   death-state frames from the rest.  This is an INDEPENDENT check only for
   arms that never saw a T-cell feature; for the others it is circular and
   is flagged as such in the output rather than being counted.

4. ENTRY DISCIPLINE (label-free).  A real death state is entered at most
   once per cell.  Many entries per cell means a "fluctuation" state.

Outputs, under {output_base_dir}/fits/:
    evaluation.csv          one row per (arm, k)
    event_detail.csv        one row per (arm, k, annotated event)
    cross_arm_agreement.csv Cohen's kappa between arms on the death indicator
    evaluation.png          recall / specificity / enrichment across arms

Usage (OccidentAnalysis; needs numpy/pandas/matplotlib/yaml/tifffile only):
    conda run -n OccidentAnalysis python evaluate_death_states.py
"""

import os
import pickle
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tifffile

_script_dir = Path(__file__).resolve().parent
with open(_script_dir / "config.yml", "r") as f:
    cfg = yaml.safe_load(f)

crop_ids = cfg["crop_ids"]
out_base_dir = cfg["output_base_dir"]
type_sep_tracks_dir = cfg["type_sep_tracks_dir"]
feature_names = cfg["emission_feature_names"]
fits_dir = os.path.join(out_base_dir, "fits")

with open(_script_dir / cfg["annotations_path"], "r") as f:
    ann = yaml.safe_load(f)
tolerance = ann["match_tolerance_frames"]
events = ann["events"]

cell_index = pd.read_csv(os.path.join(fits_dir, "cell_index.csv"))
diagnostics = np.load(os.path.join(fits_dir, "diagnostics.npy"))
active = np.load(os.path.join(fits_dir, "active_mask.npy"))
diag_idx = {name: i for i, name in enumerate(feature_names)}

col_of = {(r.crop, r.cvat_cell_id): r.column for r in cell_index.itertuples()}


def auroc(scores, positive):
    """Mann-Whitney AUROC of `scores` separating `positive` from the rest.

    Args:
        scores (np.ndarray): (n,) continuous values.
        positive (np.ndarray): (n,) boolean class labels.

    Returns:
        float: AUROC, or NaN if either class is empty.
    """
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks within ties so heavily-tied count data is scored fairly.
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0)
                 / (n_pos * n_neg))


def cohens_kappa(a, b):
    """Cohen's kappa between two boolean labelings of the same items."""
    n = len(a)
    if n == 0:
        return np.nan
    observed = float((a == b).mean())
    p_a = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((observed - p_a) / (1 - p_a)) if p_a < 1 else np.nan


def entry_frames(is_death_col, active_col):
    """Frames where a cell enters the death state from a non-death active frame.

    Args:
        is_death_col (np.ndarray): (T,) bool, True where the cell is in the state.
        active_col (np.ndarray): (T,) bool.

    Returns:
        list[int]: entry frame indices, including the first active frame if the
            cell starts out already in the state.
    """
    frames = np.where(active_col)[0]
    entries = []
    for i, t in enumerate(frames):
        if not is_death_col[t]:
            continue
        if i == 0 or not is_death_col[frames[i - 1]]:
            entries.append(int(t))
    return entries


# --- Division frames (label-free negative control) --------------------------
division_frames = {}   # column -> division frame
for crop in crop_ids:
    graph_path = os.path.join(type_sep_tracks_dir, crop, "graph.pkl")
    ids_path = os.path.join(type_sep_tracks_dir, crop, "cancer_ids.pkl")
    if not (os.path.exists(graph_path) and os.path.exists(ids_path)):
        continue
    graph = pickle.load(open(graph_path, "rb"))
    cancer_ids = set(int(i) for i in pickle.load(open(ids_path, "rb")))
    tracks = np.asarray(tifffile.imread(
        os.path.join(type_sep_tracks_dir, crop, "tracks.tiff")))[..., 1]
    for child, parent in graph.items():
        if int(child) not in cancer_ids and int(parent) not in cancer_ids:
            continue
        col = col_of.get((crop, int(child)))
        if col is None:
            continue   # daughter dropped by the min_t filter
        present = np.where((tracks == int(child)).any(axis=(1, 2)))[0]
        if len(present):
            division_frames[col] = int(present[0])
print(f"Division control: {len(division_frames)} cancer daughters retained "
      f"after filtering")

T, n_cells = active.shape
tcell_counts = diagnostics[:, :, diag_idx["dilated_t_cell_neighbors"]]

N_PERMUTATIONS = 2000
_rng = np.random.default_rng(0)


def permutation_null(entries_by_col, frame_events, n_perm=N_PERMUTATIONS):
    """Chance rates for frame recall and division false positives.

    There are only three frame-level annotated events, so "3/3" is not by
    itself evidence: a state occupying 25% of cell-frames will hit all three
    windows most of the time.  This relocates each cell's death-state entries
    uniformly at random among that cell's own active frames -- preserving the
    number of entries per cell, and therefore the state's occupancy and
    fragmentation -- and recomputes both statistics.

    Args:
        entries_by_col (dict): column -> list of entry frames.
        frame_events (list): (column, lo, hi) for frame-level annotations.
        n_perm (int): number of permutations.

    Returns:
        dict: mean chance recall, one-sided p-value, mean chance division hits.
    """
    if not frame_events:
        return dict(chance_frame_recall=np.nan, frame_recall_p=np.nan,
                    chance_division_hits=np.nan)

    active_frames = {c: np.where(active[:, c])[0] for c in range(n_cells)}
    observed_hits = sum(
        any(lo - tolerance <= t <= hi + tolerance
            for t in entries_by_col.get(col, []))
        for col, lo, hi in frame_events)

    recalls = np.zeros(n_perm)
    div_hits = np.zeros(n_perm)
    div_items = list(division_frames.items())
    for b in range(n_perm):
        shuffled = {}
        for col, entries in entries_by_col.items():
            frames = active_frames[col]
            if len(entries) == 0 or len(frames) == 0:
                shuffled[col] = []
            else:
                shuffled[col] = _rng.choice(
                    frames, size=min(len(entries), len(frames)), replace=False)
        recalls[b] = sum(
            any(lo - tolerance <= t <= hi + tolerance for t in shuffled.get(col, []))
            for col, lo, hi in frame_events)
        div_hits[b] = sum(
            1 for col, frame in div_items
            if any(abs(t - frame) <= tolerance for t in shuffled.get(col, [])))

    n_events = len(frame_events)
    return dict(
        chance_frame_recall=round(float(recalls.mean() / n_events), 3),
        frame_recall_p=round(float((recalls >= observed_hits).mean()), 4),
        chance_division_hits=round(float(div_hits.mean()), 2))


def evaluate_fit(arm, k):
    """Score one fit; returns (summary_row, per-event rows, death indicator)."""
    fit_dir = os.path.join(fits_dir, arm, f"k{k}")
    with open(os.path.join(fit_dir, "fit_summary.yml")) as fh:
        info = yaml.safe_load(fh)
    assignments = np.load(os.path.join(fit_dir, "state_assignments.npy"))
    death_state = info["death_state"]
    is_death = active & (assignments == death_state)

    uses_tcell = any("t_cell" in f for f in info["features"])

    # --- 1. Recall on annotated events ---
    event_rows, hits, frame_level, cell_hits, cell_level = [], 0, 0, 0, 0
    frame_events = []
    for ev in events:
        col = col_of.get((ev["crop"], ev["cell_id"]))
        row = dict(arm=arm, k=k, crop=ev["crop"], cell_id=ev["cell_id"],
                   confidence=ev.get("confidence"))
        if col is None:
            row.update(status="cell_filtered_out", detected=False)
            event_rows.append(row)
            continue

        entries = entry_frames(is_death[:, col], active[:, col])
        frac = float(is_death[active[:, col], col].mean())
        row["death_entries"] = len(entries)
        row["frac_frames_in_death_state"] = round(frac, 3)

        if ev.get("local_frames"):
            lo, hi = ev["local_frames"]
            matched = [t for t in entries if lo - tolerance <= t <= hi + tolerance]
            frame_level += 1
            hits += bool(matched)
            frame_events.append((col, lo, hi))
            row.update(status="frame_level", window=[lo, hi],
                       matched_entry=matched[0] if matched else None,
                       detected=bool(matched))
        else:
            # Already-dead or unresolved timing: cell-level question only.
            cell_level += 1
            detected = frac > 0.5 if ev.get("already_dead") else len(entries) > 0
            cell_hits += bool(detected)
            row.update(
                status="already_dead" if ev.get("already_dead") else "cell_level",
                detected=bool(detected))
        event_rows.append(row)

    # --- 2. Division specificity ---
    entries_by_col = {c: entry_frames(is_death[:, c], active[:, c])
                      for c in range(n_cells)}
    total_entries = sum(len(v) for v in entries_by_col.values())
    division_hits = sum(
        1 for col, frame in division_frames.items()
        if any(abs(t - frame) <= tolerance for t in entries_by_col.get(col, []))
    )

    # --- 3. T-cell enrichment ---
    tcell_auroc = auroc(tcell_counts[active], is_death[active])

    # --- 4. Entry discipline ---
    cells_with_entry = sum(1 for v in entries_by_col.values() if v)
    entries_per_entered_cell = (total_entries / cells_with_entry
                                if cells_with_entry else np.nan)

    summary = dict(
        arm=arm, k=k, num_lags=info["num_lags"],
        emission_dim=info["emission_dim"], death_state=death_state,
        occupancy=round(info["death_state_occupancy"], 4),
        self_transition=round(info["death_state_self_transition"], 3),
        frame_recall=(hits / frame_level) if frame_level else np.nan,
        frame_hits=f"{hits}/{frame_level}",
        cell_recall=(cell_hits / cell_level) if cell_level else np.nan,
        cell_hits=f"{cell_hits}/{cell_level}",
        total_entries=total_entries,
        cells_with_entry=cells_with_entry,
        entries_per_entered_cell=round(entries_per_entered_cell, 2),
        division_false_positives=f"{division_hits}/{len(division_frames)}",
        tcell_auroc=round(tcell_auroc, 3) if np.isfinite(tcell_auroc) else np.nan,
        tcell_auroc_is_circular=uses_tcell,
        log_prob=info["log_prob_final"],
    )
    summary.update(permutation_null(entries_by_col, frame_events))
    return summary, event_rows, is_death


print("=" * 60)
print("Step 5: Evaluating fitted arms")
print("=" * 60)

summary_path = os.path.join(fits_dir, "summary.csv")
fits = pd.read_csv(summary_path)

rows, event_rows, death_indicators = [], [], {}
for r in fits.itertuples():
    summary, evs, is_death = evaluate_fit(r.arm, r.k)
    rows.append(summary)
    event_rows.extend(evs)
    death_indicators[(r.arm, r.k)] = is_death

evaluation = pd.DataFrame(rows).sort_values(["arm", "k"]).reset_index(drop=True)
evaluation.to_csv(os.path.join(fits_dir, "evaluation.csv"), index=False)
pd.DataFrame(event_rows).to_csv(os.path.join(fits_dir, "event_detail.csv"), index=False)

show = ["arm", "k", "occupancy", "self_transition", "frame_hits",
        "chance_frame_recall", "frame_recall_p", "cell_hits",
        "entries_per_entered_cell", "division_false_positives",
        "chance_division_hits", "tcell_auroc", "tcell_auroc_is_circular"]
pd.set_option("display.width", 250)
print("\n" + evaluation[show].to_string(index=False))

n_tests = len(evaluation)
n_frame_events = sum(1 for e in events if e.get("local_frames"))
print(f"\nMULTIPLICITY: {n_tests} (arm, k) combinations were scored against "
      f"{n_frame_events} frame-level events.")
print(f"  A Bonferroni-corrected 0.05 bar is p < {0.05 / max(n_tests, 1):.4f}; "
      f"the smallest attainable p with {n_frame_events} events is "
      f"well above that.  Treat `frame_recall_p` as ranking information, not "
      f"as a significance test -- no arm can clear a corrected bar at this "
      f"annotation count.  More annotated events is the binding constraint on "
      f"deciding between arms.")

# --- Cross-arm agreement, per k ---
agreement_rows = []
for k in sorted(fits["k"].unique()):
    arms_at_k = [a for (a, kk) in death_indicators if kk == k]
    for i, a1 in enumerate(arms_at_k):
        for a2 in arms_at_k[i + 1:]:
            d1 = death_indicators[(a1, k)][active]
            d2 = death_indicators[(a2, k)][active]
            agreement_rows.append(dict(
                k=k, arm_a=a1, arm_b=a2,
                agreement=round(float((d1 == d2).mean()), 4),
                kappa=round(cohens_kappa(d1, d2), 4)))
if agreement_rows:
    agreement = pd.DataFrame(agreement_rows)
    agreement.to_csv(os.path.join(fits_dir, "cross_arm_agreement.csv"), index=False)
    print("\nCross-arm agreement on the death indicator (top 12 by kappa):")
    print(agreement.sort_values("kappa", ascending=False).head(12).to_string(index=False))

# --- Figure ---
arms_order = list(dict.fromkeys(evaluation["arm"]))
fig, axes = plt.subplots(1, 3, figsize=(15, 4), tight_layout=True)
for arm in arms_order:
    sub = evaluation[evaluation["arm"] == arm]
    axes[0].plot(sub["k"], sub["frame_recall"], marker="o", label=arm)
    axes[1].plot(sub["k"], sub["occupancy"], marker="o", label=arm)
    circular = sub["tcell_auroc_is_circular"].to_numpy()
    axes[2].plot(sub["k"], sub["tcell_auroc"], marker="o",
                 linestyle="--" if circular.all() else "-", label=arm)
axes[0].set_ylabel("Frame-level recall on annotated deaths")
axes[0].set_ylim(-0.05, 1.05)
axes[1].set_ylabel("Death-state occupancy")
axes[2].set_ylabel("T-cell AUROC (dashed = circular)")
axes[2].axhline(0.5, color="grey", lw=0.8, ls=":")
for ax in axes:
    ax.set_xlabel("num_states (k)")
    ax.legend(fontsize=6)
plt.savefig(os.path.join(fits_dir, "evaluation.png"), dpi=200, bbox_inches="tight")
plt.close()

print(f"\nSaved evaluation.csv / event_detail.csv / evaluation.png -> {fits_dir}")
print("=" * 60)
