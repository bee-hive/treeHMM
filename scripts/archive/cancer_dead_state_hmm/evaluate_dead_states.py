"""Step 3: Score every fit against the ground-truth death events.

The target here is a durable dead CONDITION, not a one-frame event, so the
checks differ from `cancer_death_hmm`:

1. ENTRY RECALL -- does the cell enter the dead state within
   `match_tolerance_frames` of its annotated window?  Scored against a
   permutation null that relocates each cell's entries uniformly among that
   cell's own active frames, preserving entries-per-cell and therefore
   occupancy.

2. PERSISTENCE -- after entering, what fraction of the cell's remaining
   frames stay in the state?  For an absorbing fit this is 1.0 by
   construction and only confirms the constraint took; for a free fit it is
   the real question, and a free state that scores near 1 has learned
   persistence from the data rather than been told.

3. ALREADY-DEAD CAPTURE -- the annotated cell that is already dead when its
   track starts.  A transition-based detector cannot represent it at all
   (every `cancer_death_hmm` arm scored 0 here); an absorbing model can,
   through the initial distribution.  This is the check that most directly
   distinguishes the two designs.

4. DIVISION SPECIFICITY (label-free) -- mitotic rounding produces the death
   shape signature, so the annotated cancer divisions are known non-death
   changes and any entry landing on one is a confirmed false positive.

5. CELL-LEVEL DEATH FRACTION -- the share of cells ever entering the state.
   This is the quantity the downstream per-condition comparison wants, and
   with an absorbing state it is well defined.  Reported per condition,
   against the ~33% observed in the fully-annotated B8_t50 crop.

6. T-CELL ENRICHMENT (label-free) -- AUROC of `dilated_t_cell_neighbors`
   separating dead-state frames from the rest.  No arm here uses a T-cell
   feature, so this is independent for all of them.

Outputs under {output_base_dir}/fits/: evaluation.csv, event_detail.csv,
condition_death_fractions.csv, evaluation.png

Usage (OccidentAnalysis):
    conda run -n OccidentAnalysis python evaluate_dead_states.py
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
conditions_dict = cfg["conditions_dict"]
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
crop_of_col = {r.column: r.crop for r in cell_index.itertuples()}

T, n_cells = active.shape
tcell_counts = diagnostics[:, :, diag_idx["dilated_t_cell_neighbors"]]
N_PERMUTATIONS = 2000
_rng = np.random.default_rng(0)


def auroc(scores, positive):
    """Tie-corrected Mann-Whitney AUROC; NaN if either class is empty."""
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0)
                 / (n_pos * n_neg))


SIGNATURE_FEATURES = [("d_circularity", +1.0), ("d_area_frac", -1.0),
                      ("area_over_running_max", -1.0),
                      ("running_max_abs_d_circularity", +1.0)]


def dead_state_contrast(is_dead):
    """Effect size of the dead state on the death signature, in pooled SDs.

    `fit_dead_state_hmm.py` picks the dead candidate with a score that
    z-standardizes ACROSS STATES.  That says which state is most death-like
    but nothing about whether the difference is meaningful -- and at k=2 the
    across-state z is degenerate (always +/-1), so the score is pinned at
    +/-2 no matter how similar the two states are.  A fit whose two states
    differ by 1% on every diagnostic scores identically to one whose states
    are genuinely different phenotypes.

    This measures the gap the other way: for each signature feature, the
    dead state's mean minus the rest, divided by the SD over all active
    cell-frames.  Signed so that positive means "more death-like".  A value
    near 0 means the state partitions something other than the phenotype.

    Args:
        is_dead (np.ndarray): (T, n_cells) bool, dead-state membership.

    Returns:
        float: mean signed Cohen's d over the signature features.
    """
    dead_sel = is_dead[active]
    if dead_sel.all() or not dead_sel.any():
        return 0.0
    effects = []
    for name, sign in SIGNATURE_FEATURES:
        vals = diagnostics[:, :, diag_idx[name]][active]
        sd = vals.std()
        if sd < 1e-12:
            continue
        effects.append(sign * (vals[dead_sel].mean() - vals[~dead_sel].mean()) / sd)
    return float(np.mean(effects)) if effects else 0.0


def entry_frames(is_dead_col, active_col):
    """Frames where a cell enters the dead state from a non-dead active frame."""
    frames = np.where(active_col)[0]
    entries = []
    for i, t in enumerate(frames):
        if not is_dead_col[t]:
            continue
        if i == 0 or not is_dead_col[frames[i - 1]]:
            entries.append(int(t))
    return entries


def persistence_after_entry(is_dead_col, active_col, entry):
    """Fraction of a cell's active frames at/after `entry` that stay dead."""
    frames = np.where(active_col)[0]
    after = frames[frames >= entry]
    return float(is_dead_col[after].mean()) if len(after) else np.nan


# --- Division control (label-free) ------------------------------------------
division_frames = {}
for crop in crop_ids:
    graph_path = os.path.join(type_sep_tracks_dir, crop, "graph.pkl")
    ids_path = os.path.join(type_sep_tracks_dir, crop, "cancer_ids.pkl")
    if not (os.path.exists(graph_path) and os.path.exists(ids_path)):
        continue
    graph = pickle.load(open(graph_path, "rb"))
    cancer_ids = {int(i) for i in pickle.load(open(ids_path, "rb"))}
    tracks = np.asarray(tifffile.imread(
        os.path.join(type_sep_tracks_dir, crop, "tracks.tiff")))[..., 1]
    for child, parent in graph.items():
        if int(child) not in cancer_ids and int(parent) not in cancer_ids:
            continue
        col = col_of.get((crop, int(child)))
        if col is None:
            continue
        present = np.where((tracks == int(child)).any(axis=(1, 2)))[0]
        if len(present):
            division_frames[col] = int(present[0])
print(f"Division control: {len(division_frames)} cancer daughters retained")


def permutation_null(entries_by_col, frame_events):
    """Chance entry recall and chance division hits, preserving entries/cell."""
    if not frame_events:
        return dict(chance_entry_recall=np.nan, entry_recall_p=np.nan,
                    chance_division_hits=np.nan)
    active_frames = {c: np.where(active[:, c])[0] for c in range(n_cells)}
    observed = sum(any(lo - tolerance <= t <= hi + tolerance
                       for t in entries_by_col.get(col, []))
                   for col, lo, hi in frame_events)
    recalls = np.zeros(N_PERMUTATIONS)
    div_hits = np.zeros(N_PERMUTATIONS)
    div_items = list(division_frames.items())
    for b in range(N_PERMUTATIONS):
        shuffled = {}
        for col, entries in entries_by_col.items():
            frames = active_frames[col]
            shuffled[col] = ([] if len(entries) == 0 or len(frames) == 0
                             else _rng.choice(frames,
                                              size=min(len(entries), len(frames)),
                                              replace=False))
        recalls[b] = sum(any(lo - tolerance <= t <= hi + tolerance
                             for t in shuffled.get(col, []))
                         for col, lo, hi in frame_events)
        div_hits[b] = sum(1 for col, frame in div_items
                          if any(abs(t - frame) <= tolerance
                                 for t in shuffled.get(col, [])))
    return dict(
        chance_entry_recall=round(float(recalls.mean() / len(frame_events)), 3),
        entry_recall_p=round(float((recalls >= observed).mean()), 4),
        chance_division_hits=round(float(div_hits.mean()), 2))


def evaluate_fit(arm, variant, k):
    """Score one fit; returns (summary row, per-event rows, condition rows)."""
    fit_dir = os.path.join(fits_dir, arm, variant, f"k{k}")
    with open(os.path.join(fit_dir, "fit_summary.yml")) as fh:
        info = yaml.safe_load(fh)
    assignments = np.load(os.path.join(fit_dir, "state_assignments.npy"))
    dead_state = info["dead_state"]
    is_dead = active & (assignments == dead_state)

    entries_by_col = {c: entry_frames(is_dead[:, c], active[:, c])
                      for c in range(n_cells)}

    event_rows, frame_events = [], []
    hits = frame_level = 0
    already_dead_captured = None
    cell_level_hits = cell_level = 0
    persistences = []

    for ev in events:
        col = col_of.get((ev["crop"], ev["cell_id"]))
        row = dict(arm=arm, variant=variant, k=k, crop=ev["crop"],
                   cell_id=ev["cell_id"], confidence=ev.get("confidence"))
        if col is None:
            row.update(status="cell_filtered_out", detected=False)
            event_rows.append(row)
            continue

        entries = entries_by_col[col]
        frac = float(is_dead[active[:, col], col].mean())
        row["dead_entries"] = len(entries)
        row["frac_frames_in_dead_state"] = round(frac, 3)

        if ev.get("local_frames"):
            lo, hi = ev["local_frames"]
            matched = [t for t in entries if lo - tolerance <= t <= hi + tolerance]
            frame_level += 1
            hits += bool(matched)
            frame_events.append((col, lo, hi))
            if matched:
                p = persistence_after_entry(is_dead[:, col], active[:, col],
                                            matched[0])
                persistences.append(p)
                row["persistence_after_entry"] = round(p, 3)
            row.update(status="frame_level", window=[lo, hi],
                       matched_entry=matched[0] if matched else None,
                       detected=bool(matched))
        elif ev.get("already_dead"):
            # The model should have this cell dead essentially throughout,
            # reachable only via the initial distribution.
            already_dead_captured = frac > 0.5
            row.update(status="already_dead", detected=bool(already_dead_captured))
        else:
            cell_level += 1
            detected = len(entries) > 0
            cell_level_hits += bool(detected)
            row.update(status="cell_level", detected=bool(detected))
        event_rows.append(row)

    total_entries = sum(len(v) for v in entries_by_col.values())
    cells_with_entry = sum(1 for v in entries_by_col.values() if v)
    division_hits = sum(1 for col, frame in division_frames.items()
                        if any(abs(t - frame) <= tolerance
                               for t in entries_by_col.get(col, [])))

    # Cell-level death fraction, overall and per condition.
    ever_dead = np.array([is_dead[:, c].any() for c in range(n_cells)])
    condition_rows = []
    for condition, crops in conditions_dict.items():
        cols = [c for c in range(n_cells) if crop_of_col[c] in crops]
        if cols:
            condition_rows.append(dict(
                arm=arm, variant=variant, k=k, condition=condition,
                n_cells=len(cols),
                frac_cells_ever_dead=round(float(ever_dead[cols].mean()), 4)))

    summary = dict(
        arm=arm, variant=variant, k=k, emission_dim=info["emission_dim"],
        absorbing_state=info["absorbing_state"], dead_state=dead_state,
        dead_state_is_absorbing=info["dead_state_is_absorbing"],
        occupancy=round(info["dead_state_occupancy"], 4),
        self_transition=round(info["dead_state_self_transition"], 3),
        dead_state_contrast=round(dead_state_contrast(is_dead), 3),
        entry_hits=f"{hits}/{frame_level}",
        entry_recall=(hits / frame_level) if frame_level else np.nan,
        mean_persistence_after_entry=(round(float(np.mean(persistences)), 3)
                                      if persistences else np.nan),
        already_dead_captured=already_dead_captured,
        cell_level_hits=f"{cell_level_hits}/{cell_level}",
        frac_cells_ever_dead=round(float(ever_dead.mean()), 4),
        entries_per_entered_cell=(round(total_entries / cells_with_entry, 2)
                                  if cells_with_entry else np.nan),
        division_false_positives=f"{division_hits}/{len(division_frames)}",
        tcell_auroc=round(auroc(tcell_counts[active], is_dead[active]), 3),
        log_prob=info["log_prob_final"])
    summary.update(permutation_null(entries_by_col, frame_events))
    return summary, event_rows, condition_rows


print("=" * 60)
print("Step 3: Evaluating dead-state fits")
print("=" * 60)

fits = pd.read_csv(os.path.join(fits_dir, "summary.csv"))
rows, event_rows, condition_rows = [], [], []
for r in fits.itertuples():
    s, e, c = evaluate_fit(r.arm, r.variant, int(r.k))
    rows.append(s)
    event_rows.extend(e)
    condition_rows.extend(c)

evaluation = pd.DataFrame(rows).sort_values(
    ["arm", "variant", "k"]).reset_index(drop=True)
evaluation.to_csv(os.path.join(fits_dir, "evaluation.csv"), index=False)
pd.DataFrame(event_rows).to_csv(os.path.join(fits_dir, "event_detail.csv"),
                                index=False)
pd.DataFrame(condition_rows).to_csv(
    os.path.join(fits_dir, "condition_death_fractions.csv"), index=False)

show = ["arm", "variant", "k", "occupancy", "dead_state_contrast",
        "dead_state_is_absorbing", "entry_hits", "chance_entry_recall",
        "entry_recall_p", "mean_persistence_after_entry",
        "already_dead_captured", "frac_cells_ever_dead",
        "entries_per_entered_cell", "division_false_positives", "tcell_auroc"]
pd.set_option("display.width", 260)
print("\n" + evaluation.sort_values("dead_state_contrast", ascending=False)[
    show].to_string(index=False))
print("\n`dead_state_contrast` is the dead state's separation from the rest on "
      "the death signature, in pooled SDs (signed, positive = more "
      "death-like). Read it FIRST: a fit can score well on entry recall while "
      "its states are nearly identical, in which case the state partitions "
      "something other than the phenotype and the recall is not evidence of a "
      "death detector.")

n_frame_events = sum(1 for e in events if e.get("local_frames"))
print(f"\nMULTIPLICITY: {len(evaluation)} fits scored against "
      f"{n_frame_events} frame-level events. A Bonferroni-corrected 0.05 bar "
      f"is p < {0.05 / max(len(evaluation), 1):.5f}, which {n_frame_events} "
      f"events cannot reach. Treat `entry_recall_p` as ranking information, "
      f"not a significance test.")

fig, axes = plt.subplots(1, 4, figsize=(19, 4), tight_layout=True)
for (arm, variant), sub in evaluation.groupby(["arm", "variant"], sort=False):
    style = "-" if variant == "absorbing" else "--"
    label = f"{arm} [{variant}]"
    axes[0].plot(sub["k"], sub["entry_recall"], style, marker="o", label=label)
    axes[1].plot(sub["k"], sub["occupancy"], style, marker="o", label=label)
    axes[2].plot(sub["k"], sub["mean_persistence_after_entry"], style,
                 marker="o", label=label)
    axes[3].plot(sub["k"], sub["tcell_auroc"], style, marker="o", label=label)
axes[0].set_ylabel("Entry recall on annotated deaths")
axes[0].set_ylim(-0.05, 1.05)
axes[1].set_ylabel("Dead-state occupancy")
axes[2].set_ylabel("Persistence after entry")
axes[2].set_ylim(-0.05, 1.05)
axes[3].set_ylabel("T-cell AUROC (independent)")
axes[3].axhline(0.5, color="grey", lw=0.8, ls=":")
for ax in axes:
    ax.set_xlabel("num_states (k)")
    ax.legend(fontsize=5)
plt.savefig(os.path.join(fits_dir, "evaluation.png"), dpi=200, bbox_inches="tight")
plt.close()

print(f"\nSaved evaluation.csv / event_detail.csv / "
      f"condition_death_fractions.csv / evaluation.png -> {fits_dir}")
print("=" * 60)
