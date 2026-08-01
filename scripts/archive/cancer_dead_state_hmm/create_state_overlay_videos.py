"""Step 4: Render state-overlay videos, one per (fit, crop).

Phase background, T cells as a blue layer, cancer cells tinted by HMM state,
cell IDs labelled, warmup frames grey, and the fit's identity (arm, variant,
k, lag order, features) in the title so videos cannot be confused once
separated from their directory.

Nothing derived from the ground-truth annotations is drawn -- no highlighting
of annotated cells, no markers for annotated windows -- so watching a video is
an independent read on whether the states track death rather than a prompted
one.  The dead-candidate state IS named in the legend, since that label comes
from the label-blind scoring rule in `fit_dead_state_hmm.py`, and the
absorbing state is named because it is a property of the model.

Which fits get rendered is controlled by `video_fits` in config.yml, or by
command line:
    python create_state_overlay_videos.py                        # config
    python create_state_overlay_videos.py memory                 # all its fits
    python create_state_overlay_videos.py memory:absorbing:4     # one fit

Outputs: {output_base_dir}/fits/{arm}/{variant}/k{K}/{crop}_state_overlay.mp4

Usage (OccidentAnalysis):
    conda run -n OccidentAnalysis python create_state_overlay_videos.py
"""

import os
import sys
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_script_dir = Path(__file__).resolve().parent
with open(_script_dir / "config.yml", "r") as f:
    cfg = yaml.safe_load(f)

crop_ids = cfg["crop_ids"]
tracking_crops_dir = cfg["tracking_crops_dir"]
type_sep_tracks_dir = cfg["type_sep_tracks_dir"]
out_base_dir = cfg["output_base_dir"]
video_fps = cfg["video_fps"]
video_figsize = tuple(cfg["video_figsize"])
fits_dir = os.path.join(out_base_dir, "fits")

sys.path.insert(0, cfg["imaging_pipeline_dir"])
import tifffile  # noqa: E402
from scripts.utils.PlottingUtils import (  # noqa: E402
    create_fixed_colormap, create_fixed_norm, plt_to_mp4)

SMALL_SIZE = 7
plt.rc("font", size=SMALL_SIZE)
plt.rc("axes", titlesize=8, labelsize=SMALL_SIZE)
plt.rc("xtick", labelsize=SMALL_SIZE)
plt.rc("ytick", labelsize=SMALL_SIZE)
plt.rc("legend", fontsize=SMALL_SIZE)
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.use14corefonts"] = True

WARMUP_VAL = -1
WARMUP_COLOUR = (0.5, 0.5, 0.5, 1.0)


def parse_fraction(text):
    """'3/15' -> 3.0; NaN for anything unparseable."""
    try:
        return float(str(text).split("/")[0])
    except (ValueError, AttributeError):
        return np.nan


def resolve_fits():
    """Decide which (arm, variant, k) triples to render."""
    summary = pd.read_csv(os.path.join(fits_dir, "summary.csv"))
    available = {(r.arm, r.variant, int(r.k)) for r in summary.itertuples()}

    if len(sys.argv) > 1:
        requested = []
        for token in sys.argv[1:]:
            parts = token.split(":")
            matches = [t for t in sorted(available)
                       if t[0] == parts[0]
                       and (len(parts) < 2 or t[1] == parts[1])
                       and (len(parts) < 3 or t[2] == int(parts[2]))]
            if not matches:
                sys.exit(f"Nothing fitted matches '{token}'")
            requested.extend(matches)
        return requested

    spec = cfg.get("video_fits", "auto")
    if spec == "all":
        return sorted(available)
    if isinstance(spec, list):
        return [(d["arm"], d.get("variant", "absorbing"), int(d["k"]))
                for d in spec]

    eval_path = os.path.join(fits_dir, "evaluation.csv")
    if not os.path.exists(eval_path):
        sys.exit("video_fits: auto needs evaluation.csv -- run "
                 "evaluate_dead_states.py first, or set an explicit list.")
    ev = pd.read_csv(eval_path)
    ev["div_fp"] = ev["division_false_positives"].map(parse_fraction)
    chosen = []
    for (arm, variant), group in ev.groupby(["arm", "variant"], sort=False):
        best = group.sort_values(["entry_recall_p", "div_fp", "occupancy"]).iloc[0]
        chosen.append((arm, variant, int(best["k"])))
    return chosen


fits = resolve_fits()
cell_index = pd.read_csv(os.path.join(fits_dir, "cell_index.csv"))
active_mask = np.load(os.path.join(fits_dir, "active_mask.npy"))

print("=" * 60)
print(f"Step 4: Rendering {len(fits)} fit(s) x {len(crop_ids)} crops = "
      f"{len(fits) * len(crop_ids)} videos")
for arm, variant, k in fits:
    print(f"    {arm} [{variant}] k={k}")
print("=" * 60)

crop_cache = {}
for crop in crop_ids:
    well = crop.split("_")[0]
    raw = np.asarray(tifffile.imread(
        os.path.join(tracking_crops_dir, well, crop, "crop.tiff")))
    tracks = np.asarray(tifffile.imread(
        os.path.join(type_sep_tracks_dir, crop, "tracks.tiff")))
    crop_cache[crop] = dict(raw=raw, t_cell=tracks[..., 0], cancer=tracks[..., 1])
    print(f"  loaded {crop}: raw {raw.shape}")


for arm, variant, k in fits:
    fit_dir = os.path.join(fits_dir, arm, variant, f"k{k}")
    with open(os.path.join(fit_dir, "fit_summary.yml")) as fh:
        info = yaml.safe_load(fh)
    assignments = np.load(os.path.join(fit_dir, "state_assignments.npy"))
    num_states = info["num_states"]
    dead_state = info["dead_state"]
    num_lags = info["num_lags"]
    absorbing_state = info["absorbing_state"]
    feat_str = ", ".join(info["features"])

    print("\n" + "=" * 60)
    print(f"Fit {arm} [{variant}] k={k}  (dead candidate = state {dead_state}, "
          f"occupancy {info['dead_state_occupancy']:.3f})")
    print("=" * 60)

    cmap = create_fixed_colormap(num_states)
    norm = create_fixed_norm(num_states)

    for crop in crop_ids:
        cached = crop_cache[crop]
        cancer_tracks = cached["cancer"].copy()
        t_cell_tracks, raw_tiff = cached["t_cell"], cached["raw"]
        T = cancer_tracks.shape[0]

        rows = cell_index[cell_index["crop"] == crop]
        keep_ids = rows["cvat_cell_id"].to_numpy()
        columns = rows["column"].to_numpy()
        # Cells dropped by the min_t filter carry no state; hide them.
        cancer_tracks[~np.isin(cancer_tracks, keep_ids)] = 0

        # 0 = background, -1 = warmup/uninferred, 1..K = state + 1
        state_tracks = np.zeros_like(cancer_tracks, dtype=np.int32)
        for cell_id, col in zip(keep_ids, columns):
            for t in range(T):
                pixels = cancer_tracks[t] == cell_id
                if not pixels.any():
                    continue
                state_tracks[t][pixels] = (
                    assignments[t, col] + 1 if active_mask[t, col] else WARMUP_VAL)

        def frame_fn(t, _raw=raw_tiff, _cancer=cancer_tracks, _state=state_tracks,
                     _tcell=t_cell_tracks, _cmap=cmap, _norm=norm, _crop=crop,
                     _ns=num_states, _dead=dead_state, _arm=arm, _var=variant,
                     _k=k, _lags=num_lags, _feats=feat_str, _abs=absorbing_state):
            bg = _raw[t, ..., 1] if _raw.ndim == 4 else _raw[t]
            plt.imshow(bg, cmap="gray")
            plt.imshow(np.where(_tcell[t] != 0, 1, 0), cmap="Blues", alpha=0.30)

            state_frame = _state[t]
            warmup_rgba = np.zeros((*state_frame.shape, 4))
            warmup_rgba[state_frame == WARMUP_VAL] = WARMUP_COLOUR
            warmup_rgba[..., 3] *= 0.55
            plt.imshow(warmup_rgba)
            plt.imshow(np.where(state_frame > 0, state_frame, 0),
                       cmap=_cmap, norm=_norm, alpha=0.55)

            # Cell IDs only -- nothing here may depend on the annotations.
            cancer_frame = _cancer[t]
            for track_id in np.unique(cancer_frame[cancer_frame > 0]):
                yx = np.argwhere(cancer_frame == track_id).mean(axis=0)
                plt.text(yx[1], yx[0], str(int(track_id)), color="white",
                         fontsize=SMALL_SIZE, fontweight="bold",
                         ha="center", va="center")

            handles = []
            for i in range(1, _ns + 1):
                s = i - 1
                tags = []
                if s == _dead:
                    tags.append("dead candidate")
                if _abs is not None and s == _abs:
                    tags.append("absorbing")
                label = f"{s}  <- {', '.join(tags)}" if tags else str(s)
                handles.append(mpatches.Patch(color=_cmap(i), label=label))
            handles.append(mpatches.Patch(color="grey", label="warmup"))
            handles.append(mpatches.Patch(color="tab:blue", label="T cell"))
            plt.legend(title="state", handles=handles, loc="upper left")

            plt.title(f"{_crop}, frame {t + 1}/{T}\n"
                      f"arm {_arm} [{_var}], k={_k}, dead candidate = "
                      f"state {_dead}\nnum_lags: {_lags}, features: {_feats}")
            plt.axis("off")

        video_path = os.path.join(fit_dir, f"{crop}_state_overlay.mp4")
        print(f"  {crop}: {len(keep_ids)} cells -> {os.path.basename(video_path)}")
        plt_to_mp4(frame_fn, list(range(T)), video_path,
                   fps=video_fps, figsize=video_figsize)

print("\n" + "=" * 60)
print(f"Rendered {len(fits) * len(crop_ids)} videos under {fits_dir}")
print("=" * 60)
