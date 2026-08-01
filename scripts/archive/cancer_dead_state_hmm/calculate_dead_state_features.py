"""Step 1: Per-cancer-cell, per-frame features for the dead-state HMM.

Cancer cells are the PHASE masks -- channel 1 of the type-separated CVAT
tracks -- so columns index the ground-truth death annotations directly.

Eighteen features, in four groups.  Arms select subsets; the rest stay
available as annotation-independent diagnostics.

GEOMETRY (0-9), carried over from cancer_death_hmm:
    area, circularity, velocity, t_cell_neighbors_20px,
    dilated_t_cell_neighbors, d_area_frac, d_circularity,
    win_std_log_area, win_std_circularity, win_std_displacement

MEMORY (10-11) -- monotone within a track:
    area_over_running_max          area_t / max(area over active frames <= t)
    running_max_abs_d_circularity  running max of |d_circularity|

    These exist because a post-death frame is otherwise indistinguishable
    from an ordinary small round cell.  Nothing in pure geometry tells the
    model that a cell has ALREADY had its event, which is what an absorbing
    dead state needs.

PHASE INTENSITY (12-14):
    phase_mean      mean phase inside the mask (apoptotic cells round up and
                    go bright/refractile before lysing)
    phase_std       within-mask SD (granularity / blebbing texture)
    phase_contrast  mask mean minus the mean of a disk(ring_radius) ring
                    just outside it -- the refractile halo, background-corrected

RFP INTENSITY (15-17), only cancer cells carry RFP nuclei:
    rfp_mean           mean RFP inside the mask
    rfp_cv             within-mask SD / mean -- rises as the nucleus condenses
                       and fragments into bright puncta on a dark background
    rfp_concentration  share of the mask's total RFP in its brightest
                       `rfp_top_fraction` of pixels -- a direct condensation
                       measure that is scale-free in absolute intensity

Both channels come from the TrackingCrops crop.tiff (last axis: 0 = RFP,
1 = phase), which is the image actually aligned with the tracks.  Each
channel is normalized ONCE per crop over the whole stack using fixed
percentiles, so per-frame illumination wobble cannot become the dominant
axis and crop-level offsets do not turn into a batch effect.

Saved per crop, into {output_base_dir}/{crop}/:
    cancer_cell_ids.npy         (N,)      sorted CVAT track IDs == column order
    cancer_phase_centroids.npy  (T,N,2)   crop-local (y,x); NaN where absent
    cancer_emissions_array.npy  (T,N,18)
    cancer_emissions_names.txt

Usage (OccidentAnalysis):
    conda run -n OccidentAnalysis python calculate_dead_state_features.py
"""

import os
import sys
from pathlib import Path

import yaml
import numpy as np
import tifffile
from skimage.measure import label as sklabel, regionprops
from skimage.morphology import binary_dilation, disk

_script_dir = Path(__file__).resolve().parent
with open(_script_dir / "config.yml", "r") as f:
    cfg = yaml.safe_load(f)

crop_ids = cfg["crop_ids"]
type_sep_tracks_dir = cfg["type_sep_tracks_dir"]
tracking_crops_dir = cfg["tracking_crops_dir"]
out_base_dir = cfg["output_base_dir"]
feature_names = cfg["emission_feature_names"]
radius_px = cfg["radius_px"]
dilate_radius = cfg["dilate_radius"]
window_frames = cfg["window_frames"]
ring_radius = cfg["ring_radius"]
top_fraction = cfg["rfp_top_fraction"]
lo_pct, hi_pct = cfg["intensity_percentiles"]

sys.path.insert(0, cfg["imaging_pipeline_dir"])
from scripts.utils.StatUtils import (  # noqa: E402
    calculate_centroids_per_frame_dict,
    get_cell_neighbors,
)

IDX = {name: i for i, name in enumerate(feature_names)}
RING_SELEM = disk(ring_radius)


def circularity_of(mask):
    """4*pi*area / perimeter^2 for the largest connected component.

    Args:
        mask (np.ndarray): (H, W) boolean single-cell mask.

    Returns:
        float: circularity, NaN if empty or degenerate.  A disc scores 1.0;
            discretization can push very small masks slightly above 1.
    """
    props = regionprops(sklabel(mask.astype(int)))
    if not props:
        return np.nan
    region = max(props, key=lambda p: p.area)
    if region.perimeter <= 0:
        return np.nan
    return float(4.0 * np.pi * region.area / (region.perimeter ** 2))


def normalize_stack(stack, lo_p, hi_p):
    """Scale a (T, H, W) stack to [0, 1] using percentiles of the whole stack."""
    lo, hi = np.percentile(stack, [lo_p, hi_p])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((stack.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def intensity_features(mask, phase_frame, rfp_frame):
    """Six intensity features for one cell mask in one frame.

    Args:
        mask (np.ndarray): (H, W) boolean single-cell mask.
        phase_frame (np.ndarray): (H, W) normalized phase.
        rfp_frame (np.ndarray): (H, W) normalized RFP.

    Returns:
        dict: phase_mean, phase_std, phase_contrast, rfp_mean, rfp_cv,
            rfp_concentration.
    """
    phase_vals = phase_frame[mask]
    rfp_vals = rfp_frame[mask]

    ring = binary_dilation(mask, RING_SELEM) & ~mask
    ring_mean = float(phase_frame[ring].mean()) if ring.any() else np.nan

    rfp_mean = float(rfp_vals.mean())
    # CV is undefined for a dark mask; 0 is the right answer there (no
    # structure to speak of) and keeps the feature finite.
    rfp_cv = float(rfp_vals.std() / rfp_mean) if rfp_mean > 1e-6 else 0.0

    total = float(rfp_vals.sum())
    if total > 1e-9:
        n_top = max(1, int(round(top_fraction * rfp_vals.size)))
        brightest = np.sort(rfp_vals)[-n_top:].sum()
        rfp_concentration = float(brightest / total)
    else:
        # With no signal, the brightest pixels hold exactly their pixel
        # share -- the value a uniform mask would give.
        rfp_concentration = float(top_fraction)

    phase_mean = float(phase_vals.mean())
    return dict(
        phase_mean=phase_mean,
        phase_std=float(phase_vals.std()),
        phase_contrast=(phase_mean - ring_mean) if np.isfinite(ring_mean) else 0.0,
        rfp_mean=rfp_mean,
        rfp_cv=rfp_cv,
        rfp_concentration=rfp_concentration,
    )


def trailing_std(series, active, window):
    """Trailing rolling std over a cell's active frames; 0 where undefined."""
    out = np.zeros(len(series), dtype=float)
    for t in np.where(active)[0]:
        lo = max(0, t - window + 1)
        vals = series[lo:t + 1][active[lo:t + 1]]
        vals = vals[~np.isnan(vals)]
        if len(vals) >= 2:
            out[t] = float(np.std(vals))
    return out


def count_tcell_neighbors(centroids, tcell_centroids_dict, radius, T, N):
    """Count T-cell centroids within `radius` px of each cancer centroid."""
    counts = np.zeros((T, N), dtype=float)
    for t in range(T):
        tcells = tcell_centroids_dict.get(t, {})
        if not tcells:
            continue
        tc = np.array(list(tcells.values()), dtype=float)
        cent_t = centroids[t]
        present = ~np.isnan(cent_t[:, 0])
        if not present.any():
            continue
        d = np.linalg.norm(cent_t[present, None, :] - tc[None, :, :], axis=2)
        counts[t, present] = np.sum(d <= radius, axis=1)
    return counts


print("=" * 60)
print("Step 1: Dead-state features (geometry + memory + phase/RFP intensity)")
print("=" * 60)

for crop in crop_ids:
    print("\n" + "-" * 60)
    print(f"Crop {crop}")
    well = crop.split("_")[0]

    tracks = tifffile.imread(os.path.join(type_sep_tracks_dir, crop, "tracks.tiff"))
    t_cell_tracks, cancer_tracks = tracks[..., 0], tracks[..., 1]
    T = cancer_tracks.shape[0]

    images = np.asarray(tifffile.imread(
        os.path.join(tracking_crops_dir, well, crop, "crop.tiff")))
    assert images.shape[0] == T, (
        f"{crop}: crop.tiff has {images.shape[0]} frames, tracks have {T}")
    rfp = normalize_stack(images[..., 0], lo_pct, hi_pct)
    phase = normalize_stack(images[..., 1], lo_pct, hi_pct)

    cancer_cell_ids = np.sort(np.unique(cancer_tracks[cancer_tracks > 0]))
    N = len(cancer_cell_ids)
    id_to_col = {int(cid): i for i, cid in enumerate(cancer_cell_ids)}
    print(f"  T={T}, num_cancer={N}, images {images.shape}")

    emissions = np.zeros((T, N, len(feature_names)), dtype=float)
    centroids = np.full((T, N, 2), np.nan, dtype=float)
    active = np.zeros((T, N), dtype=bool)

    # --- Per-frame geometry, contact and intensity ---
    for t in range(T):
        c_frame = cancer_tracks[t]
        tc_frame = t_cell_tracks[t]
        for cid in np.unique(c_frame[c_frame > 0]):
            col = id_to_col[int(cid)]
            mask = c_frame == cid
            active[t, col] = True

            emissions[t, col, IDX["area"]] = float(mask.sum())
            emissions[t, col, IDX["circularity"]] = circularity_of(mask)
            ys, xs = np.nonzero(mask)
            centroids[t, col] = (ys.mean(), xs.mean())
            emissions[t, col, IDX["dilated_t_cell_neighbors"]] = len(
                get_cell_neighbors(mask, tc_frame, dilate_radius, exclude_ids=None))

            for name, value in intensity_features(mask, phase[t], rfp[t]).items():
                emissions[t, col, IDX[name]] = value

    tcell_centroids_dict = calculate_centroids_per_frame_dict(t_cell_tracks)
    emissions[:, :, IDX["t_cell_neighbors_20px"]] = count_tcell_neighbors(
        centroids, tcell_centroids_dict, radius_px, T, N)

    # --- Per-cell temporal features over ACTIVE frames only ---
    for col in range(N):
        act = active[:, col]
        frames = np.where(act)[0]
        if len(frames) == 0:
            continue

        area = np.where(act, emissions[:, col, IDX["area"]], np.nan)
        circ = np.where(act, emissions[:, col, IDX["circularity"]], np.nan)
        disp = np.zeros(T, dtype=float)

        for i in range(1, len(frames)):
            t, prev = frames[i], frames[i - 1]
            if area[prev] > 0:
                emissions[t, col, IDX["d_area_frac"]] = (
                    (area[t] - area[prev]) / area[prev])
            if not (np.isnan(circ[t]) or np.isnan(circ[prev])):
                emissions[t, col, IDX["d_circularity"]] = circ[t] - circ[prev]
            step = float(np.linalg.norm(centroids[t, col] - centroids[prev, col]))
            disp[t] = step / float(t - prev)
        emissions[:, col, IDX["velocity"]] = disp

        log_area = np.where(act & (area > 0),
                            np.log(np.where(area > 0, area, 1.0)), np.nan)
        emissions[:, col, IDX["win_std_log_area"]] = trailing_std(
            log_area, act, window_frames)
        emissions[:, col, IDX["win_std_circularity"]] = trailing_std(
            circ, act, window_frames)
        emissions[:, col, IDX["win_std_displacement"]] = trailing_std(
            np.where(act, disp, np.nan), act, window_frames)

        # --- Memory features: monotone along the cell's own active frames ---
        running_max_area = 0.0
        running_max_dcirc = 0.0
        for t in frames:
            running_max_area = max(running_max_area, float(area[t]))
            if running_max_area > 0:
                emissions[t, col, IDX["area_over_running_max"]] = (
                    float(area[t]) / running_max_area)
            running_max_dcirc = max(
                running_max_dcirc,
                abs(float(emissions[t, col, IDX["d_circularity"]])))
            emissions[t, col, IDX["running_max_abs_d_circularity"]] = running_max_dcirc

    n_nan = int(np.isnan(emissions).sum())
    if n_nan:
        print(f"  WARNING: {n_nan} NaN feature values (degenerate masks) -> 0.0")
        emissions = np.nan_to_num(emissions, nan=0.0)

    out_dir = os.path.join(out_base_dir, crop)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "cancer_cell_ids.npy"), cancer_cell_ids)
    np.save(os.path.join(out_dir, "cancer_phase_centroids.npy"), centroids)
    np.save(os.path.join(out_dir, "cancer_emissions_array.npy"), emissions)
    with open(os.path.join(out_dir, "cancer_emissions_names.txt"), "w") as fh:
        fh.write("\n".join(feature_names) + "\n")

    act_vals = emissions[active]
    print(f"  emissions {emissions.shape}, {int(active.sum())} active cell-frames")
    for i, name in enumerate(feature_names):
        v = act_vals[:, i]
        print(f"    {name:<30s} mean={v.mean():9.3f}  "
              f"[{v.min():9.3f}, {v.max():9.3f}]")
    print(f"  saved -> {out_dir}")

print("\n" + "=" * 60)
print("Done.")
print("=" * 60)
