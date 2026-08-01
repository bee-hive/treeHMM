"""Step 2: DINOv2 embeddings of mask-annotated patches around each cancer cell.

SELF-CONTAINED: the cs229Dino env lacks the MarsonImagingPipeline
dependencies, so this imports only numpy / tifffile / torch / transformers
/ yaml and inlines the few helpers it needs.

For each crop:
  - read the pre-cut crop.tiff (channel 0 = RFP nuclei, channel 1 = phase)

    NOTE: this MUST come from TrackingCrops, not from slicing the per-well
    registered.tiff with the crop_id's coordinates.  Those coordinates do not
    land on the same region -- checked against the masks, the registered
    slice puts as much RFP in T-cell masks as in cancer masks (ratio to
    background 2.06 vs 2.64), whereas the TrackingCrops image gives 1.74 vs
    3.83.  Only cancer cells carry RFP, so the TrackingCrops image is the one
    aligned with the tracks.  Embeddings cut from the registered slice
    describe the wrong pixels entirely.
  - normalize phase contrast ONCE over the whole crop stack using fixed
    percentiles.  Per-frame min-max (what the older pipelines used) lets a
    single bright artifact rescale a whole frame, which turns the leading
    PCs into a clock rather than a description of the cell -- exactly the
    failure `check_dino_confounds.py` was written to catch.
  - build an RGB per frame: phase as the luminance base, with cell masks
    ALPHA-BLENDED on top so texture (blebbing, granularity, the bright halo
    of a rounded-up cell) survives instead of being replaced by flat colour.
        subject cancer cell -> red
        other cancer cells  -> blue
        T cells             -> green
    The subject gets its own colour because a 50 px patch frequently
    contains a second cancer cell, and a single "cancer red" would leave the
    embedding unable to tell which cell it is describing.
  - cut a dino_patch_px patch centered on the subject's phase centroid
    (edge-padded) and run facebook/dinov2-base over the batch.

Saved per crop, into {output_base_dir}/{crop}/:
    cancer_dino_raw.npy        (T, N, 768)  NaN where no valid patch
    cancer_dino_valid_mask.npy (T, N)       True where a real patch was embedded

Column ordering matches cancer_cell_ids.npy from Step 1.

Usage (cs229Dino):
    conda run -n cs229Dino python compute_dino_embeddings.py
"""

import os
import re
import sys
from glob import glob
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import yaml
import numpy as np
import tifffile
import torch
from transformers import AutoImageProcessor, AutoModel

_script_dir = Path(__file__).resolve().parent
with open(_script_dir / "config.yml", "r") as f:
    cfg = yaml.safe_load(f)

crop_ids = cfg["crop_ids"]
out_base_dir = cfg["output_base_dir"]
type_sep_tracks_dir = cfg["type_sep_tracks_dir"]
tracking_crops_dir = cfg["tracking_crops_dir"]
dino_model_id = cfg["dino_model_id"]
dino_model_path = cfg.get("dino_model_path")
patch_px = cfg["dino_patch_px"]
batch_size = cfg.get("dino_batch_size", 256)
mask_alpha = cfg["dino_mask_alpha"]
subject_colour = np.array(cfg["dino_subject_colour"], dtype=np.float32)
other_colour = np.array(cfg["dino_other_cancer_colour"], dtype=np.float32)
tcell_colour = np.array(cfg["dino_tcell_colour"], dtype=np.float32)
lo_pct, hi_pct = cfg["dino_norm_percentiles"]
include_rfp = cfg.get("dino_include_rfp", False)
rfp_alpha = cfg.get("dino_rgb_alpha_nuclei", 0.3)

if not cfg.get("use_dino", True):
    print("use_dino is false; skipping DINOv2 embedding step.")
    sys.exit(0)


def normalize_stack(stack, lo_p, hi_p):
    """Scale a (T, H, W) stack to [0, 1] using percentiles of the WHOLE stack.

    Args:
        stack (np.ndarray): (T, H, W) intensities.
        lo_p (float): lower percentile mapped to 0.
        hi_p (float): upper percentile mapped to 1.

    Returns:
        np.ndarray: (T, H, W) float32 in [0, 1], one shared scale for all frames.
    """
    lo, hi = np.percentile(stack, [lo_p, hi_p])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((stack.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def blend(img, mask, colour, alpha):
    """Alpha-blend a flat `colour` into `img` wherever `mask` is True.

    Args:
        img (np.ndarray): (H, W, 3) float32 in [0, 1], modified in place.
        mask (np.ndarray): (H, W) boolean.
        colour (np.ndarray): (3,) float32 RGB in [0, 1].
        alpha (float): blend strength; 0 leaves the image untouched.

    Returns:
        np.ndarray: the same array, for chaining.
    """
    if mask.any():
        img[mask] = (1.0 - alpha) * img[mask] + alpha * colour
    return img


def resolve_model_src(model_id, model_path):
    """Prefer a local snapshot directory; fall back to the repo id."""
    if model_path and os.path.isdir(os.path.join(model_path, "snapshots")):
        snaps = sorted(glob(os.path.join(model_path, "snapshots", "*")))
        if snaps:
            return snaps[-1]
    return model_id


device = "cuda" if torch.cuda.is_available() else "cpu"
model_src = resolve_model_src(dino_model_id, dino_model_path)
print(f"Loading DINOv2 from: {model_src}  (device={device})")
processor = AutoImageProcessor.from_pretrained(model_src, local_files_only=True)
model = AutoModel.from_pretrained(model_src, local_files_only=True).to(device).eval()
embed_dim = model.config.hidden_size


@torch.no_grad()
def embed_patches(patches):
    """Embed a list of (patch_px, patch_px, 3) uint8 arrays -> (len, embed_dim)."""
    out = np.zeros((len(patches), embed_dim), dtype=np.float32)
    for start in range(0, len(patches), batch_size):
        batch = patches[start:start + batch_size]
        inp = processor(images=batch, return_tensors="pt").to(device)
        res = model(**inp)
        emb = res.pooler_output
        if emb is None:
            emb = res.last_hidden_state[:, 0]
        out[start:start + len(batch)] = emb.detach().cpu().numpy()
    return out


print("=" * 60)
print(f"Step 2: DINOv2 embeddings ({patch_px}x{patch_px} mask-annotated patches)")
print("=" * 60)

half = patch_px // 2

for crop in crop_ids:
    print("\n" + "-" * 60)
    print(f"Crop {crop}")
    well = crop.split("_")[0]

    crop_img = np.asarray(tifffile.imread(
        os.path.join(tracking_crops_dir, well, crop, "crop.tiff")))  # (T, H, W, 2)
    phase = normalize_stack(crop_img[..., 1], lo_pct, hi_pct)
    rfp = normalize_stack(crop_img[..., 0], lo_pct, hi_pct) if include_rfp else None
    T = crop_img.shape[0]

    tracks = tifffile.imread(os.path.join(type_sep_tracks_dir, crop, "tracks.tiff"))
    t_cell_tracks, cancer_tracks = tracks[..., 0], tracks[..., 1]

    out_dir = os.path.join(out_base_dir, crop)
    cancer_cell_ids = np.load(os.path.join(out_dir, "cancer_cell_ids.npy"))
    centroids = np.load(os.path.join(out_dir, "cancer_phase_centroids.npy"))
    N = len(cancer_cell_ids)
    assert centroids.shape[:2] == (T, N), (
        f"centroids {centroids.shape} vs (T={T}, N={N}) mismatch for {crop}")

    raw = np.full((T, N, embed_dim), np.nan, dtype=np.float32)
    valid = np.zeros((T, N), dtype=bool)

    patches, index = [], []
    for t in range(T):
        base = np.repeat(phase[t][:, :, None], 3, axis=2).astype(np.float32)
        if include_rfp:
            # Mixed into luminance, not into a hue, so it cannot be confused
            # with the red the subject mask uses.
            base = (1.0 - rfp_alpha) * base + rfp_alpha * np.repeat(
                rfp[t][:, :, None], 3, axis=2)

        # Everything except the subject cell, painted once per frame.
        shared = blend(base.copy(), t_cell_tracks[t] > 0, tcell_colour, mask_alpha)
        all_cancer = blend(shared.copy(), cancer_tracks[t] > 0, other_colour, mask_alpha)

        for col, cid in enumerate(cancer_cell_ids):
            cy, cx = centroids[t, col]
            if np.isnan(cy):
                continue
            subject = cancer_tracks[t] == cid
            img = all_cancer.copy()
            # Repaint the subject FROM the pre-cancer image, so it comes out
            # red rather than red blended over blue.
            img[subject] = ((1.0 - mask_alpha) * shared[subject]
                            + mask_alpha * subject_colour)
            img_u8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
            padded = np.pad(img_u8, ((half, half), (half, half), (0, 0)), mode="edge")
            iy, ix = int(round(cy)), int(round(cx))
            patches.append(padded[iy:iy + patch_px, ix:ix + patch_px])
            index.append((t, col))

    print(f"  valid patches: {len(patches)} / {T * N}")
    if patches:
        emb = embed_patches(patches)
        for (t, col), e in zip(index, emb):
            raw[t, col] = e
            valid[t, col] = True

    np.save(os.path.join(out_dir, "cancer_dino_raw.npy"), raw)
    np.save(os.path.join(out_dir, "cancer_dino_valid_mask.npy"), valid)
    print(f"  saved raw {raw.shape} + valid mask -> {out_dir}")

print("\n" + "=" * 60)
print("Done.")
print("=" * 60)
