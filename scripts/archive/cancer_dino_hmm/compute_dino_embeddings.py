"""
Step 2: Compute DINOv2 embeddings of 25x25 crops centered on cancer nuclei.

SELF-CONTAINED: the cs229Dino environment lacks the MarsonImagingPipeline
dependencies, so this script imports ONLY numpy / tifffile / torch / transformers
/ yaml.  The tiny pure-numpy `get_RGB_image_with_nuclei` helper and the crop-id
slice parser are inlined here (copied from MarsonImagingPipeline PipelineUtils).

For each crop:
  - memmap the per-well registered.tiff and slice the crop region
    (channel 0 = RFP nuclei, channel 1 = phase)
  - build one RGB per full 150x150 crop frame (consistent contrast)
  - extract a 25x25 patch centered on each cancer-nucleus centroid (edge padded)
  - run facebook/dinov2-base (CLS / pooler output, 768-d) on the batched patches

Saved per crop (into {output_base_dir}/{crop}/):
    cancer_dino_raw.npy        (T, N, 768)  raw embeddings; NaN where no valid patch
    cancer_dino_valid_mask.npy (T, N)       True where a real patch was embedded

Column ordering matches cancer_cell_ids.npy / cancer_nucleus_centroids.npy from Step 1.

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

# ---------------------------------------------------------------------------
# Load shared configuration
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).resolve().parent
with open(_script_dir / "config.yml", "r") as f:
    cfg = yaml.safe_load(f)

crop_ids = cfg["crop_ids"]
out_base_dir = cfg["output_base_dir"]
registration_base_dir = cfg["registration_base_dir"]
dino_model_id = cfg["dino_model_id"]
dino_model_path = cfg.get("dino_model_path")
patch_px = cfg["dino_patch_px"]
alpha = cfg["dino_rgb_alpha"]
batch_size = cfg.get("dino_batch_size", 256)

if not cfg.get("use_dino", True):
    print("use_dino is false; skipping DINOv2 embedding step.")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Inlined helpers (pure numpy; copied from MarsonImagingPipeline to stay
# self-contained in the dependency-light cs229Dino env)
# ---------------------------------------------------------------------------
def get_RGB_image_with_nuclei(phase_image, nuclei_image, alpha=0.3):
    """Combine phase and nuclei single-channel images into an (H, W, 3) uint8 RGB.

    Each channel is min-max normalized over the whole array passed in; phase is
    duplicated across RGB at weight (1-alpha) and nuclei intensity is added into
    the red channel at weight alpha.
    """
    phase_image = (phase_image - np.min(phase_image)) / (np.max(phase_image) - np.min(phase_image))
    phase_image = (255 * phase_image).astype(np.uint8)

    nuclei_image = (nuclei_image - np.min(nuclei_image)) / (np.max(nuclei_image) - np.min(nuclei_image))
    nuclei_image = (255 * nuclei_image).astype(np.uint8)

    rgb_image = (np.stack([phase_image] * 3, axis=-1).astype(float) * (1.0 - alpha)).astype(np.uint8)
    rgb_image[..., 0] += (nuclei_image.astype(np.float32) * alpha).astype(np.uint8)
    rgb_image = np.clip(rgb_image, 0, 255)
    return rgb_image


def parse_crop_slices(crop_id):
    """Parse `B4_t50t100y200y350x750x900` -> (t0,t1,y0,y1,x0,x1).

    Replicates slice_indices_from_slice_id from MarsonImagingPipeline.
    """
    slice_part = crop_id.split("_", 1)[1]
    matches = re.findall(r"([a-zA-Z])(\d+)", slice_part)
    vals = [int(v) for _, v in matches]
    t0, t1, y0, y1, x0, x1 = vals[:6]
    return t0, t1, y0, y1, x0, x1


def resolve_model_src(model_id, model_path):
    """Prefer a local snapshot dir; fall back to the repo id (offline cache)."""
    if model_path and os.path.isdir(os.path.join(model_path, "snapshots")):
        snaps = sorted(glob(os.path.join(model_path, "snapshots", "*")))
        if snaps:
            return snaps[-1]
    return model_id


# ---------------------------------------------------------------------------
# Load DINOv2 once
# ---------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_src = resolve_model_src(dino_model_id, dino_model_path)
print(f"Loading DINOv2 from: {model_src}  (device={device})")
processor = AutoImageProcessor.from_pretrained(model_src, local_files_only=True)
model = AutoModel.from_pretrained(model_src, local_files_only=True).to(device).eval()
embed_dim = model.config.hidden_size


@torch.no_grad()
def embed_patches(patches):
    """patches: list of (patch_px, patch_px, 3) uint8 arrays -> (len, embed_dim)."""
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


def extract_patch(rgb_padded, cy, cx, half, patch):
    """Fixed `patch`x`patch` window centered at original (cy,cx) on an edge-padded RGB."""
    cy = int(round(cy)); cx = int(round(cx))
    return rgb_padded[cy:cy + patch, cx:cx + patch]


# ---------------------------------------------------------------------------
# Main loop over crops
# ---------------------------------------------------------------------------
print("=" * 60)
print("Step 2: Computing DINOv2 embeddings")
print("=" * 60)

half = patch_px // 2

for crop in crop_ids:
    print("\n" + "-" * 60)
    print(f"Crop {crop}")
    well = crop.split("_")[0]
    t0, t1, y0, y1, x0, x1 = parse_crop_slices(crop)

    reg_path = os.path.join(registration_base_dir, well, "registered.tiff")
    reg = tifffile.memmap(reg_path)
    crop_img = np.array(reg[t0:t1, y0:y1, x0:x1, :])  # (T, H, W, 2): ch0=RFP nuclei, ch1=phase
    T = crop_img.shape[0]

    out_dir = os.path.join(out_base_dir, crop)
    cancer_cell_ids = np.load(os.path.join(out_dir, "cancer_cell_ids.npy"))
    centroids = np.load(os.path.join(out_dir, "cancer_nucleus_centroids.npy"))  # (T, N, 2)
    N = len(cancer_cell_ids)
    assert centroids.shape[:2] == (T, N), (
        f"centroids {centroids.shape} vs (T={T}, N={N}) mismatch for {crop}"
    )

    raw = np.full((T, N, embed_dim), np.nan, dtype=np.float32)
    valid = np.zeros((T, N), dtype=bool)

    patches = []
    index = []  # (t, col)
    for t in range(T):
        rgb_t = get_RGB_image_with_nuclei(
            crop_img[t, ..., 1], crop_img[t, ..., 0], alpha=alpha
        )
        rgb_pad = np.pad(rgb_t, ((half, half), (half, half), (0, 0)), mode="edge")
        for col in range(N):
            cy, cx = centroids[t, col]
            if np.isnan(cy):
                continue
            patches.append(extract_patch(rgb_pad, cy, cx, half, patch_px))
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
