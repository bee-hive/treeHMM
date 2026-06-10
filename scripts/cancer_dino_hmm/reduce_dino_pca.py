"""
Step 3: PCA-reduce the DINOv2 embeddings to the top-N principal components.

One JOINT PCA is fit across all crops (so PCs are comparable across crops, like
the jointly-fit HMM).  Only valid (real-patch) cell-frame embeddings are used to
fit; invalid (absent-nucleus) entries are written as zeros in the output (matching
how the model pads inactive cells).

Saved:
  per crop:  {output_base_dir}/{crop}/cancer_dino_pca.npy      (T, N, n_dino_pcs)
  global:    {output_base_dir}/dino_pca_components.npy          (n_dino_pcs, 768)
             {output_base_dir}/dino_pca_explained_variance_ratio.npy (n_dino_pcs,)
             {output_base_dir}/dino_pca_mean.npy                (768,)

Usage (cs229Dino):
    conda run -n cs229Dino python reduce_dino_pca.py
"""

import os
import sys
from pathlib import Path

import yaml
import numpy as np
from sklearn.decomposition import PCA

_script_dir = Path(__file__).resolve().parent
with open(_script_dir / "config.yml", "r") as f:
    cfg = yaml.safe_load(f)

crop_ids = cfg["crop_ids"]
out_base_dir = cfg["output_base_dir"]
n_pcs = cfg["n_dino_pcs"]

if not cfg.get("use_dino", True):
    print("use_dino is false; skipping PCA step.")
    sys.exit(0)

print("=" * 60)
print("Step 3: PCA-reducing DINOv2 embeddings")
print("=" * 60)

# --- Gather valid embeddings across all crops ---
raws, valids = {}, {}
valid_rows = []
for crop in crop_ids:
    out_dir = os.path.join(out_base_dir, crop)
    raw = np.load(os.path.join(out_dir, "cancer_dino_raw.npy"))          # (T, N, 768)
    valid = np.load(os.path.join(out_dir, "cancer_dino_valid_mask.npy")) # (T, N)
    raws[crop] = raw
    valids[crop] = valid
    valid_rows.append(raw[valid])  # (n_valid_crop, 768)

X_valid = np.concatenate(valid_rows, axis=0)
print(f"Fitting PCA on {X_valid.shape[0]} valid cell-frame embeddings "
      f"(dim {X_valid.shape[1]}) -> {n_pcs} PCs")

n_components = min(n_pcs, X_valid.shape[0], X_valid.shape[1])
if n_components < n_pcs:
    print(f"WARNING: only {n_components} components available (< n_dino_pcs={n_pcs})")

pca = PCA(n_components=n_components).fit(X_valid)
evr = pca.explained_variance_ratio_
print("Explained variance ratio per PC:")
for i, v in enumerate(evr):
    print(f"  PC{i:2d}: {v:.4f}   (cumulative {np.cumsum(evr)[i]:.4f})")
print(f"Top-{n_components} cumulative explained variance: {evr.sum():.4f}")

# --- Transform each crop, zeroing invalid cell-frames ---
for crop in crop_ids:
    raw = raws[crop]
    valid = valids[crop]
    T, N, D = raw.shape
    pcs = np.zeros((T, N, n_components), dtype=np.float32)
    flat_valid = valid.reshape(-1)
    flat_raw = raw.reshape(-1, D)
    if flat_valid.any():
        transformed = pca.transform(flat_raw[flat_valid])
        out = pcs.reshape(-1, n_components)
        out[flat_valid] = transformed
        pcs = out.reshape(T, N, n_components)
    np.save(os.path.join(out_base_dir, crop, "cancer_dino_pca.npy"), pcs)
    print(f"  {crop}: saved cancer_dino_pca.npy {pcs.shape}")

# --- Save global PCA artifacts ---
np.save(os.path.join(out_base_dir, "dino_pca_components.npy"), pca.components_)
np.save(os.path.join(out_base_dir, "dino_pca_explained_variance_ratio.npy"), evr)
np.save(os.path.join(out_base_dir, "dino_pca_mean.npy"), pca.mean_)
print(f"\nSaved global PCA artifacts -> {out_base_dir}")

print("\n" + "=" * 60)
print("Done.")
print("=" * 60)
