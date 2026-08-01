"""Step 3: PCA-reduce the DINOv2 embeddings to the top-N principal components.

One JOINT PCA across all crops, so the PCs are comparable across crops in
the same way the jointly-fit HMM is.  Only valid (real-patch) cell-frames
are used to fit; invalid entries are written as zeros, matching how the
model pads inactive cells.

Whitening is ON by default (`dino_pca_whiten`).  Unwhitened, PC1 of a
DINOv2 embedding carries orders of magnitude more variance than PC10, and
the leading component dominates the emission geometry regardless of whether
it has anything to do with the phenotype of interest.

Saved:
  per crop:  {output_base_dir}/{crop}/cancer_dino_pca.npy   (T, N, n_dino_pcs)
  global:    {output_base_dir}/dino_pca_components.npy
             {output_base_dir}/dino_pca_explained_variance_ratio.npy
             {output_base_dir}/dino_pca_mean.npy

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
whiten = cfg.get("dino_pca_whiten", True)

if not cfg.get("use_dino", True):
    print("use_dino is false; skipping PCA step.")
    sys.exit(0)

print("=" * 60)
print(f"Step 3: PCA-reducing DINOv2 embeddings (whiten={whiten})")
print("=" * 60)

raws, valids, valid_rows = {}, {}, []
for crop in crop_ids:
    out_dir = os.path.join(out_base_dir, crop)
    raw = np.load(os.path.join(out_dir, "cancer_dino_raw.npy"))
    valid = np.load(os.path.join(out_dir, "cancer_dino_valid_mask.npy"))
    raws[crop], valids[crop] = raw, valid
    valid_rows.append(raw[valid])

x_valid = np.concatenate(valid_rows, axis=0)
print(f"Fitting PCA on {x_valid.shape[0]} valid cell-frame embeddings "
      f"(dim {x_valid.shape[1]}) -> {n_pcs} PCs")

n_components = min(n_pcs, x_valid.shape[0], x_valid.shape[1])
if n_components < n_pcs:
    print(f"WARNING: only {n_components} components available (< n_dino_pcs={n_pcs})")

pca = PCA(n_components=n_components, whiten=whiten).fit(x_valid)
evr = pca.explained_variance_ratio_
print("Explained variance ratio per PC:")
for i, v in enumerate(evr):
    print(f"  PC{i:2d}: {v:.4f}   (cumulative {np.cumsum(evr)[i]:.4f})")
print(f"Variance explained by the {n_components} kept PCs: {100 * evr.sum():.2f}%")

for crop in crop_ids:
    raw, valid = raws[crop], valids[crop]
    T, N, D = raw.shape
    flat_valid = valid.reshape(-1)
    out = np.zeros((T * N, n_components), dtype=np.float32)
    if flat_valid.any():
        out[flat_valid] = pca.transform(raw.reshape(-1, D)[flat_valid])
    pcs = out.reshape(T, N, n_components)
    np.save(os.path.join(out_base_dir, crop, "cancer_dino_pca.npy"), pcs)
    print(f"  {crop}: saved cancer_dino_pca.npy {pcs.shape}")

np.save(os.path.join(out_base_dir, "dino_pca_components.npy"), pca.components_)
np.save(os.path.join(out_base_dir, "dino_pca_explained_variance_ratio.npy"), evr)
np.save(os.path.join(out_base_dir, "dino_pca_mean.npy"), pca.mean_)
print(f"\nSaved global PCA artifacts -> {out_base_dir}")

print("\n" + "=" * 60)
print("Done.")
print("=" * 60)
