"""Step 3: PCA-reduce the 768-d DINOv2 embeddings to the top-N components.

One PCA is fit jointly across every valid cell-frame in every well, so the
component basis is shared and states are comparable across wells/conditions.

Whitening is on by default (``dino_pca_whiten``).  Unwhitened, PC1 of a DINOv2
embedding set carries far more variance than PC10, so a k-means initialization
degenerates into a 1-D split on PC1.  Whitening gives each kept component equal
say at initialization; the HMM's full-covariance Gaussian emissions can still
learn any correlation structure afterwards.

Saved per well:
    nuclei_dino_pca.npy  (T, N, n_pcs)  zeros at invalid cell-frames
Global artifacts in ``{output_base_dir}/``:
    dino_pca_components.npy, dino_pca_explained_variance_ratio.npy,
    dino_pca_mean.npy, dino_pca_scale.npy

Usage (cs229Dino):
    conda run -n cs229Dino python reduce_dino_pca.py
"""

import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import yaml
from sklearn.decomposition import PCA

_script_dir = Path(__file__).resolve().parent
with open(_script_dir / "config.yml", "r") as f:
    cfg = yaml.safe_load(f)

wells: List[str] = cfg["wells"]
out_base_dir: str = cfg["output_base_dir"]
n_pcs: int = cfg["n_dino_pcs"]
whiten: bool = cfg.get("dino_pca_whiten", True)

if not cfg.get("use_dino", True):
    print("use_dino is false; skipping PCA step.")
    sys.exit(0)

print("=" * 60)
print(f"Step 3: PCA-reducing DINOv2 embeddings (whiten={whiten})")
print("=" * 60)

raws, valids, valid_rows = {}, {}, []
for well in wells:
    out_dir = os.path.join(out_base_dir, well)
    raw = np.load(os.path.join(out_dir, "nuclei_dino_raw.npy"))          # (T, N, 768)
    valid = np.load(os.path.join(out_dir, "nuclei_dino_valid_mask.npy"))  # (T, N)
    raws[well] = raw
    valids[well] = valid
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

for well in wells:
    raw = raws[well]
    valid = valids[well]
    num_frames, num_cells, dim = raw.shape
    flat_valid = valid.reshape(-1)
    out = np.zeros((num_frames * num_cells, n_components), dtype=np.float32)
    if flat_valid.any():
        out[flat_valid] = pca.transform(raw.reshape(-1, dim)[flat_valid])
    pcs = out.reshape(num_frames, num_cells, n_components)
    np.save(os.path.join(out_base_dir, well, "nuclei_dino_pca.npy"), pcs)
    print(f"  {well}: saved nuclei_dino_pca.npy {pcs.shape}")

np.save(os.path.join(out_base_dir, "dino_pca_components.npy"), pca.components_)
np.save(os.path.join(out_base_dir, "dino_pca_explained_variance_ratio.npy"), evr)
np.save(os.path.join(out_base_dir, "dino_pca_mean.npy"), pca.mean_)
np.save(os.path.join(out_base_dir, "dino_pca_scale.npy"),
        np.sqrt(pca.explained_variance_))
print(f"\nSaved global PCA artifacts -> {out_base_dir}")

print("\n" + "=" * 60)
print("Done.")
print("=" * 60)
