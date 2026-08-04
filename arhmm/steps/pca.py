"""Step `pca`: one joint PCA reducing the embeddings to the top-k components.

**One PCA, fit across every crop at once**, on valid cell-frames only.  Fitting
per crop would put each crop in its own basis, so `dino_pc_0` would mean a
different direction in each and the joint fit would be comparing incomparable
numbers.

**Whitening is on by default and should stay on.**  Unwhitened, the leading
component's variance dwarfs the tenth's, so k-means initialization degenerates
into a one-dimensional split on PC0 and the remaining components contribute
almost nothing to the initial assignment.

Invalid cell-frames are written as **zeros, not NaN**, matching how the model
pads inactive cells -- the fit step's non-finite scrub would otherwise report
them as data it had to repair.

Written into `{cache}/pca/{key}/`:

    pca_model.npz
        components               (k, E)   the basis
        mean                     (E,)     subtracted before projection
        explained_variance_ratio (k,)
        scale                    (k,)     sqrt(explained_variance_), so the
                                          whitening can be inverted and a
                                          component mapped back to a direction
    {crop}/dino_pcs.npz
        pcs         (T, N, k) float32, zeros where invalid
        valid_mask  (T, N)    bool

Run inside the DINO environment (scikit-learn).
"""

from __future__ import annotations

import sys

import numpy as np

from arhmm import config as cfgmod
from arhmm.core import io
from arhmm.steps import step_main


def _run(cfg: dict, layout, args) -> dict:
    from sklearn.decomposition import PCA

    n_pcs = int(cfgmod.get_path(cfg, "dino.n_pcs"))
    whiten = bool(cfgmod.get_path(cfg, "dino.whiten", True))
    if not whiten:
        print("WARNING: dino.whiten is false; PC0's variance will dominate kmeans init")

    crops = cfgmod.crop_ids(cfg)
    loaded = {}
    for crop_id in crops:
        arrays = io.load_npz(layout.crop_dir("dino", crop_id) / "embeddings.npz")
        loaded[crop_id] = (arrays["raw"], arrays["valid_mask"])

    training = np.concatenate([raw[valid] for raw, valid in loaded.values()], axis=0)
    if training.shape[0] < n_pcs:
        raise ValueError(
            f"only {training.shape[0]} valid cell-frames for {n_pcs} components"
        )
    print(f"fitting one joint PCA on {training.shape[0]} valid cell-frames "
          f"x {training.shape[1]} dims -> {n_pcs} components (whiten={whiten})")

    # Fixed seed: PCA's randomized solver is only used for large inputs, but
    # pinning it costs nothing and keeps the step a pure function of its inputs.
    pca = PCA(n_components=min(n_pcs, *training.shape), whiten=whiten, random_state=0)
    pca.fit(training)

    ratios = pca.explained_variance_ratio_
    for index, ratio in enumerate(ratios):
        print(f"  PC{index}: {ratio:.4f}  (cumulative {ratios[: index + 1].sum():.4f})")

    out_root = io.ensure_dir(layout.step_dir("pca"))
    io.save_npz(
        out_root / "pca_model.npz",
        components=pca.components_.astype(np.float32),
        mean=pca.mean_.astype(np.float32),
        explained_variance_ratio=ratios.astype(np.float32),
        scale=np.sqrt(pca.explained_variance_).astype(np.float32),
    )

    summary = {}
    for crop_id, (raw, valid) in loaded.items():
        num_frames, num_cells = valid.shape
        pcs = np.zeros((num_frames, num_cells, pca.n_components_), dtype=np.float32)
        if valid.any():
            pcs[valid] = pca.transform(raw[valid]).astype(np.float32)
        crop_dir = io.ensure_dir(out_root / crop_id)
        io.save_npz(crop_dir / "dino_pcs.npz", pcs=pcs, valid_mask=valid)
        summary[crop_id] = {"valid_cell_frames": int(valid.sum())}
        print(f"  [{crop_id}] projected {int(valid.sum())} cell-frames")

    io.write_yaml(
        out_root / "summary.yml",
        {
            "n_components": int(pca.n_components_),
            "whiten": whiten,
            "training_cell_frames": int(training.shape[0]),
            "explained_variance_ratio": [float(v) for v in ratios],
            "cumulative_explained_variance": float(ratios.sum()),
            "crops": summary,
        },
    )
    return {"n_components": int(pca.n_components_),
            "cumulative_explained_variance": float(ratios.sum())}


if __name__ == "__main__":
    sys.exit(step_main("pca", "joint PCA of the DINOv2 embeddings", _run))
