"""Step `dino`: DINOv2 embeddings of patches centred on each cancer cell.

One patch per observed cell-frame **per patch size**, embedded in batches and
scattered back onto the crop's `(T, N)` column layout.  Cell-frames with no
patch -- an absent cell, or a centroid that is not finite -- are NaN and flagged
in `valid_mask`; the `pca` step turns them into zeros, matching how the model
pads inactive cells.

When `dino.patch_px` names several sizes the embeddings are **averaged over the
sizes** here, so what leaves this step is one multi-scale vector per cell-frame
in the same `(T, N, E)` shape a single size produces.  Everything downstream --
`pca`, the DINO provider, `fit` -- is therefore unaware that multi-scale exists.
The set of cell-frames does not depend on the size (see `core.dino.iter_patches`),
so the average is over a fixed set and disagreement is an error, not a case to
handle.

Written per crop, into `{cache}/dino/{key}/{crop}/`:

    embeddings.npz
        raw         (T, N, E) float32   NaN where no patch, averaged over sizes
        valid_mask  (T, N)    bool
    meta.json
        cell_ids, model_id, patch_px (list), embed_dim, transformers version

Also writes a contact sheet of sixteen sampled patches -- `contact_sheet.png`,
or one `contact_sheet_p{size}.png` per size.  The one thing that cannot be
asserted about this step is whether the patches look right -- subject red, other
cancer blue, T cells green, cell centred -- so it makes that cheap to check by
eye.

Run inside the DINO environment (torch / transformers).
"""

from __future__ import annotations

import sys

import numpy as np

from arhmm import config as cfgmod
from arhmm.core import cells as cellsmod
from arhmm.core import io
from arhmm.steps import step_main

CONTACT_SHEET_PATCHES = 16


def _contact_sheet(patches, labels, path, columns=4):
    """Save a grid of sampled patches so the composition can be checked by eye."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = int(np.ceil(len(patches) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(2.0 * columns, 2.2 * rows), squeeze=False)
    for index, ax in enumerate(axes.ravel()):
        if index < len(patches):
            ax.imshow(patches[index])
            ax.set_title(labels[index], fontsize=7)
        ax.axis("off")
    figure.suptitle("subject red, other cancer blue, T cells green", fontsize=9)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _write_contact_sheet(patches, index, cell_ids, patch_px, sizes, crop_dir):
    """Save one sampled sheet for a patch size.

    Single-size runs keep the historic `contact_sheet.png` name; multi-scale
    runs get one sheet per size, because the thing worth checking by eye is
    exactly how much context each size takes in.
    """
    step = max(1, len(patches) // CONTACT_SHEET_PATCHES)
    sampled = list(range(0, len(patches), step))[:CONTACT_SHEET_PATCHES]
    name = "contact_sheet.png" if len(sizes) == 1 else f"contact_sheet_p{patch_px}.png"
    _contact_sheet(
        [patches[i] for i in sampled],
        [f"t={index[i][0]} cell={cell_ids[index[i][1]]}" for i in sampled],
        crop_dir / name,
    )


def _run(cfg: dict, layout, args) -> dict:
    import transformers

    from arhmm.core import dino as dinomod

    params = cfgmod.get_path(cfg, "dino")
    sizes = cfgmod.dino_patch_sizes(cfg)
    processor, model, device, embed_dim = dinomod.load_model(
        params["model_id"], params.get("model_path"), None
    )
    print(f"model {params['model_id']} on {device}, embed_dim {embed_dim}, "
          f"patch {'/'.join(str(s) for s in sizes)}px, "
          f"transformers {transformers.__version__}")

    out_root = io.ensure_dir(layout.step_dir("dino"))
    summary: dict[str, dict] = {}

    for crop_id in cfgmod.crop_ids(cfg):
        feature_dir = layout.crop_dir("features", crop_id)
        arrays = io.load_npz(feature_dir / "features.npz")
        cell_ids = arrays["cell_ids"]
        centroids = arrays["centroids"]
        active = arrays["active_mask"]

        crop = cellsmod.load_crop(cfg, crop_id, with_image=True)
        if not np.array_equal(crop.cell_ids, cell_ids):
            raise ValueError(
                f"{crop_id}: cell ids differ between the features cache and the tracks; "
                f"the caches are out of step"
            )

        raw = np.full((crop.num_frames, len(cell_ids), embed_dim), np.nan, dtype=np.float32)
        valid = np.zeros((crop.num_frames, len(cell_ids)), dtype=bool)
        index: list[tuple[int, int]] = []
        rows = cols = None

        # Accumulated in place rather than stacked: an (S, T, N, E) array is
        # gigabytes per crop, and only the mean is ever written.
        for position, patch_px in enumerate(sizes):
            patches, this_index = [], []
            for t, column, patch in dinomod.iter_patches(
                crop, centroids, active, params, patch_px
            ):
                patches.append(patch)
                this_index.append((t, column))

            if position == 0:
                index = this_index
                if index:
                    rows = np.array([t for t, _ in index])
                    cols = np.array([c for _, c in index])
                    valid[rows, cols] = True
            elif this_index != index:
                # Documented not to happen -- iter_patches skips on activity,
                # centroid and mask, never on size -- so if it does, the average
                # would silently mix cell-frames and must not be written.
                raise ValueError(
                    f"{crop_id}: patch size {patch_px}px yielded {len(this_index)} "
                    f"cell-frames against {len(index)} at {sizes[0]}px; the patch "
                    f"sizes disagree about which cells are observed"
                )

            print(f"  [{crop_id}] {patch_px}px: {len(patches)} patches", flush=True)
            embeddings = dinomod.embed_patches(
                patches, processor, model, device,
                batch_size=int(params.get("batch_size", 256)),
            )
            if index:
                if position == 0:
                    raw[rows, cols] = embeddings
                else:
                    raw[rows, cols] += embeddings

            if patches:
                _write_contact_sheet(patches, this_index, cell_ids, patch_px, sizes,
                                     io.ensure_dir(out_root / crop_id))

        if index and len(sizes) > 1:
            raw[rows, cols] /= len(sizes)

        crop_dir = io.ensure_dir(out_root / crop_id)
        io.save_npz(crop_dir / "embeddings.npz", raw=raw, valid_mask=valid)
        io.write_json(
            crop_dir / "meta.json",
            {
                "crop_id": crop_id,
                "cell_ids": [int(v) for v in cell_ids],
                "model_id": params["model_id"],
                "patch_px": sizes,
                "embed_dim": embed_dim,
                "transformers": transformers.__version__,
                "num_patches": len(index) * len(sizes),
            },
        )

        summary[crop_id] = {
            "patches": len(index) * len(sizes),
            "patches_per_size": len(index),
            "valid_cell_frames": int(valid.sum()),
        }

    io.write_yaml(
        out_root / "summary.yml",
        {
            "model_id": params["model_id"],
            "patch_px": sizes,
            "embed_dim": embed_dim,
            "crops": summary,
        },
    )
    return {"crops": len(summary), "embed_dim": embed_dim, "patch_sizes": len(sizes)}


if __name__ == "__main__":
    sys.exit(step_main("dino", "DINOv2 embeddings of centroid patches", _run))
