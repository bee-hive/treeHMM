"""Step `features`: per-cell, per-frame features computed from the tracks.

Everything named in `features.compute` is calculated and saved for every
cell-frame, regardless of what the model will use.  That separation is
deliberate: the fit selects a subset, but the per-state distributions and any
downstream evaluation get to read features the model never saw, so state
profiling stays independent of the fit's own inputs.

Written per crop, into `{cache}/features/{key}/{crop}/`:

    features.npz
        cell_ids         (N,)        int32   sorted cell IDs; defines column order
        centroids        (T, N, 2)   float32 crop-local (y, x), NaN where absent
        values           (T, N, F)   float32 the features, NaN where undefined
        active_mask      (T, N)      bool    cell observed
        parent_indices   (T, N)      int32   always the cell's own column
        is_division_mask (T, N)      bool    all False -- no divisions
        is_new_root_mask (T, N)      bool    first active frame

    meta.json
        feature_names, feature_units, crop_id, cell_source, num_frames, num_cells

Strings live in the JSON sidecar rather than the npz because the three conda
environments are on different numpy majors and only numeric and bool arrays
round-trip between them.

Run inside the imaging environment.
"""

from __future__ import annotations

import sys

import numpy as np

from treearhmm import config as cfgmod
from treearhmm.core import cells as cellsmod
from treearhmm.core import io, lineage
from treearhmm.core import trackfeatures as tf
from treearhmm.steps import step_main


def _run(cfg: dict, layout, args) -> dict:
    requested = cfgmod.computed_features(cfg)
    params = cfgmod.get_path(cfg, "features.params", {}) or {}
    wants_image = tf.needs_image(requested)
    out_root = io.ensure_dir(layout.step_dir("features"))

    print(f"features: {', '.join(requested)}")
    print(f"params:   { {k: params.get(k) for k in sorted(tf.required_params(requested))} }")
    if not wants_image:
        print("no requested feature reads pixel intensities; skipping crop.tiff")

    summary: dict[str, dict] = {}
    for crop_id in cfgmod.crop_ids(cfg):
        crop = cellsmod.load_crop(cfg, crop_id, with_image=wants_image)

        def tick(t, _crop=crop_id, _total=None):
            if (t + 1) % 10 == 0 or t == 0:
                print(f"  [{_crop}] frame {t + 1}", flush=True)

        values, centroids, present = tf.compute_per_frame(
            crop.cancer, crop.tcells, crop.image, crop.cell_ids, requested, params, progress=tick
        )
        masks = lineage.build_masks(present)
        values.update(
            tf.compute_temporal(values, centroids, masks["active_mask"], requested, params)
        )

        # One (T, N, F) array in registry order, so the column layout is a
        # function of the feature set rather than of config ordering.
        stacked = np.stack([values[name] for name in requested], axis=-1).astype(np.float32)

        crop_dir = io.ensure_dir(out_root / crop_id)
        io.save_npz(
            crop_dir / "features.npz",
            cell_ids=crop.cell_ids.astype(np.int32),
            centroids=centroids.astype(np.float32),
            values=stacked,
            **{k: masks[k] for k in lineage.MASK_KEYS},
        )
        io.write_json(
            crop_dir / "meta.json",
            {
                "crop_id": crop_id,
                "cell_source": cfgmod.get_path(cfg, "cells.source"),
                "feature_names": requested,
                "feature_units": [tf.FEATURE_REGISTRY[n].units for n in requested],
                "num_frames": crop.num_frames,
                "num_cells": crop.num_cells,
            },
        )

        counts = lineage.summarize(masks)
        active_mask = masks["active_mask"]
        undefined = {
            name: int(np.isnan(values[name][active_mask]).sum()) for name in requested
        }

        # A value outside a feature's mathematical range means the estimator
        # broke down -- skimage's perimeter, for instance, is unreliable on
        # masks a few pixels across, which drives 4*pi*A/P^2 above 1.  These are
        # reported, never clipped: silently bounding them would hide that the
        # measurement is not trustworthy on those cells.
        out_of_bounds = {}
        for name in requested:
            bounds = tf.FEATURE_REGISTRY[name].bounds
            if bounds is None:
                continue
            sample = values[name][active_mask]
            finite = sample[np.isfinite(sample)]
            outside = (finite < bounds[0]) | (finite > bounds[1])
            if outside.any():
                out_of_bounds[name] = {
                    "count": int(outside.sum()),
                    "fraction": float(outside.mean()),
                    "min": float(finite.min()),
                    "max": float(finite.max()),
                }

        summary[crop_id] = {
            **counts,
            "undefined_on_active": undefined,
            "out_of_bounds_on_active": out_of_bounds,
        }
        print(
            f"  [{crop_id}] {counts['num_cells']} cells, {counts['num_frames']} frames, "
            f"{counts['active_cell_frames']} active cell-frames"
        )
        for name, count in undefined.items():
            if count:
                print(f"      {name}: {count} undefined (NaN) on active cell-frames")
        for name, report in out_of_bounds.items():
            bounds = tf.FEATURE_REGISTRY[name].bounds
            print(
                f"      WARNING {name}: {report['count']} values "
                f"({100 * report['fraction']:.1f}%) outside {bounds}, "
                f"range [{report['min']:.3g}, {report['max']:.3g}] -- the estimator "
                f"is unreliable on these cells"
            )

    io.write_yaml(
        out_root / "summary.yml",
        {
            "features": requested,
            "params": {k: params.get(k) for k in sorted(tf.required_params(requested))},
            "cell_source": cfgmod.get_path(cfg, "cells.source"),
            "crops": summary,
        },
    )
    return {"crops": len(summary)}


if __name__ == "__main__":
    sys.exit(step_main("features", "per-cell, per-frame features from the tracks", _run))
