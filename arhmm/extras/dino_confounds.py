"""Extra `dino_confounds`: is a DINO principal component really a clock?

A leading component that correlates strongly with frame index or with crop
identity is describing acquisition, not phenotype, and any state built on it is
a batch effect wearing a biological name.  This has already happened once in
this project: a pre-fix pipeline produced components with a correlation ratio of
0.665 against crop identity and |r| of 0.017 against the shape features -- the
embedding was a near-perfect crop detector and almost blind to the cells.

Two statistics per retained component, over valid cell-frames only:

    R^2 against frame index      a linear clock
    correlation ratio (eta^2)    against crop identity, which needs no linear
                                 relationship and so catches batch effects that
                                 a correlation would miss

Rules of thumb: eta^2 above ~0.3 against crop on a leading component means the
run should be read as reporting acquisition; a high R^2 against frame index on
PC0 means the same for time.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

REQUIRES: tuple[str, ...] = ("dino",)


def _r2_against_time(values: np.ndarray, frames: np.ndarray) -> float:
    """Squared Pearson correlation between a component and the frame index."""
    if values.size < 3 or np.std(values) < 1e-12 or np.std(frames) < 1e-12:
        return float("nan")
    return float(np.corrcoef(values, frames)[0, 1] ** 2)


def _eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    """Correlation ratio: the fraction of variance explained by group membership.

    Unlike a correlation this needs no ordering or linearity, so it detects a
    component that simply takes a different value in each crop.

    Args:
        values (np.ndarray): `(n,)` component values.
        groups (np.ndarray): `(n,)` group labels.

    Returns:
        float: eta^2 in [0, 1]; NaN when undefined.
    """
    if values.size < 3:
        return float("nan")
    total_variance = np.var(values)
    if total_variance < 1e-12:
        return float("nan")
    grand_mean = values.mean()
    between = 0.0
    for group in np.unique(groups):
        member = values[groups == group]
        between += member.size * (member.mean() - grand_mean) ** 2
    return float(between / (values.size * total_variance))


def run(cfg: dict, layout, out_dir: Path) -> None:
    """Write the per-component confound table and figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from arhmm import config as cfgmod
    from arhmm.core import io, viz

    columns = cfgmod.dino_column_names(cfg)
    if not columns:
        raise RuntimeError("this run has no DINO components; dino_confounds has nothing to check")

    values_blocks, frame_blocks, crop_blocks = [], [], []
    for crop_id in cfgmod.crop_ids(cfg):
        arrays = io.load_npz(layout.crop_dir("pca", crop_id) / "dino_pcs.npz")
        pcs, valid = arrays["pcs"], arrays["valid_mask"]
        num_frames = pcs.shape[0]
        frame_index = np.repeat(np.arange(num_frames)[:, None], pcs.shape[1], axis=1)
        values_blocks.append(pcs[valid])
        frame_blocks.append(frame_index[valid])
        crop_blocks.append(np.full(int(valid.sum()), crop_id))

    values = np.concatenate(values_blocks, axis=0)
    frames = np.concatenate(frame_blocks, axis=0)
    crops = np.concatenate(crop_blocks, axis=0)
    print(f"  {values.shape[0]} valid cell-frames across {len(set(crops.tolist()))} crops")

    rows = []
    for index, name in enumerate(columns):
        component = values[:, index]
        rows.append(
            {
                "component": name,
                "r2_vs_frame_index": _r2_against_time(component, frames.astype(float)),
                "eta2_vs_crop": _eta_squared(component, crops),
            }
        )

    with io.atomic_write(out_dir / "dino_confounds.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["component", "r2_vs_frame_index", "eta2_vs_crop"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})

    worst_time = max(rows, key=lambda r: (r["r2_vs_frame_index"] or 0))
    worst_crop = max(rows, key=lambda r: (r["eta2_vs_crop"] or 0))
    print(f"  worst clock:        {worst_time['component']} R^2={worst_time['r2_vs_frame_index']:.3f}")
    print(f"  worst batch effect: {worst_crop['component']} eta^2={worst_crop['eta2_vs_crop']:.3f}")
    if worst_crop["eta2_vs_crop"] > 0.3:
        print("  WARNING: a component is largely explained by crop identity; this run "
              "may be describing acquisition rather than phenotype")

    viz.apply_style()
    figure, ax = plt.subplots(figsize=(max(4.0, 0.55 * len(columns)), 3.4))
    positions = np.arange(len(columns))
    ax.bar(positions - 0.2, [r["r2_vs_frame_index"] for r in rows], 0.4,
           label="R$^2$ vs frame index", color="#4c72b0")
    ax.bar(positions + 0.2, [r["eta2_vs_crop"] for r in rows], 0.4,
           label=r"$\eta^2$ vs crop", color="#c44e52")
    ax.axhline(0.3, color="black", linestyle="--", linewidth=0.8)
    ax.text(len(columns) - 0.5, 0.31, "read with suspicion above here", fontsize=6,
            ha="right", va="bottom")
    ax.set_xticks(positions, [c.replace("dino_pc_", "PC") for c in columns])
    ax.set_ylim(0, 1)
    ax.set_ylabel("variance explained")
    ax.set_title("DINO components vs acquisition confounds")
    ax.legend(loc="upper right", frameon=False)
    figure.savefig(out_dir / "dino_confounds.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)
