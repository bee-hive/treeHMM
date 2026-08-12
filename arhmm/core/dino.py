"""DINOv2 patch construction and embedding.

**torch and transformers are imported lazily**, inside `load_model` and
`embed_patches` only.  Everything above those -- the patch composition, which is
the part with the subtle decisions and therefore the part worth testing -- is
plain numpy and imports anywhere, so its unit tests run in all three conda
environments rather than only the DINO one.

The patch handed to DINOv2 is not a raw crop.  It is the greyscale phase image
centred on the cancer cell, plus **at most one** overlay, selected by
`dino.patch_overlay`:

    rfp      RFP intensity added to the red channel at `rfp_alpha`; the
             cell-type masks are not drawn at all.
    masks    the cell-type masks alpha-blended at `mask_alpha`; RFP unused.
    none     plain phase, neither overlay.

The two are exclusive by construction rather than by preference.  `masks` spends
every hue on a cell type --

    T cells         green
    other cancer    blue
    the subject     red

-- so RFP has no channel left to occupy, which is the whole reason `rfp` mode
turns the masks off.  Note what `rfp` and `none` give up: with no mask painted,
the subject cell is identified only by being at the centre of the patch.

Two details carry most of the value in `masks` mode, and both were arrived at
the hard way:

  * **Blend, do not flat-fill.** At `mask_alpha` the phase texture inside the
    mask -- blebbing, the refractile halo, granularity -- survives into the
    embedding.  A flat fill throws exactly the signal away that makes an
    appearance embedding worth computing.
  * **Repaint the subject from the pre-blue image.** If the subject is painted
    over the composite that already has other cancer cells in blue, a subject
    overlapping a neighbour comes out red-over-blue, so the embedding encodes
    *overlap* rather than identity.  Compositing the subject from the image that
    has only the T-cell layer keeps its colour a function of the cell alone.

The patch is a fixed `patch_px` window around the rounded centroid, taken from
an edge-padded image.  Padding first means every cell gets a full-size window
with no bounds checks and no variable-size patches near the frame edge.

`dino.patch_px` may name **several** sizes, in which case the cell is cut and
embedded once per size and the embeddings are averaged, before PCA, into one
multi-scale vector.  The window is the only thing that changes: the processor
resizes every patch to the model's input resolution regardless, so a small
window is a close crop and a large one is the same cell with more context.
Averaging is what stops the choice between the two from being a fork in the
experiment -- see `configs/runs/dino_focus/_base.yml` for what that fork costs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

import numpy as np


def resolve_model_src(model_id: str, model_path: str | Path | None) -> str:
    """Resolve a local HuggingFace snapshot directory, falling back to the id.

    The cluster runs offline, so a cached snapshot is the normal case.

    Args:
        model_id (str): e.g. `facebook/dinov2-base`.
        model_path (str | Path | None): a `models--*` cache directory.

    Returns:
        str: a path or model id suitable for `from_pretrained`.
    """
    if not model_path:
        return model_id
    root = Path(model_path)
    snapshots = root / "snapshots"
    if snapshots.is_dir():
        candidates = sorted(
            (p for p in snapshots.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
        )
        if candidates:
            return str(candidates[-1])
    return str(root) if root.is_dir() else model_id


def load_model(model_id: str, model_path: str | Path | None, device: str | None = None):
    """Load DINOv2 and its image processor, offline.

    Returns:
        tuple: `(processor, model, device, embed_dim)`.
    """
    import torch
    from transformers import AutoImageProcessor, AutoModel

    source = resolve_model_src(model_id, model_path)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(source, local_files_only=True)
    model = AutoModel.from_pretrained(source, local_files_only=True).to(device).eval()
    return processor, model, device, int(model.config.hidden_size)


def embed_patches(patches: Sequence[np.ndarray], processor, model, device: str,
                  batch_size: int = 256) -> np.ndarray:
    """Embed a list of `(P, P, 3)` uint8 patches.

    Args:
        patches (Sequence[np.ndarray]): the patches.
        processor: the DINOv2 image processor.
        model: the DINOv2 model.
        device (str): torch device.
        batch_size (int): forward-pass batch size; throughput only.

    Returns:
        np.ndarray: `(len(patches), embed_dim)` float32.
    """
    import torch

    if not patches:
        return np.zeros((0, int(model.config.hidden_size)), dtype=np.float32)

    outputs = []
    with torch.no_grad():
        for start in range(0, len(patches), batch_size):
            batch = list(patches[start : start + batch_size])
            inputs = processor(images=batch, return_tensors="pt").to(device)
            result = model(**inputs)
            # transformers 5.x still provides pooler_output for DINOv2; the CLS
            # fallback is defensive against that changing.
            pooled = getattr(result, "pooler_output", None)
            if pooled is None:
                pooled = result.last_hidden_state[:, 0]
            outputs.append(pooled.detach().float().cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


def _blend(image: np.ndarray, mask: np.ndarray, colour: Sequence[float], alpha: float) -> np.ndarray:
    """Alpha-blend a colour into an RGB float image where `mask` is true."""
    out = image.copy()
    if mask.any():
        out[mask] = (1.0 - alpha) * image[mask] + alpha * np.asarray(colour, dtype=np.float32)
    return out


def effective_mask_alpha(params: dict) -> float:
    """`mask_alpha`, or 0 when the overlay mode does not paint the masks.

    `subject_patch` never sees the mode, so this is what keeps its repaint in
    step with `compose_frame`.

    Args:
        params (dict): the config's `dino` block.

    Returns:
        float: the blend weight to use for every mask in this mode.
    """
    if params.get("patch_overlay", "none") != "masks":
        return 0.0
    return float(params["mask_alpha"])


def compose_frame(
    phase: np.ndarray,
    cancer: np.ndarray,
    tcells: np.ndarray,
    rfp: np.ndarray | None,
    params: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the shared and all-cancer composites for one frame.

    Args:
        phase (np.ndarray): `(H, W)` float in [0, 1], normalized over the stack.
        cancer (np.ndarray): `(H, W)` cancer label image.
        tcells (np.ndarray): `(H, W)` T-cell label image.
        rfp (np.ndarray | None): `(H, W)` float in [0, 1].  Required by the `rfp`
            overlay, ignored by the other two.
        params (dict): the config's `dino` block.

    Returns:
        tuple:
            shared (np.ndarray): phase + T cells, **without** the cancer layer.
                Subject cells are repainted from this, so a subject overlapping
                another cancer cell does not come out red-over-blue.
            all_cancer (np.ndarray): `shared` plus every cancer cell in blue.

        Outside `masks` mode nothing is painted, so both are the same image.

    Raises:
        ValueError: the `rfp` overlay was asked for without an RFP channel.
    """
    overlay = params.get("patch_overlay", "none")
    base = np.repeat(phase[..., None].astype(np.float32), 3, axis=2)

    if overlay == "rfp":
        if rfp is None:
            raise ValueError("dino.patch_overlay is 'rfp', but no RFP channel was given")
        # Red, not luminance: with the masks off, the hue is free.
        alpha = float(params.get("rfp_alpha", 0.3))
        base[..., 0] = np.clip(base[..., 0] + alpha * rfp.astype(np.float32), 0.0, 1.0)

    if overlay != "masks":
        # Aliased deliberately: `subject_patch` copies `all_cancer` before it
        # reads `shared`, and at mask_alpha 0 its repaint is a no-op anyway.
        return base, base

    mask_alpha = float(params["mask_alpha"])
    shared = _blend(base, tcells > 0, params["tcell_colour"], mask_alpha)
    all_cancer = _blend(shared, cancer > 0, params["other_cancer_colour"], mask_alpha)
    return shared, all_cancer


def subject_patch(
    shared: np.ndarray,
    all_cancer: np.ndarray,
    subject_mask: np.ndarray,
    centroid: tuple[float, float],
    patch_px: int,
    subject_colour: Sequence[float],
    mask_alpha: float,
) -> np.ndarray:
    """Cut the fixed-size patch centred on one cell, with that cell painted red.

    Args:
        shared (np.ndarray): phase + T cells, no cancer layer.
        all_cancer (np.ndarray): shared plus all cancer cells in blue.
        subject_mask (np.ndarray): `(H, W)` bool, this cell's pixels.
        centroid (tuple[float, float]): `(y, x)` in image coordinates.
        patch_px (int): edge length of the square window.
        subject_colour (Sequence[float]): RGB for the subject.
        mask_alpha (float): blend weight.

    Returns:
        np.ndarray: `(patch_px, patch_px, 3)` uint8.
    """
    image = all_cancer.copy()
    # From `shared`, not from `all_cancer`: see the module docstring.
    image[subject_mask] = (
        (1.0 - mask_alpha) * shared[subject_mask]
        + mask_alpha * np.asarray(subject_colour, dtype=np.float32)
    )

    as_uint8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    half = patch_px // 2
    padded = np.pad(as_uint8, ((half, half), (half, half), (0, 0)), mode="edge")
    y = int(round(centroid[0]))
    x = int(round(centroid[1]))
    return padded[y : y + patch_px, x : x + patch_px]


def iter_patches(
    crop,
    centroids: np.ndarray,
    active: np.ndarray,
    params: dict,
    patch_px: int,
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield `(frame, column, patch)` for every observed cell-frame in a crop.

    Which cell-frames are yielded does **not** depend on `patch_px` -- a
    cell-frame is skipped only when it is inactive, its centroid is not finite,
    or its label mask is empty.  So calling this at several sizes gives the same
    `(frame, column)` sequence each time, which is what lets the step average
    across sizes without tracking per-size coverage.

    Args:
        crop (CropCells): the loaded crop, with its image.
        centroids (np.ndarray): `(T, N, 2)` crop-local centroids, NaN where absent.
        active (np.ndarray): `(T, N)` bool.
        params (dict): the config's `dino` block.
        patch_px (int): edge length of the square window, passed explicitly
            rather than read from `params` because `dino.patch_px` may name
            several sizes.

    Yields:
        tuple[int, int, np.ndarray]: frame index, column index, `(P, P, 3)` uint8.
    """
    patch_px = int(patch_px)
    mask_alpha = effective_mask_alpha(params)
    subject_colour = params["subject_colour"]

    for t in range(crop.num_frames):
        if not active[t].any():
            continue
        phase = crop.image[t, ..., 1]
        rfp = crop.image[t, ..., 0]
        shared, all_cancer = compose_frame(phase, crop.cancer[t], crop.tcells[t], rfp, params)
        for column, cell_id in enumerate(crop.cell_ids):
            if not active[t, column] or not np.isfinite(centroids[t, column]).all():
                continue
            mask = crop.cancer[t] == cell_id
            if not mask.any():
                continue
            yield t, column, subject_patch(
                shared, all_cancer, mask, tuple(centroids[t, column]),
                patch_px, subject_colour, mask_alpha,
            )
