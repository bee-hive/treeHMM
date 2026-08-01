"""Step 2: DINOv2 embeddings of 30x30 patches centered on each cancer nucleus.

SELF-CONTAINED: the cs229Dino environment lacks the MarsonImagingPipeline
dependencies, so this imports only numpy / tifffile / torch / transformers / yaml.

Geometry: tracking masks and the centroids in ``cell_data.parquet`` live in the
600x600 CENTER crop of ``registered.tiff`` (450, 1040, 1408, 2), over frames
FRAME_START:FRAME_END.  So for crop-local (t, y, x):

    y_reg = y + (1040 - 600)//2 = y + 220
    x_reg = x + (1408 - 600)//2 = x + 404
    t_reg = t + 50

Offsets are derived from config, not hardcoded.

Per well:
  - memmap registered.tiff, slice the 600x600 center over the frame range
  - build one RGB per frame (phase greyscale + RFP nuclei into red), so contrast
    is consistent across all patches from that frame
  - cut a 30x30 patch centered on each nucleus centroid (edge-padded)
  - run facebook/dinov2-base, take the 768-d CLS / pooler output

Saved per well (into ``{output_base_dir}/{well}/``):
    nuclei_dino_raw.npy        (T, N, 768)  NaN where no valid patch
    nuclei_dino_valid_mask.npy (T, N)       True where a real patch was embedded

Column ordering matches nuclei_cell_ids.npy / nuclei_centroids.npy from Step 1.

Usage (cs229Dino):
    conda run -n cs229Dino python compute_dino_embeddings.py
"""

import os
import sys
from glob import glob
from pathlib import Path
from typing import List

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import tifffile
import torch
import yaml
from transformers import AutoImageProcessor, AutoModel

_script_dir = Path(__file__).resolve().parent
with open(_script_dir / "config.yml", "r") as f:
    cfg = yaml.safe_load(f)

wells: List[str] = cfg["wells"]
out_base_dir: str = cfg["output_base_dir"]
patch_px: int = cfg["dino_patch_px"]
alpha: float = cfg["dino_rgb_alpha"]
batch_size: int = cfg.get("dino_batch_size", 256)

if not cfg.get("use_dino", True):
    print("use_dino is false; skipping DINOv2 embedding step.")
    sys.exit(0)

reg_dir = Path(cfg["snakemake_runs_dir"]) / cfg["registration_run"] / "output"

raw_shape = cfg["raw_video_shape"]
center_h, center_w = cfg["video_center_extract"]
frame_start = cfg["frame_start"]
frame_end = cfg["frame_end"]
rfp_channel = cfg["rfp_channel"]
phase_channel = cfg["phase_channel"]

Y_OFFSET = (raw_shape[1] - center_h) // 2
X_OFFSET = (raw_shape[2] - center_w) // 2
print(f"Crop -> registered offsets: y+{Y_OFFSET}, x+{X_OFFSET}, "
      f"t+{frame_start} (frames {frame_start}:{frame_end})")


def get_rgb_image_with_nuclei(
    phase_image: np.ndarray,
    nuclei_image: np.ndarray,
    alpha: float = 0.3
) -> np.ndarray:
    """Composite phase and RFP nuclei frames into an (H, W, 3) uint8 RGB.

    Each channel is min-max normalized over the frame passed in; phase is
    duplicated across RGB at weight (1 - alpha) and nuclei intensity is added
    into the red channel at weight alpha.

    Unlike the copy in ``scripts/cancer_dino_hmm/compute_dino_embeddings.py``,
    the composite is accumulated in float and clipped before the uint8 cast --
    adding into a uint8 array wraps around on overflow, and clipping after the
    wrap cannot undo it.

    Args:
        phase_image (np.ndarray): (H, W) phase channel.
        nuclei_image (np.ndarray): (H, W) RFP nuclei channel.
        alpha (float): weight of the nuclei channel in the red plane.

    Returns:
        np.ndarray: (H, W, 3) uint8 RGB composite.
    """
    def _norm(img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32)
        lo, hi = float(np.min(img)), float(np.max(img))
        if hi - lo < 1e-8:
            return np.zeros_like(img)
        return (img - lo) / (hi - lo)

    phase = _norm(phase_image) * 255.0
    nuclei = _norm(nuclei_image) * 255.0

    rgb = np.stack([phase] * 3, axis=-1) * (1.0 - alpha)
    rgb[..., 0] += nuclei * alpha
    return np.clip(rgb, 0, 255).astype(np.uint8)


def resolve_model_src(model_id: str, model_path: str) -> str:
    """Prefer a local snapshot directory; fall back to the repo id.

    Args:
        model_id (str): HuggingFace repo id.
        model_path (str): local hub cache directory for the model.

    Returns:
        str: path or repo id to load from.
    """
    if model_path and os.path.isdir(os.path.join(model_path, "snapshots")):
        snaps = sorted(glob(os.path.join(model_path, "snapshots", "*")))
        if snaps:
            return snaps[-1]
    return model_id


device = "cuda" if torch.cuda.is_available() else "cpu"
model_src = resolve_model_src(cfg["dino_model_id"], cfg.get("dino_model_path"))
print(f"Loading DINOv2 from: {model_src}  (device={device})")
processor = AutoImageProcessor.from_pretrained(model_src, local_files_only=True)
model = AutoModel.from_pretrained(model_src, local_files_only=True).to(device).eval()
embed_dim = model.config.hidden_size


@torch.no_grad()
def embed_patches(patches: List[np.ndarray]) -> np.ndarray:
    """Embed a list of uint8 RGB patches with DINOv2.

    Args:
        patches (List[np.ndarray]): each (patch_px, patch_px, 3) uint8.

    Returns:
        np.ndarray: (len(patches), embed_dim) float32 embeddings.
    """
    out = np.zeros((len(patches), embed_dim), dtype=np.float32)
    for start in range(0, len(patches), batch_size):
        batch = patches[start:start + batch_size]
        inp = processor(images=batch, return_tensors="pt").to(device)
        res = model(**inp)
        emb = res.pooler_output
        if emb is None:
            emb = res.last_hidden_state[:, 0]
        out[start:start + len(batch)] = emb.detach().cpu().float().numpy()
    return out


print("=" * 60)
print(f"Step 2: DINOv2 embeddings of {patch_px}x{patch_px} nucleus patches")
print("=" * 60)

half = patch_px // 2

for well in wells:
    print("\n" + "-" * 60)
    print(f"Well {well}")

    out_dir = os.path.join(out_base_dir, well)
    cell_ids = np.load(os.path.join(out_dir, "nuclei_cell_ids.npy"))
    centroids = np.load(os.path.join(out_dir, "nuclei_centroids.npy"))  # (T, N, 2)
    num_frames, num_cells = centroids.shape[:2]

    reg = tifffile.memmap(reg_dir / well / "registered.tiff")
    crop = np.asarray(
        reg[frame_start:frame_end,
            Y_OFFSET:Y_OFFSET + center_h,
            X_OFFSET:X_OFFSET + center_w, :]
    )
    if crop.shape[0] != num_frames:
        raise RuntimeError(
            f"{well}: registered.tiff slice has {crop.shape[0]} frames but "
            f"centroids have {num_frames}"
        )

    raw = np.full((num_frames, num_cells, embed_dim), np.nan, dtype=np.float32)
    valid = np.zeros((num_frames, num_cells), dtype=bool)

    patches: List[np.ndarray] = []
    index: List[tuple] = []
    for t in range(num_frames):
        present = np.where(~np.isnan(centroids[t, :, 0]))[0]
        if len(present) == 0:
            continue
        rgb = get_rgb_image_with_nuclei(
            crop[t, ..., phase_channel], crop[t, ..., rfp_channel], alpha=alpha
        )
        rgb_pad = np.pad(rgb, ((half, half), (half, half), (0, 0)), mode="edge")
        for col in present:
            cy = int(round(centroids[t, col, 0]))
            cx = int(round(centroids[t, col, 1]))
            patches.append(rgb_pad[cy:cy + patch_px, cx:cx + patch_px])
            index.append((t, col))

    print(f"  valid patches: {len(patches)} / {num_frames * num_cells}")
    if patches:
        shapes = {p.shape for p in patches}
        if shapes != {(patch_px, patch_px, 3)}:
            raise RuntimeError(f"{well}: ragged patch shapes {shapes}")
        emb = embed_patches(patches)
        rows = np.array([t for t, _ in index])
        cols = np.array([c for _, c in index])
        raw[rows, cols] = emb
        valid[rows, cols] = True

    np.save(os.path.join(out_dir, "nuclei_dino_raw.npy"), raw)
    np.save(os.path.join(out_dir, "nuclei_dino_valid_mask.npy"), valid)
    print(f"  saved raw {raw.shape} + valid mask -> {out_dir}")

print("\n" + "=" * 60)
print("Done.")
print("=" * 60)
