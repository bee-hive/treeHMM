"""
Optional division-lineage wiring for the cancer tree-AR-HMM (Step 4).

Used only when `allow_divisions: true`.  The cancer cells/columns are Caliban
nucleus IDs, but the division lineage (`graph.pkl`: child->parent) is in CVAT
cancer-ID space.  We therefore:
  1. map each CVAT cancer ID -> the Caliban nucleus it overlaps most (over all frames),
  2. translate each child->parent edge into nucleus-column space,
  3. at the daughter's first active frame, mark the division: set is_division_mask,
     point parent_indices at the parent's nucleus column, and clear is_new_root_mask.

Edges that cannot be mapped cleanly (no overlapping nucleus, parent not active the
frame before the daughter appears, self-edges) are skipped with a warning.
"""

import os
import pickle

import numpy as np
import tifffile


def _build_cvat_to_nucleus_map(cancer_cvat_tracks, nuc, cancer_cvat_ids):
    """Map each CVAT cancer ID -> Caliban nucleus ID by maximum pixel overlap."""
    mapping = {}
    for cvat_id in cancer_cvat_ids:
        cvat_mask = cancer_cvat_tracks == cvat_id
        if not cvat_mask.any():
            continue
        overlap_nuc = nuc[cvat_mask]
        overlap_nuc = overlap_nuc[overlap_nuc > 0]
        if overlap_nuc.size == 0:
            continue
        ids, counts = np.unique(overlap_nuc, return_counts=True)
        mapping[int(cvat_id)] = int(ids[np.argmax(counts)])
    return mapping


def apply_division_lineage(crop, nuc, cancer_cell_ids, id_to_col, active_mask,
                           is_division_mask, is_new_root_mask, parent_indices,
                           type_sep_tracks_dir, out_base_dir):
    """Mutate the mask arrays in place to encode division edges for one crop."""
    graph_path = os.path.join(type_sep_tracks_dir, crop, "graph.pkl")
    tracks_path = os.path.join(type_sep_tracks_dir, crop, "tracks.tiff")
    if not (os.path.exists(graph_path) and os.path.exists(tracks_path)):
        print(f"  [divisions] {crop}: graph.pkl/tracks.tiff missing; skipping.")
        return

    graph = pickle.load(open(graph_path, "rb"))                 # {child_cvat: parent_cvat}
    cancer_cvat_tracks = tifffile.imread(tracks_path)[..., 1]   # ch1 = cancer CVAT IDs
    cvat_ids = np.unique(cancer_cvat_tracks[cancer_cvat_tracks > 0])
    cvat_to_nuc = _build_cvat_to_nucleus_map(cancer_cvat_tracks, nuc, cvat_ids)

    applied, skipped = 0, 0
    for child_cvat, parent_cvat in graph.items():
        child_nuc = cvat_to_nuc.get(int(child_cvat))
        parent_nuc = cvat_to_nuc.get(int(parent_cvat))
        if child_nuc is None or parent_nuc is None:
            skipped += 1; continue
        if child_nuc == parent_nuc or child_nuc not in id_to_col or parent_nuc not in id_to_col:
            skipped += 1; continue
        c_col = id_to_col[child_nuc]
        p_col = id_to_col[parent_nuc]
        child_active = np.where(active_mask[:, c_col])[0]
        if len(child_active) == 0:
            skipped += 1; continue
        t_first = child_active[0]
        # parent must be active the frame before the daughter appears
        if t_first == 0 or not active_mask[t_first - 1, p_col]:
            skipped += 1; continue
        is_division_mask[t_first, c_col] = True
        is_new_root_mask[t_first, c_col] = False
        parent_indices[t_first, c_col] = p_col
        applied += 1

    print(f"  [divisions] {crop}: applied {applied} edges, skipped {skipped}")
