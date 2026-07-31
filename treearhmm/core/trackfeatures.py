"""The registry of features computed directly from the tracks.

**Adding a feature means adding one decorated function here.** Nothing else in
the pipeline needs to change: config validation, the model's emission vector,
the per-state distribution plots, the CSV headers and the held-out diagnostics
all read this registry.

Features fall into two stages:

    per_frame   computed from a single frame's masks and image, independently of
                any other frame (area, circularity, neighbour counts, ...).
    temporal    computed from a cell's per-frame series over its ACTIVE frames
                (velocity, deltas, trailing window statistics).

Temporal features are always taken against the previous *active* frame, so a
tracking gap produces one wide step rather than a spurious drop to zero and
back.  `SeriesBundle` deliberately exposes no way to index `t - 1` directly --
`prev`, `gap` and `window` are the only accessors -- so that rule is structural
rather than a comment a later edit can quietly violate.  A cell's first active
frame has no predecessor, which is why every run holds
`cells.warmup_frames >= 1`.

**Import weight matters.** This module is imported by `treearhmm.config`, which
every step imports, and the steps run in three different conda environments --
only one of which has scikit-image.  So `numpy` is imported at module level and
**scikit-image is imported lazily inside the functions that need it**.  Keep it
that way: the registry's metadata (names, stages, units, docs, dependencies)
must stay importable everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

PER_FRAME = "per_frame"
TEMPORAL = "temporal"


@dataclass(frozen=True)
class Feature:
    """One computed quantity, with everything the rest of the pipeline needs.

    Attributes:
        name (str): the config-facing name.
        stage (str): `PER_FRAME` or `TEMPORAL`.
        units (str): shown on plot axes; `""` for dimensionless.
        doc (str): one sentence, used as the plot panel subtitle.
        fn (Callable): `(FrameBundle) -> (N,)` or `(SeriesBundle) -> (T, N)`.
        depends (tuple[str, ...]): other features this one reads.
        uses (tuple[str, ...]): `features.params` keys it reads.  Only these are
            folded into the features cache key, so changing a parameter no
            computed feature consumes does not invalidate the cache.
        needs_image (bool): reads the phase/RFP image, not just the masks.
    """

    name: str
    stage: str
    units: str
    doc: str
    fn: Callable
    depends: tuple[str, ...] = ()
    uses: tuple[str, ...] = ()
    needs_image: bool = False


FEATURE_REGISTRY: dict[str, Feature] = {}


def _register(feature: Feature) -> None:
    if feature.name in FEATURE_REGISTRY:
        raise ValueError(f"duplicate feature registration: {feature.name!r}")
    FEATURE_REGISTRY[feature.name] = feature


def per_frame(name, *, units, doc, uses=(), needs_image=False):
    """Register a per-frame feature: `(FrameBundle) -> (N,) float`."""

    def decorate(fn):
        _register(
            Feature(name, PER_FRAME, units, doc, fn, uses=tuple(uses), needs_image=needs_image)
        )
        return fn

    return decorate


def temporal(name, *, units, doc, depends=(), uses=()):
    """Register a temporal feature: `(SeriesBundle) -> (T, N) float`."""

    def decorate(fn):
        _register(Feature(name, TEMPORAL, units, doc, fn, depends=tuple(depends), uses=tuple(uses)))
        return fn

    return decorate


# --------------------------------------------------------------------------- #
# registry queries
# --------------------------------------------------------------------------- #


def feature_names(stage: str | None = None) -> list[str]:
    """Registered feature names, optionally restricted to one stage.

    Args:
        stage (str | None): `PER_FRAME`, `TEMPORAL`, or None for all.

    Returns:
        list[str]: names in registry order.
    """
    return [f.name for f in FEATURE_REGISTRY.values() if stage is None or f.stage == stage]


def expand_requested(requested: Sequence[str]) -> list[str]:
    """The full feature set that must be computed to satisfy `requested`.

    Pulls in the per-frame features a requested temporal feature reads, even
    when the config did not name them, and returns everything in registry order
    so the cached column layout is a function of the set, not of config order.

    Args:
        requested (Sequence[str]): feature names from the config.

    Returns:
        list[str]: requested features plus their dependencies, deduplicated.

    Raises:
        KeyError: if a name is not registered.
    """
    needed: set[str] = set()
    for name in requested:
        feature = FEATURE_REGISTRY[name]
        needed.add(name)
        needed.update(feature.depends)
    return [n for n in FEATURE_REGISTRY if n in needed]


def required_params(names: Iterable[str]) -> set[str]:
    """`features.params` keys consumed by the given features."""
    used: set[str] = set()
    for name in names:
        used.update(FEATURE_REGISTRY[name].uses)
    return used


def needs_image(names: Iterable[str]) -> bool:
    """Whether any of the given features reads the phase/RFP image."""
    return any(FEATURE_REGISTRY[n].needs_image for n in names)


# --------------------------------------------------------------------------- #
# the bundles features are handed
# --------------------------------------------------------------------------- #


@dataclass
class FrameBundle:
    """One frame's masks and image, with the shared work already done.

    `props` holds the output of a single `regionprops_table` call over the whole
    frame, scattered onto this crop's fixed column order and padded with NaN
    where a cell is absent.  A feature is therefore a lookup, not a loop over
    cells -- which is the point, since the archived pipelines' per-cell Python
    loops over full-frame boolean compares were their dominant cost.

    Attributes:
        labels (np.ndarray): `(H, W)` int, cancer cells in this crop's ID space.
        other_labels (np.ndarray): `(H, W)` int, T cells (every other non-zero ID).
        image (np.ndarray | None): `(H, W, 2)` float32 in [0, 1]; channel 0 RFP,
            channel 1 phase, normalized over the WHOLE stack.  None when nothing
            requested needs it.
        cell_ids (np.ndarray): `(N,)` sorted cancer IDs -- the column order.
        centroids (np.ndarray): `(N, 2)` crop-local (y, x), NaN where absent.
        other_centroids (np.ndarray): `(M, 2)` T-cell centroids this frame.
        params (dict): `features.params`.
        props (dict[str, np.ndarray]): `(N,)` regionprops columns, NaN where absent.
        present (np.ndarray): `(N,)` bool, whether each cell appears this frame.
    """

    labels: np.ndarray
    other_labels: np.ndarray
    image: np.ndarray | None
    cell_ids: np.ndarray
    centroids: np.ndarray
    other_centroids: np.ndarray
    params: dict
    props: dict[str, np.ndarray] = field(default_factory=dict)
    present: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))

    def prop(self, key: str) -> np.ndarray:
        """A regionprops column, `(N,)` float with NaN where the cell is absent."""
        return self.props[key]


@dataclass
class SeriesBundle:
    """A crop's per-frame features, for computing temporal ones.

    The only accessors reaching into the past are `prev`, `gap` and `window`,
    and all three step over **active** frames.  There is deliberately no way to
    index `t - 1`.

    Attributes:
        values (dict[str, np.ndarray]): per-frame features, each `(T, N)`.
        centroids (np.ndarray): `(T, N, 2)`, NaN where absent.
        active (np.ndarray): `(T, N)` bool.
        params (dict): `features.params`.
    """

    values: dict[str, np.ndarray]
    centroids: np.ndarray
    active: np.ndarray
    params: dict
    _prev_idx: np.ndarray | None = None

    @property
    def prev_idx(self) -> np.ndarray:
        """`(T, N)` int: index of the previous active frame, or -1 if there is none."""
        if self._prev_idx is None:
            self._prev_idx = _previous_active_index(self.active)
        return self._prev_idx

    def gap(self) -> np.ndarray:
        """`(T, N)` float: frames since the previous active frame; NaN at the first."""
        num_frames = self.active.shape[0]
        rows = np.arange(num_frames)[:, None]
        gap = (rows - self.prev_idx).astype(np.float64)
        return np.where(self.prev_idx >= 0, gap, np.nan)

    def prev(self, name: str) -> np.ndarray:
        """`(T, N)` float: a feature's value at each cell's previous active frame."""
        return _gather_previous(self.values[name], self.prev_idx)

    def prev_centroids(self) -> np.ndarray:
        """`(T, N, 2)` float: centroid at each cell's previous active frame."""
        return np.stack(
            [_gather_previous(self.centroids[..., k], self.prev_idx) for k in range(2)], axis=-1
        )

    def displacement(self) -> np.ndarray:
        """`(T, N)` float: distance moved since the previous active frame."""
        delta = self.centroids - self.prev_centroids()
        return np.linalg.norm(delta, axis=-1)

    def window(self, series: np.ndarray, length: int) -> np.ndarray:
        """`(T, N, length)`: the last `length` ACTIVE values up to and including t.

        Args:
            series (np.ndarray): `(T, N)` values to window.
            length (int): trailing window length in active frames.

        Returns:
            np.ndarray: NaN-padded on the left where a cell has fewer than
                `length` active frames so far.
        """
        num_frames, num_cells = self.active.shape
        out = np.full((num_frames, num_cells, length), np.nan)
        for col in range(num_cells):
            frames = np.flatnonzero(self.active[:, col])
            for position, t in enumerate(frames):
                lo = max(0, position - length + 1)
                chunk = series[frames[lo : position + 1], col]
                out[t, col, length - len(chunk) :] = chunk
        return out


def _previous_active_index(active: np.ndarray) -> np.ndarray:
    """`(T, N)` index of the previous active frame per cell-frame, else -1."""
    num_frames, num_cells = active.shape
    prev = np.full((num_frames, num_cells), -1, dtype=np.int64)
    for col in range(num_cells):
        frames = np.flatnonzero(active[:, col])
        if frames.size > 1:
            prev[frames[1:], col] = frames[:-1]
    return prev


def _gather_previous(series: np.ndarray, prev_idx: np.ndarray) -> np.ndarray:
    """Gather `series` at `prev_idx`, returning NaN where there is no predecessor."""
    safe = np.where(prev_idx >= 0, prev_idx, 0)
    gathered = np.take_along_axis(series, safe, axis=0).astype(np.float64)
    return np.where(prev_idx >= 0, gathered, np.nan)


# --------------------------------------------------------------------------- #
# per-frame features
# --------------------------------------------------------------------------- #


@per_frame("area", units="px^2", doc="mask pixel count")
def _area(fb: FrameBundle) -> np.ndarray:
    return fb.prop("area")


@per_frame("perimeter", units="px", doc="perimeter of the cell mask")
def _perimeter(fb: FrameBundle) -> np.ndarray:
    return fb.prop("perimeter")


@per_frame(
    "circularity",
    units="",
    doc="4*pi*area / perimeter^2; 1.0 for a perfect disc, lower as the outline roughens",
)
def _circularity(fb: FrameBundle) -> np.ndarray:
    perimeter = fb.prop("perimeter")
    with np.errstate(divide="ignore", invalid="ignore"):
        value = 4.0 * np.pi * fb.prop("area") / np.square(perimeter)
    return np.where(perimeter > 0, value, np.nan)


@per_frame("eccentricity", units="", doc="eccentricity of the fitted ellipse; 0 is a circle")
def _eccentricity(fb: FrameBundle) -> np.ndarray:
    return fb.prop("eccentricity")


@per_frame("solidity", units="", doc="area divided by convex hull area; drops when the cell blebs")
def _solidity(fb: FrameBundle) -> np.ndarray:
    return fb.prop("solidity")


@per_frame("extent", units="", doc="area divided by bounding box area")
def _extent(fb: FrameBundle) -> np.ndarray:
    return fb.prop("extent")


@per_frame(
    "aspect_ratio",
    units="",
    doc="major over minor axis length of the fitted ellipse; 1.0 is round",
)
def _aspect_ratio(fb: FrameBundle) -> np.ndarray:
    minor = fb.prop("axis_minor_length")
    with np.errstate(divide="ignore", invalid="ignore"):
        value = fb.prop("axis_major_length") / minor
    return np.where(minor > 0, value, np.nan)


@per_frame(
    "t_cell_neighbors",
    units="count",
    uses=("neighbor_radius_px",),
    doc="T cells whose centroid lies within neighbor_radius_px of this cell's centroid",
)
def _t_cell_neighbors(fb: FrameBundle) -> np.ndarray:
    return _radius_counts(fb.centroids, fb.other_centroids, float(fb.params["neighbor_radius_px"]))


@per_frame(
    "cancer_neighbors",
    units="count",
    uses=("neighbor_radius_px",),
    doc="other cancer cells whose centroid lies within neighbor_radius_px",
)
def _cancer_neighbors(fb: FrameBundle) -> np.ndarray:
    counts = _radius_counts(fb.centroids, fb.centroids, float(fb.params["neighbor_radius_px"]))
    # Every present cell matched itself; absent cells stayed NaN.
    return counts - 1.0


@per_frame(
    "dilated_t_cell_neighbors",
    units="count",
    uses=("dilate_radius_px",),
    doc="distinct T cells touching this cell's mask dilated by dilate_radius_px",
)
def _dilated_t_cell_neighbors(fb: FrameBundle) -> np.ndarray:
    from skimage.morphology import binary_dilation, disk

    footprint = disk(int(fb.params["dilate_radius_px"]))
    counts = np.full(len(fb.cell_ids), np.nan)
    for index, cell_id in enumerate(fb.cell_ids):
        if not fb.present[index]:
            continue
        mask = fb.labels == cell_id
        touching = fb.other_labels[binary_dilation(mask, footprint)]
        counts[index] = np.count_nonzero(np.unique(touching))
    return counts


@per_frame("rfp_mean", units="a.u.", needs_image=True, doc="mean RFP intensity inside the mask")
def _rfp_mean(fb: FrameBundle) -> np.ndarray:
    return _intensity_stats(fb, channel=0)["mean"]


@per_frame("rfp_total", units="a.u.", needs_image=True, doc="summed RFP intensity inside the mask")
def _rfp_total(fb: FrameBundle) -> np.ndarray:
    return _intensity_stats(fb, channel=0)["total"]


@per_frame(
    "phase_std",
    units="a.u.",
    needs_image=True,
    doc="standard deviation of phase intensity inside the mask; rises with granularity",
)
def _phase_std(fb: FrameBundle) -> np.ndarray:
    return _intensity_stats(fb, channel=1)["std"]


# --------------------------------------------------------------------------- #
# temporal features
# --------------------------------------------------------------------------- #


@temporal(
    "velocity",
    units="px/frame",
    doc="centroid displacement since the previous active frame, divided by the frame gap",
)
def _velocity(sb: SeriesBundle) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return sb.displacement() / sb.gap()


@temporal(
    "d_area_frac",
    units="",
    depends=("area",),
    doc="(area - previous area) / previous area; scale-free, so an acute collapse is visible",
)
def _d_area_frac(sb: SeriesBundle) -> np.ndarray:
    previous = sb.prev("area")
    with np.errstate(divide="ignore", invalid="ignore"):
        value = (sb.values["area"] - previous) / previous
    return np.where(previous > 0, value, np.nan)


@temporal(
    "d_circularity",
    units="",
    depends=("circularity",),
    doc="circularity minus circularity at the previous active frame",
)
def _d_circularity(sb: SeriesBundle) -> np.ndarray:
    return sb.values["circularity"] - sb.prev("circularity")


@temporal(
    "win_std_log_area",
    units="",
    depends=("area",),
    uses=("window_frames",),
    doc="trailing standard deviation of log(area) over window_frames active frames",
)
def _win_std_log_area(sb: SeriesBundle) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        log_area = np.log(np.where(sb.values["area"] > 0, sb.values["area"], np.nan))
    return _nanstd_window(sb, log_area)


@temporal(
    "win_std_circularity",
    units="",
    depends=("circularity",),
    uses=("window_frames",),
    doc="trailing standard deviation of circularity over window_frames active frames",
)
def _win_std_circularity(sb: SeriesBundle) -> np.ndarray:
    return _nanstd_window(sb, sb.values["circularity"])


@temporal(
    "win_std_displacement",
    units="px",
    uses=("window_frames",),
    doc="trailing standard deviation of per-frame displacement over window_frames active frames",
)
def _win_std_displacement(sb: SeriesBundle) -> np.ndarray:
    return _nanstd_window(sb, sb.displacement())


def _nanstd_window(sb: SeriesBundle, series: np.ndarray) -> np.ndarray:
    """Trailing NaN-aware standard deviation; NaN until two values are available."""
    import warnings

    windowed = sb.window(series, int(sb.params["window_frames"]))
    counts = np.sum(~np.isnan(windowed), axis=-1)
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        # Windows with fewer than two values are expected early in a cell's
        # life and are masked out below; numpy's "degrees of freedom <= 0"
        # warning for them is noise, not a signal.
        warnings.simplefilter("ignore", RuntimeWarning)
        std = np.nanstd(windowed, axis=-1)
    return np.where(counts >= 2, std, np.nan)


# --------------------------------------------------------------------------- #
# shared computation helpers
# --------------------------------------------------------------------------- #


def _radius_counts(centroids: np.ndarray, others: np.ndarray, radius: float) -> np.ndarray:
    """Count `others` within `radius` of each centroid; NaN where the cell is absent.

    Args:
        centroids (np.ndarray): `(N, 2)` subject centroids, NaN where absent.
        others (np.ndarray): `(M, 2)` centroids to count.
        radius (float): distance threshold in pixels.

    Returns:
        np.ndarray: `(N,)` counts as float, NaN for absent cells.
    """
    counts = np.full(len(centroids), np.nan)
    present = ~np.isnan(centroids[:, 0])
    if not present.any() or len(others) == 0:
        counts[present] = 0.0
        return counts
    distances = np.linalg.norm(centroids[present, None, :] - others[None, :, :], axis=-1)
    counts[present] = np.sum(distances <= radius, axis=1)
    return counts


def _intensity_stats(fb: FrameBundle, channel: int) -> dict[str, np.ndarray]:
    """Per-cell mean, total and std of one image channel inside each mask.

    Computed with `np.bincount` over the label image rather than regionprops, so
    all three statistics for every cell cost one pass over the frame.

    Args:
        fb (FrameBundle): the frame.
        channel (int): 0 for RFP, 1 for phase.

    Returns:
        dict[str, np.ndarray]: `mean`, `total`, `std`, each `(N,)` with NaN where
            the cell is absent.
    """
    cache_key = f"_intensity_{channel}"
    if cache_key in fb.props:
        return fb.props[cache_key]

    num_cells = len(fb.cell_ids)
    empty = np.full(num_cells, np.nan)
    if fb.image is None:
        stats = {"mean": empty, "total": empty.copy(), "std": empty.copy()}
        fb.props[cache_key] = stats
        return stats

    flat_labels = fb.labels.ravel()
    values = fb.image[..., channel].ravel().astype(np.float64)
    size = int(fb.labels.max()) + 1
    counts = np.bincount(flat_labels, minlength=size).astype(np.float64)
    sums = np.bincount(flat_labels, weights=values, minlength=size)
    sums_sq = np.bincount(flat_labels, weights=values * values, minlength=size)

    ids = fb.cell_ids
    n = counts[ids]
    total = sums[ids]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = total / n
        variance = np.maximum(sums_sq[ids] / n - np.square(mean), 0.0)
    present = n > 0
    stats = {
        "mean": np.where(present, mean, np.nan),
        "total": np.where(present, total, np.nan),
        "std": np.where(present, np.sqrt(variance), np.nan),
    }
    fb.props[cache_key] = stats
    return stats


REGIONPROPS_COLUMNS = (
    "label",
    "area",
    "perimeter",
    "eccentricity",
    "solidity",
    "extent",
    "axis_major_length",
    "axis_minor_length",
    "centroid",
)


def frame_props(labels: np.ndarray, cell_ids: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Region properties for one frame, scattered onto a fixed column order.

    Args:
        labels (np.ndarray): `(H, W)` label image.
        cell_ids (np.ndarray): `(N,)` the crop's fixed column order.

    Returns:
        tuple:
            props (dict[str, np.ndarray]): one `(N,)` float array per property,
                NaN where the cell does not appear in this frame.
            present (np.ndarray): `(N,)` bool, whether each cell appears.
    """
    from skimage.measure import regionprops_table

    num_cells = len(cell_ids)
    names = [c for c in REGIONPROPS_COLUMNS if c != "label" and c != "centroid"]
    props = {name: np.full(num_cells, np.nan) for name in names}
    props["centroid-0"] = np.full(num_cells, np.nan)
    props["centroid-1"] = np.full(num_cells, np.nan)
    present = np.zeros(num_cells, dtype=bool)

    if not labels.any():
        return props, present

    table = regionprops_table(labels, properties=REGIONPROPS_COLUMNS)
    # `regionprops_table` returns rows for the labels actually present, in
    # ascending label order; map them onto our fixed columns.
    positions = np.searchsorted(cell_ids, table["label"])
    valid = (positions < num_cells) & (cell_ids[np.clip(positions, 0, num_cells - 1)] == table["label"])
    positions = positions[valid]
    present[positions] = True
    for name in names:
        props[name][positions] = np.asarray(table[name], dtype=np.float64)[valid]
    props["centroid-0"][positions] = np.asarray(table["centroid-0"], dtype=np.float64)[valid]
    props["centroid-1"][positions] = np.asarray(table["centroid-1"], dtype=np.float64)[valid]
    return props, present


def label_centroids(labels: np.ndarray) -> np.ndarray:
    """Centroids of every labelled object in a frame.

    Args:
        labels (np.ndarray): `(H, W)` label image.

    Returns:
        np.ndarray: `(M, 2)` of (y, x); empty when the frame has no objects.
    """
    if not labels.any():
        return np.zeros((0, 2))
    from skimage.measure import regionprops_table

    table = regionprops_table(labels, properties=("centroid",))
    return np.stack([table["centroid-0"], table["centroid-1"]], axis=-1)


def compute_per_frame(
    labels_stack: np.ndarray,
    other_stack: np.ndarray,
    image_stack: np.ndarray | None,
    cell_ids: np.ndarray,
    requested: Sequence[str],
    params: dict,
    progress: Callable[[int], None] | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Compute every requested per-frame feature, plus centroids and presence.

    Args:
        labels_stack (np.ndarray): `(T, H, W)` cancer label images.
        other_stack (np.ndarray): `(T, H, W)` T-cell label images.
        image_stack (np.ndarray | None): `(T, H, W, 2)` normalized image, or None.
        cell_ids (np.ndarray): `(N,)` the crop's fixed column order.
        requested (Sequence[str]): feature names; temporal ones are ignored here.
        params (dict): `features.params`.
        progress (Callable[[int], None] | None): called with each frame index.

    Returns:
        tuple:
            values (dict[str, np.ndarray]): `(T, N)` per requested per-frame feature.
            centroids (np.ndarray): `(T, N, 2)` crop-local (y, x), NaN where absent.
            present (np.ndarray): `(T, N)` bool.
    """
    wanted = [n for n in requested if FEATURE_REGISTRY[n].stage == PER_FRAME]
    num_frames = labels_stack.shape[0]
    num_cells = len(cell_ids)

    values = {name: np.full((num_frames, num_cells), np.nan) for name in wanted}
    centroids = np.full((num_frames, num_cells, 2), np.nan)
    present = np.zeros((num_frames, num_cells), dtype=bool)

    for t in range(num_frames):
        labels = labels_stack[t]
        props, frame_present = frame_props(labels, cell_ids)
        centroids[t, :, 0] = props["centroid-0"]
        centroids[t, :, 1] = props["centroid-1"]
        present[t] = frame_present

        bundle = FrameBundle(
            labels=labels,
            other_labels=other_stack[t],
            image=None if image_stack is None else image_stack[t],
            cell_ids=cell_ids,
            centroids=centroids[t],
            other_centroids=label_centroids(other_stack[t]),
            params=params,
            props=props,
            present=frame_present,
        )
        for name in wanted:
            values[name][t] = FEATURE_REGISTRY[name].fn(bundle)
        if progress is not None:
            progress(t)

    return values, centroids, present


def compute_temporal(
    values: dict[str, np.ndarray],
    centroids: np.ndarray,
    active: np.ndarray,
    requested: Sequence[str],
    params: dict,
) -> dict[str, np.ndarray]:
    """Compute every requested temporal feature from the per-frame series.

    Args:
        values (dict[str, np.ndarray]): per-frame features already computed.
        centroids (np.ndarray): `(T, N, 2)` centroids, NaN where absent.
        active (np.ndarray): `(T, N)` bool.
        requested (Sequence[str]): feature names; per-frame ones are ignored here.
        params (dict): `features.params`.

    Returns:
        dict[str, np.ndarray]: `(T, N)` per requested temporal feature.
    """
    wanted = [n for n in requested if FEATURE_REGISTRY[n].stage == TEMPORAL]
    if not wanted:
        return {}
    bundle = SeriesBundle(values=values, centroids=centroids, active=active, params=params)
    return {name: FEATURE_REGISTRY[name].fn(bundle) for name in wanted}
