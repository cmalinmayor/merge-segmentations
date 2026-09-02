"""Candidate algorithms for splitting two overlapping masks along a shared boundary.

Prototypes for comparison, not wired into `merge_segmentations` yet. Each function
takes two boolean masks of the same shape (2D or 3D) and returns a labeled array:
1 where `mask_a` wins, 2 where `mask_b` wins, 0 outside both masks.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.segmentation import watershed

from ._compare import _signed_distance


def split_voronoi(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    """Assign each pixel to whichever mask's signed distance field is more negative there.

    This is the same nearest-boundary-point tie-break `merge_segmentations` uses. It's a
    purely local metric -- distance to the single nearest boundary pixel -- so it can
    produce a jagged seam on irregular/concave shapes where the two fields tie along a
    ragged curve rather than a smooth one.
    """
    sdf_a = _signed_distance(mask_a)
    sdf_b = _signed_distance(mask_b)

    labels = np.zeros(mask_a.shape, dtype=np.int32)
    either = mask_a | mask_b
    a_wins = either & (sdf_a <= sdf_b)
    labels[a_wins] = 1
    labels[either & ~a_wins] = 2
    return labels


def split_watershed(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    smooth_sigma: float = 2.0,
) -> np.ndarray:
    """Flood-fill from each mask's exclusive territory across a smoothed elevation surface.

    Markers seed from `mask_a`'s and `mask_b`'s exclusive (non-overlapping) territory, so
    the flood only needs to resolve the contested area rather than re-decide pixels that
    were never ambiguous. The elevation surface is `min(sdf_a, sdf_b)`, Gaussian-smoothed:
    this is a valley (very negative) deep inside EITHER shape, rising toward a ridge where
    the two shapes' boundaries meet -- watershed needs both markers to sit in low basins
    with a ridge between them, not one marker in a global minimum and the other in a
    global maximum (`sdf_b - sdf_a` puts marker_a's basin near its peak, so it can never
    flood outward -- verified empirically: that version assigns exactly the marker's own
    pixels to it and gives everything else to the other label). Smoothing is specifically
    to avoid inheriting the raw SDFs' local jaggedness -- watershed floods along a
    monotonic surface, so smoothing it first is what should produce a smoother seam than
    `split_voronoi`.

    Does not itself guarantee that either output region stays as connected as its input
    mask was; the caller should verify with `skimage.measure.label` if that matters.
    """
    sdf_a = _signed_distance(mask_a)
    sdf_b = _signed_distance(mask_b)
    elevation = gaussian_filter(np.minimum(sdf_a, sdf_b), sigma=smooth_sigma)

    marker_a = mask_a & ~mask_b
    marker_b = mask_b & ~mask_a
    markers = np.zeros(mask_a.shape, dtype=np.int32)
    markers[marker_a] = 1
    markers[marker_b] = 2

    either = mask_a | mask_b
    labels = watershed(elevation, markers=markers, mask=either)
    return np.asarray(labels, dtype=np.int32)


def _relax_flow_field(
    source: np.ndarray,
    domain: np.ndarray,
    n_iter: int = 200,
) -> np.ndarray:
    """Iteratively diffuse a unit potential from `source` across `domain`.

    A cheap Jacobi-iteration stand-in for solving the Laplace equation (steady-state
    diffusion): repeatedly average each pixel with its neighbors, re-pinning `source`
    pixels to 1 every iteration, and zeroing anything outside `domain`. After enough
    iterations this approximates the harmonic potential that would result from solving
    the linear system directly, without needing to build a sparse matrix.
    """
    field = source.astype(np.float64).copy()
    ndim = field.ndim
    for _ in range(n_iter):
        total = np.zeros_like(field)
        counts = np.zeros_like(field)
        for axis in range(ndim):
            shifted_fwd = np.roll(field, 1, axis=axis)
            shifted_back = np.roll(field, -1, axis=axis)
            total += shifted_fwd + shifted_back
            counts += 2
        field = np.where(source, 1.0, total / counts)
        field[~domain] = 0.0
    return field


def split_flow_field(mask_a: np.ndarray, mask_b: np.ndarray, n_iter: int = 200) -> np.ndarray:
    """Assign each pixel to whichever mask's diffused potential field reaches it more strongly.

    Each mask's exclusive territory is a unit-potential source; the potential diffuses
    (via repeated local averaging, an iterative stand-in for solving the Laplace equation)
    across the union of both masks, decaying with distance in a way that follows the
    domain's shape rather than straight-line Euclidean distance -- so, unlike
    `split_voronoi`, a concave/U-shaped mask's potential has to flow around the bend
    instead of just measuring distance to the nearest point on its own boundary. Flow
    never leaves the union of the two masks, so it structurally cannot create a
    disconnected component that wasn't already present in `mask_a | mask_b`.
    """
    either = mask_a | mask_b
    marker_a = mask_a & ~mask_b
    marker_b = mask_b & ~mask_a

    field_a = _relax_flow_field(marker_a, either, n_iter=n_iter)
    field_b = _relax_flow_field(marker_b, either, n_iter=n_iter)

    labels = np.zeros(mask_a.shape, dtype=np.int32)
    a_wins = either & (field_a >= field_b)
    labels[a_wins] = 1
    labels[either & ~a_wins] = 2
    return labels
