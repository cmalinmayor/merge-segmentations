"""Visually compare candidate two-mask merge-boundary algorithms.

"Generate 2D pair" / "Generate 3D pair" buttons each build a fresh random pair of
irregular blobs with a barely-touching overlap (IoU searched to land at or below ~0.2 for
2D, ~0.12 for 3D -- the regime that matters most: today anything below `min_overlap` is
treated as unrelated and both labels are copied into the merged output as-is, which is
wrong if they actually share any pixels at all). 2D shapes come straight from
skimage.data.binary_blobs. 3D shapes are a single irregular 2D cross-section tapered
(eroded) toward the top/bottom slices, not `binary_blobs(n_dim=3, ...)` directly -- that
generates a dense field of many blobs that almost always touch each other somewhere in
the volume, so even the largest connected component reads as a chaotic multi-lobed mass
rather than one instance (see `_random_blob_3d`). The 3D IoU ceiling is lower because the
erosion taper compounds size differences between the two cross-sections super-linearly (a
bigger 2D shape both has more area and survives more erosion steps before vanishing), so a
given IoU tends to look more overlapped in 3D than the same number would in 2D.

Generating replaces whatever's currently shown: `mask_a`, `mask_b`, and one "algorithms"
labels layer stacking all 3 split outputs along a leading dimension (which napari turns
into its own slider, alongside the usual z slider for 3D) -- shown in contour mode so it
overlays the two mask layers underneath instead of occluding them. Small labels show the
pair's IoU and which algorithm the slider is currently showing, since napari's own slider
only displays a bare index. Nothing here is wired into `merge_segmentations` -- purely a
comparison tool for picking an algorithm.

Needs this project's `view` extra:

    uv run --extra view scripts/compare_merge_algorithms.py
"""

import napari
import numpy as np
from scipy.ndimage import binary_erosion, shift
from skimage.data import binary_blobs
from skimage.measure import label

from compare_segmentations._merge_algorithms import (
    split_flow_field,
    split_voronoi,
    split_watershed,
)

ALGORITHMS = {
    "voronoi": split_voronoi,
    "watershed": split_watershed,
    "flow": split_flow_field,
}

MAX_IOU = 0.2


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labeled = label(mask)
    sizes = [(labeled == i).sum() for i in range(1, labeled.max() + 1)]
    biggest = int(np.argmax(sizes)) + 1
    return labeled == biggest


def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    union = (mask_a | mask_b).sum()
    return float((mask_a & mask_b).sum() / union) if union else 0.0


def _random_blob_2d(length: int, rng: np.random.Generator) -> np.ndarray:
    # blob_size_fraction ~0.4-0.5 keeps the largest component irregular (not a smooth
    # circle) but still mostly one connected lobe rather than a spiny, hard-to-read
    # tangle -- skimage's own default-ish 0.1-0.3 range gets branchy fast.
    blob_size_fraction = rng.uniform(0.4, 0.5)
    volume_fraction = rng.uniform(0.2, 0.35)
    seed = int(rng.integers(0, 2**31 - 1))
    mask = binary_blobs(
        length=length,
        n_dim=2,
        blob_size_fraction=blob_size_fraction,
        volume_fraction=volume_fraction,
        rng=seed,
    )
    return _largest_component(mask)


def _random_blob_3d(length: int, rng: np.random.Generator) -> np.ndarray:
    """A single roughly cell-shaped 3D blob: an irregular 2D cross-section, tapered
    (eroded) toward top and bottom instead of extruded straight through z.

    `binary_blobs(n_dim=3, ...)` generates a dense field of many blobs that almost
    always touch each other somewhere in the volume -- even after keeping only the
    largest connected component, the result reads as a chaotic multi-lobed mass, not a
    single instance, which isn't representative of most segmentation targets (real cells
    are typically concave-but-simple solids, tapering toward their poles, not a tangle of
    spheres fused together). Eroding a single 2D cross-section by an amount that grows
    toward the top/bottom slices keeps it a single connected 3D shape (each z-slice's
    mask is always a subset of the base slice, so adjacent z-slices always overlap) while
    still being irregular in cross-section, not a sphere/ellipsoid.
    """
    cross_section = _random_blob_2d(length, rng)
    depth = length // 2
    center = depth // 2
    volume = np.zeros((depth, length, length), dtype=bool)
    for z in range(depth):
        erosion_amount = abs(z - center)
        eroded = cross_section
        for _ in range(erosion_amount):
            eroded = binary_erosion(eroded)
        volume[z] = eroded
    return _largest_component(volume)


def _random_blob(length: int, n_dim: int, rng: np.random.Generator) -> np.ndarray:
    if n_dim == 2:
        return _random_blob_2d(length, rng)
    if n_dim == 3:
        return _random_blob_3d(length, rng)
    raise ValueError(f"Unsupported n_dim: {n_dim}")


def generate_touching_pair(
    length: int,
    n_dim: int,
    rng: np.random.Generator,
    max_iou: float = MAX_IOU,
    max_attempts: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two different random blobs shifted together until they barely overlap.

    Searches over the fraction of the centroid-to-centroid distance to pull `mask_b`
    toward `mask_a`, in both directions (the two random blobs might already start out
    overlapping, in which case they need to be pushed apart instead), for the largest
    magnitude that still keeps IoU in `(0, max_iou]`. Retries with entirely new blobs if
    no such fraction is found within `max_attempts` (e.g. blobs so far apart that even a
    huge pull never brings them into contact).

    Args:
        length: linear size of the generated arrays.
        n_dim: 2 or 3.
        rng: source of randomness, so repeated calls (e.g. from a "generate" button)
            produce different shapes each time.
        max_iou: upper bound on the overlap IoU to search for. Defaults to `MAX_IOU`.
        max_attempts: how many fresh blob pairs to try before giving up. Defaults to 50.

    Returns:
        Two boolean masks of shape `(length,) * n_dim` with IoU in `(0, max_iou]`.

    Raises:
        RuntimeError: if no touching pair is found within `max_attempts`.
    """
    for _ in range(max_attempts):
        mask_a = _random_blob(length, n_dim, rng)
        mask_b_raw = _random_blob(length, n_dim, rng)

        centroid_a = np.array(np.nonzero(mask_a)).mean(axis=1)
        centroid_b = np.array(np.nonzero(mask_b_raw)).mean(axis=1)
        direction = centroid_a - centroid_b

        best: tuple[np.ndarray, float] | None = None
        for sign in (1.0, -1.0):
            for frac in np.linspace(0.0, 3.0, 120):
                offset = direction * frac * sign
                mask_b = shift(mask_b_raw.astype(np.float64), shift=offset, order=0) > 0.5
                iou = _iou(mask_a, mask_b)
                if 0 < iou <= max_iou and (best is None or iou > best[1]):
                    best = (mask_b, iou)
                if iou > max_iou:
                    break

        if best is not None:
            return mask_a, best[0]

    raise RuntimeError(
        f"Could not find a touching pair (IoU <= {max_iou}) within {max_attempts} attempts"
    )


def main() -> None:
    from magicgui.widgets import Container, Label, PushButton

    viewer = napari.Viewer()
    algo_names = list(ALGORITHMS)
    rng = np.random.default_rng()
    current_layers: list[napari.layers.Labels] = []
    algo_label = Label(value=f"algorithm: {algo_names[0]}")
    iou_label = Label(value="iou: -")

    def _update_algo_label(event: object = None) -> None:
        # the "algorithms" layer's own leading axis is always index 0 of ITS data,
        # which napari maps to the first slider step regardless of how many other
        # (e.g. z) dimensions the current shape pair has. viewer.dims briefly reflects
        # other layers (e.g. mask_a alone, with no algorithm axis) while layers are being
        # swapped below, so guard against a step index that doesn't correspond to a
        # real algorithm yet.
        step = viewer.dims.current_step[0]
        if 0 <= step < len(algo_names):
            algo_label.value = f"algorithm: {algo_names[step]}"

    viewer.dims.events.current_step.connect(_update_algo_label)

    def _generate(n_dim: int, length: int, max_iou: float = MAX_IOU) -> None:
        for layer in current_layers:
            viewer.layers.remove(layer)
        current_layers.clear()

        mask_a, mask_b = generate_touching_pair(length, n_dim, rng, max_iou=max_iou)
        iou_label.value = f"iou: {_iou(mask_a, mask_b):.3f}"
        seg_a = mask_a.astype(np.int32)
        seg_b = mask_b.astype(np.int32) * 2
        stacked = np.stack([ALGORITHMS[name](mask_a, mask_b) for name in algo_names])

        current_layers.append(viewer.add_labels(seg_a, name="mask_a"))
        current_layers.append(viewer.add_labels(seg_b, name="mask_b"))
        algorithms_layer = viewer.add_labels(stacked, name="algorithms")
        algorithms_layer.contour = 2
        current_layers.append(algorithms_layer)
        _update_algo_label()

    generate_2d = PushButton(text="Generate 2D pair")
    generate_2d.changed.connect(lambda: _generate(n_dim=2, length=64))
    generate_3d = PushButton(text="Generate 3D pair")
    # the erosion-based taper in _random_blob_3d compounds size differences between two
    # cross-sections super-linearly (a bigger 2D shape both has more area AND survives
    # more erosion steps before vanishing), so 3D pairs tend to look more overlapped than
    # the same IoU would in 2D -- use a lower ceiling here.
    generate_3d.changed.connect(lambda: _generate(n_dim=3, length=32, max_iou=0.12))

    viewer.window.add_dock_widget(
        Container(widgets=[generate_2d, generate_3d, iou_label, algo_label]),
        name="Comparison controls",
        area="right",
    )
    _generate(n_dim=2, length=64)

    napari.run()


if __name__ == "__main__":
    main()
