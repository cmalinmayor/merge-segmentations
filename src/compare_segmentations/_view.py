"""Visualize a set of compared segmentations (and their merge result) in napari.

Needs the `view` extra (`napari`); imported lazily so the rest of this package
works without it installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari
    import numpy as np

    from ._compare import ComparisonResult, LabelRef, PairwiseOverlap


def _flagged_labels_by_source(result: ComparisonResult) -> dict[int, set[int]]:
    """`{source: {label values in conflicts or unmatched}}` -- everything NOT merged."""
    flagged: dict[int, set[int]] = {}
    for ref in [ref for group in result.conflicts for ref in group] + result.unmatched:
        flagged.setdefault(ref.source, set()).add(ref.label)
    return flagged


class _MergeHider:
    """Hides merged labels on a Labels layer via its colormap, without touching layer.data.

    Builds the real per-label color_dict once (from the layer's default colormap) and,
    on every `update`, only flips alpha to 0/1 in the existing dict entries -- constructing
    a brand-new `DirectLabelColormap` re-validates every color and is too slow to do on
    every slider change (see motile_tracker's ContourLabels.set_opacity/refresh_colormap,
    which uses the same in-place-alpha + cache-clear trick).
    """

    def __init__(self, layer: napari.layers.Labels, seg: np.ndarray) -> None:
        import numpy as np
        from napari.utils.colormaps import DirectLabelColormap

        self._layer = layer
        original_cmap = layer.colormap
        present_labels = np.unique(seg)
        present_labels = present_labels[present_labels != 0]
        self._color_dict: dict[int | None, np.ndarray] = {
            int(label): np.asarray(original_cmap.map(label), dtype=np.float32).copy()
            for label in present_labels
        }
        self._color_dict[None] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        layer.colormap = DirectLabelColormap(color_dict=self._color_dict)

    def update(self, flagged_labels: set[int]) -> None:
        for label, color in self._color_dict.items():
            if label is None:
                continue
            color[3] = 1.0 if label in flagged_labels else 0.0
        # mutating color_dict values in place, then clearing the cache and reassigning the
        # same colormap object, rebuilds only the (cheap) GPU texture -- not the (slow)
        # per-color validation a new DirectLabelColormap(...) would trigger.
        colormap = self._layer.colormap
        colormap._clear_cache()
        self._layer.colormap = colormap


def _copy_label(
    source: napari.layers.Labels,
    target: napari.layers.Labels,
    label: int,
) -> None:
    """Copy one label's mask from `source` into `target`, in place.

    Respects `target.preserve_labels` (the napari Labels layer "preserve labels"
    checkbox) exactly as painting would: if set, pixels that are already non-zero
    in `target` are left untouched, so the copy can only fill in background.

    The copied region is written under a *new* label id, obtained the same way as
    napari's own "new label" action (`layer.data.max() + 1`, see
    `napari.layers.labels._labels_key_bindings.new_label`) -- never under `label`
    itself, since `label` is only meaningful within `source` and reusing it verbatim
    risks colliding with an unrelated object that already happens to hold that same
    id in `target`, silently fusing the two into one label instead of adding a
    distinguishable new object.
    """
    from napari.layers.labels._labels_key_bindings import new_label as _select_new_label

    mask = source.data == label
    if target.preserve_labels:
        mask &= target.data == 0
    if not mask.any():
        return
    _select_new_label(target)
    target.data[mask] = target.selected_label
    target.refresh()


def view_comparison(
    raw: np.ndarray,
    segmentations: list[np.ndarray],
    overlaps: list[PairwiseOverlap],
    threshold: float = 0.5,
    min_overlap: float = 0.1,
    min_agree: int | None = None,
    names: list[str] | None = None,
) -> napari.Viewer:
    """Open a napari viewer with the raw data, each input segmentation, and the merge result.

    Each segmentation layer only shows labels currently in `result.conflicts` or
    `result.unmatched` -- merged labels are made transparent via the layer's colormap
    (see `_MergeHider`), never removed from the underlying array. Segmentation layers
    open in contour mode (outlines only), so overlapping flagged shapes across
    segmentations stay distinguishable.

    Includes two sliders, IoU threshold and min overlap. Both reclassify and re-merge
    purely from the already-computed `overlaps` on every change -- `min_overlap` only
    filters IoU values that are already known (see `compare_segmentations`), so neither
    slider ever re-touches pixel data. This means `overlaps` must have been computed (via
    `compute_pairwise_overlaps`) with a `min_overlap` at or below the lowest value the
    slider should reach -- the default range here goes down to 0.0, so pass `overlaps`
    computed with `min_overlap=0.0` for the full slider range to work.

    Also includes a "Copy Label" widget for resolving conflicts/unmatched labels by
    hand: pick a source layer, a target layer, and a label value (defaults to the
    source layer's currently-selected label, kept in sync as it changes), then copy
    that label's mask into the target. Respects the target layer's built-in napari
    "preserve labels" checkbox (`layer.preserve_labels`) exactly as painting would --
    if set, the copy only fills in background pixels (0) in the target and never
    overwrites its existing labels.

    Args:
        raw: raw image data, same shape as each array in `segmentations`.
        segmentations: the labeled arrays to compare, all the same shape as `raw`.
        overlaps: pre-computed pairwise overlaps from `compute_pairwise_overlaps` (or loaded
            via `load_pairwise_overlaps`), computed with `min_overlap=0.0` (or otherwise at
            or below every value the min overlap slider will be moved to).
        threshold: initial IoU threshold for the threshold slider. Defaults to 0.5.
        min_overlap: initial value of the min overlap slider. Defaults to 0.1.
        min_agree: passed through to `compare_segmentations` on every change.
            Defaults to requiring all of `segmentations` to agree.
        names: one label per segmentation, used to name its layer. Defaults to
            `"segmentation 0"`, `"segmentation 1"`, etc.

    Returns:
        The napari `Viewer`, with one image layer for `raw`, one labels layer per
        segmentation (showing only conflicts/unmatched), one labels layer for the merge
        result, and the two sliders.
    """
    import napari
    from magicgui import magicgui

    from ._compare import compare_segmentations, precompute_segmentation_props

    if names is None:
        names = [f"segmentation {i}" for i in range(len(segmentations))]

    # regionprops never changes while segmentations don't, so this is computed once here
    # and reused on every slider move instead of being recomputed inside compare_segmentations.
    props = precompute_segmentation_props(segmentations)
    # a given exact set of labels blends to the same pixels regardless of threshold/min_overlap,
    # so this dict (mutated in place by merge_segmentations) lets unchanged groups skip
    # distance_transform_edt entirely across slider moves.
    blend_cache: dict[frozenset[LabelRef], tuple[tuple[slice, ...], np.ndarray]] = {}

    initial_result = compare_segmentations(
        segmentations,
        threshold=threshold,
        min_overlap=min_overlap,
        min_agree=min_agree,
        overlaps=overlaps,
        props=props,
        blend_cache=blend_cache,
    )

    viewer = napari.Viewer()
    viewer.add_image(raw, name="raw")

    seg_layers = [
        viewer.add_labels(seg, name=name) for seg, name in zip(segmentations, names, strict=True)
    ]
    for layer in seg_layers:
        # contour mode draws only label outlines, so overlapping flagged shapes across
        # segmentations stay distinguishable instead of occluding each other as filled areas.
        layer.contour = 2
    hiders = [_MergeHider(layer, seg) for layer, seg in zip(seg_layers, segmentations, strict=True)]
    merged_layer = viewer.add_labels(initial_result.merged, name="merged")

    def _apply_result(result: ComparisonResult) -> None:
        merged_layer.data = result.merged
        flagged_by_source = _flagged_labels_by_source(result)
        for source, hider in enumerate(hiders):
            hider.update(flagged_by_source.get(source, set()))

    _apply_result(initial_result)

    @magicgui(
        auto_call=True,
        iou_threshold={"widget_type": "FloatSlider", "min": 0.0, "max": 1.0, "step": 0.01},
        min_iou_overlap={"widget_type": "FloatSlider", "min": 0.0, "max": 1.0, "step": 0.01},
    )
    def _set_thresholds(
        iou_threshold: float = threshold, min_iou_overlap: float = min_overlap
    ) -> None:
        result = compare_segmentations(
            segmentations,
            threshold=iou_threshold,
            min_overlap=min_iou_overlap,
            min_agree=min_agree,
            overlaps=overlaps,
            props=props,
            blend_cache=blend_cache,
        )
        _apply_result(result)

    viewer.window.add_dock_widget(_set_thresholds, name="Thresholds", area="right")

    all_label_layers = [*seg_layers, merged_layer]

    @magicgui(
        call_button="Copy Label",
        source={"choices": all_label_layers},
        target={"choices": all_label_layers},
    )
    def _copy_label_widget(
        source: napari.layers.Labels = seg_layers[0],
        target: napari.layers.Labels = merged_layer,
        label_value: int = 1,
    ) -> None:
        if source is target:
            raise ValueError("Source and target layers must be different.")
        _copy_label(source, target, label_value)

    def _sync_label_from_source(event: object = None) -> None:
        source = _copy_label_widget.source.value
        if source is not None:
            _copy_label_widget.label_value.value = source.selected_label

    _copy_label_widget.source.changed.connect(_sync_label_from_source)
    for layer in seg_layers:
        layer.events.selected_label.connect(_sync_label_from_source)
    _sync_label_from_source()

    viewer.window.add_dock_widget(_copy_label_widget, name="Copy Label", area="right")
    return viewer
