"""Compare and reconcile label pairs across two or more segmentation arrays.

The workflow is:

1. `compute_pairwise_overlaps` -- find all pairwise label overlaps (the
   expensive, pixel-touching step; cacheable via `save_pairwise_overlaps` /
   `load_pairwise_overlaps` so a threshold change doesn't require recomputing
   it).
2. `compare_segmentations` -- group labels that mutually overlap, classify
   each group as "agree" (merge), "conflict" (flag for manual resolution), or
   "unmatched" (present in only one segmentation, flag for manual
   resolution), and merge the agreeing groups via `merge_segmentations`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops
from traccuracy.matchers._compute_overlap import get_labels_with_overlap


@dataclass(frozen=True)
class LabelRef:
    """A reference to a single label in one of the compared segmentations."""

    source: int
    """Index into the list of segmentations passed to `compare_segmentations`."""
    label: int
    """The label value within that segmentation."""


@dataclass(frozen=True)
class PairwiseOverlap:
    """A single pairwise label match, as produced by `compute_pairwise_overlaps`."""

    ref_a: LabelRef
    ref_b: LabelRef
    iou: float


@dataclass
class ComparisonResult:
    """The outcome of reconciling two or more segmentations."""

    merged: np.ndarray
    """New labeled array (0 = background) containing one new label per agreeing group."""
    conflicts: list[list[LabelRef]]
    """Groups of mutually-overlapping labels that don't fully agree, for manual resolution."""
    unmatched: list[LabelRef]
    """Labels with no qualifying overlap in any other segmentation."""
    centroids: dict[LabelRef, tuple[float, ...]]
    """Centroid of every `LabelRef` appearing in `conflicts` or `unmatched`."""


@dataclass
class SegmentationProps:
    """`regionprops` results for one segmentation, cached since they never change with
    `threshold`/`min_overlap`/`min_agree` -- only the pixel data does.
    """

    boxes: np.ndarray
    labels: np.ndarray
    slices: dict[int, tuple[slice, ...]]
    centroids: dict[int, tuple[float, ...]]


def _segmentation_props(seg: np.ndarray) -> SegmentationProps:
    boxes = []
    labels = []
    slices = {}
    centroids = {}
    for prop in regionprops(seg):
        boxes.append(prop.bbox)
        labels.append(prop.label)
        slices[prop.label] = prop.slice
        centroids[prop.label] = tuple(prop.centroid)
    return SegmentationProps(np.asarray(boxes), np.asarray(labels), slices, centroids)


def precompute_segmentation_props(segmentations: list[np.ndarray]) -> list[SegmentationProps]:
    """Precompute and cache `regionprops` data for each segmentation.

    `compare_segmentations` and `merge_segmentations` otherwise recompute this on every
    call, even though it's identical as long as the segmentation arrays themselves haven't
    changed -- pass the result here to `compare_segmentations(..., props=...)` to skip that
    work on every `threshold`/`min_overlap`/`min_agree` change (e.g. every slider move in
    `view_comparison`).

    Args:
        segmentations: list of labeled arrays, all the same shape (2D or 3D).

    Returns:
        One `SegmentationProps` per entry in `segmentations`, in the same order.
    """
    return [_segmentation_props(seg) for seg in segmentations]


def _union_slice(slices: list[tuple[slice, ...]]) -> tuple[slice, ...]:
    starts = tuple(min(s[dim].start for s in slices) for dim in range(len(slices[0])))
    stops = tuple(max(s[dim].stop for s in slices) for dim in range(len(slices[0])))
    return tuple(slice(start, stop) for start, stop in zip(starts, stops, strict=True))


def _pad_slice(sslice: tuple[slice, ...], shape: tuple[int, ...]) -> tuple[slice, ...]:
    """`sslice` padded by 1 pixel per dimension, clamped to `shape`.

    `_signed_distance` needs background pixels around a shape to measure "outside"
    distance against -- a crop tight to the shape's own bbox has none (the whole crop is
    foreground), which corrupts the SDF magnitude near the crop edge. 1 pixel is enough:
    `merge_segmentations`'s Voronoi tie-break only ever compares two groups within the
    intersection of their own crops (that's the only region either of them ever writes
    to), so the fix only needs each group's SDF to be correct at its own true boundary,
    not to reach out towards a competing group that might be far away.
    """
    padded = []
    for dim, s in enumerate(sslice):
        margin = 1
        start = max(0, s.start - margin)
        stop = min(shape[dim], s.stop + margin)
        padded.append(slice(start, stop))
    return tuple(padded)


def _match_labels(
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    iou_threshold: float,
) -> tuple[
    list[tuple[int, int, float]], dict[int, tuple[slice, ...]], dict[int, tuple[slice, ...]]
]:
    if seg_a.shape != seg_b.shape:
        raise ValueError(f"Segmentation shapes must match, got {seg_a.shape} and {seg_b.shape}")

    props_a = _segmentation_props(seg_a)
    props_b = _segmentation_props(seg_b)

    overlaps = get_labels_with_overlap(
        seg_a,
        seg_b,
        gt_boxes=props_a.boxes,
        res_boxes=props_b.boxes,
        gt_labels=props_a.labels,
        res_labels=props_b.labels,
        overlap="iou",
    )
    matches = [(a, b, iou) for a, b, iou in overlaps if iou >= iou_threshold]
    return matches, props_a.slices, props_b.slices


def find_overlapping_labels(
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    iou_threshold: float = 0.5,
) -> list[tuple[int, int, float]]:
    """Find all pairs of labels in two segmentation arrays that overlap by IoU.

    Only label pairs whose IoU is at or above `iou_threshold` are returned. Background
    (label 0) is ignored.

    Args:
        seg_a: labeled array, 2D or 3D.
        seg_b: labeled array of the same shape as `seg_a`.
        iou_threshold: minimum IoU for a pair to be included. Defaults to 0.5.

    Returns:
        List of (label_a, label_b, iou) tuples for each overlapping pair at or above
        `iou_threshold`.

    Raises:
        ValueError: if `seg_a` and `seg_b` do not have the same shape.
    """
    matches, _slices_a, _slices_b = _match_labels(seg_a, seg_b, iou_threshold)
    return matches


def _signed_distance(mask: np.ndarray) -> np.ndarray:
    """Return a signed distance field: negative inside, positive outside.

    Distances are measured to the nearest boundary pixel, so blending several
    such fields and thresholding at 0 morphs the shapes into their middle ground.
    """
    inside = distance_transform_edt(mask)
    outside = distance_transform_edt(~mask)
    return np.asarray(outside - inside)


def compute_pairwise_overlaps(
    segmentations: list[np.ndarray],
    min_overlap: float = 0.0,
) -> list[PairwiseOverlap]:
    """Find all pairwise label overlaps across a list of segmentations.

    This is the expensive, pixel-touching step of the comparison workflow. Its
    result can be saved via `save_pairwise_overlaps` and reloaded via
    `load_pairwise_overlaps`, so that re-running `compare_segmentations` with a
    different `threshold` doesn't require recomputing pixel overlaps.

    Args:
        segmentations: list of labeled arrays, all the same shape (2D or 3D).
        min_overlap: minimum IoU for a pairwise match to be recorded at all. Defaults to 0.0
            (any nonzero pixel overlap is recorded).

    Returns:
        List of `PairwiseOverlap`, one per matching label pair across every combination
        of two segmentations in the input list.

    Raises:
        ValueError: if the segmentations don't all have the same shape.
    """
    overlaps: list[PairwiseOverlap] = []
    for i in range(len(segmentations)):
        for j in range(i + 1, len(segmentations)):
            matches = find_overlapping_labels(
                segmentations[i], segmentations[j], iou_threshold=min_overlap
            )
            overlaps.extend(
                PairwiseOverlap(LabelRef(i, label_a), LabelRef(j, label_b), iou)
                for label_a, label_b, iou in matches
                if iou > 0
            )
    return overlaps


def save_pairwise_overlaps(overlaps: list[PairwiseOverlap], path: str | Path) -> None:
    """Save pairwise overlaps computed by `compute_pairwise_overlaps` to a JSON file.

    Args:
        overlaps: list of `PairwiseOverlap` to save.
        path: destination file path.
    """
    data = [
        {
            "source_a": o.ref_a.source,
            "label_a": o.ref_a.label,
            "source_b": o.ref_b.source,
            "label_b": o.ref_b.label,
            "iou": o.iou,
        }
        for o in overlaps
    ]
    Path(path).write_text(json.dumps(data, indent=2))


def load_pairwise_overlaps(path: str | Path) -> list[PairwiseOverlap]:
    """Load pairwise overlaps previously saved by `save_pairwise_overlaps`.

    Args:
        path: source file path.

    Returns:
        List of `PairwiseOverlap` reconstructed from the file.
    """
    data = json.loads(Path(path).read_text())
    return [
        PairwiseOverlap(
            LabelRef(d["source_a"], d["label_a"]),
            LabelRef(d["source_b"], d["label_b"]),
            d["iou"],
        )
        for d in data
    ]


def _group_labels(
    props: list[SegmentationProps],
    overlaps: list[PairwiseOverlap],
    threshold: float,
) -> tuple[list[set[LabelRef]], nx.Graph]:
    """Connect every pair of labels that overlap at all, for size-1 (unmatched) detection.

    This graph is intentionally permissive (any recorded overlap is an edge, not just
    ones >= `threshold`) -- it's only used to find labels with *no* overlap anywhere
    (`_group_labels` components of size 1), not to decide what merges. Merging is
    decided per label by `_find_agreeing_group`, which only trusts mutual best matches.
    """
    graph: nx.Graph = nx.Graph()
    for source, source_props in enumerate(props):
        for label in source_props.labels:
            graph.add_node(LabelRef(source, int(label)))
    for overlap in overlaps:
        graph.add_edge(
            overlap.ref_a, overlap.ref_b, iou=overlap.iou, agrees=overlap.iou >= threshold
        )

    groups = [set(component) for component in nx.connected_components(graph)]
    return groups, graph


def _best_matches(
    overlaps: list[PairwiseOverlap],
) -> dict[LabelRef, dict[int, tuple[LabelRef, float]]]:
    """For each label, its highest-IoU match in every other source it overlaps at all.

    Returns:
        `{ref: {other_source: (best_ref, iou)}}` -- for label `ref`, the best match found
        in `other_source`, with its IoU.
    """
    best: dict[LabelRef, dict[int, tuple[LabelRef, float]]] = {}
    for overlap in overlaps:
        for this_ref, other_ref in ((overlap.ref_a, overlap.ref_b), (overlap.ref_b, overlap.ref_a)):
            by_source = best.setdefault(this_ref, {})
            current = by_source.get(other_ref.source)
            if current is None or overlap.iou > current[1]:
                by_source[other_ref.source] = (other_ref, overlap.iou)
    return best


def _find_agreeing_group(
    ref: LabelRef,
    best: dict[LabelRef, dict[int, tuple[LabelRef, float]]],
    num_sources: int,
    min_agree: int,
    threshold: float,
) -> set[LabelRef] | None:
    """The mutually-agreeing group containing `ref`, if `ref` is its lowest-source-index member.

    A label `M` in another source belongs to `ref`'s group only if `ref` and `M` are each
    other's best match (mutual best) and that match is >= `threshold`. This structurally
    guarantees at most one label per source: `ref` picks at most one best label per other
    source, and requiring the pick to be mutual rules out two different same-source labels
    both claiming the same partner. Checked from only the lowest-source-index member so
    each qualifying group is returned exactly once (not once per member).

    Returns:
        The agreeing group (including `ref`) if it has >= `min_agree` distinct sources,
        else `None`. Also `None` if `ref` isn't the lowest-source-index member, so callers
        iterating over all labels naturally get each group once.
    """
    group = {ref}
    for other_source in range(num_sources):
        if other_source == ref.source:
            continue
        match = best.get(ref, {}).get(other_source)
        if match is None:
            continue
        other_ref, iou = match
        if iou < threshold:
            continue
        back_match = best.get(other_ref, {}).get(ref.source)
        if back_match is not None and back_match[0] == ref:
            group.add(other_ref)

    if len(group) < min_agree:
        return None
    if min(r.source for r in group) != ref.source:
        return None
    return group


def merge_segmentations(
    segmentations: list[np.ndarray],
    groups: list[set[LabelRef]],
    props: list[SegmentationProps] | None = None,
    blend_cache: dict[frozenset[LabelRef], tuple[tuple[slice, ...], np.ndarray]] | None = None,
) -> np.ndarray:
    """Merge groups of matched labels into their shape "middle ground".

    For each group, blends the masks of all its member labels via signed distance
    transform interpolation, thresholded at the midpoint. This preserves the painted
    shape of each object rather than approximating it as a disk/sphere, and
    generalizes the pairwise nesting guarantee (where one mask fully contains
    another, the merged mask sits between them) to any number of masks. Each group
    is blended within the union of its members' bounding boxes rather than over the
    full array.

    Args:
        segmentations: list of labeled arrays, all the same shape (2D or 3D).
        groups: list of label groups to merge, each a set of `LabelRef`. Every
            `LabelRef` must name a label present in the corresponding segmentation.
        props: pre-computed `precompute_segmentation_props(segmentations)`. If omitted,
            computed internally. Pass this in to skip recomputing `regionprops` when
            calling repeatedly with unchanged `segmentations` (e.g. from a UI slider).
        blend_cache: a dict this function reads from and writes to, keyed by
            `frozenset(group)`, caching each group's `(bounding slice, blended field)`.
            The signed-distance blend of a given exact set of labels never changes (it
            depends only on the labels' pixels, not on `threshold`/`min_overlap`), so
            pass the same dict across repeated calls (e.g. from a UI slider) to skip
            `distance_transform_edt` entirely for groups seen before. Mutated in place;
            pass `{}` once and keep reusing it, rather than passing a fresh dict per call.

    When two groups' blended shapes both claim the same pixel (their bounding boxes
    overlap in pixel space), the pixel goes to whichever group's blended signed-distance
    value is more negative there -- i.e. whichever group is more confidently "inside" at
    that point, a Voronoi-style tie-break using the same field already computed for the
    blend, rather than an arbitrary group ordering silently overwriting the other.

    Returns:
        Labeled array of the same shape as the inputs, with one new label (starting
        at 1) per group, in the order given.

    Raises:
        ValueError: if the segmentations don't all have the same shape.
    """
    shape = segmentations[0].shape
    for seg in segmentations[1:]:
        if seg.shape != shape:
            raise ValueError(f"Segmentation shapes must match, got {shape} and {seg.shape}")

    if props is None:
        props = precompute_segmentation_props(segmentations)
    slices_by_source = [p.slices for p in props]
    if blend_cache is None:
        blend_cache = {}

    merged = np.zeros(shape, dtype=segmentations[0].dtype)
    # tracks, per pixel, the SDF value of whichever group currently claims it, so a later
    # group can only take a pixel away by being more confidently "inside" there.
    best_sdf = np.full(shape, np.inf, dtype=np.float64)
    for new_label, group in enumerate(groups, start=1):
        cache_key = frozenset(group)
        cached = blend_cache.get(cache_key)
        if cached is not None:
            sslice, blended = cached
        else:
            member_slices = [slices_by_source[ref.source][ref.label] for ref in group]
            tight_slice = _union_slice(member_slices)
            # padded beyond the group's own bbox, so _signed_distance has real background
            # to measure against and the Voronoi tie-break below has room to compare
            # against a neighboring group beyond just this group's own tight bbox.
            sslice = _pad_slice(tight_slice, shape)

            sdf_sum = np.zeros(tuple(s.stop - s.start for s in sslice), dtype=np.float64)
            for ref in group:
                mask = segmentations[ref.source][sslice] == ref.label
                sdf_sum += _signed_distance(mask)
            blended = sdf_sum / len(group)
            blend_cache[cache_key] = (sslice, blended)

        region = merged[sslice]
        region_best = best_sdf[sslice]
        wins = (blended <= 0) & (blended < region_best)
        region[wins] = new_label
        region_best[wins] = blended[wins]
        merged[sslice] = region
        best_sdf[sslice] = region_best

    return merged


def compare_segmentations(
    segmentations: list[np.ndarray],
    threshold: float = 0.5,
    min_overlap: float = 0.0,
    min_agree: int | None = None,
    overlaps: list[PairwiseOverlap] | None = None,
    props: list[SegmentationProps] | None = None,
    blend_cache: dict[frozenset[LabelRef], tuple[tuple[slice, ...], np.ndarray]] | None = None,
) -> ComparisonResult:
    """Reconcile two or more segmentations of the same volume.

    Each label is matched to its single best (highest-IoU) overlap in every other
    source. A set of labels merges only if they're all mutually each other's best
    match (so a label can only ever join one such set, structurally preventing two
    labels from the same source ending up in the same merge) at or above `threshold`,
    and the set spans at least `min_agree` distinct sources -- sources with no
    qualifying match are simply absent from the set, not required.

    Labels that aren't part of any merging set are flagged for manual resolution based
    on whether they overlap (>= `min_overlap`) anything else at all:

    - **conflict**: overlaps at least one other label (directly or transitively), but
      not enough to merge. Flagged in `result.conflicts`, grouped with everything else
      it transitively overlaps.
    - **unmatched**: no qualifying overlap (>= `min_overlap`) with any other
      segmentation. Flagged in `result.unmatched`.

    Args:
        segmentations: list of labeled arrays, all the same shape (2D or 3D).
        threshold: minimum IoU for a mutual-best match to count towards merging.
            Defaults to 0.5.
        min_overlap: minimum IoU for two labels to be considered related at all (below
            this, they're treated as having no relationship for grouping purposes).
            Defaults to 0.0.
        min_agree: minimum number of distinct segmentations that must mutually agree for
            a set of labels to be merged. Must be a strict majority of `len(segmentations)`
            (so a label can belong to at most one qualifying set). Defaults to
            `len(segmentations)` (all of them must agree).
        overlaps: pre-computed pairwise overlaps from `compute_pairwise_overlaps` (or
            loaded via `load_pairwise_overlaps`). If omitted, computed internally. Pass
            this in to change `threshold` repeatedly without recomputing pixel overlaps.
        props: pre-computed `precompute_segmentation_props(segmentations)`. If omitted,
            computed internally. Pass this in (alongside `overlaps`) to skip recomputing
            `regionprops` on every call when `segmentations` hasn't changed -- e.g. every
            slider move in `view_comparison`.
        blend_cache: passed through to `merge_segmentations` -- a dict this function reads
            from and writes to, caching each merge group's blended pixels by its exact
            label set. Pass the same dict across repeated calls to skip re-blending groups
            that recur unchanged (e.g. most groups, across adjacent slider positions).

    Returns:
        A `ComparisonResult` with the merged array, conflict groups, unmatched labels,
        and centroids for every flagged label.

    Raises:
        ValueError: if the segmentations don't all have the same shape, or if
            `min_agree` is not a strict majority of `len(segmentations)`.
    """
    shape = segmentations[0].shape
    for seg in segmentations[1:]:
        if seg.shape != shape:
            raise ValueError(f"Segmentation shapes must match, got {shape} and {seg.shape}")

    num_sources = len(segmentations)
    if min_agree is None:
        min_agree = num_sources
    if min_agree <= num_sources / 2:
        raise ValueError(
            f"min_agree must be a strict majority of {num_sources} segmentations, got {min_agree}"
        )

    if overlaps is None:
        overlaps = compute_pairwise_overlaps(segmentations, min_overlap=min_overlap)
    else:
        # `overlaps` may have been computed with a lower min_overlap (e.g. 0.0, to cache
        # every nonzero pair) than the one requested here -- filtering the already-computed
        # IoUs is equivalent to recomputing at this min_overlap, without touching pixels.
        overlaps = [o for o in overlaps if o.iou >= min_overlap]

    if props is None:
        props = precompute_segmentation_props(segmentations)

    best = _best_matches(overlaps)

    merge_groups: list[set[LabelRef]] = []
    merged_refs: set[LabelRef] = set()
    all_refs = [
        LabelRef(source, int(label))
        for source, source_props in enumerate(props)
        for label in source_props.labels
    ]
    for ref in sorted(all_refs, key=lambda r: (r.source, r.label)):
        group = _find_agreeing_group(ref, best, num_sources, min_agree, threshold)
        if group is not None:
            merge_groups.append(group)
            merged_refs.update(group)

    groups, _graph = _group_labels(props, overlaps, threshold)

    conflicts: list[list[LabelRef]] = []
    unmatched: list[LabelRef] = []
    for group in groups:
        leftover = group - merged_refs
        if len(leftover) == 1:
            unmatched.extend(leftover)
        elif len(leftover) > 1:
            conflicts.append(sorted(leftover, key=lambda ref: (ref.source, ref.label)))

    merged = merge_segmentations(segmentations, merge_groups, props=props, blend_cache=blend_cache)

    centroids: dict[LabelRef, tuple[float, ...]] = {}
    for ref in [ref for group in conflicts for ref in group] + unmatched:
        centroids[ref] = props[ref.source].centroids[ref.label]

    return ComparisonResult(
        merged=merged,
        conflicts=conflicts,
        unmatched=unmatched,
        centroids=centroids,
    )
