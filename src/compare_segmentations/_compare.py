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


def _boxes_and_labels(
    seg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[int, tuple[slice, ...]]]:
    boxes = []
    labels = []
    slices = {}
    for prop in regionprops(seg):
        boxes.append(prop.bbox)
        labels.append(prop.label)
        slices[prop.label] = prop.slice
    return np.asarray(boxes), np.asarray(labels), slices


def _union_slice(slices: list[tuple[slice, ...]]) -> tuple[slice, ...]:
    starts = tuple(min(s[dim].start for s in slices) for dim in range(len(slices[0])))
    stops = tuple(max(s[dim].stop for s in slices) for dim in range(len(slices[0])))
    return tuple(slice(start, stop) for start, stop in zip(starts, stops, strict=True))


def _match_labels(
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    iou_threshold: float,
) -> tuple[
    list[tuple[int, int, float]], dict[int, tuple[slice, ...]], dict[int, tuple[slice, ...]]
]:
    if seg_a.shape != seg_b.shape:
        raise ValueError(f"Segmentation shapes must match, got {seg_a.shape} and {seg_b.shape}")

    boxes_a, labels_a, slices_a = _boxes_and_labels(seg_a)
    boxes_b, labels_b, slices_b = _boxes_and_labels(seg_b)

    overlaps = get_labels_with_overlap(
        seg_a,
        seg_b,
        gt_boxes=boxes_a,
        res_boxes=boxes_b,
        gt_labels=labels_a,
        res_labels=labels_b,
        overlap="iou",
    )
    matches = [(a, b, iou) for a, b, iou in overlaps if iou >= iou_threshold]
    return matches, slices_a, slices_b


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
    segmentations: list[np.ndarray],
    overlaps: list[PairwiseOverlap],
    threshold: float,
) -> tuple[list[set[LabelRef]], nx.Graph]:
    graph: nx.Graph = nx.Graph()
    for source, seg in enumerate(segmentations):
        for prop in regionprops(seg):
            graph.add_node(LabelRef(source, prop.label))
    for overlap in overlaps:
        graph.add_edge(
            overlap.ref_a, overlap.ref_b, iou=overlap.iou, agrees=overlap.iou >= threshold
        )

    groups = [set(component) for component in nx.connected_components(graph)]
    return groups, graph


def _classify_group(
    group: set[LabelRef],
    graph: nx.Graph,
    num_sources: int,
    threshold: float,
) -> str:
    """Classify a connected component of labels as "merge", "conflict", or "unmatched"."""
    if len(group) == 1:
        return "unmatched"

    sources = [ref.source for ref in group]
    spans_all_sources = len(sources) == num_sources and len(set(sources)) == num_sources
    all_edges_agree = all(
        data["agrees"] for _u, _v, data in graph.subgraph(group).edges(data=True)
    )
    if spans_all_sources and all_edges_agree:
        return "merge"
    return "conflict"


def merge_segmentations(
    segmentations: list[np.ndarray],
    groups: list[set[LabelRef]],
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

    slices_by_source = [_boxes_and_labels(seg)[2] for seg in segmentations]

    merged = np.zeros(shape, dtype=segmentations[0].dtype)
    for new_label, group in enumerate(groups, start=1):
        member_slices = [slices_by_source[ref.source][ref.label] for ref in group]
        sslice = _union_slice(member_slices)

        sdf_sum = np.zeros(tuple(s.stop - s.start for s in sslice), dtype=np.float64)
        for ref in group:
            mask = segmentations[ref.source][sslice] == ref.label
            sdf_sum += _signed_distance(mask)
        blended = sdf_sum / len(group)

        region = merged[sslice]
        region[blended <= 0] = new_label
        merged[sslice] = region

    return merged


def compare_segmentations(
    segmentations: list[np.ndarray],
    threshold: float = 0.5,
    min_overlap: float = 0.0,
    overlaps: list[PairwiseOverlap] | None = None,
) -> ComparisonResult:
    """Reconcile two or more segmentations of the same volume.

    Labels are grouped across segmentations by transitive pairwise overlap (any
    two labels connected by a chain of overlaps above `min_overlap` land in the
    same group). Each group is then classified:

    - **agree**: the group spans every input segmentation (exactly one label per
      source) and every pairwise overlap within it is at or above `threshold`.
      These groups are merged via `merge_segmentations` into `result.merged`.
    - **conflict**: the group has more than one member but doesn't meet the
      "agree" bar (e.g. only some segmentations overlap here, or the overlap is
      too weak). Flagged in `result.conflicts` for manual resolution.
    - **unmatched**: the label has no qualifying overlap (>= `min_overlap`) with
      any other segmentation. Flagged in `result.unmatched` for manual resolution.

    Args:
        segmentations: list of labeled arrays, all the same shape (2D or 3D).
        threshold: minimum IoU for a group to be classified as "agree" and merged.
            Defaults to 0.5.
        min_overlap: minimum IoU for two labels to be considered related at all (below
            this, they're treated as having no relationship for grouping purposes).
            Defaults to 0.0.
        overlaps: pre-computed pairwise overlaps from `compute_pairwise_overlaps` (or
            loaded via `load_pairwise_overlaps`). If omitted, computed internally. Pass
            this in to change `threshold` repeatedly without recomputing pixel overlaps.

    Returns:
        A `ComparisonResult` with the merged array, conflict groups, unmatched labels,
        and centroids for every flagged label.

    Raises:
        ValueError: if the segmentations don't all have the same shape.
    """
    shape = segmentations[0].shape
    for seg in segmentations[1:]:
        if seg.shape != shape:
            raise ValueError(f"Segmentation shapes must match, got {shape} and {seg.shape}")

    if overlaps is None:
        overlaps = compute_pairwise_overlaps(segmentations, min_overlap=min_overlap)

    groups, graph = _group_labels(segmentations, overlaps, threshold)

    merge_groups: list[set[LabelRef]] = []
    conflicts: list[list[LabelRef]] = []
    unmatched: list[LabelRef] = []
    for group in groups:
        classification = _classify_group(group, graph, len(segmentations), threshold)
        if classification == "merge":
            merge_groups.append(group)
        elif classification == "conflict":
            conflicts.append(sorted(group, key=lambda ref: (ref.source, ref.label)))
        else:
            unmatched.extend(group)

    merged = merge_segmentations(segmentations, merge_groups)

    centroids: dict[LabelRef, tuple[float, ...]] = {}
    flagged_refs = [ref for group in conflicts for ref in group] + unmatched
    flagged_by_source: dict[int, list[LabelRef]] = {}
    for ref in flagged_refs:
        flagged_by_source.setdefault(ref.source, []).append(ref)
    for source, refs in flagged_by_source.items():
        wanted = {ref.label for ref in refs}
        for prop in regionprops(segmentations[source]):
            if prop.label in wanted:
                centroids[LabelRef(source, prop.label)] = tuple(prop.centroid)

    return ComparisonResult(
        merged=merged,
        conflicts=conflicts,
        unmatched=unmatched,
        centroids=centroids,
    )
