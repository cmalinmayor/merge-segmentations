"""Small library for comparing and resolving multiple segmentations of the same image."""

from importlib.metadata import PackageNotFoundError, version

from ._compare import (
    ComparisonResult,
    LabelRef,
    PairwiseOverlap,
    SegmentationProps,
    compare_segmentations,
    compute_pairwise_overlaps,
    find_overlapping_labels,
    load_pairwise_overlaps,
    merge_segmentations,
    precompute_segmentation_props,
    save_pairwise_overlaps,
)
from ._view import view_comparison

try:
    __version__ = version("compare-segmentations")
except PackageNotFoundError:  # package is not installed
    __version__ = "uninstalled"

# Everything listed here becomes the public API and gets an API docs page.
# Implementation lives in underscore-prefixed modules; see CONTRIBUTING.md.
__all__ = [
    "ComparisonResult",
    "LabelRef",
    "PairwiseOverlap",
    "SegmentationProps",
    "__version__",
    "compare_segmentations",
    "compute_pairwise_overlaps",
    "find_overlapping_labels",
    "load_pairwise_overlaps",
    "merge_segmentations",
    "precompute_segmentation_props",
    "save_pairwise_overlaps",
    "view_comparison",
]
