"""Compare the 4 cellpose_runner pharynx parameter-sweep outputs on t7.

Runs `compare_segmentations` over the masks.zarr outputs of 4 cellpose_runner
runs (a smooth_radius x cellprob_threshold sweep on the same pharynx t7 raw
data), reports how many cells agree, conflict, or are unmatched, and opens a
napari viewer with the raw data, each input segmentation, and the merge result.

Needs this project's `view` extra (napari, tifffile, zarr, janelia-pathlib):

    uv run --extra view scripts/compare_pharynx_runs.py

The cache always holds overlaps computed at min_overlap=0.0 (every nonzero pixel
overlap), so any higher min_overlap -- including the viewer's min overlap slider
-- can be applied by filtering the cache, never by recomputing from pixels.

Pass --recompute to ignore the cached pairwise overlaps file and recompute (and
overwrite it) from the mask arrays -- use this after the segmentations
themselves have changed.
"""

import argparse
from pathlib import Path

import napari
import numpy as np
import tifffile
import zarr
from janelia_pathlib import JaneliaPath

from compare_segmentations import (
    PairwiseOverlap,
    compare_segmentations,
    compute_pairwise_overlaps,
    load_pairwise_overlaps,
    save_pairwise_overlaps,
    view_comparison,
)

RUNS_ROOT = Path("/nrs/shroff/malinmayorc/cellpose/SLS267xOH15257")
RAW_PATH = Path(
    "/nrs/shroff/Pharynx_Project/Data/SLS267xOH15257/080626_Dataset_2/Raw_Data/561/t7.tif"
)
RUN_NAMES = [
    "calculating-serpent_20260831T143851",
    "ultramarine-coot_20260831T143909",
    "quaint-wombat_20260831T143825",
    "luminous-lynx_20260831T143920",
]
OVERLAPS_PATH = RUNS_ROOT / "t7_pairwise_overlaps.json"


def _resolve(path: Path) -> JaneliaPath:
    resolved = JaneliaPath(path)
    if not resolved.exists():
        resolved.mount()
    return resolved


def _load_masks(run_name: str) -> np.ndarray:
    path = _resolve(RUNS_ROOT / run_name / "masks.zarr")
    return np.asarray(zarr.open(path, mode="r")[:])


def _load_raw() -> np.ndarray:
    path = _resolve(RAW_PATH)
    mapped = tifffile.memmap(path)
    return np.asarray(mapped)


def _load_overlaps(segmentations: list[np.ndarray], recompute: bool) -> list[PairwiseOverlap]:
    overlaps_path = _resolve(OVERLAPS_PATH)
    if not recompute and overlaps_path.exists():
        print(f"Loading cached pairwise overlaps from {overlaps_path}")
        return load_pairwise_overlaps(overlaps_path)

    overlaps = compute_pairwise_overlaps(segmentations, min_overlap=0.0)
    save_pairwise_overlaps(overlaps, overlaps_path)
    print(f"Saved pairwise overlaps to {overlaps_path}")
    return overlaps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Ignore the cached pairwise overlaps file and recompute (overwriting it).",
    )
    args = parser.parse_args()

    segmentations = [_load_masks(run_name) for run_name in RUN_NAMES]
    overlaps = _load_overlaps(segmentations, recompute=args.recompute)

    result = compare_segmentations(
        segmentations, threshold=0.5, min_overlap=0.1, min_agree=3, overlaps=overlaps
    )

    n_merged = len(np.unique(result.merged)) - 1  # exclude background
    print(f"Compared {len(RUN_NAMES)} runs: {RUN_NAMES}")
    print(f"Merged (agreeing) cells: {n_merged}")
    print(f"Conflicting groups: {len(result.conflicts)}")
    print(f"Unmatched labels: {len(result.unmatched)}")

    if result.unmatched:
        print("\nUnmatched labels by run:")
        for run_index, run_name in enumerate(RUN_NAMES):
            count = sum(1 for ref in result.unmatched if ref.source == run_index)
            print(f"  {run_name}: {count}")

    raw = _load_raw()
    view_comparison(
        raw,
        segmentations,
        overlaps,
        threshold=0.5,
        min_overlap=0.1,
        min_agree=3,
        names=RUN_NAMES,
    )
    napari.run()


if __name__ == "__main__":
    main()
