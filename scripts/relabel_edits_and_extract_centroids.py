"""Relabel manually-added objects in an edited segmentation and extract centroids.

Workflow:
1. `edited` = the manually-edited segmentation (existing merged labels untouched,
   new objects painted into background).
2. `diff = (edited != 0) & (merged == 0)` -- voxels present in `edited` but not in
   `merged`, i.e. exactly the manual edits.
3. Connected components of `diff` are relabeled starting at `edited.max() + 1`, so
   every new object gets an id that can't collide with an existing merged label
   (see the label-collision bug fixed in `compare_segmentations._view._copy_label`).
4. Centroids (regionprops) are extracted from the relabeled array and written to a
   CSV alongside the tiff, with a `confidence` column: "high" for objects that came
   from `merged` unchanged, "low" for the newly-added ones.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage
from skimage.measure import regionprops

SEGMENTATION_DIR = Path(
    "/Volumes/shroff/Pharynx_Project/Data/SLS267xOH15257/080626_Dataset_2/Segmentation"
)
EDITED_PATH = SEGMENTATION_DIR / "raw-merged-edited.tiff"
MERGED_PATH = SEGMENTATION_DIR / "raw-merged.tiff"
RELABELED_PATH = SEGMENTATION_DIR / "raw-merged-edited-relabeled.tiff"
CENTROIDS_CSV_PATH = SEGMENTATION_DIR / "raw-merged-edited_centroids.csv"


def main() -> None:
    edited = tifffile.imread(EDITED_PATH)
    merged = tifffile.imread(MERGED_PATH)

    diff_mask = (edited != 0) & (merged == 0)
    diff_components, num_new = ndimage.label(diff_mask)
    start_label = int(edited.max()) + 1
    print(f"found {num_new} new connected component(s) in the manual edits")

    relabeled = edited.copy()
    relabeled[diff_mask] = diff_components[diff_mask] + (start_label - 1)

    tifffile.imwrite(RELABELED_PATH, relabeled)
    print(f"wrote relabeled array to {RELABELED_PATH}")

    rows = []
    for prop in regionprops(relabeled):
        confidence = "low" if prop.label >= start_label else "high"
        z, y, x = prop.centroid
        rows.append((z, y, x, confidence))

    with open(CENTROIDS_CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["z", "y", "x", "confidence"])
        writer.writerows(rows)

    num_high = sum(1 for r in rows if r[3] == "high")
    num_low = sum(1 for r in rows if r[3] == "low")
    print(f"wrote {len(rows)} centroids ({num_high} high, {num_low} low) to {CENTROIDS_CSV_PATH}")


if __name__ == "__main__":
    main()
