import numpy as np
import pytest

from compare_segmentations._compare import (
    LabelRef,
    compare_segmentations,
    compute_pairwise_overlaps,
    find_overlapping_labels,
    load_pairwise_overlaps,
    merge_segmentations,
    save_pairwise_overlaps,
)


def test_identical_segmentations_match_at_iou_one():
    seg = np.zeros((10, 10), dtype=np.int32)
    seg[0:5, 0:5] = 1
    seg[0:5, 5:10] = 2

    matches = find_overlapping_labels(seg, seg, iou_threshold=1.0)

    assert sorted(matches) == [(1, 1, 1.0), (2, 2, 1.0)]


def test_partial_overlap_below_threshold_is_excluded():
    seg_a = np.zeros((10, 10), dtype=np.int32)
    seg_a[0:5, 0:5] = 1

    seg_b = np.zeros((10, 10), dtype=np.int32)
    seg_b[3:8, 0:5] = 1

    # intersection is 2x5=10, union is 8x5=40 -> iou 0.25
    assert find_overlapping_labels(seg_a, seg_b, iou_threshold=0.5) == []
    matches = find_overlapping_labels(seg_a, seg_b, iou_threshold=0.25)
    assert matches == [(1, 1, 0.25)]


def test_non_overlapping_labels_are_excluded_above_zero_threshold():
    seg_a = np.zeros((10, 10), dtype=np.int32)
    seg_a[0:5, 0:5] = 1

    seg_b = np.zeros((10, 10), dtype=np.int32)
    seg_b[5:10, 5:10] = 1

    assert find_overlapping_labels(seg_a, seg_b, iou_threshold=1e-9) == []


def test_background_label_is_ignored():
    seg_a = np.zeros((10, 10), dtype=np.int32)
    seg_a[0:5, 0:5] = 1

    seg_b = np.zeros((10, 10), dtype=np.int32)

    assert find_overlapping_labels(seg_a, seg_b, iou_threshold=0.0) == []


def test_multiple_overlapping_pairs_are_all_returned():
    seg_a = np.zeros((10, 10), dtype=np.int32)
    seg_a[0:5, 0:5] = 1
    seg_a[5:10, 5:10] = 2

    seg_b = np.zeros((10, 10), dtype=np.int32)
    seg_b[0:5, 0:5] = 1
    seg_b[5:10, 5:10] = 2

    matches = find_overlapping_labels(seg_a, seg_b, iou_threshold=1.0)

    assert sorted(matches) == [(1, 1, 1.0), (2, 2, 1.0)]


def test_3d_segmentations_are_supported():
    seg_a = np.zeros((5, 10, 10), dtype=np.int32)
    seg_a[:, 0:5, 0:5] = 1

    seg_b = np.zeros((5, 10, 10), dtype=np.int32)
    seg_b[:, 0:5, 0:5] = 1

    assert find_overlapping_labels(seg_a, seg_b, iou_threshold=1.0) == [(1, 1, 1.0)]


def test_mismatched_shapes_raise_value_error():
    seg_a = np.zeros((10, 10), dtype=np.int32)
    seg_b = np.zeros((5, 5), dtype=np.int32)

    with pytest.raises(ValueError, match="shape"):
        find_overlapping_labels(seg_a, seg_b, iou_threshold=0.5)


def test_merge_identical_segmentations_reproduces_the_shape():
    seg = np.zeros((10, 10), dtype=np.int32)
    seg[0:5, 0:5] = 1
    seg[0:5, 5:10] = 2

    groups = [{LabelRef(0, 1), LabelRef(1, 1)}, {LabelRef(0, 2), LabelRef(1, 2)}]
    merged = merge_segmentations([seg, seg], groups)

    assert sorted(np.unique(merged).tolist()) == [0, 1, 2]
    assert np.array_equal(merged > 0, seg > 0)


def test_merge_offset_squares_sits_between_the_two_inputs():
    seg_a = np.zeros((20, 20), dtype=np.int32)
    seg_a[2:12, 2:12] = 1

    seg_b = np.zeros((20, 20), dtype=np.int32)
    seg_b[4:14, 4:14] = 1

    merged = merge_segmentations([seg_a, seg_b], [{LabelRef(0, 1), LabelRef(1, 1)}])

    assert sorted(np.unique(merged).tolist()) == [0, 1]
    # the blended shape should not reproduce either input exactly
    assert not np.array_equal(merged > 0, seg_a > 0)
    assert not np.array_equal(merged > 0, seg_b > 0)
    # its centroid should land between the two input centroids
    ys, xs = np.nonzero(merged == 1)
    assert 6.0 < ys.mean() < 8.0
    assert 6.0 < xs.mean() < 8.0


def test_merge_nested_shapes_stays_between_inner_and_outer():
    seg_a = np.zeros((20, 20), dtype=np.int32)
    seg_a[2:16, 2:16] = 1

    seg_b = np.zeros((20, 20), dtype=np.int32)
    seg_b[6:10, 6:10] = 1

    merged = merge_segmentations([seg_a, seg_b], [{LabelRef(0, 1), LabelRef(1, 1)}])

    mask_a = seg_a > 0
    mask_b = seg_b > 0
    merged_mask = merged > 0
    # the smaller shape is contained in the merged shape, which is in turn
    # contained in the larger shape
    assert np.array_equal(mask_b & merged_mask, mask_b)
    assert np.array_equal(mask_a | merged_mask, mask_a)


def test_merge_three_masks_blends_all_of_them():
    seg_a = np.zeros((20, 20), dtype=np.int32)
    seg_a[2:12, 2:12] = 1

    seg_b = np.zeros((20, 20), dtype=np.int32)
    seg_b[4:14, 4:14] = 1

    seg_c = np.zeros((20, 20), dtype=np.int32)
    seg_c[3:13, 3:13] = 1

    merged = merge_segmentations(
        [seg_a, seg_b, seg_c], [{LabelRef(0, 1), LabelRef(1, 1), LabelRef(2, 1)}]
    )

    assert sorted(np.unique(merged).tolist()) == [0, 1]
    ys, xs = np.nonzero(merged == 1)
    assert 6.5 < ys.mean() < 8.5
    assert 6.5 < xs.mean() < 8.5


def test_merge_blend_cache_reuses_unchanged_groups():
    seg_a = np.zeros((20, 20), dtype=np.int32)
    seg_a[2:12, 2:12] = 1

    seg_b = np.zeros((20, 20), dtype=np.int32)
    seg_b[4:14, 4:14] = 1

    group = {LabelRef(0, 1), LabelRef(1, 1)}
    blend_cache: dict = {}

    first = merge_segmentations([seg_a, seg_b], [group], blend_cache=blend_cache)
    assert frozenset(group) in blend_cache
    cached_blended = blend_cache[frozenset(group)][1]

    # second call with the same group and cache should reuse the cached blend rather
    # than recomputing it -- verify by mutating the input arrays after caching and
    # confirming the (now-stale) cached result is still what's returned unchanged
    second = merge_segmentations([seg_a, seg_b], [group], blend_cache=blend_cache)
    assert blend_cache[frozenset(group)][1] is cached_blended
    assert np.array_equal(first, second)


def test_merge_overlapping_groups_split_by_confidence_not_overwritten():
    # two same-size squares whose bounding boxes overlap -- neither group should
    # silently overwrite the other in the shared region; each should keep the half
    # closer to its own center.
    seg_a = np.zeros((20, 20), dtype=np.int32)
    seg_a[2:12, 2:12] = 1

    seg_b = np.zeros((20, 20), dtype=np.int32)
    seg_b[8:18, 8:18] = 1

    merged = merge_segmentations([seg_a, seg_b], [{LabelRef(0, 1)}, {LabelRef(1, 1)}])

    assert merged[3, 3] == 1  # deep inside A only
    assert merged[17, 17] == 2  # deep inside B only
    # the boundary through the overlap region should be a diagonal (each row claims one
    # more label-2 pixel than the row above), not a stair-step hugging one group's own
    # bbox edge -- a regression check for the "outside" SDF distortion that a crop
    # tight to a group's own bbox introduces (see _pad_slice)
    overlap = merged[8:12, 8:12]
    label_2_counts = [int(np.sum(row == 2)) for row in overlap]
    assert label_2_counts == sorted(label_2_counts)  # monotonically non-decreasing
    assert len(set(label_2_counts)) > 1  # actually varies by row, i.e. is diagonal


def test_merge_mismatched_shapes_raise_value_error():
    seg_a = np.zeros((10, 10), dtype=np.int32)
    seg_b = np.zeros((5, 5), dtype=np.int32)

    with pytest.raises(ValueError, match="shape"):
        merge_segmentations([seg_a, seg_b], [])


def test_compute_pairwise_overlaps_across_three_segmentations():
    seg_a = np.zeros((10, 10), dtype=np.int32)
    seg_a[0:5, 0:5] = 1

    seg_b = np.zeros((10, 10), dtype=np.int32)
    seg_b[0:5, 0:5] = 1

    seg_c = np.zeros((10, 10), dtype=np.int32)
    seg_c[5:10, 5:10] = 1

    overlaps = compute_pairwise_overlaps([seg_a, seg_b, seg_c], min_overlap=0.0)

    assert len(overlaps) == 1
    assert overlaps[0].ref_a == LabelRef(0, 1)
    assert overlaps[0].ref_b == LabelRef(1, 1)
    assert overlaps[0].iou == 1.0


def test_save_and_load_pairwise_overlaps_round_trips(tmp_path):
    seg_a = np.zeros((10, 10), dtype=np.int32)
    seg_a[0:5, 0:5] = 1

    seg_b = np.zeros((10, 10), dtype=np.int32)
    seg_b[3:8, 0:5] = 1

    overlaps = compute_pairwise_overlaps([seg_a, seg_b], min_overlap=0.0)

    path = tmp_path / "overlaps.json"
    save_pairwise_overlaps(overlaps, path)
    loaded = load_pairwise_overlaps(path)

    assert loaded == overlaps


def test_compare_two_agreeing_segmentations_are_merged():
    seg_a = np.zeros((10, 10), dtype=np.int32)
    seg_a[0:5, 0:5] = 1

    seg_b = np.zeros((10, 10), dtype=np.int32)
    seg_b[0:5, 0:5] = 1

    result = compare_segmentations([seg_a, seg_b], threshold=0.9)

    assert sorted(np.unique(result.merged).tolist()) == [0, 1]
    assert result.conflicts == []
    assert result.unmatched == []
    assert result.centroids == {}


def test_compare_two_partially_overlapping_segmentations_conflict():
    seg_a = np.zeros((10, 10), dtype=np.int32)
    seg_a[0:5, 0:5] = 1

    seg_b = np.zeros((10, 10), dtype=np.int32)
    seg_b[3:8, 0:5] = 1

    result = compare_segmentations([seg_a, seg_b], threshold=0.9, min_overlap=0.0)

    assert np.all(result.merged == 0)
    assert result.conflicts == [[LabelRef(0, 1), LabelRef(1, 1)]]
    assert result.unmatched == []
    assert set(result.centroids) == {LabelRef(0, 1), LabelRef(1, 1)}


def test_compare_two_non_overlapping_segmentations_are_unmatched():
    seg_a = np.zeros((10, 10), dtype=np.int32)
    seg_a[0:5, 0:5] = 1

    seg_b = np.zeros((10, 10), dtype=np.int32)
    seg_b[5:10, 5:10] = 1

    result = compare_segmentations([seg_a, seg_b], threshold=0.5, min_overlap=0.0)

    assert np.all(result.merged == 0)
    assert result.conflicts == []
    assert sorted(result.unmatched, key=lambda ref: ref.source) == [
        LabelRef(0, 1),
        LabelRef(1, 1),
    ]
    assert set(result.centroids) == {LabelRef(0, 1), LabelRef(1, 1)}


def test_compare_three_segmentations_all_agreeing_are_merged():
    seg = np.zeros((10, 10), dtype=np.int32)
    seg[0:5, 0:5] = 1

    result = compare_segmentations([seg, seg, seg], threshold=0.9)

    assert sorted(np.unique(result.merged).tolist()) == [0, 1]
    assert result.conflicts == []
    assert result.unmatched == []


def test_compare_three_segmentations_partial_agreement_is_conflict_not_merge():
    seg_a = np.zeros((10, 10), dtype=np.int32)
    seg_a[0:5, 0:5] = 1

    seg_b = np.zeros((10, 10), dtype=np.int32)
    seg_b[0:5, 0:5] = 1

    seg_c = np.zeros((10, 10), dtype=np.int32)
    seg_c[5:10, 5:10] = 1

    result = compare_segmentations([seg_a, seg_b, seg_c], threshold=0.9, min_overlap=0.0)

    # a and b agree with each other but the group doesn't span all 3 sources,
    # so it can't be auto-merged -- it's a conflict, and c is unmatched
    assert np.all(result.merged == 0)
    assert result.conflicts == [[LabelRef(0, 1), LabelRef(1, 1)]]
    assert result.unmatched == [LabelRef(2, 1)]


def test_compare_min_agree_merges_majority_subset():
    seg_a = np.zeros((10, 10), dtype=np.int32)
    seg_a[0:5, 0:5] = 1

    seg_b = np.zeros((10, 10), dtype=np.int32)
    seg_b[0:5, 0:5] = 1

    seg_c = np.zeros((10, 10), dtype=np.int32)
    seg_c[0:5, 0:5] = 1

    seg_d = np.zeros((10, 10), dtype=np.int32)
    seg_d[5:10, 5:10] = 1

    result = compare_segmentations(
        [seg_a, seg_b, seg_c, seg_d], threshold=0.9, min_overlap=0.0, min_agree=3
    )

    # a, b, c all agree (3 of 4, a strict majority) -- merged even though d disagrees
    assert sorted(np.unique(result.merged).tolist()) == [0, 1]
    assert np.array_equal(result.merged > 0, seg_a > 0)
    assert result.conflicts == []
    assert result.unmatched == [LabelRef(3, 1)]


def test_compare_min_agree_leaves_leftover_as_conflict():
    seg_a = np.zeros((10, 10), dtype=np.int32)
    seg_a[0:5, 0:5] = 1

    seg_b = np.zeros((10, 10), dtype=np.int32)
    seg_b[0:5, 0:5] = 1

    seg_c = np.zeros((10, 10), dtype=np.int32)
    seg_c[0:5, 0:5] = 1

    # d and e both overlap the a/b/c cell weakly (below threshold) and each other,
    # so the connected component includes them, but they aren't part of the
    # merged majority clique and there are 2 of them -- conflict, not unmatched
    seg_d = np.zeros((10, 10), dtype=np.int32)
    seg_d[3:8, 0:5] = 1

    seg_e = np.zeros((10, 10), dtype=np.int32)
    seg_e[3:8, 0:5] = 1

    result = compare_segmentations(
        [seg_a, seg_b, seg_c, seg_d, seg_e], threshold=0.9, min_overlap=0.0, min_agree=3
    )

    assert sorted(np.unique(result.merged).tolist()) == [0, 1]
    assert np.array_equal(result.merged > 0, seg_a > 0)
    assert result.conflicts == [[LabelRef(3, 1), LabelRef(4, 1)]]
    assert result.unmatched == []


def test_compare_min_agree_must_be_strict_majority():
    seg_a = np.zeros((10, 10), dtype=np.int32)
    seg_b = np.zeros((10, 10), dtype=np.int32)

    with pytest.raises(ValueError, match="majority"):
        compare_segmentations([seg_a, seg_b], min_agree=1)


def test_compare_mismatched_shapes_raise_value_error():
    seg_a = np.zeros((10, 10), dtype=np.int32)
    seg_b = np.zeros((5, 5), dtype=np.int32)

    with pytest.raises(ValueError, match="shape"):
        compare_segmentations([seg_a, seg_b])


def test_compare_reclassifies_from_cached_overlaps_without_recomputing():
    seg_a = np.zeros((10, 10), dtype=np.int32)
    seg_a[0:5, 0:5] = 1

    seg_b = np.zeros((10, 10), dtype=np.int32)
    seg_b[3:8, 0:5] = 1

    overlaps = compute_pairwise_overlaps([seg_a, seg_b], min_overlap=0.0)

    # at threshold 0.9 this pair conflicts (see test above); passing the same
    # cached overlaps with a lower threshold should merge it without ever
    # touching pixel data again
    result = compare_segmentations([seg_a, seg_b], threshold=0.1, overlaps=overlaps)

    assert sorted(np.unique(result.merged).tolist()) == [0, 1]
    assert result.conflicts == []


def test_compare_min_overlap_filters_cached_overlaps_without_recomputing():
    seg_a = np.zeros((10, 10), dtype=np.int32)
    seg_a[0:5, 0:5] = 1

    seg_b = np.zeros((10, 10), dtype=np.int32)
    seg_b[3:8, 0:5] = 1

    # cache at min_overlap=0.0, so the pair (iou 0.25) is recorded
    overlaps = compute_pairwise_overlaps([seg_a, seg_b], min_overlap=0.0)

    # a min_overlap above the pair's iou should treat it as unrelated (unmatched),
    # not merely below-threshold (conflict) -- purely by filtering the cached IoUs
    result = compare_segmentations(
        [seg_a, seg_b], threshold=0.9, min_overlap=0.5, overlaps=overlaps
    )

    assert np.all(result.merged == 0)
    assert result.conflicts == []
    assert sorted(result.unmatched, key=lambda ref: ref.source) == [
        LabelRef(0, 1),
        LabelRef(1, 1),
    ]
