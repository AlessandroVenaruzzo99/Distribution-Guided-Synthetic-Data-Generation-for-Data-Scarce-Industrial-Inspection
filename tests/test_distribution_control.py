"""Instance-level distribution control (bin-based allocation) and the RFS baseline."""
import math

import numpy as np
import pandas as pd

from utils_augm import (
    allocate_copies_to_target,
    bin_stats_for_class,
    compute_rfs_repeat_factors_targeted,
    distribute_uniform_from_df,
    get_bin_image_allocation_df,
    make_bins_for_class,
)
from utils_bal import compute_image_copy_allocation


def test_make_bins_cover_all_counts():
    counts = [1, 3, 4, 8, 12, 20, np.nan]
    bins = make_bins_for_class(counts, R=4, first_upper=5)
    assert bins[0] == (0, 5)
    assert bins[-1][1] == 20
    # contiguous, non-overlapping integer bins
    for (lo_a, hi_a), (lo_b, _) in zip(bins, bins[1:]):
        assert lo_b == hi_a + 1
    valid = [c for c in counts if not math.isnan(c)]
    for c in valid:
        assert sum(lo <= c <= hi for lo, hi in bins) == 1


def test_single_bin_when_counts_are_small():
    assert make_bins_for_class([1, 2, 3], R=4, first_upper=5) == [(0, 3)]
    assert make_bins_for_class([np.nan, np.nan], R=4, first_upper=5) == []


def test_bin_stats_ignore_images_of_other_classes():
    names = ["a", "b", "c", "d"]
    counts = [2, np.nan, 10, 4]
    stats = bin_stats_for_class(counts, [(0, 5), (6, 10)], names)
    assert stats[0]["image_names"] == ["a", "d"]
    assert stats[0]["total_objects"] == 6
    assert stats[0]["avg_objects_per_image"] == 3
    assert stats[1]["image_names"] == ["c"]


def test_allocation_reaches_target_with_bounded_overshoot():
    names = [f"img{i}" for i in range(6)]
    counts = [1, 2, 2, 7, 9, 15]
    bins = make_bins_for_class(counts, R=3, first_upper=3)
    stats = bin_stats_for_class(counts, bins, names)
    target = 200
    allocated, current, added, overshoot = allocate_copies_to_target(stats, target)

    assert current == sum(counts)
    assert current + added >= target
    # overshoot is bounded by the largest average density of a bin
    assert 0 <= overshoot <= max(b["avg_objects_per_image"] for b in allocated)

    # allocation is proportional to each bin's share of objects (before the remainder step)
    delta = target - current
    for b in allocated:
        w = b["total_objects"] / current
        ideal = delta * w / b["avg_objects_per_image"]
        assert math.floor(ideal) <= b["copies"] <= math.floor(ideal) + 1


def test_no_copies_for_class_already_at_target():
    stats = bin_stats_for_class([50, 60], [(0, 100)], ["x", "y"])
    allocated, current, added, overshoot = allocate_copies_to_target(stats, 100)
    assert current == 110 and added == 0
    assert all(b["copies"] == 0 for b in allocated)


def test_uniform_distribution_within_bin():
    df = pd.DataFrame([{"class_id": "A", "range": "0-5", "copies_to_add": 7, "image_names": ["a", "b", "c"]}])
    alloc = distribute_uniform_from_df(df, random_seed=0)
    assert sum(alloc.values()) == 7
    assert max(alloc.values()) - min(alloc.values()) <= 1


def test_compute_image_copy_allocation_end_to_end():
    # Two classes, one image per class row (no inter-class co-occurrence)
    df = pd.DataFrame(
        {
            "A": [30, 25, 40, np.nan, np.nan],
            "B": [np.nan, np.nan, np.nan, 2, 5],
        },
        index=["a1", "a2", "a3", "b1", "b2"],
    )
    alloc, by_bin = compute_image_copy_allocation(df, R=2, first_upper=3, target_total_objects=95, verbose=False)
    a_copies = alloc[alloc.class_name == "A"].num_copies.sum() if len(alloc) else 0
    b_rows = alloc[alloc.class_name == "B"]
    assert a_copies == 0            # majority class already at target
    assert set(b_rows.image_name) <= {"b1", "b2"}
    assert b_rows.num_copies.sum() > 0
    assert set(by_bin) == {"A", "B"}


def test_rfs_repeat_factors():
    df = pd.DataFrame(
        {"A": [10, 10, np.nan], "B": [np.nan, np.nan, 5]},
        index=["a1", "a2", "b1"],
    )
    factors, info = compute_rfs_repeat_factors_targeted(df, target_total_objects=20)
    assert info["total_objects"] == {"A": 20, "B": 5}
    assert info["scale_factors"]["A"] == 1.0
    assert info["scale_factors"]["B"] == 4.0
    assert factors == {"a1": 1, "a2": 1, "b1": 4}
