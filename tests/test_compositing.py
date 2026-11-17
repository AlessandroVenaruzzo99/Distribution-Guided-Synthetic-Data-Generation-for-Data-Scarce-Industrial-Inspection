"""Overlap statistics, background augmentation and masked object pasting on tiny synthetic arrays."""
import random

import numpy as np

from utils_augm import (
    augment_background,
    compute_class_overlap_thresholds,
    compute_overlap_between_bboxes,
    load_overlap_thresholds,
    paste_object_masked,
    rects_overlap_ratio,
    save_overlap_thresholds,
)


def test_rects_overlap_ratio():
    assert rects_overlap_ratio((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert rects_overlap_ratio((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert abs(rects_overlap_ratio((0, 0, 10, 10), (5, 0, 15, 10)) - 50 / 150) < 1e-9


def test_overlap_between_yolo_boxes():
    boxes = [[0, 0.25, 0.25, 0.2, 0.2], [0, 0.30, 0.25, 0.2, 0.2], [0, 0.8, 0.8, 0.1, 0.1]]
    max_iou, pairs = compute_overlap_between_bboxes(boxes)
    assert len(pairs) == 3
    assert abs(max_iou - (0.15 * 0.2) / (2 * 0.04 - 0.15 * 0.2)) < 1e-9
    assert compute_overlap_between_bboxes(boxes[:1]) == (0.0, [])


def test_class_thresholds_roundtrip(tmp_path):
    images, labels = tmp_path / "images", tmp_path / "labels"
    for cls, rows in {
        "classA": ["0 0.25 0.25 0.2 0.2", "0 0.30 0.25 0.2 0.2"],
        "classB": ["1 0.2 0.2 0.1 0.1", "1 0.8 0.8 0.1 0.1"],
    }.items():
        (images / cls).mkdir(parents=True)
        (labels / cls).mkdir(parents=True)
        (images / cls / "Image (1).jpg").write_bytes(b"")  # content is never read
        (labels / cls / "Image (1).txt").write_text("\n".join(rows) + "\n")

    thr, stats = compute_class_overlap_thresholds(images, labels, default_threshold=0.3, percentile=95)
    assert stats["classA"]["status"] == "computed"
    assert 0 < thr["classA"] <= 0.95
    assert thr["classB"] == 0.3  # no overlapping pair -> fallback value

    out = tmp_path / "thr.json"
    save_overlap_thresholds(thr, stats, out)
    loaded, _ = load_overlap_thresholds(out)
    assert loaded == thr


def test_paste_object_masked_only_writes_masked_pixels():
    target = np.zeros((40, 40, 3), np.uint8)
    obj = np.full((10, 10, 3), 200, np.uint8)
    mask = np.zeros((10, 10), np.uint8)
    mask[2:8, 2:8] = 255
    out = paste_object_masked(target, obj, mask, 0.5, 0.5)
    assert (target > 0).any(axis=2).sum() == 36
    assert np.allclose(out["bbox"], [0.5, 0.5, 0.25, 0.25])


def test_background_augmentation_keeps_shape_and_polygon_inside():
    random.seed(0)
    np.random.seed(0)
    img = np.full((32, 48, 3), 128, np.uint8)
    poly = np.array([[2, 2], [45, 2], [45, 29], [2, 29]], np.float32)
    opts = {"flip": True, "brightness": True, "contrast": True, "blur": True, "noise": True,
            "rotation": False, "color_jitter": True}
    for _ in range(5):
        aug, apoly = augment_background(img, poly, opts)
        assert aug.shape == img.shape and aug.dtype == np.uint8
        assert apoly.shape == poly.shape
        assert (apoly[:, 0] >= 0).all() and (apoly[:, 0] < 48).all()
        assert (apoly[:, 1] >= 0).all() and (apoly[:, 1] < 32).all()
