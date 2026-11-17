"""Optional rule-based auto-labeling on a tiny synthetic image."""
import cv2
import numpy as np

from utils_labeling import generate_yolo_labels, segment_objects


def test_contours_to_yolo_labels():
    img = np.full((100, 200), 255, np.uint8)
    cv2.rectangle(img, (20, 20), (59, 49), 0, -1)     # 40x30 dark object
    cv2.rectangle(img, (150, 70), (152, 72), 0, -1)   # tiny blob, filtered by min_area
    contours, _ = cv2.findContours(segment_objects(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    params = {"min_area": 500, "max_area": 5000, "min_aspect_ratio": 0.1, "max_aspect_ratio": 5,
              "w_eps": 0.9, "h_eps": 0.9}
    labels = generate_yolo_labels(contours, 200, 100, 3, params)
    assert len(labels) == 1
    cls, cx, cy, w, h = labels[0].split()
    assert cls == "3"
    assert abs(float(cx) - 0.2) < 0.03 and abs(float(cy) - 0.35) < 0.05
