"""Object-aware k-fold split on a tiny fake dataset + syntax check of every pipeline script."""
import py_compile
from pathlib import Path

from utils import is_synthetic_image, split_yolo_dataset_kfold

SRC_DIR = Path(__file__).resolve().parent.parent / "src"


def _make_dataset(root):
    images, labels = root / "images", root / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for i in range(1, 11):
        (images / f"Image ({i}).jpg").write_bytes(b"")
        n = 0 if i == 10 else i  # one background-only image
        (labels / f"Image ({i}).txt").write_text("".join("0 0.5 0.5 0.1 0.1\n" for _ in range(n)))
    return images, labels


def test_kfold_split_partitions_images(tmp_path):
    images, labels = _make_dataset(tmp_path / "src")
    k = 5
    seen_val = []
    for fold in range(k):
        train, val = tmp_path / f"f{fold}" / "train", tmp_path / f"f{fold}" / "val"
        split_yolo_dataset_kfold(images, labels, train, val, k, fold, seed=42)
        tr = {p.name for p in (train / "images").iterdir()}
        va = {p.name for p in (val / "images").iterdir()}
        assert not tr & va
        assert len(tr | va) == 10
        assert "Image (10).jpg" in tr  # zero-object image always in train
        seen_val.extend(va)
    # every annotated image is validated exactly once across the folds
    assert sorted(seen_val) == sorted(f"Image ({i}).jpg" for i in range(1, 10))


def test_synthetic_image_detection():
    assert not is_synthetic_image("Image (12).jpg")
    assert is_synthetic_image("generated_0001.png")


def test_all_scripts_compile():
    for script in SRC_DIR.glob("*.py"):
        py_compile.compile(str(script), doraise=True)
