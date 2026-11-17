# Distribution-Guided Synthetic Data Generation for Data-Scarce Industrial Inspection

This repository contains the official implementation of the framework proposed in the paper: 
**"Distribution-Guided Synthetic Data Generation for Data-Scarce Industrial Inspection"** (accepted for publication in *The International Journal of Advanced Manufacturing Technology*, Springer).

**Authors:** Alessandro Venaruzzo, Cigdem Beyan, Matteo Cristani — Department of Computer Science, University of Verona, Italy

**DOI:** [10.1007/s00170-026-19352-3](https://doi.org/10.1007/s00170-026-19352-3)

## 📌 Overview
Our framework addresses the challenge of data scarcity in industrial quality inspection. Unlike standard augmentation techniques, this pipeline explicitly models:
1. **Instance-level distribution control**: Regulates per-class frequencies.
2. **Dataset-driven spatial compositing**: Uses empirical overlap and placement statistics from original training data to generate realistic synthetic scenes.
3. **Automated background reconstruction**: Ensures compatibility with structured industrial acquisition conditions.

## 🗂 Repository structure

```
├── src/                              # Pipeline scripts (run from inside this folder)
│   ├── main_k-fold.py                # Entry point: 5-fold CV -> split, generation, dataset build, training, evaluation
│   ├── split_data.py                 # Object-aware k-fold split (synthetic/diffusion images kept in train only)
│   │
│   ├── data_balance_augm.py          # Proposed method: instance-level distribution control + constrained compositing
│   │                                 #   (also runs the RFS baseline and the sampling-only configuration)
│   ├── utils_bal.py                  # Bin-based allocation of synthetic images per class
│   ├── utils_augm.py                 # Background reconstruction, object extraction, IoU-constrained pasting,
│   │                                 #   adaptive overlap thresholds, foreground/background augmentation, RFS
│   ├── utils_augm_synch.py           # Same as utils_augm.py with optional synchronized fg/bg augmentation
│   │
│   ├── copy_paste_augm_REGEN.py      # Ablation: compositing with a fixed overlap threshold
│   ├── copy_paste_augm_REGEN_AOT.py  # Ablation: compositing with dataset-derived (adaptive) overlap thresholds
│   ├── copy_paste_augm_Synch_AllNoDB.py  # Ablation: all compositing modules, without distribution control
│   ├── copy_paste_augm_Vanilla.py    # Baseline: vanilla copy-paste
│   ├── utils_cp_VANILLA.py           # Vanilla copy-paste implementation
│   │
│   ├── collecting_images.py          # Builds the YOLO dataset of a fold (no synthetic generation)
│   ├── augm_collecting_images.py     # Builds the YOLO dataset of a fold (real + synthetic images)
│   ├── yaml_generator.py             # Writes configs/data.yaml for the current fold
│   ├── update_json.py                # Points the Ultralytics settings to the current fold
│   ├── edit_default_yaml.py          # Ultralytics training defaults (no built-in augmentation)
│   ├── edit_default_yaml_cp_paste.py # Ultralytics training defaults used with synthetic data
│   ├── train_pred_val.py             # Training from scratch, inference and validation (YOLOv5s / YOLO11s)
│   ├── evaluate_cv_performance.py    # Mean/std of the metrics across folds
│   ├── class_avg_metrics.py          # Per-class mean AP helper
│   ├── Plot_comparison_Seaborn.py    # Training curves (mean ± SD across folds) for several baselines
│   ├── labeling_raw_data.py          # Optional: rule-based auto-labeling (YOLO boxes from contours)
│   ├── utils_labeling.py             # Helpers for the optional auto-labeling step
│   └── utils.py                      # Shared helpers (seeding, dataset building, k-fold split, metrics)
├── tests/                            # Lightweight CPU-only unit tests (no training)
├── configs/                          # Generated data.yaml + auto-labeling parameter template
├── data/                             # Dataset location (git-ignored, see below)
├── coco_dataset/, results/           # Generated per-fold datasets and training results (git-ignored)
└── requirements.txt
```

## 🛠 Installation

```bash
git clone https://github.com/AlessandroVenaruzzo99/Distribution-Guided-Synthetic-Data-Generation-for-Data-Scarce-Industrial-Inspection.git
cd Distribution-Guided-Synthetic-Data-Generation-for-Data-Scarce-Industrial-Inspection
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> `edit_default_yaml*.py` overwrite the `cfg/default.yaml` of the **installed** Ultralytics package (a backup
> `default_backup.yaml` is created next to it), so always run the pipeline inside a dedicated virtual environment.

## 📁 Dataset layout

Images must be annotated in YOLO bounding-box format, with one folder per class:

```
data/
├── images/
│   ├── raw/<dataset_name>/<class_name>/Image (N).jpg   # images used for the k-fold split
│   └── stored/<class_name>/Image (N).jpg               # same images, used for the overlap statistics
└── labels/
    └── stored/<class_name>/Image (N).txt               # YOLO labels: class cx cy w h (normalized)
```

If `data/labels/stored` does not exist, `main_k-fold.py` runs the optional rule-based auto-labeling step
(`labeling_raw_data.py`): it copies the class folders of `data/images/raw/<dataset_name>/` into `data/images/stored/` and
derives YOLO boxes from object contours using the per-class parameters in
`configs/Labeling_yaml_config/<class_name>_labeling_parameters.yaml` (see the example template).
The annotations used for the paper were made manually; the auto-labeling step is only a convenience for new data.

The open-access dataset used for the additional experiments (Manufacturing-Backlight) is available on
[Roboflow Universe](https://universe.roboflow.com/alessandro-venaruzzo-4oaiw/manufacturing-backlight-dataset).

## 🚀 Usage

All experiment switches are at the top of `src/main_k-fold.py` (`copy_paste_augm`, `data_balancing`,
`use_rfs_custom_method`, `use_oversampling`, `vanilla`, `adaptive_overlap_threshold`, `all_no_db_sych`, `detector`, ...).
The defaults reproduce the full proposed configuration (random background selection, background and foreground
augmentation, adaptive overlap thresholds, instance-level distribution control); set `use_rfs_custom_method = True`
for the RFS baseline or `data_balancing = False` for the vanilla copy-paste / ablation runs.
The generation hyper-parameters (target instances per class, number of bins, foreground/background augmentation
options, overlap percentile) are defined at the top of the corresponding generation script.

```bash
cd src
python3 main_k-fold.py
```

Training curves for several runs (one sub-folder per method, each with its own `results/`):

```bash
python3 Plot_comparison_Seaborn.py --baselines_dir /path/to/runs
```

## 🧪 Tests

The unit tests check the distribution-control allocation, the RFS repeat factors, the overlap statistics,
the compositing helpers and the k-fold split on tiny synthetic inputs. They run on CPU in a few seconds and
do not need `torch` or `ultralytics`.

```bash
pip install -r requirements-test.txt
OMP_NUM_THREADS=1 python -m pytest tests -q
```

## 📊 Data Availability & Reproducibility

The industrial dataset used in the paper is proprietary and cannot be publicly
shared due to industrial collaboration constraints BUT the similar alternative Open is provided at the following link: https://universe.roboflow.com/alessandro-venaruzzo-4oaiw/manufacturing-backlight-dataset

## 📖 Citation

If you use this code, please cite the paper (see also [CITATION.cff](CITATION.cff)):

```bibtex
@article{venaruzzo_distribution_guided,
  title   = {Distribution-Guided Synthetic Data Generation for Data-Scarce Industrial Inspection},
  author  = {Venaruzzo, Alessandro and Beyan, Cigdem and Cristani, Matteo},
  journal = {The International Journal of Advanced Manufacturing Technology},
  publisher = {Springer},
  doi     = {10.1007/s00170-026-19352-3},
  note    = {In press}
}
```

## ⚖️ License

This project is licensed under the MIT License (see [LICENSE](LICENSE)).
