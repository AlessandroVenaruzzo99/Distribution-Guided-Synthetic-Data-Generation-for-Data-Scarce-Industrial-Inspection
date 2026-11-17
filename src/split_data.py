############################################################################
# Train-Val dataset split for CV tasks using YOLOv5---> Seed = NONE
############################################################################
# To train Yolo5 on the new dataset is usefull to create a dataset split in 80% train set and 20% validation set,
# It is important to note that some images contain many hinges while other which contain only few pieces,
# the split should be based on the number of objects present in images rather than the number of images.

#NB: for now the split is done initially on the number of images but with regards to the number of objects
# the results obtained is a objects split around 75% train set and 25% validation set

from utils import * 

# -----------------------------------------------
# Parse arguments for k-fold cross-validation
# -----------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, required=True, help="Current fold number (0-based)")
parser.add_argument("--k_folds", type=int, required=True, help="Total number of folds")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--use_synthetic", type=str, default="False", help="If True, separate synthetic images (train-only)")
args = parser.parse_args()

fold = args.fold
k_folds = args.k_folds
seed = args.seed
use_synthetic = args.use_synthetic.lower() in ("true", "1", "yes")

# Base directory: two levels up from the current file
base_dir = Path(__file__).resolve().parent.parent

# Directory containing all class folders
raw_dir = base_dir / "data" / "images" / "raw"
raw_subfolders = [f for f in raw_dir.iterdir() if f.is_dir()]
if len(raw_subfolders) != 1:
    raise FileNotFoundError(f"Expected exactly one subfolder in {raw_dir}, found {len(raw_subfolders)}")
raw_classes_dir = raw_subfolders[0]

# Loop over each class
for class_dir in raw_classes_dir.iterdir():
    if class_dir.is_dir():
        class_name = class_dir.name
        
        print(f"\n🔍 Processing class: {class_name} for fold {fold + 1}/{k_folds}")
        
        dataset_dir = class_dir
        labels_output_dir = base_dir / "data" / "labels" / "stored" / class_name

        # Use fold instead of run in the output folder name
        train_dir = base_dir / "data" / "images" / "processed" / f"fold{fold}_CV" / "stored_dataset" / "train" / class_name
        val_dir   = base_dir / "data" / "images" / "processed" / f"fold{fold}_CV" / "stored_dataset" / "val" / class_name

        # Always perform the split for k-fold (ensure clean splits)
        if train_dir.exists():
            shutil.rmtree(train_dir)
        if val_dir.exists():
            shutil.rmtree(val_dir)

        # Perform k-fold split — with or without synthetic separation
        if use_synthetic:
            print(f"  🧬 Synthetic separation ENABLED — synthetic images → train only")
            split_yolo_dataset_kfold_with_synthetic(dataset_dir, labels_output_dir, train_dir, val_dir, k_folds, fold, seed)
        else:
            split_yolo_dataset_kfold(dataset_dir, labels_output_dir, train_dir, val_dir, k_folds, fold, seed)
        
        folder_path_train = train_dir / "labels"
        folder_path_val   = val_dir / "labels"

        print("  📊 Total rows in train labels:", count_total_rows(folder_path_train))
        print("  📊 Total rows in val labels:", count_total_rows(folder_path_val))