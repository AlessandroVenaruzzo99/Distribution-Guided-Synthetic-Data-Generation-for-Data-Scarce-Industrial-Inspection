########################################
from utils_cp_VANILLA import *

# -----------------------------------------------
# Parse command line arguments
# -----------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, required=True, help="Fold number for cross-validation")
args = parser.parse_args()
fold = args.fold

# -----------------------------------------------
# Base directory: two levels up from the current file
base_dir = Path(__file__).resolve().parent.parent

# Define the number of augmentation foldsaugm_loop_folds
n_copies = 1
overlap_threshold = 0.6
num_augmentations = 1
preserve_holes=True
# -----------------------------------------------
stored_datset_train_dir = base_dir / "data" / "images" / "processed" / f"fold{fold}_CV" / "stored_dataset" / "train"
vanilla_augm_dir = base_dir / "data" / "images" / "processed" / f"fold{fold}_CV" / "copy_paste_augm"

# -----------------------------------------------
# Run the vanilla copy-paste pipeline
# -----------------------------------------------
run_vanilla_pipeline_for_all_classes(
    stored_datset_train_dir=stored_datset_train_dir,
    output_dir=vanilla_augm_dir,
    n_copies=n_copies,
    overlap_threshold=overlap_threshold,
    num_augmentations=num_augmentations,
    preserve_holes=preserve_holes
)


print("\n✅ Vanilla Copy-Paste Augmentation Completed!")