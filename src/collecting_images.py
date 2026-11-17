######################################## 
from utils import *



# -----------------------------------------------
# Parse the --fold argument from the command line
# -----------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, required=True, help="Fold number for cross-validation")
args = parser.parse_args()
fold = args.fold  # This is the fold number passed from your main loop
# -----------------------------------------------



###########################################################################################
# Create the coco_dataset directory if it doesn't exist
base_dir = Path(__file__).resolve().parent.parent
coco_dataset_dir = base_dir / "coco_dataset" / f"fold{fold}_CV" 
# Always rebuild fold directory to avoid stale/partial data from previous runs
if coco_dataset_dir.exists():
    shutil.rmtree(coco_dataset_dir)

# create the coco_dataset directory
coco_dataset_dir.mkdir(parents=True, exist_ok=True)
#define path to get images and labels and allow merging operation
proc_dir_IN = base_dir / "data" / "images" / "processed" / f"fold{fold}_CV"
train_stored_dataset_dir_IN = proc_dir_IN / "stored_dataset" / "train"
val_stored_dataset_dir_IN = proc_dir_IN / "stored_dataset" / "val"
output_dir_IN = base_dir / "coco_dataset" / f"fold{fold}_CV"

# Build the COCO dataset structure
build_coco_dataset(train_stored_dataset_dir_IN, val_stored_dataset_dir_IN, output_dir_IN)
print(" ✅ The coco_dataset directory has been rebuilt successfully. ✅")


###########################################################################################
# Count dataset objects for train and validation set.
folder_path_train = base_dir /"coco_dataset" / f"fold{fold}_CV" / "train" / "labels"
folder_path_val = base_dir /"coco_dataset" / f"fold{fold}_CV" / "val" / "labels"

print("	📊 Total rows in train labels for coco_dataset:", count_total_rows(folder_path_train))
print("	📊 Total rows in val labels  for coco_dataset:", count_total_rows(folder_path_val))
###########################################################################################






