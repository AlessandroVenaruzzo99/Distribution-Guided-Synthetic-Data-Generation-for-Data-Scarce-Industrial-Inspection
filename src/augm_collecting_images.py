######################################## 
from utils import *


# -----------------------------------------------
# Parse the --fold and --data_balancing argument from the command line
# -----------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, required=True, help="fold number for cross-validation")
parser.add_argument("--data_balancing", type=str, required=True, help="Enable data balancing before copy-paste augmentation (True/False)")
parser.add_argument("--vanilla", type=str, required=False, help="Whether vanilla copy-paste was used (True/False)")
args = parser.parse_args()
fold = args.fold
data_balancing = args.data_balancing.lower() == 'true'
vanilla = args.vanilla.lower() == 'true'

# -----------------------------------------------   

###########################################################################################
# Create the coco_dataset directory if it doesn't exist
base_dir = Path(__file__).resolve().parent.parent
# check if the coco_dataset directory already exists
coco_dataset_dir = base_dir / "coco_dataset" / f"fold{fold}_CV" 
if not coco_dataset_dir.exists():
    # create the coco_dataset directory
    coco_dataset_dir.mkdir(parents=True, exist_ok=True)
    #define path to get images and labels and allow merging operation
    proc_dir_IN = base_dir / "data" / "images" / "processed" / f"fold{fold}_CV" 
    cp_pst_augm_dir_IN = proc_dir_IN / "copy_paste_augm"
    stored_dtset_dir_IN = proc_dir_IN / "stored_dataset"
    output_dir_IN = base_dir / "coco_dataset" / f"fold{fold}_CV" 
    
    # check if the data_balancing is enabled
    if data_balancing == True:
        print(" ⚠️ Data balancing enabled, starting building coco_dataset with data balancing ⚠️ ...")
        # Build the COCO dataset structure with data balancing
        build_coco_dataset_balance(cp_pst_augm_dir_IN, stored_dtset_dir_IN, output_dir_IN)
    else:
        # check if vanilla copy-paste was used
        if vanilla == True:
            print(" ⚠️ Vanilla copy-paste enabled, starting building coco_dataset with vanilla augmentation ⚠️ ...") 
            # Build the COCO dataset structure with vanilla copy-paste
            build_coco_dataset_vanilla(cp_pst_augm_dir_IN, stored_dtset_dir_IN, output_dir_IN)
        else:
            print(" ⚠️ Non-vanilla copy-paste enabled, starting building coco_dataset with advanced augmentation ⚠️ ...") 
            # Build the COCO dataset structure with non-vanilla copy-paste
            build_coco_dataset_REGEN(cp_pst_augm_dir_IN, stored_dtset_dir_IN, output_dir_IN)
else:
    print(" ⚠️ The coco_dataset directory already exists. Skipping operation ... ⚠️")


###########################################################################################
# Count dataset objects for train and validation set.
folder_path_train = base_dir /"coco_dataset" / f"fold{fold}_CV" / "train" / "labels"
folder_path_val = base_dir /"coco_dataset" / f"fold{fold}_CV" / "val" / "labels"

print("	📊 Total rows in train labels for coco_dataset:", count_total_rows(folder_path_train))
print("	📊 Total rows in val labels  for coco_dataset:", count_total_rows(folder_path_val))
###########################################################################################






