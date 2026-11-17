########################################
from utils_augm import *


# -----------------------------------------------
# Parse the --fold argument from the command line
# -----------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, required=True, help="Fold number for cross-validation")
args = parser.parse_args()
fold = args.fold  # This is the fold number passed from your main loop
# -----------------------------------------------

# Base directory: two levels up from the current file
base_dir = Path(__file__).resolve().parent.parent

# Define the number of augmentation foldsaugm_loop_folds
augm_loop_runs = 1  # Number of times to fold the augmentation loop
rand_backg_select = False  # Whether to use random background selection
ARISE_AUGM = True  # Whether to use ARISE augmentation
OVERLAP_THRESHOLD = 0.5  # Overlap threshold for pasting objects
PRESERVS_HOLES = True  # Whether to preserve holes in the pasted objects

# -----------------------------------------------
# Object Augmentation Options
# -----------------------------------------------
OBJECT_AUGM_OPTIONS = {
    'enabled': False,
    'gaussian_blur': False,
    'gaussian_blur_kernel': 3,
    'scale_jitter': False,
    'scale_range': (0.8, 1.2),
    'rotation': False,            # ❌ DISABLED - incompatible for now with bbox size preservation
    'rotation_range': (-15, 15),  
    'brightness': False,
    'brightness_range': (-10, 10),
    'contrast': False,
    'contrast_range': (0.8, 1.2),
    'color_jitter': False,
    'hue_range': ( 0, 0),
    'saturation_range': (0.9, 1.1),
    'noise': False,
    'noise_std': 5,
    'flip_horizontal': False,     
    'flip_vertical': False,       
}

# -----------------------------------------------
# Background Augmentation Options
# -----------------------------------------------
BACKGROUND_AUGM_OPTIONS = {
    'flip': False,
    'brightness': False,
    'contrast': False,
    'blur': False,
    'noise': False,
    'rotation': False,
    'color_jitter': False
}

# -----------------------------------------------
stored_datset_train_dir = base_dir / "data" / "images" / "processed" / f"fold{fold}_CV"  / "stored_dataset" / "train"
erase_crop_augm_dir = base_dir / "data" / "images" / "processed" / f"fold{fold}_CV"  / "erase_crop_augm"
copy_paste_augm_dir = base_dir / "data" / "images" / "processed" / f"fold{fold}_CV"  / "copy_paste_augm"

# -----------------------------------------------
# Run the REGEN copy-paste pipeline
# -----------------------------------------------
# Store outputs in separate folders
# erase cropping labeled splitted dataset
erase_cropp_pipeline_for_all_classes(stored_datset_train_dir, erase_crop_augm_dir)

# copy paste augmentation with rnd background and not rnd background
run_pipeline_for_all_classes_cp(stored_datset_train_dir, copy_paste_augm_dir, erase_crop_augm_dir,
                              x_runs_times=augm_loop_runs, 
                              rnd_bckgrd=rand_backg_select, 
                              arise_augm=ARISE_AUGM,
                              overlap_threshold=OVERLAP_THRESHOLD, 
                              object_augm_options=OBJECT_AUGM_OPTIONS, 
                              preservs_holes=PRESERVS_HOLES,
                              background_augm_options=BACKGROUND_AUGM_OPTIONS) 

print("\n✅ REGEN Copy-Paste Augmentation Completed!")