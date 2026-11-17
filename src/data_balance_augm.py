########################################
from utils_augm import *
from utils_bal import * 

##############################################################################################
######################################################################################################################################################################################
# Define a function which integrate the data balancing step into the augmentation pipeline
######################################################################################################################################################################################
# -----------------------------------------------
# Parse the --fold argument from the command line
# -----------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, required=True, help="fold number for cross-validation")
parser.add_argument("--rfs", type=str, default="False", help="Use custom RFS allocation strategy (True/False)")
parser.add_argument("--oversampling", type=str, default="False", help="Use pure oversampling (no synthetic generation) (True/False)")
args = parser.parse_args()
fold = args.fold  # This is the fold number passed from your main loop
use_rfs = args.rfs.lower() == "true"
use_oversampling = args.oversampling.lower() == "true"
# -----------------------------------------------

# Base directory: two levels up from the current file
base_dir = Path(__file__).resolve().parent.parent

# Define the number of augmentation foldsaugm_loop_folds
target_total_objects = 3000  # Number of  target objects per class
R = 4  # Number of bins
first_upper = 7  # Upper limit of first bin
rand_backg_select = True  # Random Background Selection (RB) - Full configuration
ARISE_AUGM = True  # Whether to use ARISE augmentation
PRESERVS_HOLES = True  # Whether to preserve holes in the pasted objects
synch_augmentation = False  # Enable synchronized augmentation for coherent object and background transformations

# -----------------------------------------------
# Adaptive Overlap Threshold Settings
# -----------------------------------------------
USE_ADAPTIVE_OVERLAP = True  # Set to False to use fixed threshold for all classes
FIXED_OVERLAP_THRESHOLD = 0.5  # Used only if USE_ADAPTIVE_OVERLAP is False
OVERLAP_PERCENTILE = 95  # Percentile of max overlaps to use (higher = more permissive)
SAVE_THRESHOLD_STATS = True  # Whether to save computed thresholds to JSON

# Paths for stored data (used to compute adaptive thresholds)
stored_images_dir = base_dir / "data" / "images" / "stored"
stored_labels_dir = base_dir / "data" / "labels" / "stored"
threshold_output_path = base_dir / "data" / "overlap_thresholds.json"

# -----------------------------------------------
# Compute or Load Adaptive Overlap Thresholds
# -----------------------------------------------
if USE_ADAPTIVE_OVERLAP:
    # Check if thresholds were already computed
    if threshold_output_path.exists():
        print(f"\n📂 Loading pre-computed overlap thresholds from: {threshold_output_path}")
        OVERLAP_THRESHOLD, threshold_stats = load_overlap_thresholds(threshold_output_path)
        if OVERLAP_THRESHOLD:
            print("✅ Loaded class-specific thresholds:")
            for cls, thresh in OVERLAP_THRESHOLD.items():
                print(f"   {cls}: {thresh:.4f}")
        else:
            print("⚠️  Failed to load thresholds, computing fresh...")
            OVERLAP_THRESHOLD, threshold_stats = compute_class_overlap_thresholds(
                stored_images_dir, 
                stored_labels_dir,
                default_threshold=FIXED_OVERLAP_THRESHOLD,
                percentile=OVERLAP_PERCENTILE
            )
            if SAVE_THRESHOLD_STATS:
                save_overlap_thresholds(OVERLAP_THRESHOLD, threshold_stats, threshold_output_path)
    else:
        # Compute adaptive thresholds from stored data
        print("\n🔄 Computing adaptive overlap thresholds from stored data...")
        OVERLAP_THRESHOLD, threshold_stats = compute_class_overlap_thresholds(
            stored_images_dir, 
            stored_labels_dir,
            default_threshold=FIXED_OVERLAP_THRESHOLD,
            percentile=OVERLAP_PERCENTILE
        )
        if SAVE_THRESHOLD_STATS:
            save_overlap_thresholds(OVERLAP_THRESHOLD, threshold_stats, threshold_output_path)
else:
    # Use fixed threshold for all classes
    OVERLAP_THRESHOLD = FIXED_OVERLAP_THRESHOLD
    print(f"\n🎯 Using fixed overlap threshold: {OVERLAP_THRESHOLD}")

# Object augmentation options (applied to cropped objects before pasting)
# -----------------------------------------------
# Default values = "Full" configuration (RB + BA + OV + FA + DC)
OBJECT_AUGM_OPTIONS = {
    'enabled': True,   # Foreground Augmentation (FA)
    'gaussian_blur': True,
    'gaussian_blur_kernel': 3,
    'scale_jitter': True,
    'scale_range': (0.8, 1.2),
    'rotation': False,            # ❌ DISABLED - incompatible for now with bbox size preservation
    'rotation_range': (-15, 15),  
    'brightness': True,
    'brightness_range': (-10, 10),
    'contrast': True,
    'contrast_range': (0.8, 1.2),
    'color_jitter': True,
    'hue_range': ( 0, 0),
    'saturation_range': (0.9, 1.1),
    'noise': True,
    'noise_std': 5,
    'flip_horizontal': True,     
    'flip_vertical': True,       
}
# -----------------------------------------------
# Background Augmentation Options
# -----------------------------------------------
BACKGROUND_AUGM_OPTIONS = {   # Background Augmentation (BA)
    'flip': True,
    'brightness': True,
    'brightness_range': (-30, 30),
    'contrast': True,
    'contrast_range': (0.8, 1.2),
    'blur': True,
    'blur_kernel': 3,
    'noise': True,
    'noise_std': 10,
    'rotation': False,
    'rotation_range': (-5, 5),
    'color_jitter': True,
    'hue_range': (-10, 10),
    'saturation_range': (0.8, 1.2)
}


# -----------------------------------------------
# Data Balancing and Augmentation Pipeline
# -----------------------------------------------
OVERSAMPLING = use_oversampling  # Whether to use oversampling instead of synthetic generation
RFS = use_rfs  # Whether to use RFS allocation strategy


def build_rfs_synthetic_allocation(class_object_df: pd.DataFrame, target_total_objects: int) -> pd.DataFrame:
    """
    Convert custom RFS repeat factors into the image_copy_allocation format expected
    by the synthetic copy-paste pipeline.

    To mirror the non-synthetic RFS branch behavior, we map repeat_factor directly
    to num_copies in the synthetic pipeline.
    """
    image_repeat_factors, class_info = compute_rfs_repeat_factors_targeted(
        class_object_df, target_total_objects
    )

    print("\n" + "=" * 60)
    print("Using CUSTOM RFS allocation for synthetic generation")
    print("=" * 60)
    for class_id in sorted(class_info['total_objects'].keys()):
        current = class_info['total_objects'][class_id]
        factor = class_info['scale_factors'][class_id]
        projected = int(current * factor)
        print(f"Class {class_id}: current={current}, factor={factor:.2f}x, projected~{projected}")

    allocation_records = []
    for img_name in class_object_df.index:
        repeat_factor = int(image_repeat_factors.get(img_name, 1))
        # Keep the same counting semantics used by run_pipeline_for_all_classes_RFS.
        copies_to_add = max(0, repeat_factor)
        if copies_to_add == 0:
            continue

        for class_id in class_object_df.columns:
            count = class_object_df.loc[img_name, class_id]
            count_val = float(count) if pd.notna(count) else 0
            if count_val > 0:
                allocation_records.append(
                    {
                        "image_name": img_name,
                        "class_name": class_id,
                        "num_copies": copies_to_add,
                    }
                )

    image_copy_allocation_rfs = pd.DataFrame(
        allocation_records, columns=["image_name", "class_name", "num_copies"]
    )
    print(f"RFS synthetic allocation rows: {len(image_copy_allocation_rfs)}")
    return image_copy_allocation_rfs

stored_datset_train_dir = base_dir / "data" / "images" / "processed" / f"fold{fold}_CV"  / "stored_dataset" / "train"
erase_crop_augm_dir = base_dir / "data" / "images" / "processed" / f"fold{fold}_CV"  / "erase_crop_augm"


class_object_df = build_image_class_df(stored_datset_train_dir)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(class_object_df.to_string())


# Compute image copy allocation using the utility function
image_copy_allocation, image_copy_allocation_by_class_and_bin = compute_image_copy_allocation(
    class_object_df, 
    R, 
    first_upper, 
    target_total_objects, 
    verbose=True
)


# ##############################################################################################
# #------------Add Data Augmentation Pipeline----------------    

# print image_copy_allocation for debugging
print("\nFinal image copy allocation for augmentation:")
print(image_copy_allocation)

# check if oversampling is selected
if OVERSAMPLING:
    print("\nOversampling selected: Synthetic generation will be skipped where oversampling is possible.")
    # For oversampling instead of synthetic generation:
    copy_paste_augm_dir = base_dir / "data" / "images" / "processed" / f"fold{fold}_CV" / "copy_paste_augm"

    if RFS:
        run_pipeline_for_all_classes_RFS(
            stored_datset_train_dir, 
            copy_paste_augm_dir, 
            image_copy_allocation,
            target_total_objects=target_total_objects
        )
    else:
        run_pipeline_for_all_classes_balance_meth_oversample(
            stored_datset_train_dir, 
            copy_paste_augm_dir, 
            image_copy_allocation
        )
else:
    print("\nSynthetic generation selected: All copies will be generated synthetically.")
    if RFS:
        image_copy_allocation_for_pipeline = build_rfs_synthetic_allocation(
            class_object_df, target_total_objects
        )
    else:
        image_copy_allocation_for_pipeline = image_copy_allocation

    # Store outputs in separate folders
    # erase cropping labeled splitted dataset
    erase_cropp_pipeline_for_all_classes(stored_datset_train_dir, erase_crop_augm_dir)

    copy_paste_augm_dir = base_dir / "data" / "images" / "processed" / f"fold{fold}_CV"  / "copy_paste_augm"
    # copy paste augmentation with rnd background and not rnd background

    run_pipeline_for_all_classes_balance_meth(
        stored_datset_train_dir, 
        copy_paste_augm_dir, 
        erase_crop_augm_dir, 
        image_copy_allocation_for_pipeline,
        rnd_bckgrd=rand_backg_select, 
        arise_augm=ARISE_AUGM,
        overlap_threshold=OVERLAP_THRESHOLD, 
        object_augm_options=OBJECT_AUGM_OPTIONS, 
        preservs_holes=PRESERVS_HOLES,
        background_augm_options=BACKGROUND_AUGM_OPTIONS,
        synch_augmentation=synch_augmentation  # Add this
    )

