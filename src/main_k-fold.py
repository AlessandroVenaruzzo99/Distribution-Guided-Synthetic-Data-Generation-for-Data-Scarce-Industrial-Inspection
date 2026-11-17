from utils import * 


####################################################################################################################
# The Main file construction should be define to consent the updating of folders and version of the project
####################################################################################################################

# The main.py file is the entry point of the project. It orchestrates the entire workflow, from data preprocessing to model training.
# Each step is encapsulated in a function, which can be defined in separate modules or files.
# This modular approach allows for better organization and maintainability of the codebase.

def main():
    """ K-fold Cross-Validation for YOLOv5 """
    
    # Check that the annotated dataset is in place (see README, "Dataset layout").
    # If YOLO labels are missing, they are generated with the optional rule-based auto-labeling step.
    #   data/images/raw/<dataset_name>/<class_name>/*.jpg   -> images, one folder per class
    #   data/images/stored/<class_name>/*.jpg               -> same images (used for overlap statistics)
    #   data/labels/stored/<class_name>/*.txt               -> YOLO-format annotations
    base_dir = Path(__file__).resolve().parent.parent
    stored_dataset_dir = base_dir / "data" / "images" / "stored"
    stored_labels_dir = base_dir / "data" / "labels" / "stored"

    if not stored_labels_dir.exists():
        print("⚠️ No labels found in data/labels/stored: running the optional auto-labeling step 🖥️...")
        subprocess.run(["python3", "labeling_raw_data.py"], check=True)

    if not stored_dataset_dir.exists() or not stored_labels_dir.exists():
        sys.exit(
            "❌ Annotated dataset not found. Expected images in data/images/stored/<class_name>/ "
            "and YOLO labels in data/labels/stored/<class_name>/. See README.md for the expected layout."
        )

    ##########################################################################
    # K-Fold Cross-Validation Configuration
    ##########################################################################
    k_folds = 5  # Number of folds for k-fold CV
    copy_paste_augm = True  # Whether to perform copy-paste augmentation (True/False)
    data_balancing = True
    use_rfs_custom_method = False  # True=Custom RFS baseline (image-level), False=proposed bin-based allocation
    use_oversampling = False  # Keep False to run synthetic generation for both methods
    vanilla = False  # Use vanilla copy-paste augmentation (fixed overlap threshold) 
    adaptive_overlap_threshold = True  # Use class-specific overlap thresholds computed from stored data
    all_no_db_sych = True  # Use synchronized augmentation without databalancing
    fine_tuning = False
    use_image_weights = True # use image weights during training, based on the number of objects in the image (more objects = higher weight)
    use_synthetic_data = True  # When True, synthetic images go to train only (never to validation)
    detector = "yolov5s"  # Ultralytics detector: "yolov5s" (main experiments) or "yolo11s"

    seed = 42  # Fixed seed for reproducible splits
    set_seed(seed)
    
    for fold in range(k_folds):
        print(f"\n{'='*60}")
        print(f"STARTING FOLD {fold + 1}/{k_folds}")
        print(f"{'='*60}")
        
        ###########################################################################
        # Dataset split, augmentation and merge 
        ###########################################################################
        print("Splitting data 🖥️...")
        subprocess.run(["python3", "split_data.py", "--fold", str(fold), "--k_folds", str(k_folds), "--seed", str(seed), "--use_synthetic", str(use_synthetic_data)], check=True)
        
        # Rest of your pipeline remains the same, but update the run parameter to fold
        if copy_paste_augm:
            if data_balancing:
                print("Data balancing enabled, starting data balancing 🖥️...")
                print(f"Data balancing mode: {'Custom RFS' if use_rfs_custom_method else 'Bin-based'} | {'Oversampling' if use_oversampling else 'Synthetic generation'}")
                subprocess.run(
                    [
                        "python3",
                        "data_balance_augm.py",
                        "--fold",
                        str(fold),
                        "--rfs",
                        str(use_rfs_custom_method),
                        "--oversampling",
                        str(use_oversampling),
                    ],
                    check=True,
                )
            else:
                print("Copy paste augmentation enabled, starting data augmentation 🖥️...")
                if vanilla:
                    print("Using Vanilla Copy-Paste Augmentation ✅📊...")
                    subprocess.run(["python3", "copy_paste_augm_Vanilla.py", "--fold", str(fold)], check=True)
                elif adaptive_overlap_threshold:

                    if all_no_db_sych:
                        print("Using All augmentation settings less the Data Balancing Strategy 📊...")
                        subprocess.run(["python3", "copy_paste_augm_Synch_AllNoDB.py", "--fold", str(fold)], check=True)
                    else: 
                        print("Using REGEN Copy-Paste Augmentation with ADAPTIVE OVERLAP THRESHOLDS 📊...")
                        subprocess.run(["python3", "copy_paste_augm_REGEN_AOT.py", "--fold", str(fold)], check=True)
                else:
                    print("Using REGEN Copy-Paste Augmentation with FIXED overlap threshold 📊...")
                    subprocess.run(["python3", "copy_paste_augm_REGEN.py", "--fold", str(fold)], check=True)
                
            print("Starting dataset creation 🖥️...")
            subprocess.run(["python3", "augm_collecting_images.py", "--fold", str(fold), "--data_balancing", str(data_balancing), "--vanilla", str(vanilla)], check=True)
            
            print("Generating YAML file 🖥️...")
            subprocess.run(["python3", "yaml_generator.py", "--fold", str(fold)], check=True)
            print("Editing Json configuration file 🖥️...")
            subprocess.run(["python3", "update_json.py", "--fold", str(fold)], check=True)
            print("Editing default.yaml configuration file 🖥️...")
            subprocess.run(["python3", "edit_default_yaml_cp_paste.py", "--fold", str(fold)], check=True)
        else:
            print("Copy paste augmentation disabled, starting dataset creation 🖥️...")
            subprocess.run(["python3", "collecting_images.py", "--fold", str(fold)], check=True)
            
            print("Generating YAML file 🖥️...")
            subprocess.run(["python3", "yaml_generator.py", "--fold", str(fold)], check=True)
            print("Editing Json configuration file 🖥️...")
            subprocess.run(["python3", "update_json.py", "--fold", str(fold)], check=True)
            print("Editing default.yaml configuration file 🖥️...")
            subprocess.run(["python3", "edit_default_yaml.py", "--fold", str(fold)], check=True)

        ###########################################################################
        # Perform the train predict and validation of the model
        ###########################################################################
        # Pass seed for reproducible training
        print(f"Perform k-fold train, prediction and val of the model, for fold {fold + 1} 🖥️...")
        subprocess.run(["python3", "train_pred_val.py", "--fold", str(fold), "--fine_tuning", str(fine_tuning), "--k_folds", str(k_folds),  "--seed", str(seed), "--image_weights", str(use_image_weights), "--model", detector], check=True)
        print(f"Evaluate performance of the model, for fold {fold + 1} 📊...")
    
    
    ##########################################################################
    # K-Fold Cross-Validation Complete
    ##########################################################################
    print(f"\n{'='*60}")
    print(f"K-FOLD CROSS-VALIDATION COMPLETE ({k_folds} folds) ✅🎉")
    print(f"{'='*60}")

    # Evaluate model performance by mean values of CV metrics
    print("Evaluating model performance across Cross Validation🖥️...")
    subprocess.run(["python3", "evaluate_cv_performance.py", "--fold", str(fold), "--k_folds", str(k_folds)], check=True)    

  
if __name__ == '__main__':
    main()
