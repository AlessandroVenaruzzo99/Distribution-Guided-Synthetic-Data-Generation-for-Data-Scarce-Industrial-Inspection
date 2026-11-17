import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import shutil
import yaml
import json
from ultralytics import YOLO
import subprocess
from pathlib import Path
import random
import re
import argparse
import csv
import pandas as pd
import glob
from scipy import ndimage
import math


########################################################################################

###################################################
# Classification of images in Labeling_raw_data 
###################################################

# Classify all the images in the raw dataset directory
# Create a custom dataset class

#Note: possible to Normalize the image to [-1, 1] or [0, 1] simply by changing transform = normalization
#      on-fly processing of the image
class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform =  transforms.Compose([
            transforms.Resize((640, 640)),  # Resize to 640x640
            transforms.ToTensor(),  # Convert PIL image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
        ])
        # Filter out directories and non-image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'} # Add other extensions if needed
        self.image_files = [
            f for f in os.listdir(root_dir)
            if os.path.isfile(os.path.join(root_dir, f)) and os.path.splitext(f)[1].lower() in image_extensions
        ]
        # Optional: Print the list of files found to verify filtering
        # print(f"CustomImageDataset found {len(self.image_files)} image files in {root_dir}:")
        # for f in self.image_files:
        #     print(f"  - {f}")


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')  # Ensure the image is in RGB format
        if self.transform:
            image = self.transform(image)
        return image

########################################################################################
 ##########################################################################
# Set seeds for reproducibility
##########################################################################

def set_seed(seed):
    """Set seeds for reproducible results"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


########################################################################################
#########################################
# Data preprocessing
#########################################

def analyze_folders(folder_path):
    base = Path(folder_path)

    if not base.exists() or not base.is_dir():
        print(f"❌ The provided path '{folder_path}' is not a valid directory.")
        return

    # List all directories in the given path
    folders = [f for f in base.iterdir() if f.is_dir()]
    print(f"Total folders in '{folder_path}': {len(folders)}\n")

    # For each folder, list its files and show count
    for folder in folders:
        files = [f for f in folder.iterdir() if f.is_file()]
        print(f"📁 {folder.name} — {len(files)} file(s)")
        for file in files:
            print(f"  - {file.name}")
        print()


# Add this new function to process and save images
def process_and_save_images(input_folder_path: Path, output_folder_path: Path):
    """
    Processes images from input_folder_path using CustomImageDataset
    and saves them to output_folder_path.
    """
    print(f"✨ Starting image processing from '{input_folder_path}'...")

    # Create the output directory if it doesn't exist
    output_folder_path.mkdir(parents=True, exist_ok=True)
    print(f"📂 Ensuring output directory exists: '{output_folder_path}'")

    # Create an instance of the dataset
    # The transform is hardcoded in the __init__ method
    try:
        image_dataset = CustomImageDataset(root_dir=input_folder_path)
        if len(image_dataset) == 0:
             print(f"⚠️ Input folder '{input_folder_path}' is empty. No images to process.")
             return
    except FileNotFoundError:
        print(f"❌ Error: Input folder '{input_folder_path}' not found.")
        return
    except Exception as e:
        print(f"❌ Error creating CustomImageDataset: {e}")
        return


    # Create a DataLoader to iterate over the dataset
    # Use batch size 1 to process and save images individually
    # shuffle=False is important to maintain order corresponding to image_files
    dataloader = DataLoader(image_dataset, batch_size=1, shuffle=False, num_workers=2) # Adjust num_workers as needed

    print(f"Processing {len(image_dataset)} images...")

    # Iterate through the dataloader and save processed images
    for i, processed_image_tensor in enumerate(dataloader):
        try:
            # Get the original filename
            original_filename = image_dataset.image_files[i]
            output_image_path = output_folder_path / original_filename

            # Convert the processed tensor back to a savable format (PIL Image)
            # Tensor shape is [1, 3, H, W], squeeze to [3, H, W]
            processed_image_tensor = processed_image_tensor.squeeze(0)

            # Reverse normalization (optional, depending on if you want normalized images saved)
            # If you want the image to look visually correct, reverse normalization
            # If you want the tensor values as they are after normalization, skip this
            # For saving a standard image file (like JPG/PNG), reversing is usually needed.
            mean = torch.tensor(image_dataset.transform.transforms[-1].mean).view(3, 1, 1)
            std = torch.tensor(image_dataset.transform.transforms[-1].std).view(3, 1, 1)
            processed_image_tensor = processed_image_tensor * std + mean

            # Clamp values to [0, 1] range
            processed_image_tensor = torch.clamp(processed_image_tensor, 0, 1)

            # Convert from [C, H, W] tensor to [H, W, C] NumPy array
            processed_image_np = processed_image_tensor.permute(1, 2, 0).numpy()

            # Convert from float [0, 1] to uint8 [0, 255]
            processed_image_np = (processed_image_np * 255).astype(np.uint8)

            # Convert NumPy array to PIL Image
            processed_image_pil = Image.fromarray(processed_image_np)

            # Save the processed image
            processed_image_pil.save(output_image_path)

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(image_dataset)} images.")

        except Exception as e:
            print(f"❌ Error processing image {image_dataset.image_files[i]}: {e}")
            # Continue processing other images

    print(f"✅ Finished processing and saving images to '{output_folder_path}'.")


###################################################
# Collecting images in Coco_dataset  
###################################################


# Option data balancing Arise paste augmentation
def find_copy_file_name_augm(run_dir: Path) -> list[Path]:
    # Find the first file that matches the pattern 
    Image_info = []
    Label_info = []
    for file in run_dir.iterdir():
        if file.is_file() and re.search(r"Image\s*\(\d+\)_run\d+.(jpg|png)", file.name):
            Image_info.append(os.path.abspath(str(file.resolve())))
        if file.is_file() and re.search(r"Image\s*\(\d+\)_run\d+.txt", file.name):
            Label_info.append(os.path.abspath(str(file.resolve())))
    return Image_info, Label_info

# Option for copy paste augmentation with correct folder access
def build_coco_dataset_balance(cp_pst_augm_dir_IN, stored_dtset_dir_IN, output_dir_IN):
    copy_paste_augm_dir = cp_pst_augm_dir_IN
    stored_dataset_dir = stored_dtset_dir_IN
    output_dir = output_dir_IN

    # Create COCO structure
    (output_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (output_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

    # === Copy from copy_paste_augm/* to coco_dataset/train/* ===
    for class_dir in copy_paste_augm_dir.iterdir():
        for image_folder in class_dir.iterdir():
            if image_folder.is_dir():
                # image_folder 
                Image_info, Label_info = find_copy_file_name_augm(image_folder)
                for info_file in Image_info:
                    shutil.copy2(info_file, output_dir / "train" / "images")
                for info_label in Label_info:
                    shutil.copy2(info_label, output_dir / "train" / "labels")

    # === Copy from stored_dataset/train/* and stored_dataset/val/* ===
    for split in ["train", "val"]:
        split_path = stored_dataset_dir / split
        for class_dir in split_path.iterdir():
            images_dir = class_dir / "images"
            labels_dir = class_dir / "labels"
            for file in images_dir.iterdir():
                if file.is_file() and file.suffix == ".jpg" and "_debug" not in file.name:
                    dest_path = output_dir / split / "images"
                    shutil.copy2(file, dest_path)
            for file in labels_dir.iterdir():
                if file.is_file() and file.suffix == ".txt":
                    dest_path = output_dir / split / "labels"
                    shutil.copy2(file, dest_path)










########################################################################################################################
# Option for copy paste augmentation
def find_copy_file_name_augm(run_dir: Path) -> list[Path]:
    # Find the first file that matches the pattern 
    Image_info = []
    Label_info = []
    for file in run_dir.iterdir():
        if file.is_file() and re.search(r"Image\s*\(\d+\)_run\d+.(jpg|png)", file.name):
            Image_info.append(os.path.abspath(str(file.resolve())))
        if file.is_file() and re.search(r"Image\s*\(\d+\)_run\d+.txt", file.name):
            Label_info.append(os.path.abspath(str(file.resolve())))
    return Image_info, Label_info

# Opttion for copy paste augmentation
def build_coco_dataset_vanilla( cp_pst_augm_dir_IN,stored_dtset_dir_IN, output_dir_IN ):
    
    copy_paste_augm_dir = cp_pst_augm_dir_IN
    stored_dataset_dir = stored_dtset_dir_IN
    output_dir = output_dir_IN

    # Create COCO structure
    (output_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (output_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

    # ===  Copy from copy_paste_augm/* to coco_dataset/train/* === 
    for class_dir in copy_paste_augm_dir.iterdir():
        for class_ in class_dir.iterdir():
            for run in class_.iterdir():
                Image_info, Label_info = find_copy_file_name_augm(run)
                for info_file in Image_info:
                    shutil.copy2(info_file, output_dir / "train" / "images")
                for info_label in Label_info:
                    shutil.copy2(info_label, output_dir / "train" / "labels")


    # === Copy from stored_dataset/train/* and stored_dataset/val/* ===
    for split in ["train", "val"]:
        split_path = stored_dataset_dir / split
        for class_dir in split_path.iterdir():
            images_dir = class_dir / "images"
            labels_dir = class_dir / "labels"
            for file in images_dir.iterdir():
                if file.is_file() and file.suffix == ".jpg" and "_debug" not in file.name:
                    dest_path = output_dir / split / "images"
                    shutil.copy2(file, dest_path)
            for file in labels_dir.iterdir():
                if file.is_file() and file.suffix == ".txt":
                    dest_path = output_dir / split / "labels"
                    shutil.copy2(file, dest_path)

def build_coco_dataset_REGEN(cp_pst_augm_dir_IN, stored_dtset_dir_IN, output_dir_IN):
    """
    Build COCO dataset structure from copy-paste augmented images and stored dataset.
    Handles path structure: copy_paste_augm/<class_name>/run1/Image (17)/
    
    Args:
        cp_pst_augm_dir_IN: Path to copy_paste_augm directory
        stored_dtset_dir_IN: Path to stored_dataset directory
        output_dir_IN: Path to output COCO dataset directory
    """
    copy_paste_augm_dir = cp_pst_augm_dir_IN
    stored_dataset_dir = stored_dtset_dir_IN
    output_dir = output_dir_IN

    # Create COCO structure
    (output_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (output_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

    # === Copy from copy_paste_augm/* to coco_dataset/train/* ===
    # Structure: copy_paste_augm/<class_name>/run1/Image (17)/
    for class_dir in copy_paste_augm_dir.iterdir():
        if not class_dir.is_dir():
            continue
        for run_dir in class_dir.iterdir():
            if not run_dir.is_dir():
                continue
            for image_folder in run_dir.iterdir():
                if image_folder.is_dir():
                    Image_info, Label_info = find_copy_file_name_augm(image_folder)
                    for info_file in Image_info:
                        shutil.copy2(info_file, output_dir / "train" / "images")
                    for info_label in Label_info:
                        shutil.copy2(info_label, output_dir / "train" / "labels")

    # === Copy from stored_dataset/train/* and stored_dataset/val/* ===
    for split in ["train", "val"]:
        split_path = stored_dataset_dir / split
        if not split_path.exists():
            continue
        for class_dir in split_path.iterdir():
            if not class_dir.is_dir():
                continue
            images_dir = class_dir / "images"
            labels_dir = class_dir / "labels"
            if images_dir.exists():
                for file in images_dir.iterdir():
                    if file.is_file() and file.suffix == ".jpg" and "_debug" not in file.name:
                        dest_path = output_dir / split / "images"
                        shutil.copy2(file, dest_path)
            if labels_dir.exists():
                for file in labels_dir.iterdir():
                    if file.is_file() and file.suffix == ".txt":
                        dest_path = output_dir / split / "labels"
                        shutil.copy2(file, dest_path)



################################################################################################################################

# option for Yolov5 implicit live augmentation
def build_coco_dataset( train_stored_dataset_dir_IN,val_stored_dataset_dir_IN, output_dir_IN ):
    
    train_stored_labeled_dir = train_stored_dataset_dir_IN
    val_stored_dataset_dir = val_stored_dataset_dir_IN
    output_dir = output_dir_IN

    # Create COCO structure
    (output_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (output_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}

    # ===  Copy from copy_paste_augm/* to coco_dataset/train/* === 
    for class_dir in train_stored_labeled_dir.iterdir():
        images_dir = class_dir / "images"
        labels_dir = class_dir / "labels"
        if images_dir.exists():
            for file in images_dir.iterdir():
                if file.is_file() and file.suffix.lower() in image_exts and "_debug" not in file.name:
                    dest_path = output_dir / "train" / "images"
                    shutil.copy2(file, dest_path)
        if labels_dir.exists():
            for file in labels_dir.iterdir():
                if file.is_file() and file.suffix == ".txt":
                    dest_path = output_dir / "train" / "labels"
                    shutil.copy2(file, dest_path)

    # === Copy from stored_dataset/train/* and stored_dataset/val/* ===
    for class_dir in val_stored_dataset_dir.iterdir():
        images_dir = class_dir / "images"
        labels_dir = class_dir / "labels"
        if images_dir.exists():
            for file in images_dir.iterdir():
                if file.is_file() and file.suffix.lower() in image_exts and "_debug" not in file.name:
                    dest_path = output_dir / "val" / "images"
                    shutil.copy2(file, dest_path)
        if labels_dir.exists():
            for file in labels_dir.iterdir():
                if file.is_file() and file.suffix == ".txt":
                    dest_path = output_dir / "val" / "labels"
                    shutil.copy2(file, dest_path)




########################################################################################

###################################################
# Split the dataset into train and test sets
###################################################

def split_yolo_dataset_by_objects(dataset_dir, labels_dir, train_dir, val_dir, train_ratio, seed):
    """
    Split a YOLO-format dataset into train/val sets by object count.
    All images with zero objects are always assigned to the train set.

    Args:
        dataset_dir (str): Path to directory with images (e.g. .jpg, .png).
        labels_dir (str): Path to directory with YOLO .txt label files.
        train_dir (str): Path to base directory for training split (will contain 'images' and 'labels').
        val_dir (str): Path to base directory for validation split (will contain 'images' and 'labels').
        train_ratio (float): Desired fraction of objects in the train set (default 0.8).
        seed (int, optional): Random seed for reproducibility.

    This function copies images and labels into the new directories, handling missing label files as zero-object.
    """
    dataset_dir = Path(dataset_dir)
    labels_dir = Path(labels_dir)
    train_dir = Path(train_dir)
    val_dir = Path(val_dir)

    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    all_images = [p for p in dataset_dir.iterdir() if p.is_file() and p.suffix.lower() in image_exts]

    # Count objects in each image
    image_info = []
    total_objects = 0
    for img_path in all_images:
        lbl_path = labels_dir / (img_path.stem + '.txt')
        count = 0
        if lbl_path.is_file():
            with lbl_path.open('r') as f:
                lines = [line.strip() for line in f if line.strip()]
                count = len(lines)
        image_info.append((img_path, lbl_path, count))
        total_objects += count

    print(f"Total images: {len(image_info)}, Total objects: {total_objects}")
    target_train = int(total_objects * train_ratio)
    print(f"Target train objects (approx {train_ratio*100:.0f}%): {target_train}")

    if seed is not None:
        random.seed(seed)
    random.shuffle(image_info)

    train_images = []
    val_images = []
    train_obj_count = 0
    val_obj_count = 0

    # Separate zero-object images (always to train)
    zero_object_images = []
    nonzero_object_images = []
    for img_path, lbl_path, count in image_info:
        if count == 0:
            zero_object_images.append((img_path, lbl_path, count))
        else:
            nonzero_object_images.append((img_path, lbl_path, count))

    # Assign nonzero-object images by object count
    for img_path, lbl_path, count in nonzero_object_images:
        if train_obj_count < target_train:
            if train_obj_count + count <= target_train:
                train_images.append((img_path, lbl_path))
                train_obj_count += count
            else:
                overshoot = (train_obj_count + count) - target_train
                shortfall = target_train - train_obj_count
                if overshoot <= shortfall:
                    train_images.append((img_path, lbl_path))
                    train_obj_count += count
                else:
                    val_images.append((img_path, lbl_path))
                    val_obj_count += count
        else:
            val_images.append((img_path, lbl_path))
            val_obj_count += count

    # Add all zero-object images to train set
    for img_path, lbl_path, _ in zero_object_images:
        train_images.append((img_path, lbl_path))

    print(f"Assigned {len(train_images)} images with {train_obj_count} objects to TRAIN (plus {len(zero_object_images)} zero-object images)")
    print(f"Assigned {len(val_images)} images with {val_obj_count} objects to VAL")

    train_img_dir = train_dir / 'images'
    train_lbl_dir = train_dir / 'labels'
    val_img_dir = val_dir / 'images'
    val_lbl_dir = val_dir / 'labels'

    for directory in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        os.makedirs(directory, exist_ok=True)

    for img_path, lbl_path in train_images:
        shutil.copy2(img_path, train_img_dir / img_path.name)
        if lbl_path.is_file():
            shutil.copy2(lbl_path, train_lbl_dir / lbl_path.name)
        else:
            (train_lbl_dir / (img_path.stem + '.txt')).touch()

    for img_path, lbl_path in val_images:
        shutil.copy2(img_path, val_img_dir / img_path.name)
        if lbl_path.is_file():
            shutil.copy2(lbl_path, val_lbl_dir / lbl_path.name)
        else:
            (val_lbl_dir / (img_path.stem + '.txt')).touch()

    print("Dataset split complete.")

################################################################################################################################################################################
############################################################ K-Fold Cross Validation Split ################################################
#######################################################################################################################################################################
def split_yolo_dataset_kfold(dataset_dir, labels_dir, train_dir, val_dir, k_folds, current_fold, seed=42):
    """
    Split a YOLO-format dataset into train/val sets using k-fold cross-validation.
    All images with zero objects are always assigned to the train set.
    
    Args:
        dataset_dir (str): Path to directory with images
        labels_dir (str): Path to directory with YOLO .txt label files
        train_dir (str): Path to base directory for training split
        val_dir (str): Path to base directory for validation split
        k_folds (int): Total number of folds
        current_fold (int): Current fold number (0 to k_folds-1)
        seed (int): Random seed for reproducibility across all folds
    """
    dataset_dir = Path(dataset_dir)
    labels_dir = Path(labels_dir)
    train_dir = Path(train_dir)
    val_dir = Path(val_dir)

    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    all_images = [p for p in dataset_dir.iterdir() if p.is_file() and p.suffix.lower() in image_exts]

    # Count objects in each image and separate zero-object images
    image_info = []
    zero_object_images = []
    total_objects = 0
    for img_path in all_images:
        lbl_path = labels_dir / (img_path.stem + '.txt')
        count = 0
        if lbl_path.is_file():
            with lbl_path.open('r') as f:
                lines = [line.strip() for line in f if line.strip()]
                count = len(lines)
        
        if count == 0:
            zero_object_images.append((img_path, lbl_path, count))
        else:
            image_info.append((img_path, lbl_path, count))
        total_objects += count

    print(f"Total images: {len(all_images)}, Total objects: {total_objects}")
    print(f"Zero-object images (will go to train): {len(zero_object_images)}")
    
    # Sort by object count for better distribution
    image_info.sort(key=lambda x: x[2])
    
    # Use consistent seed for reproducible splits across all folds
    if seed is not None:
        random.seed(seed)
        random.shuffle(image_info)

    # Calculate fold size based on non-zero object images only
    fold_size = len(image_info) // k_folds
    remainder = len(image_info) % k_folds
    
    # Create fold boundaries
    fold_starts = []
    start = 0
    for i in range(k_folds):
        fold_starts.append(start)
        # Distribute remainder across first few folds
        current_fold_size = fold_size + (1 if i < remainder else 0)
        start += current_fold_size
    fold_starts.append(len(image_info))  # End boundary
    
    # Determine validation and training sets from non-zero object images
    val_start = fold_starts[current_fold]
    val_end = fold_starts[current_fold + 1]
    
    val_images = image_info[val_start:val_end]
    train_images = image_info[:val_start] + image_info[val_end:]
    
    # Add all zero-object images to training set
    train_images.extend(zero_object_images)
    
    # Count objects in each split
    train_obj_count = sum(count for _, _, count in train_images)
    val_obj_count = sum(count for _, _, count in val_images)
    
    print(f"Fold {current_fold + 1}/{k_folds}:")
    print(f"  Training: {len(train_images)} images with {train_obj_count} objects (includes {len(zero_object_images)} zero-object images)")
    print(f"  Validation: {len(val_images)} images with {val_obj_count} objects")

    # Create directories
    train_img_dir = train_dir / 'images'
    train_lbl_dir = train_dir / 'labels'
    val_img_dir = val_dir / 'images'
    val_lbl_dir = val_dir / 'labels'

    for directory in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        os.makedirs(directory, exist_ok=True)

    # Copy training files
    for img_path, lbl_path, _ in train_images:
        shutil.copy2(img_path, train_img_dir / img_path.name)
        if lbl_path.is_file():
            shutil.copy2(lbl_path, train_lbl_dir / lbl_path.name)
        else:
            (train_lbl_dir / (img_path.stem + '.txt')).touch()

    # Copy validation files
    for img_path, lbl_path, _ in val_images:
        shutil.copy2(img_path, val_img_dir / img_path.name)
        if lbl_path.is_file():
            shutil.copy2(lbl_path, val_lbl_dir / lbl_path.name)
        else:
            (val_lbl_dir / (img_path.stem + '.txt')).touch()

    print("K-fold dataset split complete.")


################################################################################################################################################################################
############################################ Synthetic Image Classification ####################################################################################################
################################################################################################################################################################################

def is_synthetic_image(img_path):
    """
    Classify an image as synthetic (diffusion-generated) vs original.
    
    Detection criteria (either is sufficient):
      1. Filename does NOT match the original naming pattern 'Image (N).ext'
      2. Image resolution is exactly 256x256 (fallback confirmation)
    
    Args:
        img_path (Path or str): Path to the image file.
    
    Returns:
        bool: True if the image is synthetic, False if original.
    """
    img_path = Path(img_path)
    name = img_path.name
    
    # Primary check: original images match 'Image (N).ext' pattern
    original_pattern = re.compile(r'^Image\s*\(\d+\)\.(jpg|jpeg|png|bmp|gif)$', re.IGNORECASE)
    if not original_pattern.match(name):
        return True  # Does not match original naming → synthetic
    
    return False  # Matches original naming → real image


def split_yolo_dataset_kfold_with_synthetic(dataset_dir, labels_dir, train_dir, val_dir, k_folds, current_fold, seed=42):
    """
    Split a YOLO-format dataset into train/val sets using k-fold cross-validation,
    ensuring that synthetic (diffusion-generated) images are NEVER placed in the
    validation set — they go exclusively to the training set.
    
    The k-fold splitting logic is applied only to original images.
    Zero-object original images are always assigned to the training set.
    
    Args:
        dataset_dir (str): Path to directory with images
        labels_dir (str): Path to directory with YOLO .txt label files
        train_dir (str): Path to base directory for training split
        val_dir (str): Path to base directory for validation split
        k_folds (int): Total number of folds
        current_fold (int): Current fold number (0 to k_folds-1)
        seed (int): Random seed for reproducibility across all folds
    """
    dataset_dir = Path(dataset_dir)
    labels_dir = Path(labels_dir)
    train_dir = Path(train_dir)
    val_dir = Path(val_dir)

    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    all_images = [p for p in dataset_dir.iterdir() if p.is_file() and p.suffix.lower() in image_exts]

    # ── Partition into original and synthetic images ──
    original_images = []
    synthetic_images = []
    for img_path in all_images:
        if is_synthetic_image(img_path):
            synthetic_images.append(img_path)
        else:
            original_images.append(img_path)

    print(f"  📷 Total images found: {len(all_images)}")
    print(f"      ├─ Original: {len(original_images)}")
    print(f"      └─ Synthetic (train-only): {len(synthetic_images)}")

    # ── Count objects for original images and separate zero-object ones ──
    original_info = []
    zero_object_originals = []
    total_original_objects = 0
    for img_path in original_images:
        lbl_path = labels_dir / (img_path.stem + '.txt')
        count = 0
        if lbl_path.is_file():
            with lbl_path.open('r') as f:
                lines = [line.strip() for line in f if line.strip()]
                count = len(lines)
        
        if count == 0:
            zero_object_originals.append((img_path, lbl_path, count))
        else:
            original_info.append((img_path, lbl_path, count))
        total_original_objects += count

    # ── Count objects for synthetic images (all go to train) ──
    synthetic_info = []
    total_synthetic_objects = 0
    for img_path in synthetic_images:
        lbl_path = labels_dir / (img_path.stem + '.txt')
        count = 0
        if lbl_path.is_file():
            with lbl_path.open('r') as f:
                lines = [line.strip() for line in f if line.strip()]
                count = len(lines)
        synthetic_info.append((img_path, lbl_path, count))
        total_synthetic_objects += count

    print(f"  🏷️  Original objects: {total_original_objects} (across {len(original_images)} images)")
    print(f"  🏷️  Synthetic objects: {total_synthetic_objects} (across {len(synthetic_images)} images, all → train)")
    print(f"  ⚠️  Zero-object original images (always → train): {len(zero_object_originals)}")

    # ── K-fold split on original (non-zero) images only ──
    # Sort then shuffle with fixed seed for reproducible splits
    original_info.sort(key=lambda x: x[2])
    if seed is not None:
        random.seed(seed)
        random.shuffle(original_info)

    fold_size = len(original_info) // k_folds
    remainder = len(original_info) % k_folds
    
    fold_starts = []
    start = 0
    for i in range(k_folds):
        fold_starts.append(start)
        current_fold_size = fold_size + (1 if i < remainder else 0)
        start += current_fold_size
    fold_starts.append(len(original_info))
    
    val_start = fold_starts[current_fold]
    val_end = fold_starts[current_fold + 1]
    
    val_images = original_info[val_start:val_end]
    train_images = original_info[:val_start] + original_info[val_end:]
    
    # Add zero-object originals to train
    train_images.extend(zero_object_originals)
    
    # Add ALL synthetic images to train
    train_images.extend(synthetic_info)
    
    # ── Report split statistics ──
    train_obj_count = sum(count for _, _, count in train_images)
    val_obj_count = sum(count for _, _, count in val_images)
    train_original_count = len([1 for img, _, _ in train_images if not is_synthetic_image(img)])
    train_synthetic_count = len([1 for img, _, _ in train_images if is_synthetic_image(img)])
    
    print(f"\n  📊 Fold {current_fold + 1}/{k_folds}:")
    print(f"      Training: {len(train_images)} images ({train_original_count} original + {train_synthetic_count} synthetic) with {train_obj_count} objects")
    print(f"      Validation: {len(val_images)} images (all original) with {val_obj_count} objects")

    # ── Create directories and copy files ──
    train_img_dir = train_dir / 'images'
    train_lbl_dir = train_dir / 'labels'
    val_img_dir = val_dir / 'images'
    val_lbl_dir = val_dir / 'labels'

    for directory in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        os.makedirs(directory, exist_ok=True)

    # Copy training files
    for img_path, lbl_path, _ in train_images:
        shutil.copy2(img_path, train_img_dir / img_path.name)
        if lbl_path.is_file():
            shutil.copy2(lbl_path, train_lbl_dir / lbl_path.name)
        else:
            (train_lbl_dir / (img_path.stem + '.txt')).touch()

    # Copy validation files (guaranteed to be original images only)
    for img_path, lbl_path, _ in val_images:
        shutil.copy2(img_path, val_img_dir / img_path.name)
        if lbl_path.is_file():
            shutil.copy2(lbl_path, val_lbl_dir / lbl_path.name)
        else:
            (val_lbl_dir / (img_path.stem + '.txt')).touch()

    print("  ✅ K-fold dataset split with synthetic separation complete.")


########################################################################################

# Verifing the labels Count the total number of rows in all .txt files in a folder
def count_total_rows(folder_path):
    """Counts total rows of each txt file and hence counts the number of objs for image"""
    total_rows = 0
    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        # Check if it's a .txt file
        if file_name.endswith(".txt") and os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                total_rows += sum(1 for _ in file)  # Count lines in the file

    return total_rows



########################################################################################


###################################################
# Look for a folder by name to update the json file
###################################################

def find_folder(folder_name, search_path='/'):
    """
    Recursively search for a folder by name starting from a given path.

    :param folder_name: Name of the folder to search for.
    :param search_path: Path to start the search from (default is root '/').
    :return: Full path to the folder if found, otherwise None.
    """
    for root, dirs, files in os.walk(search_path):
        if folder_name in dirs:
            return os.path.join(root, folder_name)
    return None



########################################################################################

###################################################
# Create Infered images with Ground Truth boxes
###################################################


def draw_ground_truth_boxes(image_folder, labels_folder, inference_folder, output_folder, class_names=None):
    os.makedirs(output_folder, exist_ok=True)
    
    image_paths = glob.glob(os.path.join(image_folder, "*.jpg"))  # or .png, adjust if needed
    # For each image, use the corresponding inference image as background
    for image_path in image_paths:
        filename = os.path.basename(image_path)
        infered_image_path = os.path.join(inference_folder, filename)
        if not os.path.exists(infered_image_path):
            print(f"Warning: Inference image not found for {filename}, skipping.")
            continue
        img = cv2.imread(infered_image_path)
        if img is None:
            print(f"Warning: Could not read inference image {infered_image_path}, skipping.")
            continue
        h, w = img.shape[:2]
        label_path = os.path.join(labels_folder, filename.replace(".jpg", ".txt"))
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    # Convert YOLO format to rectangle
                    x1 = int((x_center - width / 2) * w)
                    y1 = int((y_center - height / 2) * h)
                    x2 = int((x_center + width / 2) * w)
                    y2 = int((y_center + height / 2) * h)
                    color = (0, 255, 0)  # green for GT boxes
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    if class_names:
                        cv2.putText(img, class_names[int(class_id)], (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imwrite(os.path.join(output_folder, filename), img)




################################################################################################################################################
####### K-Fold Cross-Validation Performance Evaluation #########################################################################################
################################################################################################################################################

###############################################
## Evaluate Cross-Validation Performance on mean_metrics.csv
###############################################
def calculate_and_append_mean(file_path):
    """
    Reads a metrics CSV, calculates the mean of numeric columns,
    and appends an empty row and a 'mean' row to the file.
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    # 1. Read the CSV file
    df = pd.read_csv(file_path)

    # 2. Cleanup: Remove existing 'mean' or empty rows if the script is run multiple times
    # We keep only rows where the 'folds' column starts with 'fold'
    df = df[df['folds'].astype(str).str.startswith('fold')]

    # 3. Calculate the mean for numeric columns (excluding the 'folds' column)
    # We select all columns from index 1 to the end
    numeric_cols = df.iloc[:, 1:]
    mean_values = numeric_cols.mean()

    # 4. Create the 'mean' row
    # We start with a dictionary setting the 'folds' name
    mean_row = {'folds': 'mean'}
    # Add the calculated mean values to the dictionary
    for col, val in mean_values.items():
        mean_row[col] = val

    # 5. Prepare the DataFrames to append
    # Create a DataFrame for the mean row
    mean_df = pd.DataFrame([mean_row])
    
    # Create a DataFrame for the empty separator row (to match your example format)
    # We create a row with None/NaN values
    separator_df = pd.DataFrame([{col: None for col in df.columns}])

    # 6. Concatenate: Original Data + Separator + Mean
    final_df = pd.concat([df, separator_df, mean_df], ignore_index=True)

    # 7. Save back to the CSV
    # float_format='%.16f' ensures high precision is kept like in your example
    final_df.to_csv(file_path, index=False)
    
    print(f"Successfully updated {file_path} with mean values.")
################################################################
## Evaluate Cross-Validation Performance on class_metrics.csv
################################################################à
def parse_array_string(s):
    """
    Converts a string like '[ 0.5  0.2  1.0 ]' into a numpy array.
    """
    if not isinstance(s, str):
        return np.array([])
    
    # Remove brackets and extra whitespace
    s_clean = s.replace('[', '').replace(']', '').strip()
    
    # Split by whitespace (handles multiple spaces automatically)
    try:
        values = [float(x) for x in s_clean.split()]
        return np.array(values)
    except ValueError:
        return np.array([])

def format_array_string(arr):
    """
    Converts a numpy array back to string format '[ val1 val2 ... ]'.
    Using 6 decimal places for precision.
    """
    # Create a space-separated string of values
    inner_str = " ".join([f"{x:.6f}" for x in arr])
    return f"[{inner_str}]"

def calculate_class_means(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    # 1. Read the CSV
    df = pd.read_csv(file_path)

    # 2. Cleanup: Keep only rows where 'folds' starts with 'fold'
    # This prevents errors if you run the script multiple times
    df_folds = df[df['folds'].astype(str).str.startswith('fold')].copy()

    # Dictionary to hold the final mean values
    mean_row_data = {'folds': 'mean'}

    # 3. Iterate over metric columns (skipping the 'folds' column)
    metric_cols = [c for c in df.columns if c != 'folds']

    for col in metric_cols:
        # Convert the column of strings into a list of numpy arrays
        # arrays_list will be a list of arrays: [array([class1_f0, class2_f0]), array([class1_f1, class2_f1]), ...]
        arrays_list = df_folds[col].apply(parse_array_string).tolist()
        
        # Check if we have valid data
        if not arrays_list or len(arrays_list) == 0:
            mean_row_data[col] = None
            continue

        try:
            # Stack arrays vertically (Rows=Folds, Cols=Classes)
            # matrix shape: (num_folds, num_classes)
            matrix = np.vstack(arrays_list)
            
            # Calculate mean down the vertical axis (average over folds for each class)
            class_means = np.mean(matrix, axis=0)
            
            # Convert back to string format to store in CSV
            mean_row_data[col] = format_array_string(class_means)
            
        except ValueError as e:
            print(f"Skipping column '{col}' due to shape mismatch or data error: {e}")
            mean_row_data[col] = "Error"

    # 4. Prepare DataFrames to append
    mean_df = pd.DataFrame([mean_row_data])
    
    # Empty separator row with None/NaN
    separator_df = pd.DataFrame([{col: None for col in df.columns}])

    # 5. Concatenate
    final_df = pd.concat([df_folds, separator_df, mean_df], ignore_index=True)

    # 6. Save
    final_df.to_csv(file_path, index=False)
    print(f"Successfully calculated class-wise means for {file_path}")
