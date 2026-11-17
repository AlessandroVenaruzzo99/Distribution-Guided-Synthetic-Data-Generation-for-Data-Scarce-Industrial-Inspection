from utils import * 
import os


# -----------------------------------------------
# Parse the --fold argument from the command line
# -----------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, required=True, help="Fold number for cross-validation")
args = parser.parse_args()
fold = args.fold  # This is the fold number passed from your main loop
# -----------------------------------------------
# Define base directory (two levels up from current file)
base_dir = Path(__file__).resolve().parent.parent
# Automatically determine class names from subfolders in the raw data directory
raw_data_dir = os.path.join(base_dir, "data", "labels", "stored")
class_names = sorted([d for d in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir, d))])

# Count total objects for each class using count_total_rows from utils.py
class_counts = []
for class_name in class_names:
    class_dir = os.path.join(raw_data_dir, class_name)
    total_count = count_total_rows(class_dir)
    class_counts.append((class_name, total_count))

# Sort classes by total object count in descending order
class_counts.sort(key=lambda x: x[1], reverse=True)
sorted_class_names = [name for name, _ in class_counts]
print(f"🐞 DEBUG: Sorted class names by object count: {sorted_class_names}")
data_names = {i: [name] for i, name in enumerate(sorted_class_names)}

# Define the data for the YAML file
data = {
    # Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

    # COCO128 dataset https://www.kaggle.com/datasets/ultralytics/coco128 (first 128 images from COCO train2017) by Ultralytics
    # Example usage: python train.py --data coco128.yaml
    # parent
    # ├── yolov5
    # └── datasets
    #     └── coco128  ← downloads here (7 MB)
    "path": str(base_dir / "coco_dataset" / f"fold{fold}_CV"),  # Dataset root directory (cast to str if needed by YAML)
    # "path": "../coco_dataset",  # Dataset root directory
    "train": "train/images",  # Train images (relative to 'path')
    "val": "val/images",      # Validation images (relative to 'path')
    "test": None,             # Test images (optional)
    "names": data_names,  # Class names dictionary
}

# Define the output path for the YAML config file
yaml_file_path = base_dir / "configs" / "data.yaml"

# Always overwrite the YAML file
print("Updating file, creating a new one for synchronization with k-fold Cross Validation 🖥️...")

with open(yaml_file_path, "w") as yaml_file:
    yaml.dump(data, yaml_file, default_flow_style=False)
# Print the path to the saved YAML file
# Optionally, print the contents of the YAML file
with open(yaml_file_path, "r") as yaml_file:
    print("✅ YAML file overwritten, contents:")
    print(yaml_file.read())
# Print the path to the saved YAML file
print(f"YAML file saved to: {yaml_file_path}")