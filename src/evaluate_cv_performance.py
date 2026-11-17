from utils import *

#################################################################################
# Create the coco_dataset directory if it doesn't exist
base_dir = Path(__file__).resolve().parent.parent

target_file1 = base_dir / "results" / "mean_metrics.csv"
calculate_and_append_mean(target_file1)
target_file2 = base_dir / "results" / "class_metrics.csv"
calculate_class_means(target_file2)
