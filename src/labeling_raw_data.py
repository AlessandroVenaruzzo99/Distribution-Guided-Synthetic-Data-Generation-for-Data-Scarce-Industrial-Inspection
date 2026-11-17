##################################################
#  Bounding Box Creation
##################################################
# This script is designed to process a dataset of images, classify them, and create bounding boxes around detected objects.
# It uses the OpenCV library for image processing and the YOLOv5 format for labeling.


# Techniques of Annotation =================================================================================================================

# Now, assuming you've chosen a type of annotation and format, it's time to establish clear and objective LABELING RULESaa.
# These rules are like a roadmap for consistency and accuracy throughout the annotation process.
 
#  Key aspects of these RULES include:

#    - Clarity and Detail: Make sure your instructions are clear. Use examples and illustrations to understand what's expected.
#    - Consistency: Keep your annotations uniform. Set standard criteria for annotating different types of data, so all annotations follow the same rules.
#    - Reducing Bias: Stay neutral. Train yourself to be objective and minimize personal biases to ensure fair annotations.
#    - Efficiency: Work smarter, not harder. Use tools and workflows that automate repetitive tasks, making the annotation process faster and more efficient.
#    - Regularly reviewing and updating your labeling rules will help keep your annotations accurate, consistent, and aligned with your project goals.

# =====================================================================================================================




##################################################

from utils_labeling import *
# ================
#Param
save_visualization = True  # If True, we show bounding boxes in a window or in a plot
# ================

# * Problem Detected 
# The resize operation give an image which is too poor to apply Canny Edge Detection.

# =======================================
# Configuration
# =======================================
# Get the base directory (two levels up from this file)
base_dir = Path(__file__).resolve().parent.parent

# Construct the desired paths
raw_dir = base_dir / "data" / "images" / "raw"
# Expect exactly one folder inside raw_dir
raw_subfolders = [f for f in raw_dir.iterdir() if f.is_dir()]
if len(raw_subfolders) != 1:
    raise FileNotFoundError(f"Expected exactly one subfolder in {raw_dir}, found {len(raw_subfolders)}")
dataset_dir = raw_subfolders[0]

# Define the directory to store class-specific data and labeling parameters
store_dir = base_dir / "data" / "images" / "stored"
os.makedirs(store_dir, exist_ok=True)
# Configuration directory
config_dir = base_dir / "configs" / "Labeling_yaml_config" 

# Classify all the images in the raw dataset directory
image_clf_lab = ImageClfLab(dataset_dir, store_dir, config_dir)


# =======================================
# Process the dataset
# =======================================
# Create output directory if it doesn't exist
# define the directory to store the images and labels

class_dir = base_dir / "data" / "images" / "stored" 
class_label_dir = base_dir / "data" /"labels" /"stored"
save_visualization = True  # If True, we show bounding boxes in a window or in a plot

if not class_label_dir.exists() or (class_label_dir.exists() and not any(class_label_dir.iterdir())):
    print(f"Processing dataset from: {class_dir} to: {class_label_dir}")
    process_dataset_new(class_dir, class_label_dir, image_clf_lab, save_visualization)
else:
    print(f"Folder '{class_label_dir}' already exists and is not empty, skipping operations.")

# =====================================================================================================================
# Results Structure

# For each image, you’ll end up with a corresponding .txt file in labels_yolov5/.
# Each file contains one line per detected object, in YOLOv5 format.

# The YOLO format *.txt file specifications are precise:

# - One row per object bounding box.
# - Each row must contain: class_index x_center y_center width height.
# - Coordinates must be normalized to a range between 0 and 1. To achieve this, divide the pixel values of x_center and width by the image's total width, and divide y_center and height by the image's total height.
# - Class indices are zero-indexed (i.e., the first class is represented by 0, the second by 1, and so forth).

# Example lines (assuming 3 objects found):

# - 0 0.123456 0.234567 0.012345 0.045678  (bounding box 1 )
# - 0 0.223456 0.334567 0.032145 0.025678  ...
# - 0 0.423456 0.534567 0.082345 0.065432  ...    

# VISUALIZATIONS folder at the same level of labels_yolov5/ will contain images with bounding boxes drawn on them.
