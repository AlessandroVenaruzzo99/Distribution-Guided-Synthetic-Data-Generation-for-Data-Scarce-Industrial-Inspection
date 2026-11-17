########################################################################################

# =======================================
# Bounding Box Creation and Labeling
# =======================================

# The labeling process is based on the YOLOv5 format, which requires bounding box coordinates in a specific format.
# The bounding box coordinates are normalized to the image dimensions, and the labels are saved in a text file with the same name as the image.
# The labeling process involves the following steps:
# 1. Preprocess the image (grayscale, Gaussian blur, edge detection).
# 2. Detect feeder borders using Hough Lines.
# 3. Segment objects using thresholding and morphological operations.
# 4. Generate YOLOv5 labels from contours using class-specific labeling parameters.

from utils import * 
from pathlib import Path  # Import Pathlib for path handling

# ===================================================================
# # Aim to develop a big dataset which particularie on the objects and allow to store big number of classes
# # for reach a enormous class' dataset and enable FOUNDATION MODEL creation
# ===================================================================

# Before of create a new class i should check if that class already exist or not
# in a dataset where i store all the classes and labeling specifications for each class
#-------> for now 1 step at the time, just manage a folder with different classes (classification and ad hoc labeling)


# ==============================================================================================================================
# =======================================
# Class Creation to store the images and labels
# =======================================

class ImageClfLab:
    def __init__(self, dataset_dir, store_dir, config_dir):
        self.dataset_dir = dataset_dir
        self.store_dir = store_dir
        self.config_dir = config_dir  # Store config_dir as an instance variable
        self.class_ids = {}
        self.labeling_parameters = {}
        self._initialize_classes()
        self._prepare_store_dir(self.config_dir)  # Pass config_dir to _prepare_store_dir()

    def _initialize_classes(self):
        # Check if dataset_dir contains subfolders
        if not os.path.exists(self.dataset_dir) or not os.listdir(self.dataset_dir):
            print("⚠️ Clustering definition: work in progress... You still need to provide folders containing different classes of objects and its own Yaml file for each class⚠️.")
            print("Please create subfolders in the dataset directory for each class.")
            print("🛑 Exiting... classification not possible🛑.")
            return
        
        # Assign a class_id to each subfolder
        folders = [f for f in os.listdir(self.dataset_dir) if os.path.isdir(os.path.join(self.dataset_dir, f))]
        # Check if the class already exists
        # If not, create a new class
        self.class_ids = {folder: idx for idx, folder in enumerate(folders)}

        # Print class assignments
        print(f"Found {len(folders)} folders in the dataset directory 🗂️.")
        print("Class IDs assigned to each folder:")
        for folder, class_id in self.class_ids.items():
            print(f"Folder: {folder}🗂️, Class ID: {class_id} 🖥️...")

    def _prepare_store_dir(self, config_dir):
        # Default labeling parameters
        for class_name, class_id in self.class_ids.items():  # Include class_id in the loop

            class_dir = self.store_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)  # Ensure the class directory exists

            # Copy files from Raw_data_v01 to the class directory
            class_source_dir = self.dataset_dir / class_name
            if class_source_dir.exists():
                for file_name in class_source_dir.iterdir():
                    dest_file = class_dir / file_name.name
                    if file_name.is_file() and not dest_file.exists():
                        shutil.copy(file_name, dest_file)

            # Check for labeling parameters file in configs directory
            config_file = config_dir / f"{class_name}_labeling_parameters.yaml"
            params_file = class_dir / f"{class_name}_labeling_parameters.yaml"

            if config_file.exists():
                print(f"⚠️Found existing YAML config for class '{class_name}' in configs directory ✅. Copying to class directory🗂️.")
                shutil.copy(config_file, params_file)
                with open(params_file, "r") as f:
                    self.labeling_parameters[class_name] = yaml.safe_load(f)
            elif params_file.exists():
                with open(params_file, "r") as f:
                    self.labeling_parameters[class_name] = yaml.safe_load(f)
                self.labeling_parameters[class_name]["class_id"] = class_id
            else:
                # Notify user and create default YAML file
                print(f"⚠️ Labeling parameters file not found for class '{class_name}'. Creating a default file📄.")
                default_params = {
                    "class_name": class_name,  # Include class name explicitly
                    "class_id": class_id,      # Add class_id to the parameters
                    "min_area": 1800,
                    "max_area": 7500,
                    "min_aspect_ratio": 0.001,
                    "max_aspect_ratio": 2.4,
                    "w_eps": 0.9,
                    "h_eps": 0.9
                    }
                with open(params_file, "w") as f:
                    yaml.dump(default_params, f, default_flow_style=False)
                self.labeling_parameters[class_name] = default_params
                print(f"Default labeling parameters file created for class '{class_name}'📝 at {params_file} 🗂️.")
                
                # Open the YAML file in Visual Studio Code and wait until it is closed
                try:
                    subprocess.run(["code", "--wait", str(params_file)], check=True)
                except FileNotFoundError:
                    print("🛑 Visual Studio Code is not installed or not available in PATH.🛑")


# ==============================================================================================================================


# =======================================
# Bounding Box Creation and Labeling
# =======================================

# The labeling process is based on the YOLOv5 format, which requires bounding box coordinates in a specific format.
# The bounding box coordinates are normalized to the image dimensions, and the labels are saved in a text file with the same name as the image.
# The labeling process involves the following steps:
# 1. Preprocess the image (grayscale, Gaussian blur, edge detection).
# 2. Detect feeder borders using Hough Lines.
# 3. Segment objects using thresholding and morphological operations.
# 4. Generate YOLOv5 labels from contours using class-specific labeling parameters.


# 
def preprocess_image(image_path):
    """Load and preprocess the image."""
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image_gray is None:
        raise IOError(f"Could not read image from {image_path}.")
    
    # Reduce noise using Gaussian Blur
    image_gray = cv2.GaussianBlur(image_gray, (5, 5), 0.5)
    
    # Apply edge detection
    edges = cv2.Canny(image_gray, 30, 150)
    return image_gray, edges

def detect_feeder_border(edges, image_gray):
    """Detect feeder borders using Hough Lines and create a mask."""
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=200, minLineLength=300, maxLineGap=10)
    mask = np.zeros_like(image_gray)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(mask, (x1, y1), (x2, y2), 255, 5)  # Draw detected lines
    
    # Use the mask to remove the feeder's border
    feeder_area = cv2.bitwise_and(image_gray, image_gray, mask=cv2.bitwise_not(mask))
    return feeder_area

def segment_objects(feeder_area):
    """Segment objects using thresholding and morphological operations."""
    _, thresh = cv2.threshold(feeder_area, 80, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=3)
    thresh = cv2.erode(thresh, kernel, iterations=2)
    return thresh


def generate_yolo_labels(contours, width_img, height_img, class_id, labeling_params):
    """Generate YOLOv5 labels from contours using class-specific labeling parameters."""
    yolo_labels = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        aspect_ratio = w / float(h)
        
        # Retrieve class-specific parameters
        min_area = labeling_params["min_area"]
        max_area = labeling_params["max_area"]
        min_aspect_ratio = labeling_params["min_aspect_ratio"]
        max_aspect_ratio = labeling_params["max_aspect_ratio"]
        w_eps = labeling_params["w_eps"] # epsilon for width 
        h_eps = labeling_params["h_eps"] # epsilon for height

        # Calculate the area and aspect ratio
        if (min_area < area < max_area and
            w < w_eps * width_img and
            h < h_eps * height_img and
            min_aspect_ratio < aspect_ratio < max_aspect_ratio):
            
            # Convert to YOLO format
            x_center = (x + w / 2) / width_img
            y_center = (y + h / 2) / height_img
            w_norm = w / width_img
            h_norm = h / height_img
            
            label_str = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
            yolo_labels.append(label_str)
    return yolo_labels

def load_labeling_params(class_name, image_clf_lab):
    """Load labeling parameters from the ImageClfLab instance for the given class."""
    if class_name not in image_clf_lab.labeling_parameters:
        raise ValueError(f"🛑 Labeling parameters for class '{class_name}' not found in ImageClfLab 📝.")
    print(f"Loaded labeling parameters for class '{class_name}'	📝: {image_clf_lab.labeling_parameters[class_name]}")
    return image_clf_lab.labeling_parameters[class_name]

def process_image(image_path, labels_output_dir, class_name, save_visualization, image_clf_lab):
    """Process a single image and save YOLO labels."""
    # Load labeling parameters from the ImageClfLab instance
    labeling_params = load_labeling_params(class_name, image_clf_lab)
    print(f"Processing image: {image_path}	🖼️  with class: {class_name}")
    class_id = labeling_params["class_id"]
    # Preprocess the image
    image_gray, edges = preprocess_image(image_path)                                    # Gaussian Blur and Canny Edge Detection
    height_img, width_img = image_gray.shape[:2]                                        # Get image dimensions needed to found out relatives coord of BB(bounding box)
    feeder_area = detect_feeder_border(edges, image_gray)                               # Hough Line Detection
    thresh = segment_objects(feeder_area)                                               # Segment objects using thresholding and morphological operations
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
    
    # Generate YOLO labels
    # return object class ID and bounding box coordinates
    yolo_labels = generate_yolo_labels(contours, width_img, height_img, class_id, labeling_params) 
    
    # Save YOLO labels to file in a class-specific subdirectory
    class_label_dir = os.path.join(labels_output_dir, class_name)  # Create subdirectory for the class
    os.makedirs(class_label_dir, exist_ok=True)                   # Ensure the subdirectory exists
    print(f"Saving YOLO labels to: {class_label_dir}🗃️")
    base_name = os.path.splitext(os.path.basename(image_path))[0] # Extract the image name (discard extension)
    label_filename = f"{base_name}.txt"                           # Create a txt file with the extracted name
    label_file_path = os.path.join(class_label_dir, label_filename)  # Add the file to the class-specific subdirectory
    with open(label_file_path, "w") as f:                         # Opens a file at label_file_path in write mode to write the YOLO labels.
        for label_line in yolo_labels:                            # Save iteratively each bounding box coords taken
            f.write(label_line + "\n")
    print(f"YOLO labels saved to: {label_file_path}")                

    # Optional visualization, save all image with their bounding box 
    if save_visualization:
        img_color = cv2.imread(image_path)
        if img_color is not None:
            for label in yolo_labels:
                _, x_center, y_center, w_norm, h_norm = map(float, label.split())
                x = int((x_center - w_norm / 2) * width_img)
                y = int((y_center - h_norm / 2) * height_img)
                w = int(w_norm * width_img)
                h = int(h_norm * height_img)
                cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Save the visualization to disk
            output_dir = os.path.join(os.path.dirname(labels_output_dir), "visualizations")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{base_name}_bbox.png")
            cv2.imwrite(output_path, img_color)
            print(f"Visualization saved to: {output_path} 🗂️")
    
    # Return the YOLO labels for object count
    return yolo_labels


# processes all images in a dataset directory, leveraging the ImageClfLab class to handle class-specific parameters for bounding box creation and labeling
def process_dataset_new(dataset_dir, labels_output_dir, image_clf_lab, save_visualization):
    """Process all images in a dataset directory using ImageClfLab for class-specific parameters."""
    os.makedirs(labels_output_dir, exist_ok=True) # Creates the specified output directory if it does not already exist, only the labels_yolov5 dir.

    # Iterates over each class (identified by class_name), which is assumed to correspond to a subfolder within the dataset directory. 
    for class_name, class_id in image_clf_lab.class_ids.items():
        class_dir = os.path.join(dataset_dir, class_name) # Constructs the full path to the current class's folder (class_dir) 

        # fetches the class-specific parameters
        labeling_params = image_clf_lab.labeling_parameters[class_name]
        # Iterates over all files in the class directory.
        for filename in os.listdir(class_dir):
            print(f"Processing file: {filename}📄")
            print(f"Class dir: {class_dir}")
            print(f"Class name: {class_name}")
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(class_dir, filename)
                process_image(image_path, labels_output_dir, class_name, save_visualization, image_clf_lab)

# ==============================================================================================================================