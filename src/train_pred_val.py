from utils import*

##########################################################################
# YOLOv5 Manual Cross-Validation
###########################################################################
show_imgs = False


# -----------------------------------------------
# Parse the --fold argument from the command line
# -----------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, required=True, help="fold number for cross-validation")
parser.add_argument("--fine_tuning", type=str, required=True, help="Boolean value to set fine_tuning or training from scratch (True/False)")
parser.add_argument("--k_folds", type=int, required=True, help="Total number of folds for k-fold cross-validation")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--image_weights", type=str, default='False', help="Boolean value to set image_weights for training (True/False)")
parser.add_argument("--model", type=str, default="yolov5s", help="Ultralytics detector to train, e.g. yolov5s (main experiments) or yolo11s")
args = parser.parse_args()
fold = args.fold  # This is the fold number passed from your main loop
fine_tuning = args.fine_tuning.lower() == 'true' # Properly parse the fine_tuning argument as a boolean
image_weights = args.image_weights.lower() == 'true' # Properly parse the image_weights argument as a boolean
k_folds = args.k_folds  # Total number of folds
seed = args.seed  # Random seed
print(f"🛠️  Starting YOLOv5 training, prediction and validation for fold {fold}/{k_folds} with fine_tuning={fine_tuning}, image_weights={image_weights}, and seed={seed} 🛠️ ")
# -----------------------------------------------
###########################################################################
# 1. Train for Each fold: Fine Tuning or From Scratch
###########################################################################

# NOTE: Memory ∝ imgsz2×batch
# NOTE: (Training time per epoch) ∝ [(Number of images in dataset)/(batch size)] ×(processing time per batch)
# NOTE: (Processing time per batch)∝ imgsz^2
# Therefore:
# NOTE: (Training time per epoch) ∝ (imgsz^2)/(batch)


if fine_tuning:
    # Fine tuning: load pretrained weights
    model = YOLO(f'{args.model}.pt')  # pretrained weights, e.g. yolov5s.pt / yolo11s.pt
    print(f"🚀 Training YOLOv5 model (fine tuning) for fold {fold}...🚀")
    model.train(
        data=os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "configs", "data.yaml"),
        epochs=20,
        imgsz= 672,
        batch= 8,
        patience=10,
        iou=0.7,
        seed=seed,
        # image_weights=image_weights,
        project=os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "results", f"fold{fold}_CV", "train"),
        name=f"yolo_train_{fold}",
    )
else:
    # Train from scratch: no pretrained weights
    model = YOLO(f'{args.model}.yaml')  # architecture only, e.g. yolov5s.yaml / yolo11s.yaml
    print(f"🚀 Training YOLOv5 model (from scratch) for fold {fold}...🚀")
    model.train(
        data=os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "configs", "data.yaml"),
        epochs=50,
        imgsz=672,
        batch=8,
        patience=10,
        iou=0.7,
        seed=seed,
        # image_weights=image_weights,
        project=os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "results", f"fold{fold}_CV", "train"),
        name=f"yolo_train_{fold}",
    )


###########################################################################
# Post-training: Open confusion matrix (still works with either fold)
###########################################################################
# Variables from your training configuration
train_project = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "results", f"fold{fold}_CV", "train")  # Custom parent folder

# Automatically increment the fold name
train_name = f"yolo_train_{fold}"
print(f"fold name: {train_name} 🗂️")

# Construct the path to the image file
train_image_path = os.path.join(train_project, train_name, 'confusion_matrix_normalized.png')

# Open the image using the appropriate command based on the operating system
import platform

if show_imgs:
    if platform.system() == "Darwin":  # macOS
        subprocess.run(f"open {train_image_path}", shell=True)
    elif platform.system() == "Linux":  # Linux
        subprocess.run(f"xdg-open {train_image_path}", shell=True)
    elif platform.system() == "Windows":  # Windows
        subprocess.run(f"start {train_image_path}", shell=True)
    else:
        print(f"🛑 Unsupported operating system: {platform.system()} 🛑")

############################### Create Plots for Training Results ####################################

# Load results.csv from your YOLOv5 training fold
csv_path = os.path.join(
    os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
    "results", f"fold{fold}_CV", "train", train_name, "results.csv"
)
df = pd.read_csv(csv_path)

# Create output directory
train_plt_project = os.path.join(
    os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
    "results", f"fold{fold}_CV", "train", "epoch_plot"
)
os.makedirs(train_plt_project, exist_ok=True)

# ========== Plot Recall vs Epoch ==========
plt.figure(figsize=(8, 4))
plt.plot(df["epoch"], df["metrics/recall(B)"], label="Recall", color="blue")
plt.xlabel("Epoch")
plt.ylabel("Recall")
plt.title("Recall vs Epoch")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(train_plt_project, "recall_vs_epoch.png"))
print("📈 Saved: Recall vs Epoch")

# ========== Plot Precision vs Epoch ==========
plt.figure(figsize=(8, 4))
plt.plot(df["epoch"], df["metrics/precision(B)"], label="Precision", color="green")
plt.xlabel("Epoch")
plt.ylabel("Precision")
plt.title("Precision vs Epoch")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(train_plt_project, "precision_vs_epoch.png"))
print("📈 Saved: Precision vs Epoch")

# ========= Plot F1 Score vs Epoch (calculated) =========
precision = df["metrics/precision(B)"]
recall = df["metrics/recall(B)"]
f1 = 2 * precision * recall / (precision + recall + 1e-16)

plt.figure(figsize=(8, 4))
plt.plot(df["epoch"], f1, label="F1 Score", color="orange")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.title("F1 Score vs Epoch")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(train_plt_project, "f1_vs_epoch.png"))
print("📈 Saved: F1 Score vs Epoch")

# ========== Plot mAP50-95 vs Epoch ==========
plt.figure(figsize=(8, 4))
plt.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP@0.5:0.95", color="red")
plt.xlabel("Epoch")
plt.ylabel("mAP@0.5:0.95")
plt.title("mAP50-95 vs Epoch")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(train_plt_project, "map50_95_vs_epoch.png"))
print("📈 Saved: mAP50-95 vs Epoch")


###########################################################################
# 2. Prediction (Inference) for Each fold
###########################################################################


# Find the latest yolo_train_i directory
if os.path.exists(train_project):
    weights_path = os.path.join(train_project, train_name, "weights", "best.pt")
else:
    raise FileNotFoundError("🛑 No yolo_train_i directories found in results for inference.")


# Load last trained model, to be used for inference, accordingly to the main fold 
# NOTE: i can't support models bigger than 5.4 GB (yolov5s only supported)
model = YOLO(weights_path)

print(f"🚀 Inference YOLOv5 model for fold {fold}...🚀")

# Inference on validation set
results = model.predict(
    source=os.path.join(
        os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
        "coco_dataset", f"fold{fold}_CV", "val", "images"  # Make sure this folder contains supported image files
    ),
    imgsz=672,
    conf=0.4,  
    seed=seed,
    iou=0.7,
    save=True,
    show_boxes=True, 
    save_txt=True,
    save_conf=True,
    save_crop=True,
    
    show_labels=False,   # Do not show class names
    show_conf=False,     

    project=os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "results", f"fold{fold}_CV", "inference"),
    name=f"inference_results_{fold}",
)


#######################################
# Open obtained results
#######################################

# Automatically increment the fold name
inf_name=f"inference_results_{fold}"
print(f"Inference folder's name: {inf_name} 🗂️")

# Create copy of the inference results with ground truth overlay
val_img_dir = os.path.join(
    os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
    "coco_dataset", f"fold{fold}_CV", "val", "images"
)
val_lbl_dir = os.path.join(
    os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
    "coco_dataset", f"fold{fold}_CV", "val", "labels"
)

inference_results_dir = os.path.join(
    os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
    "results", f"fold{fold}_CV", "inference", inf_name
)

# Create output directory for ground truth overlay images
output_gt_img_dir = os.path.join(
    os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
    "results", f"fold{fold}_CV", "inference", f"inference_results_{fold}", "ground_truth_overlay"
)

draw_ground_truth_boxes(val_img_dir, val_lbl_dir, inference_results_dir, output_gt_img_dir)
print(f"✅ Ground truth overlay images saved to: {output_gt_img_dir}")

########################################################################à
# 3. Validation for Each fold
###########################################################################

# Find the latest yolo_train_i directory
if os.path.exists(train_project):
    weights_path = os.path.join(train_project, train_name, "weights", "best.pt")
else:
    raise FileNotFoundError("🛑 No yolo_train_i directories found in results for validation.")


# Load last trained model, to be used for inference, accordingly to the main fold 
# NOTE: i can't support models bigger than 5.4 GB (yolov5s only supported)
model = YOLO(weights_path)

print(f"🚀 Validation YOLOv5 model for fold {fold}...🚀")

# fold validation (evaluation) on val set
metrics = model.val(
    data=os.path.join(
        os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
        "configs", "data.yaml"
    ),
    split='val',                # Use the 'val' split from data.yaml
    imgsz=672,
    iou=0.7,
    conf=0.001,
    seed=seed,
    save_json=True,             # Save COCO-format predictions.json
    save_hybrid=False,          # Optionally: save labels + predictions
    project=os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))),"results", f"fold{fold}_CV", "validation",
    ),
    name=f"val_results_{fold}"
)

#######################################
# Open obtained results
#######################################

# define a way to automatically 
# go to the path ../../results/val_results_i
# and execute the command 

# Variables from your training configuration
val_project = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "results", f"fold{fold}_CV", "validation")  # Custom parent folder
# Automatically increment the fold name
val_name = f"val_results_{fold}"  # Incremented fold name
print(f"Validation folder's name: {val_name} 🗂️")

# Construct the path to the image file
val_image_path = os.path.join(val_project, val_name, 'confusion_matrix_normalized.png')

# Open the image using the appropriate command based on the operating system
import platform

if show_imgs:
    if platform.system() == "Darwin":  # macOS
        subprocess.run(f"open {val_image_path}", shell=True)
    elif platform.system() == "Linux":  # Linux
        subprocess.run(f"xdg-open {val_image_path}", shell=True)
    elif platform.system() == "Windows":  # Windows
        subprocess.run(f"start {val_image_path}", shell=True)
    else:
        print(f"🛑 Unsupported operating system: {platform.system()} 🛑")


################################## Evaluation Metrics ##################################
# Print evaluation metrics for the fold

print(f"🏁 Evaluation Metrics for fold: {fold} 🏁")


##################### Save detailed metrics ######################


#################### Class specific Evaluation Metrics ####################
# Calculate F1 score, precision, and recall
precision = metrics.box.p
recall = metrics.box.r
f1 = metrics.box.f1 # The 1e-16 is added to the denominator to prevent a potential division by zero error.
ap50 = metrics.box.ap50  # Average Precision at IoU=0.50
ap = metrics.box.ap  # Average Precision across IoU thresholds

# Print evaluation metrics
print(f"Precision evaluation Metrics for fold {precision} 🟢 ")  # Precision
print(f"Recall evaluation Metrics for fold {recall} 🟢 ")     # Recall
print(f"F1 evaluation Metrics for fold {f1} 🟢 ")        # F1 score
print(f"Average Precision at IoU=0.50 {ap50} 🟢 ")  # Average Precision at IoU=0.5  
print(f"Average Precision across IoU[0.5:0.95] thresholds {ap} 🟢 ")  # Average Precision across IoU thresholds


##################### Save on csv file ######################
# Save single class metrics to CSV
cls_csv_path = os.path.join(
    os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
    "results", "class_metrics.csv"
)
header = ["folds","cl_ap", "cl_ar", "cl_af1", "cl_ap[50]", "cl_ap_IoU[50-95]"]
values = [f"fold {fold}",precision, recall, f1, ap50, ap]

if not os.path.exists(cls_csv_path):
    # Create the CSV file and write the header
    with open(cls_csv_path, mode="w+", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
with open(cls_csv_path, mode="a+", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(values)

print(f" ✅ Saved class metrics to {cls_csv_path} 📂")


###################### Mean Evaluation Metrics ######################
# Calculate mean precision, recall, and F1 score across all classes
mean_precision = metrics.box.mp
mean_recall = metrics.box.mr
mean_f1 = 2 * mean_precision * mean_recall / (mean_precision + mean_recall + 1e-16)
mean_ap50 = metrics.box.map50  # Mean Average Precision at IoU=0.50
mean_ap = metrics.box.map  # Mean Average Precision across IoU thresholds
# Print evaluation metrics
print(f"Mean Precision evaluation across all classes  {mean_precision} ✅ " ) # Precision
print(f"Mean Recall evaluation across all classes  {mean_recall} ✅ ")  # Recall
print(f"Mean F1 evaluation across all classes {mean_f1} ✅ ")      # F1 score 
print(f"Mean Average Precision at IoU=0.50 {mean_ap50} ✅ ")  # Mean Average Precision at IoU=0.50
print(f"Mean Average Precision across IoU[0.5:0.95] thresholds {mean_ap} ✅ ")  # Mean Average Precision across IoU thresholds

##################### Save on csv file ######################

# Compute total object box count for the fold (average per epoch)
box_count_file = os.path.join(train_project, train_name, "yolo_box_count.txt")
total_boxes_all_epochs = 0
if os.path.exists(box_count_file):
    with open(box_count_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.isdigit():
                total_boxes_all_epochs += int(line)
else:
    print(f"⚠️ Warning: {box_count_file} not found.")

# The df (from results.csv loaded earlier) tells us how many epochs actually completed
epochs_completed = len(df) if 'df' in locals() and not df.empty else 1
avg_objects_per_epoch = int(total_boxes_all_epochs / epochs_completed)

print(f"📦 Total objects seen over {epochs_completed} epochs: {total_boxes_all_epochs}")
print(f"📊 Average objects per epoch for fold {fold}: {avg_objects_per_epoch}")

# Save mean metrics to CSV
mn_csv_path = os.path.join(
    os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
    "results", "mean_metrics.csv"
)
header = ["folds","mp", "mr", "mf1", "map_IoU[50]", "map_IoU[50-95]", "avg_objects_per_epoch"]
values = [f"fold {fold}", mean_precision, mean_recall, mean_f1, mean_ap50, mean_ap, avg_objects_per_epoch]

if not os.path.exists(mn_csv_path):
    # Create the CSV file and write the header
    with open(mn_csv_path, mode="w+", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
with open(mn_csv_path, mode="a+", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(values)

print(f" ✅ Saved mean metrics to {mn_csv_path} 📂")