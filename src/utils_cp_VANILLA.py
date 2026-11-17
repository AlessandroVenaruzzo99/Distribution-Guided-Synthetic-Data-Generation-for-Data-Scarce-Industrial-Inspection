from utils import *  

################################################
# load params and info about image and boxes
################################################

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"🛑 Image not found: {path} 🗂️")
    return img

def load_bboxes(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    return [list(map(float, line.strip().split())) for line in lines]

# Transform normalized bounding box to pixel coordinates
def normalized_to_pixel_bbox(bbox, img_w, img_h):
    cls, cx, cy, w, h = bbox
    cx, cy = int(cx * img_w), int(cy * img_h)
    bw, bh = int(w * img_w), int(h * img_h)
    x1 = max(cx - bw//2, 0)
    y1 = max(cy - bh//2, 0)
    x2 = min(cx + bw//2, img_w - 1)
    y2 = min(cy + bh//2, img_h - 1)
    return int(cls), x1, y1, x2, y2

################################################
# detecting feeder's borders
################################################

def detect_table_border_light(image, morph_kernel_size=15, min_contour_area=10000, debug_path=None):
    """Detects the table border in an image using contour detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    debug_img = image.copy()
    cv2.drawContours(debug_img, contours, -1, (0,255,0), 2)

    if debug_path:
        cv2.imwrite(debug_path, debug_img)

    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > min_contour_area:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(float)
                s = pts.sum(axis=1)
                diff = np.diff(pts, axis=1).ravel()
                TL = pts[np.argmin(s)]
                BR = pts[np.argmax(s)]
                TR = pts[np.argmin(diff)]
                BL = pts[np.argmax(diff)]
                return np.array([TL, TR, BR, BL], dtype=np.float32)

    # Fallback: use image borders as polygon
    print(f"⚠️ Table border not found. Using image borders as fallback with margin=5.")
    return get_image_border_as_polygon(image, margin=5)

def get_image_border_as_polygon(image, margin=5):
    """
    Returns the image borders as a polygon (with optional margin).
    Use this as fallback when table border detection fails.
    
    Args:
        image: Input image (BGR format)
        margin: Pixels to shrink from edges (default 5) to avoid edge artifacts
    
    Returns:
        np.array of shape (4, 2) with corners [TL, TR, BR, BL]
    """
    h, w = image.shape[:2]
    TL = [margin, margin]
    TR = [w - margin - 1, margin]
    BR = [w - margin - 1, h - margin - 1]
    BL = [margin, h - margin - 1]
    return np.array([TL, TR, BR, BL], dtype=np.float32)


################################################
# Overlap and placement utilities
################################################

def rects_overlap_ratio(r1, r2):
    """Compute IoU of two rectangles (x1, y1, x2, y2)."""
    x_left = max(r1[0], r2[0])
    y_top = max(r1[1], r2[1])
    x_right = min(r1[2], r2[2])
    y_bottom = min(r1[3], r2[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    r1_area = (r1[2] - r1[0]) * (r1[3] - r1[1])
    r2_area = (r2[2] - r2[0]) * (r2[3] - r2[1])
    union_area = r1_area + r2_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0.0

def all_pts_in_poly(rect, poly_pts):
    """Check if all corners of a rectangle are inside a polygon."""
    x1, y1, x2, y2 = rect
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return all(cv2.pointPolygonTest(poly_pts, pt, False) >= 0 for pt in corners)


################################################
# Feeder cleanliness detection
################################################

def is_feeder_clean_with_labels(image_path, label_path,
                                white_threshold=200,
                                dust_tolerance=0.975,
                                morph_kernel_size=5):
    """
    Detects if feeder is clean (perfectly white) or dusty, excluding areas with objects (from YOLO labels).
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    
    # Step 1: Detect feeder polygon
    for ks in [15, 11, 21, 31, 41]:
        corners = detect_table_border_light(img, morph_kernel_size=ks, min_contour_area=10000)
        if corners is not None:
            break
    
    if corners is None or len(corners) != 4:
        print(f"[WARN] Could not detect feeder in {image_path}")
        return None, None

    # Step 2: Create feeder mask
    feeder_mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.fillPoly(feeder_mask, [corners.astype(int)], 255)

    # Step 3: Create object mask from YOLO labels
    obj_mask = np.zeros_like(gray, dtype=np.uint8)
    if os.path.exists(label_path):
        bboxes = load_bboxes(label_path)
        for bbox in bboxes:
            cls, x1, y1, x2, y2 = normalized_to_pixel_bbox(bbox, w, h)
            cv2.rectangle(obj_mask, (x1, y1), (x2, y2), 255, -1)

    # Step 4: Remove objects from feeder mask
    dust_check_mask = cv2.bitwise_and(feeder_mask, cv2.bitwise_not(obj_mask))

    # Step 5: Threshold to find bright pixels
    _, bright_mask = cv2.threshold(gray, white_threshold, 255, cv2.THRESH_BINARY)

    # Step 6: Morphological open to clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)

    # Step 7: Calculate white ratio in dust-check area
    white_pixels = np.sum(bright_mask[dust_check_mask > 0] > 0)
    total_pixels = np.sum(dust_check_mask > 0)
    white_ratio = white_pixels / total_pixels if total_pixels > 0 else 0

    return white_ratio >= dust_tolerance, white_ratio


################################################
# Object mask extraction
################################################

def extract_object_mask(obj_crop, is_clean=True, preserve_holes=True):
    """
    Extract a binary mask for the object in the cropped image.
    
    Args:
        obj_crop: Cropped object image (BGR format)
        is_clean: Whether the feeder background is clean (white) or dusty
        preserve_holes: Whether to preserve holes in the object mask
    
    Returns:
        Binary mask (same size as obj_crop, 255=object, 0=background)
    """
    blurred = cv2.GaussianBlur(obj_crop, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    
    if is_clean:
        # Clean feeder: use combined thresholding approach
        morph_close_iter = 1
        morph_open_iter = 2
        
        _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary_adaptive = cv2.adaptiveThreshold(gray, 255,
                                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, 11, 2)
        edges = cv2.Canny(blurred, 30, 100)
        
        # Combine Otsu, adaptive, and edge-based masks
        combined = cv2.bitwise_or(binary_otsu, binary_adaptive)
        combined = cv2.bitwise_or(combined, edges)
        
        # Morphology to connect broken parts
        kernel = np.ones((3, 3), np.uint8)
        m = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=morph_close_iter)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=morph_open_iter)
        
        # Find contours with hierarchy
        contours, hierarchy = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if contours is None or hierarchy is None:
            return np.ones_like(gray) * 255  # Fallback: use entire crop
        
        # Build final mask
        final_mask = np.zeros_like(m)
        for idx, hier in enumerate(hierarchy[0]):
            parent = hier[3]
            if parent < 0:
                # External contour → fill it
                cv2.drawContours(final_mask, contours, idx, 255, -1)
            elif not preserve_holes:
                # Hole → erase it only if not preserving holes
                cv2.drawContours(final_mask, contours, idx, 0, -1)
    else:
        # Dusty feeder: use adaptive thresholding
        morph_close_iter = 1
        
        binary_adaptive = cv2.adaptiveThreshold(gray, 255,
                                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, 11, 2)
        
        kernel = np.ones((3, 3), np.uint8)
        closed_mask = cv2.morphologyEx(binary_adaptive, cv2.MORPH_CLOSE, kernel, iterations=morph_close_iter)
        
        contours, hierarchy = cv2.findContours(closed_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours or hierarchy is None:
            return np.ones_like(gray) * 255  # Fallback
        
        # Find the largest contour (main object)
        contour_areas = [cv2.contourArea(c) for c in contours]
        main_contour_idx = np.argmax(contour_areas)
        
        final_mask = np.zeros_like(gray)
        cv2.drawContours(final_mask, contours, main_contour_idx, 255, -1)
        
        # Preserve holes if requested
        if preserve_holes and hierarchy is not None:
            for i, hier in enumerate(hierarchy[0]):
                if hier[3] == main_contour_idx:
                    cv2.drawContours(final_mask, contours, i, 0, -1)
    
    return final_mask


################################################
# Masked pasting function
################################################

def paste_object_masked(target_image, obj_img, mask, x, y):
    """
    Paste only object pixels (where mask > 0) from obj_img into target_image at position (x, y).
    
    Args:
        target_image: Target image (will be modified in place)
        obj_img: Object image to paste
        mask: Binary mask (same size as obj_img, >0 = object pixels)
        x: Top-left x coordinate for pasting
        y: Top-left y coordinate for pasting
    
    Returns:
        Actual bounding box of pasted region (x1, y1, x2, y2)
    """
    h_target, w_target = target_image.shape[:2]
    h_obj, w_obj = obj_img.shape[:2]
    
    # Calculate paste region with boundary clipping
    paste_x1 = max(0, x)
    paste_y1 = max(0, y)
    paste_x2 = min(w_target, x + w_obj)
    paste_y2 = min(h_target, y + h_obj)
    
    # Calculate corresponding region in object crop
    crop_x1 = paste_x1 - x
    crop_y1 = paste_y1 - y
    crop_x2 = crop_x1 + (paste_x2 - paste_x1)
    crop_y2 = crop_y1 + (paste_y2 - paste_y1)
    
    # Get ROI from target and crop regions
    roi = target_image[paste_y1:paste_y2, paste_x1:paste_x2]
    crop_region = obj_img[crop_y1:crop_y2, crop_x1:crop_x2]
    mask_region = mask[crop_y1:crop_y2, crop_x1:crop_x2]
    
    # Prepare mask for broadcasting (handle 3-channel images)
    mask_bool = (mask_region > 0)
    if crop_region.ndim == 3 and mask_bool.ndim == 2:
        mask_bool = mask_bool[:, :, np.newaxis]
    
    # Copy only masked pixels
    np.copyto(roi, crop_region, where=mask_bool)
    target_image[paste_y1:paste_y2, paste_x1:paste_x2] = roi
    
    return (paste_x1, paste_y1, paste_x2, paste_y2)


################################################
# Crop objects from image with masks
################################################

def crop_objects_with_masks(image, bboxes, is_clean=True, preserve_holes=True):
    """
    Crop objects from image and extract their masks.
    
    Args:
        image: Source image (BGR format)
        bboxes: List of normalized bboxes [class, cx, cy, w, h]
        is_clean: Whether the feeder is clean
        preserve_holes: Whether to preserve holes in masks
    
    Returns:
        List of tuples: (class_id, cropped_image, mask, pixel_bbox)
    """
    h, w = image.shape[:2]
    cropped_objects = []
    
    for bbox in bboxes:
        cls, x1, y1, x2, y2 = normalized_to_pixel_bbox(bbox, w, h)
        crop = image[y1:y2, x1:x2].copy()
        
        if crop.size > 0:
            mask = extract_object_mask(crop, is_clean=is_clean, preserve_holes=preserve_holes)
            cropped_objects.append((cls, crop, mask, (x1, y1, x2, y2)))
    
    return cropped_objects


################################################
# Vanilla Copy-Paste with masked pasting
################################################

def paste_objects_vanilla_masked(
    image, 
    corners, 
    cropped_objects, 
    existing_bboxes,
    n_copies=1,
    overlap_threshold=0.3,
    max_trials=100
):
    """
    Paste cropped objects onto the original image n_copies times each using masked pasting.
    
    Args:
        image: Target image (will be modified in place)
        corners: Polygon corners defining valid paste area
        cropped_objects: List of (class_id, crop_img, mask, original_bbox)
        existing_bboxes: List of existing bboxes as (x1, y1, x2, y2) to avoid
        n_copies: Number of times to paste each object
        overlap_threshold: Max allowed IoU with existing objects
        max_trials: Max placement attempts per object
    
    Returns:
        List of new bboxes as (class_id, x1, y1, x2, y2)
    """
    h_img, w_img = image.shape[:2]
    poly = corners.astype(np.int32).reshape(-1, 1, 2)
    
    # Track all placed rectangles (existing + new)
    placed_rects = list(existing_bboxes)
    new_bboxes = []
    
    for cls, obj_crop, obj_mask, original_bbox in cropped_objects:
        oh, ow = obj_crop.shape[:2]
        
        for copy_idx in range(n_copies):
            # Optionally flip the object and mask
            obj_to_paste = obj_crop.copy()
            mask_to_paste = obj_mask.copy()
            
            if random.random() < 0.5:
                obj_to_paste = cv2.flip(obj_to_paste, 1)
                mask_to_paste = cv2.flip(mask_to_paste, 1)
            
            placed = False
            for _ in range(max_trials):
                # Random position within image bounds
                x = random.randint(0, max(0, w_img - ow))
                y = random.randint(0, max(0, h_img - oh))
                R = (x, y, x + ow, y + oh)
                
                # Check if within valid polygon area
                if not all_pts_in_poly(R, poly):
                    continue
                
                # Check overlap with all existing rectangles
                max_overlap = max(
                    (rects_overlap_ratio(R, existing) for existing in placed_rects),
                    default=0.0
                )
                if max_overlap > overlap_threshold:
                    continue
                
                # Paste the object using masked pasting
                actual_bbox = paste_object_masked(image, obj_to_paste, mask_to_paste, x, y)
                
                placed_rects.append(R)
                new_bboxes.append((cls, actual_bbox[0], actual_bbox[1], actual_bbox[2], actual_bbox[3]))
                placed = True
                break
            
            if not placed:
                print(f"⚠️ Could not place copy {copy_idx+1}/{n_copies} of class {cls} after {max_trials} tries")
    
    return new_bboxes


def save_augmented_image_and_labels(image, original_bboxes, new_bboxes, output_img_path, output_txt_path):
    """
    Save augmented image and combined labels (original + new).
    
    Args:
        image: Augmented image
        original_bboxes: List of original bboxes as (cls, x1, y1, x2, y2)
        new_bboxes: List of new bboxes as (cls, x1, y1, x2, y2)
        output_img_path: Output image path
        output_txt_path: Output label path
    """
    h_img, w_img = image.shape[:2]
    
    cv2.imwrite(output_img_path, image)
    
    all_bboxes = list(original_bboxes) + list(new_bboxes)
    
    with open(output_txt_path, 'w') as f:
        for cls, x1, y1, x2, y2 in all_bboxes:
            cx = (x1 + x2) / (2 * w_img)
            cy = (y1 + y2) / (2 * h_img)
            w = (x2 - x1) / w_img
            h = (y2 - y1) / h_img
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


################################################
# Main Vanilla Pipeline Functions
################################################
def process_single_image_vanilla(
    image_path, 
    bbox_path, 
    output_dir, 
    n_copies, 
    overlap_threshold,
    max_trials,
    preserve_holes,
    aug_idx= 1
):
    """
    Process a single image with vanilla copy-paste augmentation using masked pasting.
    
    Args:
        image_path: Path to source image
        bbox_path: Path to label file
        output_dir: Output directory (images and labels will be saved directly here)
        n_copies: Number of copies per object
        overlap_threshold: Max allowed IoU overlap
        max_trials: Max placement attempts
        preserve_holes: Whether to preserve holes in object masks
        aug_idx: Augmentation index for naming
    
    Returns:
        Tuple of (output_img_path, output_txt_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    img = load_image(image_path)
    h, w = img.shape[:2]
    bboxes = load_bboxes(bbox_path)
    
    # Detect table border
    for ks in [15, 11, 21, 31, 41]:
        corners = detect_table_border_light(img, morph_kernel_size=ks, min_contour_area=10000)
        if corners is not None:
            break
    
    if corners is None:
        corners = get_image_border_as_polygon(img, margin=5)
    
    # Detect if feeder is clean or dusty
    is_clean, _ = is_feeder_clean_with_labels(image_path, bbox_path)
    if is_clean is None:
        is_clean = True  # Default to clean if detection fails
    
    # Crop objects with masks from original image
    cropped_objects = crop_objects_with_masks(img, bboxes, is_clean=is_clean, preserve_holes=preserve_holes)
    
    # Get existing bboxes in pixel coordinates
    existing_bboxes_pixel = []
    original_bboxes_with_class = []
    for bbox in bboxes:
        cls, x1, y1, x2, y2 = normalized_to_pixel_bbox(bbox, w, h)
        existing_bboxes_pixel.append((x1, y1, x2, y2))
        original_bboxes_with_class.append((cls, x1, y1, x2, y2))
    
    # Make a copy of the image for augmentation
    aug_img = img.copy()
    
    # Paste objects using masked pasting
    new_bboxes = paste_objects_vanilla_masked(
        aug_img,
        corners,
        cropped_objects,
        existing_bboxes_pixel,
        n_copies=n_copies,
        overlap_threshold=overlap_threshold,
        max_trials=max_trials
    )
    
    # Save results directly in output_dir (no subfolder creation here)
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    
    output_img_path = os.path.join(output_dir, f"{img_name}_run{aug_idx}.jpg")
    output_txt_path = os.path.join(output_dir, f"{img_name}_run{aug_idx}.txt")
    
    save_augmented_image_and_labels(
        aug_img,
        original_bboxes_with_class,
        new_bboxes,
        output_img_path,
        output_txt_path
    )
    
    return output_img_path, output_txt_path

def run_vanilla_pipeline(
    raw_data_dir, 
    labels_dir, 
    output_dir, 
    n_copies, 
    overlap_threshold,
    num_augmentations,
    preserve_holes,
    max_trials=100
):
    """
    Run vanilla copy-paste pipeline for all images in a directory.
    
    Args:
        raw_data_dir: Directory containing source images
        labels_dir: Directory containing label files
        output_dir: Output directory
        n_copies: Number of copies per object
        overlap_threshold: Max allowed IoU overlap
        num_augmentations: Number of augmented versions per image
        preserve_holes: Whether to preserve holes in object masks
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for img_file in os.listdir(raw_data_dir):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        img_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(raw_data_dir, img_file)
        label_path = os.path.join(labels_dir, img_name + ".txt")
        
        if not os.path.exists(label_path):
            print(f"⚠️ Label missing for {img_name}, skipping...")
            continue
        
        for aug_idx in range(1, num_augmentations + 1):
            # Create a run folder to group augmentations (required by collection script)
            # Structure: .../CLASS/run_{aug_idx}/{img_name}_aug{aug_idx}/
            run_dir = os.path.join(output_dir, f"run_{aug_idx}")
            output_subdir = os.path.join(run_dir, f"{img_name}_run{aug_idx}")
            
            try:
                out_img, out_txt = process_single_image_vanilla(
                    img_path,
                    label_path,
                    output_subdir,
                    n_copies=n_copies,
                    overlap_threshold=overlap_threshold,
                    max_trials = max_trials,
                    preserve_holes=preserve_holes,
                    aug_idx=aug_idx
                )
                print(f"✅ Created: {out_img}")
            except Exception as e:
                print(f"❌ Error processing {img_name}: {e}")

def run_vanilla_pipeline_for_all_classes(
    stored_datset_train_dir, 
    output_dir, 
    n_copies, 
    overlap_threshold,
    num_augmentations,
    preserve_holes
):
    """
    Run vanilla copy-paste pipeline for all classes in the dataset.
    
    Args:
        stored_datset_train_dir: Path to training dataset with class subfolders
        output_dir: Output directory
        n_copies: Number of copies per object
        overlap_threshold: Max allowed IoU overlap
        num_augmentations: Number of augmented versions per image
        preserve_holes: Whether to preserve holes in object masks
    """
    stored_datset_train_dir = Path(stored_datset_train_dir)
    output_dir = Path(output_dir)
    
    if not stored_datset_train_dir.exists():
        print(f"❌ Directory does not exist: {stored_datset_train_dir}")
        return
    
    class_folders = [f for f in stored_datset_train_dir.iterdir() if f.is_dir()]
    
    for class_folder in class_folders:
        class_name = class_folder.name
        images_dir = class_folder / "images"
        labels_dir = class_folder / "labels"
        class_output_dir = output_dir / class_name
        
        if not images_dir.exists():
            print(f"⚠️ Images directory not found for {class_name}, skipping...")
            continue
        
        if class_output_dir.exists() and any(class_output_dir.iterdir()):
            print(f"⚠️ Output directory {class_output_dir} not empty, skipping...")
            continue
        
        print(f"\n📁 Processing class: {class_name}")
        
        run_vanilla_pipeline(
            raw_data_dir=images_dir,
            labels_dir=labels_dir,
            output_dir=class_output_dir,
            n_copies=n_copies,
            overlap_threshold=overlap_threshold,
            num_augmentations=num_augmentations,
            preserve_holes=preserve_holes
        )

        
        print(f"✅ Completed class: {class_name}")
    
    print(f"\n🎉 All classes processed! Output at: {output_dir}")