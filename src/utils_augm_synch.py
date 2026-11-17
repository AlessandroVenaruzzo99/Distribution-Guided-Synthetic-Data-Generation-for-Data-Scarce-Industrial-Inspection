from utils import * 

################################################
# load params and info about image and boxes
################################################

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"🛑 Image not found: {path} 🗂️")
    return img


################################################
# Adaptive Overlap Threshold Computation
################################################

def compute_overlap_between_bboxes(bboxes, img_w=1.0, img_h=1.0):
    """
    Compute the maximum IoU overlap between any pair of bounding boxes in an image.
    Uses the same rects_overlap_ratio logic used during pasting.
    
    Args:
        bboxes: List of normalized bounding boxes [class, center_x, center_y, width, height]
        img_w: Image width (default 1.0 for normalized coordinates)
        img_h: Image height (default 1.0 for normalized coordinates)
    
    Returns:
        max_overlap: Maximum IoU between any pair of boxes (0.0 if < 2 boxes)
        all_overlaps: List of all pairwise IoU values
    """
    if len(bboxes) < 2:
        return 0.0, []
    
    # Convert all bboxes from YOLO format (cx, cy, w, h) to corner format (x1, y1, x2, y2)
    # NO clamping - work with raw normalized coordinates
    pixel_bboxes = []
    for bbox in bboxes:
        cls, cx, cy, w, h = bbox
        cx_px, cy_px = cx * img_w, cy * img_h
        bw, bh = w * img_w, h * img_h
        x1 = cx_px - bw / 2
        y1 = cy_px - bh / 2
        x2 = cx_px + bw / 2
        y2 = cy_px + bh / 2
        pixel_bboxes.append((x1, y1, x2, y2))
    
    # Compute all pairwise IoU values using the same logic as rects_overlap_ratio
    all_overlaps = []
    max_overlap = 0.0
    
    for i in range(len(pixel_bboxes)):
        for j in range(i + 1, len(pixel_bboxes)):
            r1, r2 = pixel_bboxes[i], pixel_bboxes[j]
            
            # Compute IoU (same as rects_overlap_ratio)
            x_left = max(r1[0], r2[0])
            y_top = max(r1[1], r2[1])
            x_right = min(r1[2], r2[2])
            y_bottom = min(r1[3], r2[3])
            
            if x_right <= x_left or y_bottom <= y_top:
                iou = 0.0  # No overlap
            else:
                intersection_area = (x_right - x_left) * (y_bottom - y_top)
                r1_area = (r1[2] - r1[0]) * (r1[3] - r1[1])
                r2_area = (r2[2] - r2[0]) * (r2[3] - r2[1])
                union_area = r1_area + r2_area - intersection_area
                iou = intersection_area / union_area if union_area > 0 else 0.0
            
            all_overlaps.append(iou)
            if iou > max_overlap:
                max_overlap = iou
    
    return max_overlap, all_overlaps


def compute_class_overlap_thresholds(stored_images_dir, stored_labels_dir, default_threshold=0.3, percentile=95):
    """
    Compute adaptive overlap thresholds for each class based on real object overlaps in stored data.
    
    For each class, analyzes all images to find the maximum overlap ratio among objects
    and returns a dictionary mapping class names to their computed overlap thresholds.
    
    Args:
        stored_images_dir: Path to stored images directory (e.g., data/images/stored)
        stored_labels_dir: Path to stored labels directory (e.g., data/labels/stored)
        default_threshold: Default threshold to use if no overlaps are found (default: 0.3)
        percentile: Percentile of max overlaps to use as threshold (default: 95)
                   Using percentile instead of absolute max to avoid outliers
    
    Returns:
        dict: {class_name: overlap_threshold} for each class
        dict: {class_name: overlap_stats} with detailed statistics
    """
    stored_images_dir = Path(stored_images_dir)
    stored_labels_dir = Path(stored_labels_dir)
    
    class_thresholds = {}
    class_stats = {}
    
    # Get all class folders
    class_folders = [f for f in stored_images_dir.iterdir() if f.is_dir()]
    
    print("\n" + "="*70)
    print("🔍 COMPUTING ADAPTIVE OVERLAP THRESHOLDS FROM STORED DATA")
    print("="*70)
    
    for class_folder in class_folders:
        class_name = class_folder.name
        images_dir = class_folder
        labels_dir = stored_labels_dir / class_name
        
        if not labels_dir.exists():
            print(f"⚠️  Labels directory not found for class '{class_name}': {labels_dir}")
            class_thresholds[class_name] = default_threshold
            class_stats[class_name] = {'status': 'no_labels_dir', 'threshold': default_threshold}
            continue
        
        # Collect all max overlaps for this class
        max_overlaps_per_image = []
        all_overlaps_flat = []
        images_with_overlap = 0
        total_images = 0
        
        # Iterate through all images in the class folder
        for img_file in images_dir.iterdir():
            if not img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                continue
            
            total_images += 1
            img_name = img_file.stem
            label_path = labels_dir / f"{img_name}.txt"
            
            if not label_path.exists():
                continue
            
            # Load bboxes and compute overlaps
            try:
                bboxes = load_bboxes(str(label_path))
                if len(bboxes) < 2:
                    continue  # Need at least 2 objects to compute overlap
                
                # Get image dimensions (use standard dimensions or load image)
                # For efficiency, we can use normalized coords directly
                # Since IoU is scale-invariant, we can use normalized coords with w=h=1
                img_w, img_h = 1.0, 1.0  # Use normalized coordinates
                
                max_overlap, overlaps = compute_overlap_between_bboxes(bboxes, img_w, img_h)
                
                if max_overlap > 0:
                    max_overlaps_per_image.append(max_overlap)
                    all_overlaps_flat.extend(overlaps)
                    images_with_overlap += 1
                    
            except Exception as e:
                print(f"  ⚠️  Error processing {img_file.name}: {e}")
                continue
        
        # Compute threshold for this class
        if max_overlaps_per_image:
            # Use percentile of max overlaps to avoid outliers
            threshold = np.percentile(max_overlaps_per_image, percentile)
            # Add small margin (10%) to allow similar overlaps during augmentation
            threshold = min(threshold * 1.1, 0.95)  # Cap at 0.95 to avoid complete overlaps
            
            stats = {
                'status': 'computed',
                'total_images': total_images,
                'images_with_overlap': images_with_overlap,
                'max_overlap_found': max(max_overlaps_per_image),
                'mean_max_overlap': np.mean(max_overlaps_per_image),
                'percentile_used': percentile,
                f'p{percentile}_max_overlap': np.percentile(max_overlaps_per_image, percentile),
                'threshold': threshold,
                'num_overlap_pairs': len(all_overlaps_flat)
            }
        else:
            threshold = default_threshold
            stats = {
                'status': 'no_overlaps_found',
                'total_images': total_images,
                'images_with_overlap': 0,
                'threshold': default_threshold,
                'note': 'Using default threshold (no overlapping objects found in dataset)'
            }
        
        class_thresholds[class_name] = threshold
        class_stats[class_name] = stats
        
        # Print summary for this class
        print(f"\n📦 Class: {class_name}")
        print(f"   📊 Total images: {total_images}")
        print(f"   🔗 Images with overlapping objects: {stats.get('images_with_overlap', 0)}")
        if stats['status'] == 'computed':
            print(f"   📈 Max overlap found: {stats['max_overlap_found']:.4f}")
            print(f"   📊 Mean of max overlaps: {stats['mean_max_overlap']:.4f}")
            print(f"   📐 {percentile}th percentile: {stats[f'p{percentile}_max_overlap']:.4f}")
        print(f"   ✅ Computed threshold: {threshold:.4f}")
    
    print("\n" + "="*70)
    print("📋 OVERLAP THRESHOLD SUMMARY")
    print("="*70)
    for cls, thresh in class_thresholds.items():
        status = class_stats[cls]['status']
        print(f"   {cls}: {thresh:.4f} ({status})")
    print("="*70 + "\n")
    
    return class_thresholds, class_stats


def save_overlap_thresholds(class_thresholds, class_stats, output_path):
    """
    Save computed overlap thresholds to a JSON file for future reference.
    
    Args:
        class_thresholds: Dictionary of class -> threshold
        class_stats: Dictionary of class -> detailed statistics
        output_path: Path to save the JSON file
    """
    import json
    
    output_data = {
        'thresholds': class_thresholds,
        'statistics': {}
    }
    
    # Convert numpy types to native Python types for JSON serialization
    for cls, stats in class_stats.items():
        output_data['statistics'][cls] = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in stats.items()
        }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"💾 Overlap thresholds saved to: {output_path}")


def load_overlap_thresholds(json_path):
    """
    Load previously computed overlap thresholds from a JSON file.
    
    Args:
        json_path: Path to the JSON file
    
    Returns:
        dict: {class_name: overlap_threshold}
        dict: {class_name: statistics} or None if not available
    """
    import json
    
    json_path = Path(json_path)
    if not json_path.exists():
        return None, None
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return data.get('thresholds', {}), data.get('statistics', None)

def load_bboxes(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    return [list(map(float, line.strip().split())) for line in lines]

# Transform normalized bounding box to pixel coordinates
# bbox format: [class, center_x, center_y, width, height]
# Returns: [class, x1, y1, x2, y2]
# where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner
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
# detecting feeder's borders and delete objs inside itself
################################################
################################################################################################
def detect_table_border_light(image, morph_kernel_size=15, min_contour_area=10000, debug_path=None):
    """Detects the table border in an image using contour detection.
    Returns (corners, fallback_used) tuple where fallback_used is True if image borders were used.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Save debug image if path provided
    if debug_path:
        debug_img = image.copy()
        cv2.drawContours(debug_img, contours, -1, (0,255,0), 2)
        cv2.imwrite(debug_path, debug_img)

    # Detection logic - runs regardless of debug_path
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
                return np.array([TL, TR, BR, BL], dtype=np.float32), False  # corners found, no fallback

    # Fallback: use image borders as polygon when table border detection fails
    print(f"⚠️ Table border not found. Using image borders as fallback with margin=5.")
    return get_image_border_as_polygon(image, margin=5), True  # fallback used

# def detect_table_border_light(image, morph_kernel_size=15, min_contour_area=10000, debug_path=None):
#     """Detects the table border in an image using contour detection."""
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5,5), 0)
#     _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     kern = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Save debug image if path provided
#     if debug_path:
#         debug_img = image.copy()
#         cv2.drawContours(debug_img, contours, -1, (0,255,0), 2)
#         cv2.imwrite(debug_path, debug_img)

#     # Detection logic - runs regardless of debug_path
#     if contours:
#         c = max(contours, key=cv2.contourArea)
#         if cv2.contourArea(c) > min_contour_area:
#             peri = cv2.arcLength(c, True)
#             approx = cv2.approxPolyDP(c, 0.02 * peri, True)
#             if len(approx) == 4:
#                 pts = approx.reshape(4, 2).astype(float)
#                 s = pts.sum(axis=1)
#                 diff = np.diff(pts, axis=1).ravel()
#                 TL = pts[np.argmin(s)]
#                 BR = pts[np.argmax(s)]
#                 TR = pts[np.argmin(diff)]
#                 BL = pts[np.argmax(diff)]
#                 return np.array([TL, TR, BR, BL], dtype=np.float32)

#     # Fallback: use image borders as polygon when table border detection fails
#     print(f"⚠️ Table border not found. Using image borders as fallback with margin=5.")
#     return get_image_border_as_polygon(image, margin=5)


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



#################################################################################################
############################## Clear objects inside the detected table border ######################################################
def clear_objects_inside(image, corners, object_thresh=100, inpaint_radius=4, min_obj_area=150):
    """Clears objects inside the detected table border by inpainting."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    poly_mask = np.zeros_like(gray)
    cv2.fillPoly(poly_mask, [corners.reshape(-1,1,2).astype(np.int32)], 255)
    _, binary = cv2.threshold(gray, object_thresh, 255, cv2.THRESH_BINARY_INV)
    binary &= poly_mask
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    bin_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kern)
    nlab, labs, stats, _ = cv2.connectedComponentsWithStats(bin_close, 8)
    obj_mask = np.zeros_like(binary)
    for lbl in range(1, nlab):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_obj_area:
            obj_mask[labs == lbl] = 255
    obj_mask = cv2.dilate(obj_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    return cv2.inpaint(image, obj_mask, inpaint_radius, cv2.INPAINT_TELEA)


############################# Clear objects by bbox ######################################################
def clear_objects_by_bbox(image, bboxes, inpaint_radius=11, dilate_kernel_size=11):
    """
    Clears objects based on ground truth bounding boxes by inpainting only bbox regions.
    
    Args:
        image: Input image (BGR format)
        bboxes: List of normalized bounding boxes [class, center_x, center_y, width, height]
        inpaint_radius: Radius for inpainting
        dilate_kernel_size: Size of dilation kernel to slightly expand bbox regions
        debug_path: Optional path to save the object mask for debugging
    
    Returns:
        Image with objects cleared using inpainting
    """
    h, w = image.shape[:2]
    obj_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Create mask from ground truth bounding boxes
    for bbox in bboxes:
        cls, x1, y1, x2, y2 = normalized_to_pixel_bbox(bbox, w, h)
        cv2.rectangle(obj_mask, (x1, y1), (x2, y2), 255, -1)
    
    # Optional: Dilate the mask slightly to ensure complete object removal
    if dilate_kernel_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size))
        obj_mask = cv2.dilate(obj_mask, kernel)

    
    # Inpaint only the masked regions
    # return cv2.inpaint(image, obj_mask, inpaint_radius, cv2.INPAINT_TELEA) #INPAINT_NS
    return cv2.inpaint(image, obj_mask, inpaint_radius, cv2.INPAINT_NS)

###################################################################
# Detect Dusty Feeder and Clean Feeder 
###################################################################
def is_feeder_clean_with_labels(image_path, label_path,
                                white_threshold=198,
                                dust_tolerance=0.985,
                                morph_kernel_size=5):
    """
    Detects if feeder is clean (perfectly white) or dusty, excluding areas with objects (from YOLO labels).
    If no polygon borders are detected (fallback used), defaults to is_clean = True.
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    debug_path = image_path.replace('.jpg', '_debug.jpg')
    
    # Step 1: Detect feeder polygon
    fallback_used = False
    for ks in [15, 11, 21, 31, 41]:
        try:
            corners, fallback_used = detect_table_border_light(img, morph_kernel_size=ks, min_contour_area=10000)
            break
        except RuntimeError:
            continue
    else:
        # All attempts failed, use fallback
        print(f"⚠️ All kernel sizes failed for {image_path}. Defaulting to is_clean=True.")
        return True, 1.0

    if corners is None or len(corners) != 4:
        print(f"[WARN] Could not detect feeder in {image_path}. Defaulting to is_clean=True.")
        return True, 1.0

    # If fallback was used (image borders instead of detected polygon), default to clean
    if fallback_used:
        print(f"⚠️ Fallback used for {image_path}. Defaulting to is_clean=True (no polygon detected).")
        return True, 1.0

    # Step 2: Create feeder mask
    feeder_mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.fillPoly(feeder_mask, [corners.astype(int)], 255)

    # Step 3: Create object mask from YOLO labels
    obj_mask = np.zeros_like(gray, dtype=np.uint8)
    num_bboxes = 0
    if os.path.exists(label_path):
        bboxes = load_bboxes(label_path)
        num_bboxes = len(bboxes)
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

    # DEBUG: Print detailed info
    print(f"    🔬 DEBUG: white_threshold={white_threshold}, dust_tolerance={dust_tolerance}")
    print(f"    🔬 DEBUG: total_pixels={total_pixels}, white_pixels={white_pixels}, num_bboxes={num_bboxes}")
    print(f"    🔬 DEBUG: white_ratio={white_ratio:.4f}, needs >= {dust_tolerance} to be CLEAN")

    return white_ratio >= dust_tolerance, white_ratio



###############################  CROP MIX AUGMENTATION  #################################################################
# Check for overlaps and pasting cropped objects inside the detected table area
################################################################################################

def rects_overlap(r1, r2):
    """Check if two rectangles overlap."""
    return not (r1[2] <= r2[0] or r1[0] >= r2[2] or r1[3] <= r2[1] or r1[1] >= r2[3])

def rects_overlap_ratio(r1, r2):
    """
    Compute the intersection-over-union (IoU) of two rectangles.
    Each rectangle is (x1, y1, x2, y2).
    Returns a float between 0 and 1.
    """
    x_left = max(r1[0], r2[0])
    y_top = max(r1[1], r2[1])
    x_right = min(r1[2], r2[2])
    y_bottom = min(r1[3], r2[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0  # No overlap

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

################################# Process image for Table border detection and object cropping ########################

def process_single_image(image_path, bbox_path, output_dir):
    """Process a single image: crop objects, detect table border, clear objects inside, and save results."""
    os.makedirs(output_dir, exist_ok=True)
    img = load_image(image_path)
    h, w = img.shape[:2]
    bboxes = load_bboxes(bbox_path)

    # 1. Save cropped objects & bounding boxes
    for idx, bbox in enumerate(bboxes):
        cls, x1, y1, x2, y2 = normalized_to_pixel_bbox(bbox, w, h)
        # Crop dell'intera bounding box                                
        crop = img[y1:y2, x1:x2]
        cv2.imwrite(os.path.join(output_dir, f'object_{idx:02d}.jpg'), crop)
        with open(os.path.join(output_dir, f'object_{idx:02d}.txt'), 'w') as f:
            f.write(f"{cls} {x1} {y1} {x2} {y2}\n")

    # 2. Detect table border with retries
    debug_path = os.path.join(output_dir, "debug_contours.jpg")
    for ks in [15, 11, 21, 31, 41]:
        try:
            corners, _ = detect_table_border_light(img, morph_kernel_size=ks, min_contour_area=10000, debug_path=debug_path)
            break
        except RuntimeError:
            continue
    else:
        raise RuntimeError(f"🛑 Table border not found in {image_path}🗂️. Debug image saved at {debug_path}🗂️")

    # Save apexes
    with open(os.path.join(output_dir, 'Polygon_apexes.txt'), 'w') as f:
        for (x, y) in corners:
            f.write(f"{x:.2f} {y:.2f}\n")

    # 3. Clear objects inside table & save erased image
    clean_img = clear_objects_inside(img, corners)
    erased_img_path = os.path.join(output_dir, 'Erased_img.jpg')
    cv2.imwrite(erased_img_path, clean_img)

    return erased_img_path, corners


###################################### Process image for Table border detection and object cropping BB ########################
def process_single_image_bb(image_path, bbox_path, output_dir):
    """Process a single image: crop objects, detect table border, clear objects by bounding boxes, and save results."""
    os.makedirs(output_dir, exist_ok=True)
    img = load_image(image_path)
    h, w = img.shape[:2]
    bboxes = load_bboxes(bbox_path)

    # 1. Save cropped objects & bounding boxes
    for idx, bbox in enumerate(bboxes):
        cls, x1, y1, x2, y2 = normalized_to_pixel_bbox(bbox, w, h)
        # Crop dell'intera bounding box                                
        crop = img[y1:y2, x1:x2]
        cv2.imwrite(os.path.join(output_dir, f'object_{idx:02d}.jpg'), crop)
        with open(os.path.join(output_dir, f'object_{idx:02d}.txt'), 'w') as f:
            f.write(f"{cls} {x1} {y1} {x2} {y2}\n")

    # 2. Detect table border with retries
    debug_path = os.path.join(output_dir, "debug_contours.jpg")
    for ks in [15, 11, 21, 31, 41]:
        try:
            corners, _ = detect_table_border_light(img, morph_kernel_size=ks, min_contour_area=10000, debug_path=debug_path)            
            break
        except RuntimeError:
            continue
    else:
        raise RuntimeError(f"🛑 Table border not found in {image_path}🗂️. Debug image saved at {debug_path}🗂️")

    # 3. Clear objects by bounding boxes & save erased image
    clean_img = clear_objects_by_bbox(img, bboxes)
    erased_img_path = os.path.join(output_dir, 'Erased_img.jpg')
    cv2.imwrite(erased_img_path, clean_img)

    return erased_img_path, corners

##############################################################################################################
################### Crop Mix pasting objects  ########################
##############################################################################################################
def paste_objects_inside_table_crop_mix(erased_img_path, corners, objects_folder, output_img_path, output_txt_path, max_trials=100):
    """Paste objects inside the detected table area in the erased image."""
    img = cv2.imread(erased_img_path)
    h_img, w_img = img.shape[:2]
    poly = corners.astype(np.int32).reshape(-1, 1, 2)

    placed_rects = []
    new_bboxes = []

    for fn in sorted(os.listdir(objects_folder)):
        if not fn.startswith("object_") or not fn.lower().endswith(".jpg"):
            continue

        base = os.path.splitext(fn)[0]
        txt_path = os.path.join(objects_folder, base + ".txt")
        # actual bb coordinates, class, top left corner, bottom right corner
        cls, x0, y0, x1, y1 = map(int, open(txt_path).read().split())
        obj = cv2.imread(os.path.join(objects_folder, fn), cv2.IMREAD_UNCHANGED)

        # Generates a random float between 0 and 1, and with a 50% probability, the following block is executed.
        # If the condition is met, cv2.flip(obj, 1) is called, which flips the image obj horizontally (left-right).
        if random.random() < 0.5:
            obj = cv2.flip(obj, 1)
        # height and width of the object image used to check image bounderies
        oh, ow = obj.shape[:2]
        
        # Try to place the object inside the table area
        for _ in range(max_trials):
            # Generates random x and y coordinates within the image boundaries minus object BB dimensions.
            # Reference frame starts in 0,0, top-left corner of the image. -->x     down-(y>0) 
            x = random.randint(0, w_img - ow)
            y = random.randint(0, h_img - oh)
            R = (x, y, x + ow, y + oh)

            if not all_pts_in_poly(R, poly):
                continue
            if any(rects_overlap(R, existing) for existing in placed_rects):
                continue


            img[y:y+oh, x:x+ow] = obj
            placed_rects.append(R)
            # Class, top left corner, bottom right corner of the object bounding box
            new_bboxes.append((cls, x, y, x+ow, y+oh))
            break
        else:
            print(f"⚠️ Could not place {base} inside table after {max_trials} tries⚠️")

    cv2.imwrite(output_img_path, img)
    with open(output_txt_path, 'w') as f:
        # Here x1,y1 are top-left corner and x2,y2 are bottom-right corner of the bounding box
        for cls, xtl, ytl, xbr, ybr in new_bboxes:
            cx = (xtl + xbr) / (2 * w_img)
            cy = (ytl + ybr) / (2 * h_img)
            w = (xbr - xtl) / w_img
            h = (ybr - ytl) / h_img
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")






# ################################################################################################
# ##################### ARISE AUGMENTATION  #########################
# ################################################################################################
def paste_object_masked(target_image, obj_img, mask, new_x_center, new_y_center):
    """
    Paste only object pixels (where mask==1) from obj_img into target_image at new center.
    Args:
        target_image (np.ndarray): Target image (HWC, BGR or RGB).
        obj_img (np.ndarray): Object image (same size as mask).
        mask (np.ndarray): Binary mask (same size as obj_img, 0=background, 255=object).
        new_x_center (float): Normalized x_center.
        new_y_center (float): Normalized y_center.
    Returns:
        dict: Updated bbox in YOLO format { 'bbox': [x_center, y_center, w, h] }.
    """
    h, w = target_image.shape[:2]
    crop_h, crop_w = obj_img.shape[:2]

    xc = int(new_x_center * w)
    yc = int(new_y_center * h)
    x1_img = max(0, xc - crop_w // 2)
    y1_img = max(0, yc - crop_h // 2)
    x2_img = min(w, x1_img + crop_w)
    y2_img = min(h, y1_img + crop_h)

    x1_crop = 0
    y1_crop = 0
    x2_crop = crop_w - (crop_w - (x2_img - x1_img))
    y2_crop = crop_h - (crop_h - (y2_img - y1_img))

    roi = target_image[y1_img:y2_img, x1_img:x2_img]
    crop_region = obj_img[y1_crop:y2_crop, x1_crop:x2_crop]
    mask_region = mask[y1_crop:y2_crop, x1_crop:x2_crop]

    # Prepare mask for broadcasting
    mask_bool = (mask_region > 0)
    if crop_region.ndim == 3 and mask_bool.ndim == 2:
        mask_bool = mask_bool[:, :, np.newaxis]

    np.copyto(roi, crop_region, where=mask_bool)
    target_image[y1_img:y2_img, x1_img:x2_img] = roi

    pasted_bbox = [
        new_x_center,
        new_y_center,
        (x2_img - x1_img) / w,
        (y2_img - y1_img) / h
    ]
    return {'bbox': pasted_bbox}


def paste_objects_inside_table_Arise(erased_img_path, corners, objects_folder, output_img_path, output_txt_path, overlap_threshold, jit_gauss_augm, preservs_holes, is_clean, max_trials=100): # max trials only for crop mix
    """Paste objects inside the detected tabl
        cls, x1, y1, x2, y2 = normalized_to_pixel_bbox(bbox, w, h)
        # Crop dell'intera bounding box                                
        crop = img[y1:y2, x1:x2]e area in the erased image."""
    OVERLAP_THRESHOLD = overlap_threshold
    JIT_GAUSS_AUGM  = jit_gauss_augm
    PRESEVS_HOLES = preservs_holes 
    IS_CLEAN = is_clean
    morph_close_iter = 1

    # # Print the IS_CLEAN status for this image
    # feeder_status = "CLEAN (white background)" if IS_CLEAN else "DUSTY (noisy background)"
    # print(f"🔍 Feeder status: {feeder_status} | IS_CLEAN = {IS_CLEAN}")

    img = cv2.imread(erased_img_path)
    h_img, w_img = img.shape[:2]
    poly = corners.astype(np.int32).reshape(-1, 1, 2)

    placed_rects = []
    new_bboxes = []

    # Iterate through each object file in the objects folder
    # The objects folder contains cropped objects from the erased image
    # Each object file is named as object_XX.jpg and has a corresponding .txt file
    # The .txt file contains the class and bounding box coordinates of the object
    # The objects are pasted inside the detected table area in the erased image
    for fn in sorted(os.listdir(objects_folder)):
        if not fn.startswith("object_") or not fn.lower().endswith(".jpg"):
            continue

        base = os.path.splitext(fn)[0] # filename without its extension for each object file in the objects_folder.
        txt_path = os.path.join(objects_folder, base + ".txt") # path to the corresponding .txt file
        cls, x0, y0, x1, y1 = map(int, open(txt_path).read().split())
        obj = cv2.imread(os.path.join(objects_folder, fn), cv2.IMREAD_UNCHANGED) # refers to the cropped object image that is being pasted into the erased image.


        ##################### Extract the object image #####################
        # Apply Gaussian blur to reduce noise to the image 
        blurred = cv2.GaussianBlur(obj, (5, 5), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        # 2) Create a robust initial binary mask
        # add option to differentiate between dust feeders and clean feeders
        # Detect clean/dusty feeder using labels

        if IS_CLEAN == True:
            morph_close_iter=1
            morph_open_iter=2
            fill_holes=True
            # If feeder is clean
            _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            binary_adaptive = cv2.adaptiveThreshold(gray, 255,
                                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    cv2.THRESH_BINARY_INV, 11, 2)
            edges = cv2.Canny(blurred, 30, 100)
            # Combine Otsu, adaptive, and edge-based masks
            combined = cv2.bitwise_or(binary_otsu, binary_adaptive)
            combined = cv2.bitwise_or(combined, edges)
            # Morphology to connect broken parts of the object
            kernel = np.ones((3, 3), np.uint8)
            m = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=morph_close_iter)
            m      = cv2.morphologyEx(m,       cv2.MORPH_OPEN,  kernel, iterations=morph_open_iter)

            # 4) Find all contours with hierarchy (so we can detect holes)
            contours, hierarchy = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if contours is None or hierarchy is None:
                raise ValueError("No contours found. Check your pre‐processing.")

            # 5) Build final mask by drawing outer contours filled, and holes optionally cleared
            final_mask = np.zeros_like(m)
            for idx, hier in enumerate(hierarchy[0]):
                parent = hier[3]
                if parent < 0:
                    # this is an external contour → fill it
                    cv2.drawContours(final_mask, contours, idx, 255, -1)
                elif fill_holes:
                    # this is a hole → erase it
                    cv2.drawContours(final_mask, contours, idx,   0, -1)
                # else: skip erasing holes, so holes remain

        else:
            # If feeder is dusty
            binary_adaptive = cv2.adaptiveThreshold(gray, 255,
                                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    cv2.THRESH_BINARY_INV, 11, 2)

            # Morphology to connect broken parts of the object
            kernel = np.ones((3, 3), np.uint8)
            closed_mask = cv2.morphologyEx(binary_adaptive, cv2.MORPH_CLOSE, kernel, iterations=morph_close_iter)


            # 4) Find all contours and their hierarchy
            contours, hierarchy = cv2.findContours(closed_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            if not contours or hierarchy is None:
                return np.zeros_like(gray) # Return empty mask if no contours found

            # 5) Find the largest contour, which we assume is the main object
            contour_areas = [cv2.contourArea(c) for c in contours]
            main_contour_idx = np.argmax(contour_areas)

            # 6) Create the final mask
            # Start by drawing the main object contour as a solid, filled shape.
            final_mask = np.zeros_like(gray)
            cv2.drawContours(final_mask, contours, main_contour_idx, 255, -1)

            # 7) If preserve_holes is True, find and draw the real holes
            if PRESEVS_HOLES:
                # Iterate through all contours
                for i, hier in enumerate(hierarchy[0]):
                    # A contour is a hole if it has a parent, and its parent is the main contour
                    if hier[3] == main_contour_idx:
                        # Draw the hole in black
                        cv2.drawContours(final_mask, contours, i, 0, -1)

        # Convert the final mask to 3 channels if it is single channel
        image_rgb = cv2.cvtColor(obj, cv2.COLOR_BGR2RGB)

        """Apply binary mask to the image, keeping only the foreground."""
        if len(final_mask.shape) == 2:
            final_mask = final_mask[:, :, np.newaxis]
            obtained_mask = cv2.bitwise_and(image_rgb, image_rgb, mask=final_mask.astype(np.uint8))


#################### Paste Cropped Objects Inside the Table Area ####################

        # Apply object augmentation using the augment_object function
        # JIT_GAUSS_AUGM is actually object_augm_options dict passed from caller
        if isinstance(JIT_GAUSS_AUGM, dict):
            # Use the new augment_object function with all augmentation options
            obj, final_mask, scale_factor = augment_object(obj, final_mask.squeeze(), JIT_GAUSS_AUGM)
            # Ensure mask has correct shape after augmentation
            if len(final_mask.shape) == 2:
                final_mask = final_mask[:, :, np.newaxis]
        else:
            # Legacy behavior: simple flip and noise/jitter
            scale_factor = 1.0
            if random.random() < 0.5:
                obj = cv2.flip(obj, 1)
                final_mask = cv2.flip(final_mask, 1)
            if JIT_GAUSS_AUGM and random.random() < 0.5:
                noise = np.random.normal(0, 25, obj.shape).astype(np.float32)
                noisy_obj = obj.astype(np.float32) + noise
                obj = np.clip(noisy_obj, 0, 255).astype(np.uint8)
                alpha = 1.0 + random.uniform(-0.2, 0.2)
                beta = random.uniform(-0.2, 0.2) * 255
                obj = cv2.convertScaleAbs(obj, alpha=alpha, beta=beta)

        # Get object dimensions AFTER augmentation (includes scale changes)
        oh, ow = obj.shape[:2]

        # Ensure scaled object fits within image bounds
        if ow > w_img or oh > h_img:
            # Downscale to fit while preserving aspect ratio
            fit_scale = min(w_img / ow, h_img / oh) * 0.95  # 5% margin
            new_w = int(ow * fit_scale)
            new_h = int(oh * fit_scale)
            if new_w > 0 and new_h > 0:
                obj = cv2.resize(obj, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                final_mask = cv2.resize(final_mask.squeeze(), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                if len(final_mask.shape) == 2:
                    final_mask = final_mask[:, :, np.newaxis]
                oh, ow = obj.shape[:2]
                print(f"⚠️ Object too large after scaling, resized to fit: {new_w}x{new_h}")

        for _ in range(max_trials):
            x = random.randint(0, w_img - ow)
            y = random.randint(0, h_img - oh)
            R = (x, y, x + ow, y + oh)

            if not all_pts_in_poly(R, poly):
                continue
            if any(rects_overlap_ratio(R, existing) > OVERLAP_THRESHOLD for existing in placed_rects):
                continue

            # Calculate normalized center coordinates for the mask
            new_x_center = (x + ow / 2) / w_img
            new_y_center = (y + oh / 2) / h_img

            # Paste only the masked object
            paste_object_masked(img, obj, final_mask.squeeze(), new_x_center, new_y_center)

            placed_rects.append(R)
            new_bboxes.append((cls, x, y, x+ow, y+oh))
            break
        else:
            print(f"⚠️ Could not place {base} inside table after {max_trials} tries⚠️")            


########### Write the new image and bounding boxes to the output paths ############
    # Save the modified image and bounding boxes
    cv2.imwrite(output_img_path, img)
    with open(output_txt_path, 'w') as f:
        for cls, x1, y1, x2, y2 in new_bboxes:
            cx = (x1 + x2) / (2 * w_img)
            cy = (y1 + y2) / (2 * h_img)
            w = (x2 - x1) / w_img
            h = (y2 - y1) / h_img
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        




 
################################################################################################
# Run the entire pipeline of cropping, clearing, and pasting objects
#################################################################################################
######################## Erasing objects inside the detected table area ########################
def erase_cropp_pipeline(raw_data_dir, labels_dir, output_dir, runs_times):
    """Run the entire pipeline for copy-paste augmentation."""
    class_path = raw_data_dir
  
    for img_file in os.listdir(class_path):
        if not img_file.lower().endswith(".jpg"):
            continue

        img_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(class_path, img_file)
        label_path = os.path.join(labels_dir, img_name + ".txt")

        if not os.path.exists(label_path):
            print(f"Label missing: {label_path}")
            continue

        output_image_dir = os.path.join(output_dir, img_name)
        erased_img_path, corners = process_single_image(img_path, label_path, output_image_dir)
        print(f"Erased image 🖼️ saved at: {erased_img_path}🗂️ ✅")

        cropped_img_path = os.path.join(output_image_dir, f"{img_name}_run{runs_times}.jpg")
        cropped_label_path = os.path.join(output_image_dir, f"{img_name}_run{runs_times}.txt")

        print(f"Cropped image 🖼️ saved at: {cropped_img_path}🗂️ ✅")
        print(f"Cropped Label 🏷️ saved at: {cropped_label_path}🗂️ ✅")



def erase_cropp_pipeline_for_all_classes(stored_datset_train_dir, copy_paste_augm_dir):
    """
    For each class in coco_dataset_builder/train/, run the augmentation pipeline and store results in copy_paste_augm.
    Skips classes that already have non-empty output folders.
    """
    coco_builder_train_dir = Path(stored_datset_train_dir)
    copy_paste_augm_dir = Path(copy_paste_augm_dir)
    if not coco_builder_train_dir.exists():
        print(f"❌ Directory does not exist: {coco_builder_train_dir}")
        return
    class_folders = [f for f in coco_builder_train_dir.iterdir() if f.is_dir()]
    
    for class_folder in class_folders:
        class_name = class_folder.name
        raw_data_dir = class_folder / "images"
        labels_dir = class_folder / "labels"
        output_dir_base = copy_paste_augm_dir / class_name

        if output_dir_base.exists() and any(output_dir_base.iterdir()):
            print(f"⚠️  Output directory {output_dir_base} is not empty, it'll make use of the old augmented dataset. Skipping operations ...")
            continue
        else:
            print(f"✅  Output directory {output_dir_base} is empty. Proceeding with the run.")
            os.makedirs(output_dir_base, exist_ok=True)
            output_dir = output_dir_base
            erase_cropp_pipeline(
                raw_data_dir=raw_data_dir,
                labels_dir=labels_dir,
                output_dir=output_dir,
                runs_times=1
            )
            print(f"{class_name} case run done ✅")
        print(f"All runs for {class_name} case completed ✅.")







################################################################################################################################################
#########################  Paste object ########################
################################################################################################################################################
def random_flip_image_and_polygon(image, polygon):
    """Randomly flip image and polygon horizontally or vertically."""
    flip_type = random.choice(['none', 'h', 'v'])
    h, w = image.shape[:2]
    if flip_type == 'none':
        return image, polygon
    elif flip_type == 'h':
        flipped_img = cv2.flip(image, 1)
        flipped_poly = polygon.copy()
        flipped_poly[:, 0] = w - polygon[:, 0] - 1
        return flipped_img, flipped_poly
    elif flip_type == 'v':
        flipped_img = cv2.flip(image, 0)
        flipped_poly = polygon.copy()
        flipped_poly[:, 1] = h - polygon[:, 1] - 1
        return flipped_img, flipped_poly
    
def get_erase_img_poly_txt_selector(erase_cropp_imgs_dir, img_name=None, rnd_background=True):
    """
    Selects (erased_img_path, polygon_txt_path, objects_folder) tuple.
    If rnd_background is True, randomly pick a class and image folder.
    If rnd_background is False, pick the folder corresponding to img_name.
    """
    class_folders = [os.path.join(erase_cropp_imgs_dir, d) for d in os.listdir(erase_cropp_imgs_dir)
                        if os.path.isdir(os.path.join(erase_cropp_imgs_dir, d))]
    if not class_folders:
        print(f"❌ No class folders found in {erase_cropp_imgs_dir} 🗂️")
        return []

    if rnd_background:
        chosen_class = random.choice(class_folders)
        image_folders = [os.path.join(chosen_class, d) for d in os.listdir(chosen_class)
                            if os.path.isdir(os.path.join(chosen_class, d))]
        if not image_folders:
            print(f"❌ No Image folders found in {chosen_class} 🗂️")
            return []
        chosen_image_folder = random.choice(image_folders)
    else:
        # Find the class and image folder corresponding to img_name
        chosen_image_folder = None
        for class_folder in class_folders:
            candidate = os.path.join(class_folder, img_name)
            if os.path.isdir(candidate):
                chosen_image_folder = candidate
                break
        if chosen_image_folder is None:
            print(f"❌ No folder found for image {img_name} in {erase_cropp_imgs_dir} 🗂️")
            return []

    erased_img_path = os.path.join(chosen_image_folder, "Erased_img.jpg")
    polygon_txt_path = os.path.join(chosen_image_folder, "Polygon_apexes.txt")

    if os.path.exists(erased_img_path) and os.path.exists(polygon_txt_path):
        return [(erased_img_path, polygon_txt_path, chosen_image_folder)]
    else:
        print(f"❌ Erased image or polygon file not found in {chosen_image_folder} 🗂️")
        return []










###################################################################################################################################
############################ Loop ext-1 Copy-Paste-Method REGEN ################################################################################################################
###################################################################################################################################
def run_pipeline_cp(stored_dataset, stored_labels, erase_img_dir, class_name, output_dir, runs_times, rnd_bckgrd, arise_augm, overlap_threshold, object_augm_options, preservs_holes, background_augm_options, synch_augmentation=False):
    """
    Run the copy-paste augmentation pipeline using precomputed erased images and polygons.
    Enable or disable randomly selects of an erased image and polygon as background for the sybthetic image.
    Pick the current image name to load all cropped objects in the corresponding folder.
    Paste the cropped objects inside the detected table area in the erased image.
    """
    ARISE_AUGM = arise_augm

    # Default object augmentation options
    if object_augm_options is None:
        object_augm_options = {'enabled': False}

    # Default background augmentation options
    if background_augm_options is None:
        background_augm_options = {
            'flip': True,
            'brightness': False,
            'contrast': False,
            'blur': False,
            'noise': False,
            'rotation': False,
            'color_jitter': False
        }

    if erase_img_dir is None:
        raise ValueError("❌ erase_crop_augm_dir must be provided to use precomputed erased images.")
    
    class_path = stored_dataset
    rand_bckgrd = rnd_bckgrd
    # for loop to iterate through all images in the class folder
    for img_file in os.listdir(class_path):
        if not img_file.lower().endswith(".jpg"):
            continue

        # Get the image name without extension
        img_name = os.path.splitext(img_file)[0]
        
        # Generate synchronized augmentation options if enabled
        if synch_augmentation:
            current_obj_options, current_bg_options = generate_synchronized_augm_options(
                object_augm_options, background_augm_options
            )
        else:
            current_obj_options = object_augm_options
            current_bg_options = background_augm_options
        
        ########################## Check if the image has dust on itself ########################
        image_path = os.path.join(class_path, img_file)
        label_path = os.path.join(stored_labels, img_name + ".txt")
        if not os.path.exists(label_path):
            print(f"❌ Label missing for image: {img_name} in {stored_labels} 🗂️")
            continue

        # Choose which paste function to use
        is_clean, white_ratio = is_feeder_clean_with_labels( image_path, label_path )
        
        # Print clean/dusty detection result
        feeder_status = "CLEAN" if is_clean else "DUSTY"
        print(f"📊 Image: {img_name} | Status: {feeder_status} | White ratio: {white_ratio:.4f}")

        ################## BACKGORUND AUGMENTATION ########################################################## 
        # get erase image path, polygon apexes path and objects folder
        erased_entries = get_erase_img_poly_txt_selector(erase_img_dir,  img_name, rnd_background=rand_bckgrd)
        # get erased image path, polygon apexes path and objects folder
        if not erased_entries:
            print(f"❌ No erased image or polygon found for {img_name} in {erase_img_dir} 🗂️")
            continue
        erased_img_path, polygon_txt_path, chosen_folder = erased_entries[0]

        # Load erased image and polygon for background augmentation
        erased_img = cv2.imread(erased_img_path)
        original_corners = np.loadtxt(polygon_txt_path, dtype=np.float32).reshape(-1, 2)
        
        # Apply background augmentation using augment_background function
        augmented_erased_img, augmented_corners = augment_background(
            erased_img, 
            original_corners, 
            augment_options=current_bg_options
        )
        
        # Save augmented background temporarily
        augmented_erase_img_path = os.path.join(os.path.dirname(erased_img_path), "augmented.jpg")
        cv2.imwrite(augmented_erase_img_path, augmented_erased_img)
        augmented_polygon_txt_path = os.path.join(os.path.dirname(polygon_txt_path), "augmented_polygon.txt")
        np.savetxt(augmented_polygon_txt_path, augmented_corners, fmt='%.2f')
        
        # Update paths to use augmented versions
        erased_img_path = augmented_erase_img_path
        polygon_txt_path = augmented_polygon_txt_path

        # check if background is choosed rnd or not
        print(f"Current image: {img_name} 🖼️. Selected background image: {chosen_folder}🖼️🗂️.")
        
        # Load polygon apexes
        corners = np.loadtxt(polygon_txt_path, dtype=np.float32).reshape(-1, 2)

        # Always use the cropped objects from the current image
        objects_folder = os.path.join(erase_img_dir, class_name, img_name) # objects folder: is the tuple where the cropped objects are stored
        if not os.path.isdir(objects_folder):
            print(f"❌ Objects folder not found for image {img_name}: {objects_folder}")
            continue
        
        ################# PASTE OBJECTS INSIDE THE TABLE AREA ##########################################################
        # Prepare output paths, create output directory if it does not exist
        output_image_dir = os.path.join(output_dir, img_name)
        os.makedirs(output_image_dir, exist_ok=True)
        augmented_img_path = os.path.join(output_image_dir, f"{img_name}_run{runs_times}.jpg")
        augmented_label_path = os.path.join(output_image_dir, f"{img_name}_run{runs_times}.txt")
        
        if ARISE_AUGM:
            paste_objects_inside_table_Arise(erased_img_path, corners, objects_folder, augmented_img_path, augmented_label_path, overlap_threshold, current_obj_options, preservs_holes, is_clean)
        else:
            paste_objects_inside_table_crop_mix(erased_img_path, corners, objects_folder, augmented_img_path, augmented_label_path)

        print(f"Erased image 🖼️ pick from path: {erased_img_path}🗂️")
        print(f"Synthetic image 🖼️ saved at: {augmented_img_path}🗂️ ✅")
        print(f"Label 🏷️ saved at: {augmented_label_path}🗂️ ✅")


############################ Loop ext ################################################################################################################
def run_pipeline_for_all_classes_cp(stored_datset_train_dir, copy_paste_augm_dir, erase_crop_augm_dir, x_runs_times, rnd_bckgrd, arise_augm, overlap_threshold, object_augm_options=None, preservs_holes=None, background_augm_options=None, synch_augmentation=False):
    """
    For each class in stored_datset_train_dir/train/, run the augmentation pipeline and store results in copy_paste_augm.
    Skips classes that already have non-empty output folders.
    
    Args:
        overlap_threshold: Can be either:
            - A single float value (used for all classes)
            - A dictionary {class_name: threshold} for class-specific thresholds
    """
    coco_builder_train_dir = Path(stored_datset_train_dir)
    copy_paste_augm_dir = Path(copy_paste_augm_dir)
    # check if coco_builder_train_dir exists
    if not coco_builder_train_dir.exists():
        print(f"❌ Directory does not exist, can't perform augmentation: {coco_builder_train_dir}")
        exit(1)

    # Handle overlap_threshold: can be dict or float
    use_class_specific_threshold = isinstance(overlap_threshold, dict)
    if use_class_specific_threshold:
        print("🎯 Using CLASS-SPECIFIC overlap thresholds")
    else:
        print(f"🎯 Using GLOBAL overlap threshold: {overlap_threshold}")

    # Iterate through each class folder in coco_builder_train_dir
    class_folders = [f for f in coco_builder_train_dir.iterdir() if f.is_dir()]

    for class_folder in class_folders:
        class_name = class_folder.name
        raw_data_dir = class_folder / "images"
        labels_dir = class_folder / "labels"
        output_dir_base = copy_paste_augm_dir / class_name

        # Get class-specific threshold or use global threshold
        if use_class_specific_threshold:
            class_threshold = overlap_threshold.get(class_name, 0.3)  # Default to 0.3 if class not found
            print(f"   📏 {class_name}: Using overlap threshold = {class_threshold:.4f}")
        else:
            class_threshold = overlap_threshold

        if output_dir_base.exists() and any(output_dir_base.iterdir()):
            print(f"⚠️  Output directory {output_dir_base} is not empty, it'll make use of the old augmented dataset. Skipping operations ...")
            continue
        else:
            print(f"✅  Output directory {output_dir_base} is empty. Proceeding with the runs.")
            os.makedirs(output_dir_base, exist_ok=True)
            for run_idx in range(1, x_runs_times + 1):
                output_dir = output_dir_base / f"run{run_idx}"
                os.makedirs(output_dir, exist_ok=True)
                run_pipeline_cp(
                    stored_dataset= raw_data_dir,
                    stored_labels= labels_dir,
                    erase_img_dir = erase_crop_augm_dir,
                    class_name= class_name,
                    output_dir= output_dir,
                    runs_times= run_idx,
                    rnd_bckgrd= rnd_bckgrd,
                    arise_augm= arise_augm,
                    overlap_threshold= class_threshold,  # Use class-specific threshold
                    object_augm_options= object_augm_options,
                    preservs_holes= preservs_holes,
                    background_augm_options= background_augm_options,
                    synch_augmentation=synch_augmentation
                )
                print(f"{class_name} case run {run_idx} done ✅")
        print(f"All runs for {class_name} case completed ✅.")



def augment_background(image, polygon, augment_options=None):
    """
    Apply multiple augmentations to the background image and update polygon coordinates.
    
    Args:
        image: Background image (BGR format)
        polygon: Polygon corners as numpy array (N, 2)
        augment_options: Dict with augmentation flags and ranges, e.g.:
            {
                'flip': True,           # Random horizontal/vertical flip
                'brightness': True,     # Random brightness adjustment
                'brightness_range': (-30, 30),  # Brightness shift range
                'contrast': True,       # Random contrast adjustment
                'contrast_range': (0.8, 1.2),  # Contrast factor range
                'blur': True,           # Random Gaussian blur
                'blur_kernel': 3,       # Blur kernel size
                'noise': True,          # Random Gaussian noise
                'noise_std': 10,        # Noise standard deviation
                'rotation': True,       # Small random rotation (±5°)
                'rotation_range': (-5, 5),  # Rotation angle range
                'color_jitter': True,   # Random hue/saturation shift
                'hue_range': (-10, 10),  # Hue shift range
                'saturation_range': (0.8, 1.2)  # Saturation shift range
            }
    
    Returns:
        tuple: (augmented_image, augmented_polygon)
    """
    if augment_options is None:
        augment_options = {
            'flip': True,
            'brightness': True,
            'brightness_range': (-30, 30),
            'contrast': True,
            'contrast_range': (0.8, 1.2),
            'blur': False,
            'blur_kernel': 3,
            'noise': False,
            'noise_std': 10,
            'rotation': False,
            'rotation_range': (-5, 5),
            'color_jitter': True,
            'hue_range': (-10, 10),
            'saturation_range': (0.8, 1.2)
        }
    
    aug_img = image.copy()
    aug_poly = polygon.copy()
    h, w = aug_img.shape[:2]
    
    # 1. Random Flip (horizontal or vertical)
    if augment_options.get('flip', True):
        flip_type = random.choice(['none', 'h', 'v'])
        if flip_type == 'h':
            aug_img = cv2.flip(aug_img, 1)
            aug_poly[:, 0] = w - aug_poly[:, 0] - 1
        elif flip_type == 'v':
            aug_img = cv2.flip(aug_img, 0)
            aug_poly[:, 1] = h - aug_poly[:, 1] - 1
    
    # 2. Random Brightness adjustment
    if augment_options.get('brightness', True) and random.random() < 0.5:
        bright_range = augment_options.get('brightness_range', (-30, 30))
        beta = random.uniform(bright_range[0], bright_range[1])
        aug_img = cv2.convertScaleAbs(aug_img, alpha=1.0, beta=beta)
    
    # 3. Random Contrast adjustment
    if augment_options.get('contrast', True) and random.random() < 0.5:
        contrast_range = augment_options.get('contrast_range', (0.8, 1.2))
        alpha = random.uniform(contrast_range[0], contrast_range[1])
        aug_img = cv2.convertScaleAbs(aug_img, alpha=alpha, beta=0)
    
    # 4. Random Gaussian Blur
    if augment_options.get('blur', False) and random.random() < 0.3:
        kernel_size = augment_options.get('blur_kernel', 3)
        if kernel_size % 2 == 0:
            kernel_size += 1
        aug_img = cv2.GaussianBlur(aug_img, (kernel_size, kernel_size), 0)
    
    # 5. Random Gaussian Noise
    if augment_options.get('noise', False) and random.random() < 0.3:
        noise_std = augment_options.get('noise_std', 10)
        noise = np.random.normal(0, noise_std, aug_img.shape).astype(np.float32)
        aug_img = np.clip(aug_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # 6. Small Random Rotation (±5°) - also updates polygon
    if augment_options.get('rotation', False) and random.random() < 0.3:
        rot_range = augment_options.get('rotation_range', (-5, 5))
        angle = random.uniform(rot_range[0], rot_range[1])
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aug_img = cv2.warpAffine(aug_img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        # Transform polygon points
        ones = np.ones((aug_poly.shape[0], 1))
        poly_hom = np.hstack([aug_poly, ones])
        aug_poly = (M @ poly_hom.T).T
    
    # 7. Random Color Jitter (Hue/Saturation)
    if augment_options.get('color_jitter', True) and random.random() < 0.5:
        hsv = cv2.cvtColor(aug_img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hue_range = augment_options.get('hue_range', (-10, 10))
        sat_range = augment_options.get('saturation_range', (0.8, 1.2))
        hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(hue_range[0], hue_range[1])) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(sat_range[0], sat_range[1]), 0, 255)
        aug_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    return aug_img, aug_poly

################################### Improved Augment Object  ####################################################

def augment_object(obj_img, obj_mask=None, augment_options=None):
    """
    Apply augmentations to a cropped object before pasting.
    
    WARNING: Rotation is NOT recommended for axis-aligned bounding box detection
    as it would require recalculating bbox dimensions.
    """
    if augment_options is None:
        augment_options = {'enabled': False}
    
    if not augment_options.get('enabled', False):
        return obj_img, obj_mask, 1.0
    
    aug_img = obj_img.copy()
    aug_mask = obj_mask.copy() if obj_mask is not None else None
    scale_factor = 1.0
    
    # 1. Random Horizontal Flip (SAFE - bbox size unchanged)
    if augment_options.get('flip_horizontal', False) and random.random() < 0.9:
        aug_img = cv2.flip(aug_img, 1)
        if aug_mask is not None:
            aug_mask = cv2.flip(aug_mask, 1)
    
    # 2. Random Vertical Flip (SAFE - bbox size unchanged)
    if augment_options.get('flip_vertical', False) and random.random() < 0.9:
        aug_img = cv2.flip(aug_img, 0)
        if aug_mask is not None:
            aug_mask = cv2.flip(aug_mask, 0)
    
    # 3. Random Scale Jitter (SAFE - bbox scales proportionally)
    if augment_options.get('scale_jitter', False) and random.random() < 1.:
        scale_range = augment_options.get('scale_range', (0.8, 1.2))
        scale_factor = random.uniform(scale_range[0], scale_range[1])
        new_w = int(aug_img.shape[1] * scale_factor)
        new_h = int(aug_img.shape[0] * scale_factor)
        if new_w > 0 and new_h > 0:
            aug_img = cv2.resize(aug_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            if aug_mask is not None:
                aug_mask = cv2.resize(aug_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    # 4. Random Rotation - DISABLED FOR AABB DETECTION
    if augment_options.get('rotation', False):
        print("⚠️ WARNING: Rotation is disabled for axis-aligned bounding box detection.")
        print("   Enable only if using oriented bounding boxes (OBB).")
        # Rotation code commented out to prevent accidental use
        # rot_range = augment_options.get('rotation_range', (-15, 15))
        # angle = random.uniform(rot_range[0], rot_range[1])
        # ... rotation would break AABB labels
    
    # 5. Random Brightness (SAFE - no geometric change)
    if augment_options.get('brightness', False) and random.random() < 0.9:
        bright_range = augment_options.get('brightness_range', (-20, 20))
        beta = random.uniform(bright_range[0], bright_range[1])
        aug_img = cv2.convertScaleAbs(aug_img, alpha=1.0, beta=beta)
    
    # 6. Random Contrast (SAFE - no geometric change)
    if augment_options.get('contrast', False) and random.random() < 0.9:
        contrast_range = augment_options.get('contrast_range', (0.9, 1.1))
        alpha = random.uniform(contrast_range[0], contrast_range[1])
        aug_img = cv2.convertScaleAbs(aug_img, alpha=alpha, beta=0)
    
    # 7. Random Color Jitter (SAFE - no geometric change)
    if augment_options.get('color_jitter', False) and random.random() < 0.9:
        hsv = cv2.cvtColor(aug_img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hue_range = augment_options.get('hue_range', (-10, 10))
        sat_range = augment_options.get('saturation_range', (0.9, 1.1))
        hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(hue_range[0], hue_range[1])) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(sat_range[0], sat_range[1]), 0, 255)
        aug_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # 8. Gaussian Blur (SAFE - no geometric change)
    if augment_options.get('gaussian_blur', False) and random.random() < 0.9:
        kernel_size = augment_options.get('gaussian_blur_kernel', 3)
        if kernel_size % 2 == 0:
            kernel_size += 1
        aug_img = cv2.GaussianBlur(aug_img, (kernel_size, kernel_size), 0)
    
    # 9. Gaussian Noise (SAFE - no geometric change)
    if augment_options.get('noise', False) and random.random() < 0.9:
        noise_std = augment_options.get('noise_std', 5)
        noise = np.random.normal(0, noise_std, aug_img.shape).astype(np.float32)
        aug_img = np.clip(aug_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    return aug_img, aug_mask, scale_factor



###################################################################################################################################
############################ Loop ext-1 Balance-Method ################################################################################################################
###################################################################################################################################

def generate_synchronized_augm_options(base_object_options, base_background_options):
    """
    Generate synchronized augmentation options for coherent object and background augmentation.
    
    Args:
        base_object_options: Base object augmentation options dict
        base_background_options: Base background augmentation options dict
    
    Returns:
        tuple: (synced_object_options, synced_background_options)
    """
    import random
    
    # Generate fixed values for coherence
    brightness_value = random.uniform(-30, 30)
    contrast_value = random.uniform(0.7, 1.3)
    hue_value = random.uniform(-15, 15)
    saturation_value = random.uniform(0.7, 1.3)
    noise_std_value = random.uniform(5, 15)
    blur_kernel_value = random.choice([3, 5])
    rotation_value = random.uniform(-5, 5)
    
    # Set ranges to fixed values for exact coherence
    brightness_range = (brightness_value, brightness_value)
    contrast_range = (contrast_value, contrast_value)
    hue_range = (hue_value, hue_value)
    saturation_range = (saturation_value, saturation_value)
    noise_std = noise_std_value
    blur_kernel = blur_kernel_value
    rotation_range = (rotation_value, rotation_value)
    
    # Copy base options
    obj_opts = base_object_options.copy()
    bg_opts = base_background_options.copy()
    
    # Set shared values
    obj_opts['brightness_range'] = brightness_range
    obj_opts['contrast_range'] = contrast_range
    obj_opts['hue_range'] = hue_range
    obj_opts['saturation_range'] = saturation_range
    obj_opts['noise_std'] = noise_std
    obj_opts['gaussian_blur_kernel'] = blur_kernel
    
    bg_opts['brightness_range'] = brightness_range
    bg_opts['contrast_range'] = contrast_range
    bg_opts['hue_range'] = hue_range
    bg_opts['saturation_range'] = saturation_range
    bg_opts['noise_std'] = noise_std
    bg_opts['blur_kernel'] = blur_kernel
    bg_opts['rotation_range'] = rotation_range
    
    return obj_opts, bg_opts

def run_pipeline_balance_meth(
    stored_dataset, 
    stored_labels, 
    erase_img_dir, 
    class_name, 
    image_copy_allocation,  
    output_dir, 
    rnd_bckgrd, 
    arise_augm, 
    overlap_threshold, 
    object_augm_options,  
    preservs_holes,
    background_augm_options,
    synch_augmentation=False
):
    """
    Run the copy-paste augmentation pipeline using precomputed erased images and polygons.
    
    Args:
        stored_dataset: Path to stored dataset images
        stored_labels: Path to stored labels
        erase_img_dir: Path to erased/cropped images
        class_name: Current class name
        image_copy_allocation: DataFrame with allocation info
        output_dir: Output directory
        rnd_bckgrd: Whether to use random background
        arise_augm: Whether to use ARISE augmentation
        overlap_threshold: Overlap threshold for pasting
        object_augm_options: Dict with object augmentation settings
        preservs_holes: Whether to preserve holes
        background_augm_options: Dict with background augmentation settings
    """
    ARISE_AUGM = arise_augm

    # Default object augmentation options
    if object_augm_options is None:
        object_augm_options = {'enabled': False}

    # Default background augmentation options
    if background_augm_options is None:
        background_augm_options = {
            'flip': True,
            'brightness': True,
            'contrast': True,
            'blur': False,
            'noise': False,
            'rotation': False,
            'color_jitter': True
        }

    if erase_img_dir is None:
        raise ValueError("❌ erase_crop_augm_dir must be provided to use precomputed erased images.")
    
    class_path = stored_dataset
    rand_bckgrd = rnd_bckgrd

    # Filter allocation for current class
    df_class = image_copy_allocation[image_copy_allocation['class_name'] == class_name]
    for _, row in df_class.iterrows():
        img_name = row['image_name']
        num_copies = int(row['num_copies'])

        image_path = find_image_with_extension(class_path, img_name)
        label_path = os.path.join(stored_labels, img_name + ".txt")
        if not os.path.exists(image_path) or not os.path.exists(label_path):
            print(f"❌ Image or label missing for: {img_name} in {class_path} or {stored_labels} 🗂️")
            continue

        is_clean, white_ratio = is_feeder_clean_with_labels(image_path, label_path)

        for run_idx in range(1, num_copies + 1):
            # Generate synchronized augmentation options if enabled
            if synch_augmentation:
                current_obj_options, current_bg_options = generate_synchronized_augm_options(
                    object_augm_options, background_augm_options
                )
            else:
                current_obj_options = object_augm_options
                current_bg_options = background_augm_options
            
            erased_entries = get_erase_img_poly_txt_selector(erase_img_dir, img_name, rnd_background=rand_bckgrd)
            if not erased_entries:
                print(f"❌ No erased image or polygon found for {img_name} in {erase_img_dir} 🗂️")
                continue
            erased_img_path, polygon_txt_path, chosen_folder = erased_entries[0]

            # Load erased image and polygon
            erased_img = cv2.imread(erased_img_path)
            original_corners = np.loadtxt(polygon_txt_path, dtype=np.float32).reshape(-1, 2)
            
            ################## ENHANCED BACKGROUND AUGMENTATION ##################
            augmented_erased_img, augmented_corners = augment_background(
                erased_img, 
                original_corners, 
                augment_options=current_bg_options
            )
            ######################################################################
            
            # Save augmented background temporarily
            augmented_erase_img_path = os.path.join(os.path.dirname(erased_img_path), f"augmented_{run_idx}.jpg")
            cv2.imwrite(augmented_erase_img_path, augmented_erased_img)
            augmented_polygon_txt_path = os.path.join(os.path.dirname(polygon_txt_path), f"augmented_polygon_{run_idx}.txt")
            np.savetxt(augmented_polygon_txt_path, augmented_corners, fmt='%.2f')
            
            erased_img_path = augmented_erase_img_path
            polygon_txt_path = augmented_polygon_txt_path

            print(f"Current image: {img_name} 🖼️. Selected background image: {chosen_folder}🖼️🗂️.")

            corners = np.loadtxt(polygon_txt_path, dtype=np.float32).reshape(-1, 2)
            objects_folder = os.path.join(erase_img_dir, class_name, img_name)
            if not os.path.isdir(objects_folder):
                print(f"❌ Objects folder not found for image {img_name}: {objects_folder}")
                continue

            output_image_dir = os.path.join(output_dir, f"{img_name}_run{run_idx}")
            os.makedirs(output_image_dir, exist_ok=True)
            augmented_img_path = os.path.join(output_image_dir, f"{img_name}_run{run_idx}.jpg")
            augmented_label_path = os.path.join(output_image_dir, f"{img_name}_run{run_idx}.txt")

            if ARISE_AUGM:
                paste_objects_inside_table_Arise(
                    erased_img_path, corners, objects_folder, augmented_img_path, augmented_label_path,
                    overlap_threshold, current_obj_options, preservs_holes, is_clean
                )
            else:
                paste_objects_inside_table_crop_mix(
                    erased_img_path, corners, objects_folder, augmented_img_path, augmented_label_path
                )

            print(f"Erased image 🖼️ pick from path: {erased_img_path}🗂️")
            print(f"Synthetic image 🖼️ saved at: {augmented_img_path}🗂️ ✅")
            print(f"Label 🏷️ saved at: {augmented_label_path}🗂️ ✅")



############################ Loop ext ################################################################################################################
def run_pipeline_for_all_classes_balance_meth(
    stored_datset_train_dir, 
    copy_paste_augm_dir, 
    erase_crop_augm_dir,  
    image_copy_allocation,
    rnd_bckgrd, 
    arise_augm, 
    overlap_threshold, 
    object_augm_options=None,  # Changed from jit_gauss_augm
    preservs_holes=None,
    background_augm_options=None,
    synch_augmentation=False
):
    """
    For each class in stored_datset_train_dir/train/, run the augmentation pipeline.
    
    Args:
        overlap_threshold: Can be either:
            - A single float value (used for all classes)
            - A dictionary {class_name: threshold} for class-specific thresholds
    """
    coco_builder_train_dir = Path(stored_datset_train_dir)
    copy_paste_augm_dir = Path(copy_paste_augm_dir)
    if not coco_builder_train_dir.exists():
        print(f"❌ Directory does not exist, can't perform augmentation: {coco_builder_train_dir}")
        exit(1)

    # Handle overlap_threshold: can be dict or float
    use_class_specific_threshold = isinstance(overlap_threshold, dict)
    if use_class_specific_threshold:
        print("🎯 Using CLASS-SPECIFIC overlap thresholds")
    else:
        print(f"🎯 Using GLOBAL overlap threshold: {overlap_threshold}")

    class_folders = [f for f in coco_builder_train_dir.iterdir() if f.is_dir()]

    for class_folder in class_folders:
        class_name = class_folder.name
        raw_data_dir = class_folder / "images"
        labels_dir = class_folder / "labels"
        output_dir_base = copy_paste_augm_dir / class_name

        # Get class-specific threshold or use global threshold
        if use_class_specific_threshold:
            class_threshold = overlap_threshold.get(class_name, 0.3)  # Default to 0.3 if class not found
            print(f"   📏 {class_name}: Using overlap threshold = {class_threshold:.4f}")
        else:
            class_threshold = overlap_threshold

        if output_dir_base.exists() and any(output_dir_base.iterdir()):
            print(f"⚠️  Output directory {output_dir_base} is not empty. Skipping...")
            continue
        else:
            print(f"✅  Output directory {output_dir_base} is empty. Proceeding with the runs.")
            os.makedirs(output_dir_base, exist_ok=True)

            run_pipeline_balance_meth(
                stored_dataset=raw_data_dir,
                stored_labels=labels_dir,
                erase_img_dir=erase_crop_augm_dir,
                class_name=class_name,
                image_copy_allocation=image_copy_allocation,  
                output_dir=output_dir_base,
                rnd_bckgrd=rnd_bckgrd,
                arise_augm=arise_augm,
                overlap_threshold=class_threshold,  # Use class-specific threshold
                object_augm_options=object_augm_options,
                preservs_holes=preservs_holes,
                background_augm_options=background_augm_options,
                synch_augmentation=synch_augmentation
            )

        print(f"All runs for {class_name} case completed ✅.")


































##############################################################################################################
################## Oversampling RSF Method  ########################
##############################################################################################################

def compute_rfs_repeat_factors_targeted(
    class_object_df: pd.DataFrame,
    target_total_objects: int
) -> tuple:
    """
    Compute repeat factors to reach target_total_objects per class.
    Unlike standard RFS, this considers OBJECT counts, not just image presence.
    
    Args:
        class_object_df: DataFrame with index=image_name, columns=class_id, values=object counts
        target_total_objects: target number of objects per class
    
    Returns:
        tuple: (image_repeat_factors dict, class_info dict)
    """
    import math
    
    # Compute total objects per class (THIS IS WHAT YOU CARE ABOUT)
    class_total_objects = {}
    for class_id in class_object_df.columns:
        class_total_objects[class_id] = int(class_object_df[class_id].sum())
    
    # Compute scale factor needed for each class to reach target
    class_scale_factors = {}
    for class_id, current_count in class_total_objects.items():
        if current_count >= target_total_objects:
            class_scale_factors[class_id] = 1.0  # Already at or above target
        elif current_count > 0:
            class_scale_factors[class_id] = target_total_objects / current_count
        else:
            class_scale_factors[class_id] = 1.0
    
    # For each image, compute repeat factor based on the class it belongs to
    image_repeat_factors = {}
    for img_name in class_object_df.index:
        max_factor = 1.0
        for class_id in class_object_df.columns:
            count = class_object_df.loc[img_name, class_id]
            count_val = float(count) if pd.notna(count) else 0
            if count_val > 0:
                # This image contributes to this class, use its scale factor
                max_factor = max(max_factor, class_scale_factors[class_id])
        image_repeat_factors[img_name] = max(1, round(max_factor))
    
    class_info = {
        'total_objects': class_total_objects,
        'scale_factors': class_scale_factors
    }
    
    return image_repeat_factors, class_info


def run_pipeline_for_all_classes_RFS(
    stored_datset_train_dir: Path,
    copy_paste_augm_dir: Path,
    image_copy_allocation: pd.DataFrame,
    t: float = 0.001,
    target_total_objects: int = 3000
):
    """
    Run augmentation pipeline using targeted RFS method to reach target_total_objects per class.
    
    Args:
        stored_datset_train_dir: Path to the training dataset (with class subfolders)
        copy_paste_augm_dir: Path to output augmented images
        image_copy_allocation: DataFrame with original allocation (for reference)
        t: RFS threshold parameter (kept for API compatibility, not used in targeted mode)
        target_total_objects: Target number of objects per class
    """
    coco_builder_train_dir = Path(stored_datset_train_dir)
    copy_paste_augm_dir = Path(copy_paste_augm_dir)
    
    if not coco_builder_train_dir.exists():
        print(f"❌ Directory does not exist: {coco_builder_train_dir}")
        return
    
    # Create output directory
    copy_paste_augm_dir.mkdir(parents=True, exist_ok=True)
    
    # Build image-class counts from the dataset
    class_object_df = build_image_class_df(stored_datset_train_dir)
    
    # Compute targeted repeat factors
    image_repeat_factors, class_info = compute_rfs_repeat_factors_targeted(
        class_object_df, target_total_objects
    )
    
    # Print statistics
    print("\n" + "="*60)
    print("Targeted RFS Statistics")
    print("="*60)
    print(f"\nTarget objects per class: {target_total_objects}")
    
    print("\nClass statistics:")
    for class_id in sorted(class_info['total_objects'].keys()):
        current = class_info['total_objects'][class_id]
        factor = class_info['scale_factors'][class_id]
        projected = int(current * factor)
        print(f"  Class {class_id}:")
        print(f"    Current objects: {current}")
        print(f"    Scale factor: {factor:.2f}x")
        print(f"    Projected objects: ~{projected}")
    
    print("\nImage repeat factors (sample, showing images with factor > 1):")
    images_with_copies = [(img, f) for img, f in image_repeat_factors.items() if f > 1]
    for img_name, factor in sorted(images_with_copies, key=lambda x: -x[1])[:10]:
        print(f"  {img_name}: repeat {factor}x")
    if len(images_with_copies) > 10:
        print(f"  ... and {len(images_with_copies) - 10} more images with factor > 1")
    
    # Process each class folder
    class_folders = [f for f in coco_builder_train_dir.iterdir() if f.is_dir()]
    
    total_original = 0
    total_augmented = 0
    
    for class_folder in class_folders:
        class_name = class_folder.name
        images_dir = class_folder / "images"
        labels_dir = class_folder / "labels"
        output_dir_base = copy_paste_augm_dir / class_name
        
        if not images_dir.exists():
            print(f"⚠️ Images directory not found for {class_name}, skipping...")
            continue
        
        if output_dir_base.exists() and any(output_dir_base.iterdir()):
            print(f"⚠️ Output directory {output_dir_base} is not empty. Skipping...")
            continue
        
        output_dir_base.mkdir(parents=True, exist_ok=True)
        
        class_original = 0
        class_copies = 0
        
        # Copy and repeat images based on repeat factors
        for img_file in images_dir.glob("*"):
            if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue
            
            img_name = img_file.stem
            label_file = labels_dir / f"{img_name}.txt"
            
            repeat_factor = image_repeat_factors.get(img_name, 1)
            class_original += 1
            total_original += 1
            
            for copy_idx in range(1, repeat_factor + 1):
                new_img_name = f"{img_name}_run{copy_idx}"
                output_image_dir = output_dir_base / new_img_name
                output_image_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy image
                new_img_path = output_image_dir / f"{new_img_name}{img_file.suffix}"
                shutil.copy2(img_file, new_img_path)
                
                # Copy label if exists
                if label_file.exists():
                    new_label_path = output_image_dir / f"{new_img_name}.txt"
                    shutil.copy2(label_file, new_label_path)
                
                if copy_idx > 1:
                    class_copies += 1
                    total_augmented += 1
        
        print(f"\n📊 Class {class_name}:")
        print(f"   Original images: {class_original}")
        print(f"   Copies added: {class_copies}")
        print(f"   Total: {class_original + class_copies}")
    
    print(f"\n{'='*60}")
    print("Targeted RFS Augmentation Complete ✅")
    print(f"{'='*60}")
    print(f"Original images: {total_original}")
    print(f"Additional copies created: {total_augmented}")
    print(f"Total images after RFS: {total_original + total_augmented}")
    print(f"Output directory: {copy_paste_augm_dir}")

##############################################################################################################
################## Oversampling Balance Method  ########################
##############################################################################################################
def find_image_with_extension(directory, base_name):
    """
    Find an image file with any common extension given a base name.
    
    Args:
        directory: Path to the directory containing images
        base_name: Image name without extension
    
    Returns:
        Full path to the image if found, None otherwise
    """
    extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.bmp', '.BMP']
    for ext in extensions:
        candidate = os.path.join(directory, base_name + ext)
        if os.path.exists(candidate):
            return candidate
    return None

def run_pipeline_balance_meth_oversample(
    stored_dataset, 
    stored_labels, 
    class_name, 
    image_copy_allocation,  
    output_dir
):
    """
    Balance classes by oversampling (duplicating) existing images instead of generating synthetic ones.
    For each image in image_copy_allocation, copy it the specified number of times.
    """
    class_path = stored_dataset

    # Filter allocation for current class
    df_class = image_copy_allocation[image_copy_allocation['class_name'] == class_name]
    
    for _, row in df_class.iterrows():
        img_name = row['image_name']
        num_copies = int(row['num_copies'])

        src_image_path = find_image_with_extension(class_path, img_name)
        src_label_path = os.path.join(stored_labels, img_name + ".txt")
        
        if not os.path.exists(src_image_path) or not os.path.exists(src_label_path):
            print(f"❌ Image or label missing for: {img_name} in {class_path} or {stored_labels} 🗂️")
            continue

        # Copy the image and label num_copies times
        for run_idx in range(1, num_copies + 1):
            output_image_dir = os.path.join(output_dir, f"{img_name}_run{run_idx}")
            os.makedirs(output_image_dir, exist_ok=True)
            
            dst_image_path = os.path.join(output_image_dir, f"{img_name}_run{run_idx}.jpg")
            dst_label_path = os.path.join(output_image_dir, f"{img_name}_run{run_idx}.txt")
            
            # Copy image
            shutil.copy2(src_image_path, dst_image_path)
            # Copy label
            shutil.copy2(src_label_path, dst_label_path)
            
            print(f"Oversampled image 🖼️ saved at: {dst_image_path}🗂️ ✅")
            print(f"Oversampled label 🏷️ saved at: {dst_label_path}🗂️ ✅")


def run_pipeline_for_all_classes_balance_meth_oversample(
    stored_datset_train_dir, 
    oversample_augm_dir,  
    image_copy_allocation
):
    """
    For each class in stored_datset_train_dir, run the oversampling pipeline and store results in oversample_augm_dir.
    Uses image_copy_allocation DataFrame to specify which images to duplicate and how many times.
    Skips classes that already have non-empty output folders.
    image_copy_allocation columns: ['class_name', 'image_name', 'num_copies']
    """
    coco_builder_train_dir = Path(stored_datset_train_dir)
    oversample_augm_dir = Path(oversample_augm_dir)
    
    if not coco_builder_train_dir.exists():
        print(f"❌ Directory does not exist, can't perform oversampling: {coco_builder_train_dir}")
        exit(1)

    # Iterate through each class folder
    class_folders = [f for f in coco_builder_train_dir.iterdir() if f.is_dir()]

    for class_folder in class_folders:
        class_name = class_folder.name
        raw_data_dir = class_folder / "images"
        labels_dir = class_folder / "labels"
        output_dir_base = oversample_augm_dir / class_name

        if output_dir_base.exists() and any(output_dir_base.iterdir()):
            print(f"⚠️  Output directory {output_dir_base} is not empty. Skipping operations ...")
            continue
        else:
            print(f"✅  Output directory {output_dir_base} is empty. Proceeding with oversampling.")
            os.makedirs(output_dir_base, exist_ok=True)

            run_pipeline_balance_meth_oversample(
                stored_dataset=raw_data_dir,
                stored_labels=labels_dir,
                class_name=class_name,
                image_copy_allocation=image_copy_allocation,  
                output_dir=output_dir_base
            )

        print(f"All oversampling for {class_name} case completed ✅.")











##############################################################################################################
################## Explicit_Balancing_Method  ########################
##############################################################################################################



def build_image_class_df(stored_datset_train_dir):
    """
    Builds a DataFrame of object counts:
    index = image_name
    columns = class_id
    values = object counts

    Uses the global variable stored_datset_train_dir, which should be a Path object
    pointing to the 'train' folder containing class subfolders with .txt label files.
    """
    records = []

    # Convert Path to string for os.listdir compatibility
    folder_path = str(stored_datset_train_dir)

    for class_folder in os.listdir(folder_path):
        class_folder_path = os.path.join(folder_path, class_folder)
        if not os.path.isdir(class_folder_path):
            continue  # Skip non-folder entries
        # access to labels folder inside each class folder
        class_id = class_folder  # Assuming folder names are the class IDs  
        labels_path = os.path.join(class_folder_path, "labels")
        # get access to each label file inside the labels folder
        if not os.path.exists(labels_path):
            continue # Skip if 'labels' folder doesn't exist
        for file_name in os.listdir(labels_path):
            if file_name.endswith(".txt"):
                file_path = os.path.join(labels_path, file_name)
                with open(file_path, "r", encoding="utf-8") as f:
                    object_count = len(f.readlines())
                image_name = file_name.replace(".txt", "")
                records.append((image_name, class_id, object_count))

    # Build DataFrame
    df = pd.DataFrame(records, columns=["image_name", "class_id", "object_count"])
    df = df.pivot(index="image_name", columns="class_id", values="object_count")
    return df




###########################################################################################
###########################################################################################
def make_bins_for_class(counts, R, first_upper):
    valid_counts = [c for c in counts if pd.notna(c)]
    if not valid_counts:
        return []
    M = max(valid_counts)
    if M <= first_upper or R == 1:
        # Only one bin needed
        return [(0, M)]
    remaining = R - 1
    span = (M - first_upper) / remaining if remaining > 0 else 0
    bins = [(0, first_upper)]
    prev_upper = first_upper
    for k in range(1, R):
        # For last bin, always end at M
        if k == R-1:
            upper = M
        else:
            target = first_upper + k*span
            upper = int(round(target))
            if upper <= prev_upper:
                upper = prev_upper + 1
        bins.append((prev_upper+1, upper))
        prev_upper = upper
    return bins



def bin_stats_for_class(counts, bins, image_names):
    # Only consider images that actually have a label file for this class (ignore None/NaN)
    valid = [(img, c) for img, c in zip(image_names, counts) if pd.notna(c)]
    valid_counts = [c for _, c in valid]
    valid_images = [img for img, _ in valid]
    stats = []
    for lo, hi in bins:
        idx = [i for i, c in enumerate(valid_counts) if lo <= c <= hi]
        num_images = len(idx)
        total_objects = sum(valid_counts[i] for i in idx)
        avg = total_objects / num_images if num_images > 0 else (lo + hi) / 2.0
        bin_images = [valid_images[i] for i in idx]
        stats.append({
            'range': f'{lo}-{hi}',
            'lo': lo, 'hi': hi,
            'image_names': bin_images,
            'num_images': num_images,
            'total_objects': total_objects,
            'avg_objects_per_image': avg
        })
    return stats

###########################################################################################


def allocate_copies_to_target(bin_stats, target_total_objects, method='proportional_objects'):
    """
    Allocate integer copies per bin to reach target_total_objects.
    method: 'proportional_images' or 'proportional_objects' (weights for distribution)
    Returns updated bin_stats with 'copies' column and achieved totals.
    """
    current_total = sum(b['total_objects'] for b in bin_stats)
    delta = target_total_objects - current_total
    if delta <= 0:
        # No copies needed
        for b in bin_stats:
            b['copies'] = 0
        return bin_stats, current_total, 0, 0.0  # bin_stats, initial_total, added_objects, overshoot

    # Compute weights p_b
    if method == 'proportional_images':
        total_images = sum(b['num_images'] for b in bin_stats)
        if total_images == 0:
            weights = [1/len(bin_stats)] * len(bin_stats)
        else:
            weights = [b['num_images'] / total_images for b in bin_stats]
    elif method == 'proportional_objects':
        total_objs = sum(b['total_objects'] for b in bin_stats)
        if total_objs == 0:
            weights = [1/len(bin_stats)] * len(bin_stats)
        else:
            weights = [b['total_objects'] / total_objs for b in bin_stats]
    else:
        raise ValueError("method must be 'proportional_images' or 'proportional_objects'")

    # Avoid divide by zero: if avg == 0 set to small epsilon (or use range midpoint)
    avg_list = [b['avg_objects_per_image'] if b['avg_objects_per_image'] > 0 else (b['lo']+b['hi'])/2.0 for b in bin_stats]

    # initial fractional allocation (float copies)
    copies_float = [(delta * w) / avg if avg > 0 else 0 for w, avg in zip(weights, avg_list)]
    # initial integer copies: floor
    copies_int = [math.floor(c) for c in copies_float]
    # Achieved objects with this integer allocation
    achieved = sum(c * avg for c, avg in zip(copies_int, avg_list))
    remaining = delta - achieved

    # fractional parts for tie-breaking / priority
    fractional_parts = [cf - ci for cf, ci in zip(copies_float, copies_int)]

    # Now distribute remaining (which is in "objects") by giving +1 copy to bins in order of
    # descending fractional_part (to keep proportions), breaking ties by larger avg_objects_per_image.
    # Stop when remaining <= 0.
    additional_copies = [0] * len(bin_stats)
    idx_order = sorted(range(len(bin_stats)), key=lambda i: (fractional_parts[i], avg_list[i]), reverse=True)

    pointer = 0
    if remaining > 0:
        max_iters = 100000
        iters = 0
        while remaining > 0 and iters < max_iters:
            i = idx_order[pointer % len(idx_order)]
            additional_copies[i] += 1
            remaining -= avg_list[i]
            pointer += 1
            iters += 1
        if iters >= max_iters:
            raise RuntimeError("Too many iterations in distribution loop. Check avg estimates.")

    # finalize copies
    final_copies = [ci + ac for ci, ac in zip(copies_int, additional_copies)]
    # compute added objects
    added_objects = sum(c * avg for c, avg in zip(final_copies, avg_list))
    achieved_total = current_total + added_objects
    overshoot = achieved_total - target_total_objects

    # write back to bin_stats
    for b, copies, avg in zip(bin_stats, final_copies, avg_list):
        b['copies'] = int(copies)
        b['estimated_added_objects'] = copies * avg

    return bin_stats, current_total, added_objects, overshoot


def get_bin_image_allocation_df(class_bin_stats_dict):
    """
    Given a dictionary mapping class_id to bin_stats (list of dicts for each bin),
    returns a DataFrame with columns:
        ['class_id', 'range', 'copies_to_add', 'image_names']
    """
    records = []
    for class_id, bin_stats in class_bin_stats_dict.items():
        for b in bin_stats:
            records.append({
                'class_id': class_id,
                'range': b['range'],
                'copies_to_add': b.get('copies', 0),
                'image_names': b.get('image_names', [])
            })
    return pd.DataFrame(records)





##############################################################################################
##############################################################################################




def distribute_uniform_from_df(bin_image_allocation_df, random_seed=None):
    """
    For each bin in bin_image_allocation_df, distribute 'copies_to_add' as evenly as possible among 'image_names'.
    If perfect distribution isn't possible, randomly assign the remaining copies to images.
    Returns a dict: {image_name: total_copies_to_add}
    """
    if random_seed is not None:
        random.seed(random_seed)

    image_copy_counts = {}

    for _, row in bin_image_allocation_df.iterrows():
        images = row['image_names']
        copies_needed = int(row['copies_to_add'])
        num_images = len(images)
        if num_images == 0 or copies_needed == 0:
            continue

        base = copies_needed // num_images
        remainder = copies_needed % num_images

        # Assign base copies to each image
        for img in images:
            image_copy_counts[img] = image_copy_counts.get(img, 0) + base

        # Randomly assign the remaining copies
        if remainder > 0:
            chosen = random.sample(images, remainder)
            for img in chosen:
                image_copy_counts[img] += 1

    return image_copy_counts




############################################################################################################################################################################################

############################################################################################################################################################################################
