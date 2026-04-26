import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from config import SAM_CKPT_PATH

def load_sam_predictor(device="cuda" if torch.cuda.is_available() else "cpu"):
    """Loads the SAM predictor using the ViT-B checkpoint."""
    sam = sam_model_registry["vit_b"](checkpoint=SAM_CKPT_PATH)
    sam.to(device=device)
    return SamPredictor(sam)

def compute_iou(boxA, boxB):
    """Computes Intersection over Union for Non-Maximum Suppression."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(max(1, boxAArea + boxBArea - interArea))

def segment_hieroglyphs_igsm(image_path, predictor, is_carved=True):
    """
    Focused Generic Segmentation Method (IGSM).
    Uses Otsu binarization to find points, then prompts SAM.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
        
    original_color = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Otsu Binarization to find foreground regions
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 2. Find connected regions
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    
    # Determine number of points per region based on paper specs
    num_points = np.random.randint(6, 12) if is_carved else np.random.randint(3, 9)
    
    predictor.set_image(cv2.cvtColor(original_color, cv2.COLOR_BGR2RGB))
    
    all_boxes = []
    
    for i in range(1, num_labels): # Skip background (0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 50: # Filter tiny noise
            continue
            
        y_coords, x_coords = np.where(labels == i)
        coords = list(zip(x_coords, y_coords))
        
        # Sample random points from the component
        sample_size = min(num_points, len(coords))
        sampled_indices = np.random.choice(len(coords), size=sample_size, replace=False)
        
        input_points = np.array([coords[idx] for idx in sampled_indices])
        input_labels = np.ones(len(input_points)) # 1 indicates a foreground point
        
        # 3. Prompt SAM
        masks, scores, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )
        
        # Take the best mask (highest confidence score)
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx]
        
        # Find bounding box of the mask
        y_mask, x_mask = np.where(best_mask)
        if len(x_mask) > 0 and len(y_mask) > 0:
            x_min, x_max = np.min(x_mask), np.max(x_mask)
            y_min, y_max = np.min(y_mask), np.max(y_mask)
            all_boxes.append([x_min, y_min, x_max - x_min, y_max - y_min])
            
    # 4. Non-Maximum Suppression (IoU threshold = 0.5)
    keep_indices = []
    for i in range(len(all_boxes)):
        keep = True
        for j in range(len(keep_indices)):
            idx = keep_indices[j]
            if compute_iou(all_boxes[i], all_boxes[idx]) > 0.5:
                keep = False
                break
        if keep:
            keep_indices.append(i)
            
    cropped_hieroglyphs = []
    final_boxes = []
    
    for idx in keep_indices:
        x, y, w, h = all_boxes[idx]
        crop = original_color[y:y+h, x:x+w]
        cropped_hieroglyphs.append(crop)
        final_boxes.append((x, y, w, h))
        
    final_boxes.sort(key=lambda b: (b[1], -b[0]))
    return cropped_hieroglyphs, final_boxes
