import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from config import SAM_CKPT_PATH


def load_sam_predictor(device=None):
    """Loads the SAM predictor using the ViT-B checkpoint."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_b"](checkpoint=SAM_CKPT_PATH)
    sam.to(device=device)
    return SamPredictor(sam)


def compute_iou(boxA, boxB):
    """Computes Intersection over Union for Non-Maximum Suppression."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(max(1, boxAArea + boxBArea - interArea))


def segment_hieroglyphs_igsm(image_path, predictor, is_carved=True):
    """
    Focused Generic Segmentation Method (IGSM).
    Uses Otsu binarization to find point prompts, then queries SAM
    per connected region. Applies NMS to remove duplicate detections.

    Args:
        image_path (str): Path to the stela image.
        predictor (SamPredictor): Loaded SAM predictor instance.
        is_carved (bool): True for carved stone (6-11 pts), False for painted (3-8 pts).

    Returns:
        tuple[list[np.ndarray], list[tuple]]:
            - List of cropped hieroglyph images (BGR)
            - List of bounding boxes as (x, y, w, h), sorted top-to-bottom right-to-left
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    original_color = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Otsu Binarization to find foreground regions
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    # Set image once for all SAM queries
    predictor.set_image(cv2.cvtColor(original_color, cv2.COLOR_BGR2RGB))

    all_boxes = []

    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 50:  # Filter tiny noise regions
            continue

        # FIX: Sample num_points per region, not once globally
        num_points = np.random.randint(6, 12) if is_carved else np.random.randint(3, 9)

        y_coords, x_coords = np.where(labels == i)
        coords = list(zip(x_coords, y_coords))

        # Sample random points from the connected component
        sample_size = min(num_points, len(coords))
        sampled_indices = np.random.choice(len(coords), size=sample_size, replace=False)

        input_points = np.array([coords[idx] for idx in sampled_indices])
        input_labels = np.ones(len(input_points))  # 1 = foreground point

        # 3. Prompt SAM with the sampled points
        masks, scores, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )

        # Take the highest confidence mask
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx]

        # Derive bounding box from the mask
        y_mask, x_mask = np.where(best_mask)
        if len(x_mask) > 0 and len(y_mask) > 0:
            x_min, x_max = int(np.min(x_mask)), int(np.max(x_mask))
            y_min, y_max = int(np.min(y_mask)), int(np.max(y_mask))
            all_boxes.append([x_min, y_min, x_max - x_min, y_max - y_min])

    # 4. Non-Maximum Suppression (IoU threshold = 0.5)
    keep_indices = []
    for i in range(len(all_boxes)):
        keep = True
        for j in keep_indices:
            if compute_iou(all_boxes[i], all_boxes[j]) > 0.5:
                keep = False
                break
        if keep:
            keep_indices.append(i)

    # FIX: Build crops and boxes as pairs then sort together to keep alignment
    detections = []
    for idx in keep_indices:
        x, y, w, h = all_boxes[idx]
        crop = original_color[y:y + h, x:x + w]
        detections.append(((x, y, w, h), crop))

    # Sort top-to-bottom, right-to-left (matches Egyptian reading order)
    detections.sort(key=lambda d: (d[0][1], -d[0][0]))

    final_boxes = [d[0] for d in detections]
    cropped_hieroglyphs = [d[1] for d in detections]

    return cropped_hieroglyphs, final_boxes
