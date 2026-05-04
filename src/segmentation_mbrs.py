import cv2
import numpy as np

def segment_hieroglyphs_mbrs(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    original_color = cv2.imread(image_path)
    img_h, img_w = img.shape
    img_area = img_h * img_w

    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    high_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5 * high_thresh
    edges = cv2.Canny(blurred, low_thresh, high_thresh)

    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Adaptive size bounds based on image dimensions
    # Min: a hieroglyph must be at least 1.5% of the shorter side in both w and h
    min_dim = min(img_h, img_w)
    min_size = max(10, int(0.015 * min_dim))
    # Max: no single detection larger than 5% of image area
    max_area = 0.05 * img_area

    detections = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        box_area = w * h
        # RESEARCH FIX: Relaxed strict dimension check to allow thin glyphs (like z1, N35) as long as total area is valid.
        if (w > 5 and h > 5) and box_area > (min_size ** 2) and box_area < max_area:
            crop = original_color[y:y+h, x:x+w]
            detections.append(((x, y, w, h), crop))

    detections.sort(key=lambda d: (d[0][1], -d[0][0]))
    bounding_boxes = [d[0] for d in detections]
    cropped_hieroglyphs = [d[1] for d in detections]

    return cropped_hieroglyphs, bounding_boxes
