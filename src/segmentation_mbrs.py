import cv2
import numpy as np

def segment_hieroglyphs_mbrs(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
        
    original_color = cv2.imread(image_path)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    high_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5 * high_thresh
    edges = cv2.Canny(blurred, low_thresh, high_thresh)
    
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # FIX: Group coordinates and crops together before sorting
    detections = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 10:
            crop = original_color[y:y+h, x:x+w]
            detections.append(((x, y, w, h), crop))
            
    # Sort pairs together — top-to-bottom, right-to-left
    detections.sort(key=lambda d: (d[0][1], -d[0][0]))
    
    bounding_boxes = [d[0] for d in detections]
    cropped_hieroglyphs = [d[1] for d in detections]
    
    return cropped_hieroglyphs, bounding_boxes
