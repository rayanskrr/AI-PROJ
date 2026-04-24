import cv2
import numpy as np

def segment_hieroglyphs_mbrs(image_path):
    """
    Method Based on Region Segmentation (MBRS).
    Uses Canny edge detection to isolate hieroglyphs.
    """
    # 1. Read image in grayscale for edge detection
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
        
    # Read original color image for the final crop
    original_color = cv2.imread(image_path)
    
    # 2. Apply Gaussian Blur to reduce noise before edge detection
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # 3. Canny Edge Detection
    # Using Otsu's thresholding to dynamically find optimal thresholds for Canny
    high_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5 * high_thresh
    edges = cv2.Canny(blurred, low_thresh, high_thresh)
    
    # 4. Dilate the edges to connect broken parts of a single hieroglyph
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # 5. Find contours (connected regions)
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cropped_hieroglyphs = []
    bounding_boxes = []
    
    # 6. Extract the regions based on contours
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter out extremely small noise (adjustable based on image resolution)
        if w > 10 and h > 10:
            # Crop the original colored image using the bounding box
            crop = original_color[y:y+h, x:x+w]
            cropped_hieroglyphs.append(crop)
            bounding_boxes.append((x, y, w, h))
            
    # Sort bounding boxes top-to-bottom, right-to-left as per standard Egyptian reading rules
    bounding_boxes.sort(key=lambda b: (b[1], -b[0]))
    
    return cropped_hieroglyphs, bounding_boxes
