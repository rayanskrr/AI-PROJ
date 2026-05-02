import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from augmentation import get_val_test_transforms
from segmentation_mbrs import segment_hieroglyphs_mbrs

def predict_pipeline(image_path, class_names, models, method='mbrs',
                     predictor=None, device="cuda"):
    """
    End-to-End Inference Pipeline:
    Image -> Segmentation -> CVV Ensemble -> Gardiner Codes -> Visualization

    Args:
        image_path (str): Path to the stela image.
        class_names (list[str]): Ordered list of Gardiner code class names.
        models (list): Pre-loaded CVV ensemble models (must be 3).
        method (str): 'mbrs' or 'igsm'.
        predictor: Pre-loaded SAM predictor (required if method='igsm').
        device (str): 'cuda' or 'cpu'.

    Returns:
        list[tuple]: Each entry is (gardiner_code, confidence, (x, y, w, h))
    """
    print(f"\n--- Starting Inference on {image_path} ---")

    # 1. Segmentation
    if method == 'mbrs':
        print("Running MBRS Segmentation...")
        crops, boxes = segment_hieroglyphs_mbrs(image_path)
    elif method == 'igsm':
        from segmentation_igsm import segment_hieroglyphs_igsm
        if predictor is None:
            raise ValueError("IGSM requires a loaded SAM predictor.")
        print("Running IGSM Segmentation...")
        crops, boxes = segment_hieroglyphs_igsm(image_path, predictor)
    else:
        raise ValueError("Method must be 'mbrs' or 'igsm'.")

    if not crops:
        print("No hieroglyphs detected in the image.")
        return []

    print(f"Detected {len(crops)} potential hieroglyphs.")

    # FIX 1: Models passed in from outside — not loaded here every call
    if not models or len(models) != 3:
        raise RuntimeError("Expected exactly 3 pre-loaded CVV ensemble models.")

    # 2. Prepare transforms
    transform = get_val_test_transforms()

    predictions = []

    # Read original image for visualization
    original_img = cv2.imread(image_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    # FIX 2: Use device type string safely for autocast (works on both cuda and cpu)
    device_type = device.split(':')[0]

    # 3. Classification via Soft Voting
    print("Running classification ensemble...")
    with torch.no_grad():
        for i, crop in enumerate(crops):
            # OpenCV crop (numpy BGR) → PIL Image (RGB)
            pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

            # Apply transforms and add batch dimension: [1, C, H, W]
            input_tensor = transform(pil_img).unsqueeze(0).to(device)

            num_classes = len(class_names)
            ensemble_probs = torch.zeros((1, num_classes), device=device)

            for model in models:
                with torch.autocast(device_type=device_type):
                    output = model(input_tensor)
                probs = torch.softmax(output.float(), dim=1)
                ensemble_probs += probs

            # Soft Voting: average probabilities, pick highest
            ensemble_probs /= len(models)
            best_prob, pred_idx = torch.max(ensemble_probs, 1)

            pred_class = class_names[pred_idx.item()]
            predictions.append((pred_class, best_prob.item(), boxes[i]))

    # 4. Visualize Results
    plt.figure(figsize=(12, 12))
    plt.imshow(original_img)
    ax = plt.gca()

    print("\n--- Final Predictions (Right-to-Left, Top-to-Bottom) ---")
    for idx, (pred_class, prob, box) in enumerate(predictions):
        x, y, w, h = box
        print(f"Symbol {idx + 1}: {pred_class} (Confidence: {prob:.4f})")

        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y - 5, f"{pred_class}", color='red', fontsize=12, weight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    plt.axis('off')
    plt.title(f"Inference Results ({method.upper()}) — CVV Soft Voting")
    plt.show()

    return predictions
