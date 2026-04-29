import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix
from model import build_model
from config import CKPT_DIR

def load_cvv_ensemble(num_classes, device="cuda"):
    """Loads the 3 best checkpoints from the CVV training slots."""
    models = []
    for slot in range(1, 4):
        ckpt_path = os.path.join(CKPT_DIR, f"cvv_slot_{slot}_best.pth")
        model = build_model(num_classes)
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.to(device)
            model.eval() # Set to evaluation mode!
            models.append(model)
        else:
            print(f"Warning: Checkpoint not found at {ckpt_path}. Did Slot {slot} finish training?")
    return models

def evaluate_ensemble(models, test_loader, class_names, device="cuda"):
    """
    Evaluates the ensemble using Soft Voting (averaging probabilities).
    Calculates accuracy, balanced accuracy, precision, recall, and F1.
    """
    if not models:
        raise ValueError("No models loaded for evaluation.")

    all_preds = []
    all_labels = []

    with torch.no_grad(): # No gradients needed for testing, saves VRAM!
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Get probabilities from all models
            ensemble_probs = torch.zeros((images.size(0), len(class_names)), device=device)
            for model in models:
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                ensemble_probs += probs
            
            # Soft voting: average the probabilities and take the max
            ensemble_probs /= len(models)
            _, preds = torch.max(ensemble_probs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate Sklearn Metrics
    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    print("\n" + "="*40)
    print("--- Ensemble Evaluation Results ---")
    print("="*40)
    print(f"Accuracy:          {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"Weighted Precision:{precision:.4f}")
    print(f"Weighted Recall:   {recall:.4f}")
    print(f"Weighted F1 Score: {f1:.4f}")
    print("="*40)
    
    return all_labels, all_preds

def plot_confusion_matrix(labels, preds, class_names, save_path="confusion_matrix.png"):
    """Plots and saves the confusion matrix."""
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(24, 24)) # Massive figure because we have 164 classes
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual Gardiner Code')
    plt.xlabel('Predicted Gardiner Code')
    plt.title('CVV Ensemble Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\nConfusion matrix saved successfully to {save_path}")
