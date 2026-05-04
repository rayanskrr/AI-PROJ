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
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            model.to(device)
            model.eval()
            models.append(model)
        else:
            print(f"Warning: Checkpoint not found at {ckpt_path}.")
    return models

def evaluate_ensemble(models, test_loader, class_names, device="cuda"):
    """
    Evaluates the ensemble using BOTH Soft Voting and Hard Voting.
    """
    if not models:
        raise ValueError("No models loaded for evaluation.")

    all_labels = []
    all_soft_preds = []
    all_hard_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            batch_size = images.size(0)
            ensemble_probs = torch.zeros((batch_size, len(class_names)), device=device)
            model_preds = torch.zeros((len(models), batch_size), dtype=torch.long, device=device)

            for i, model in enumerate(models):
                with torch.autocast(device_type='cuda'):
                    outputs = model(images)

                probs = torch.softmax(outputs.float(), dim=1)

                # For Soft Voting: Accumulate probabilities
                ensemble_probs += probs

                # For Hard Voting: Store each model's discrete prediction
                _, preds = torch.max(probs, 1)
                model_preds[i] = preds

            # --- SOFT VOTING CALCULATION ---
            ensemble_probs /= len(models)
            _, soft_preds = torch.max(ensemble_probs, 1)

            # --- HARD VOTING CALCULATION ---
            # torch.mode returns (values, indices). We want the values (the majority class).
            hard_preds, _ = torch.mode(model_preds, dim=0)

            all_soft_preds.extend(soft_preds.cpu().numpy())
            all_hard_preds.extend(hard_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    def print_metrics(labels, preds, title):
        acc = accuracy_score(labels, preds)
        bal_acc = balanced_accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='macro', zero_division=0
        )
        print(f"\n{'='*40}")
        print(f"--- {title} Results ---")
        print(f"{'='*40}")
        print(f"Accuracy:          {acc:.4f}")
        print(f"Balanced Accuracy: {bal_acc:.4f}")
        print(f"Macro Precision:   {precision:.4f}")
        print(f"Macro Recall:      {recall:.4f}")
        print(f"Macro F1 Score:    {f1:.4f}")
        print(f"{'='*40}")

    print_metrics(all_labels, all_soft_preds, "Soft Voting")
    print_metrics(all_labels, all_hard_preds, "Hard Voting")

    return all_labels, all_soft_preds, all_hard_preds

def plot_confusion_matrix(labels, preds, class_names, save_path="confusion_matrix.png"):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(24, 24))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual Gardiner Code')
    plt.xlabel('Predicted Gardiner Code')
    plt.title('CVV Ensemble Confusion Matrix (Soft Voting)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\nConfusion matrix saved successfully to {save_path}")
