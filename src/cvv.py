import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from dataset import HieroglyphDataset, balance_training_data
from augmentation import get_train_transforms, get_val_test_transforms
from model import build_model
from config import CKPT_DIR, BATCH_SIZE, MAX_EPOCHS, PATIENCE, LEARNING_RATE, WEIGHT_DECAY

def train_cvv_slots(image_paths, class_to_idx, num_classes, device="cuda"):
    os.makedirs(CKPT_DIR, exist_ok=True)
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    for slot, (train_idx, val_idx) in enumerate(kf.split(image_paths)):
        print(f"\n{'='*40}\nStarting CVV Slot {slot + 1} / 3\n{'='*40}")

        train_paths = [image_paths[i] for i in train_idx]
        val_paths = [image_paths[i] for i in val_idx]

        # FIX: Explicitly call the data balancer
        train_paths = balance_training_data(train_paths)

        train_dataset = HieroglyphDataset(train_paths, class_to_idx, transform=get_train_transforms())
        val_dataset = HieroglyphDataset(val_paths, class_to_idx, transform=get_val_test_transforms())

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

        model = build_model(num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

        # FIX: Updated GradScaler API
        scaler = torch.amp.GradScaler(cuda)

        best_val_acc = 0.0
        epochs_no_improve = 0

        for epoch in range(MAX_EPOCHS):
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                # FIX: Updated autocast API
                with torch.autocast(device_type='cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total_train += labels.size(0)
                correct_train += predicted.eq(labels).sum().item()

            train_acc = correct_train / total_train

            model.eval()
            correct_val = 0
            total_val = 0
            val_loss = 0.0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)

                    _, predicted = outputs.max(1)
                    total_val += labels.size(0)
                    correct_val += predicted.eq(labels).sum().item()

            val_acc = correct_val / total_val
            scheduler.step()

            print(f"Epoch [{epoch+1}/{MAX_EPOCHS}] - Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                ckpt_path = os.path.join(CKPT_DIR, f"cvv_slot_{slot+1}_best.pth")
                torch.save(model.state_dict(), ckpt_path)
                print(f"--> Saved new best model for Slot {slot+1} to {ckpt_path}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= PATIENCE:
                    print(f"Early stopping triggered for Slot {slot+1} at epoch {epoch+1}")
                    break

        # FIX: Explicit memory release to prevent T4 OOM crashes between slots
        del model, optimizer, scheduler, scaler, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()

    print("\nAll CVV Slots Completed!")
