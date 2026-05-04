import os
import random
from collections import Counter
from PIL import Image
from torch.utils.data import Dataset

def get_filtered_dataset(dataset_dir, min_images=7):
    """Filters classes with < 7 images and ignores non-image files."""
    valid_extensions = ('.png', '.jpg', '.jpeg', '.JPG', '.PNG')
    class_names = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])

    valid_paths = []
    class_to_idx = {}
    idx = 0

    for class_name in class_names:
        class_dir = os.path.join(dataset_dir, class_name)
        # FIX: Only grab files with valid image extensions
        images = [os.path.join(class_dir, f) for f in os.listdir(class_dir)
                  if f.lower().endswith(valid_extensions)]

        if len(images) >= min_images:
            valid_paths.extend(images)
            class_to_idx[class_name] = idx
            idx += 1

    return valid_paths, class_to_idx

def balance_training_data(train_paths):
    """Oversamples minority classes to the average and caps majority classes at 2x average."""
    class_counts = Counter([os.path.basename(os.path.dirname(p)) for p in train_paths])
    if not class_counts:
        return train_paths

    avg_count = int(sum(class_counts.values()) / len(class_counts))
    cap_count = avg_count * 2

    balanced_paths = []
    paths_by_class = {c: [] for c in class_counts.keys()}

    for p in train_paths:
        c = os.path.basename(os.path.dirname(p))
        paths_by_class[c].append(p)

    for c, paths in paths_by_class.items():
        count = len(paths)
        if count < avg_count:
            balanced_paths.extend(paths)
            diff = avg_count - count
            balanced_paths.extend(random.choices(paths, k=diff))
        elif count > cap_count:
            balanced_paths.extend(random.sample(paths, cap_count))
        else:
            balanced_paths.extend(paths)

    random.shuffle(balanced_paths)
    return balanced_paths

class HieroglyphDataset(Dataset):
    def __init__(self, image_paths, class_to_idx, transform=None):
        self.image_paths = image_paths
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        class_name = os.path.basename(os.path.dirname(img_path))
        label = self.class_to_idx[class_name]

        # Force convert to RGB to prevent crashes from grayscale or RGBA images
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
