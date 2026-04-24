import os
import glob
import random
from collections import Counter
from PIL import Image
from torch.utils.data import Dataset
from config import MIN_IMAGES_PER_CLASS

def get_filtered_dataset(dataset_dir):
    """Filters out classes with fewer than the minimum required images."""
    all_images = glob.glob(os.path.join(dataset_dir, '*/*.*'))

    # Count images per class
    labels = [os.path.basename(os.path.dirname(p)) for p in all_images]
    class_counts = Counter(labels)

    # Keep only classes with >= MIN_IMAGES_PER_CLASS
    valid_classes = {cls for cls, count in class_counts.items() if count >= MIN_IMAGES_PER_CLASS}

    filtered_images = [p for p in all_images if os.path.basename(os.path.dirname(p)) in valid_classes]

    # Create consistent class to index mapping
    class_to_idx = {cls: idx for idx, cls in enumerate(sorted(valid_classes))}

    return filtered_images, class_to_idx

def balance_training_data(train_paths):
    """
    Balances the dataset by oversampling under-represented classes
    and capping over-represented classes to 2x the average count.
    """
    labels = [os.path.basename(os.path.dirname(p)) for p in train_paths]
    class_counts = Counter(labels)

    avg_count = int(len(train_paths) / len(class_counts))
    max_count = avg_count * 2

    balanced_paths = []
    class_to_paths = {}

    for p, l in zip(train_paths, labels):
        class_to_paths.setdefault(l, []).append(p)

    for cls, paths in class_to_paths.items():
        if len(paths) > max_count:
            # Cap at 2x average
            balanced_paths.extend(random.sample(paths, max_count))
        elif len(paths) < avg_count:
            # Oversample to reach the average
            balanced_paths.extend(paths)
            shortfall = avg_count - len(paths)
            balanced_paths.extend(random.choices(paths, k=shortfall))
        else:
            balanced_paths.extend(paths)

    # Shuffle the final balanced dataset
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
        label_str = os.path.basename(os.path.dirname(img_path))
        label = self.class_to_idx[label_str]

        # Convert to RGB to handle any grayscale/RGBA anomalies safely
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
