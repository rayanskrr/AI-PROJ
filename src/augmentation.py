import torchvision.transforms as T
from config import IMAGE_SIZE, MEAN, STD

def get_train_transforms():
    """Augmentations applied only to training data based on paper specs."""
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.RandomHorizontalFlip(p=0.5),
        # 10px translation on a 224px image is approx 0.045
        T.RandomAffine(
            degrees=5,
            translate=(10/IMAGE_SIZE, 10/IMAGE_SIZE),
            scale=(0.90, 1.10)
        ),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])

def get_val_test_transforms():
    """Deterministic transforms for validation and test sets."""
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
