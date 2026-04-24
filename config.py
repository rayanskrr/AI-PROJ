import os

# Paths
DATASET_DIR = '/content/EgyptianHieroglyphicText'
CKPT_DIR = '/content/drive/MyDrive/hieroglyph_ckpts'
SAM_CKPT_PATH = '/content/drive/MyDrive/sam_ckpts/sam_vit_b.pth'

# Architecture & Training Hyperparameters
IMAGE_SIZE = 224
BATCH_SIZE = 32
MIN_IMAGES_PER_CLASS = 7
MAX_EPOCHS = 15
PATIENCE = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# ImageNet Normalization stats for ConvNeXt
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
