import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

def build_model(num_classes):
    """
    Builds a ConvNeXt-tiny model with a modified classification head.
    Uses pretrained ImageNet weights as the starting point (as per paper specs).
    """
    # Load the pretrained ConvNeXt-tiny model
    weights = ConvNeXt_Tiny_Weights.DEFAULT
    model = convnext_tiny(weights=weights)

    # In ConvNeXt, the classifier is a Sequential block.
    # We need to find the number of input features to the final Linear layer.
    in_features = model.classifier[2].in_features

    # Replace the final layer with a new one matching our exact class count
    model.classifier[2] = nn.Linear(in_features, num_classes)

    return model
