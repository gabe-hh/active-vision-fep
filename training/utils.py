import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import ImageEnhance, ImageFilter
import math

def init_weights(m):
    if isinstance(m, nn.Linear):
        # For Linear layers
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        # For Convolutional layers
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def reshape_to_heatmap(tensor):
    length = tensor.numel()
    sqrt_len = int(math.sqrt(length))

    # Find the factors of length closest to its square root
    for i in range(sqrt_len, 0, -1):
        if length % i == 0:
            factor1 = i
            factor2 = length // i
            break

    # Reshape the tensor
    return tensor.view(1, factor1, factor2)

def enhance_image(images):
    # Define the transformations
    tensor_to_pil = transforms.ToPILImage()  # Handles single images only
    pil_to_tensor = transforms.PILToTensor()
    
    enhanced_images = []

    # Process each image in the batch
    for i in range(images.size(0)):  # Iterate over the batch dimension
        image = images[i]
        pil_image = tensor_to_pil(image)
        smoothed_image = pil_image.filter(ImageFilter.SMOOTH)
        enhancer = ImageEnhance.Contrast(smoothed_image)
        contrast_enhanced_image = enhancer.enhance(1.5)
        edge_enhanced_image = contrast_enhanced_image.filter(ImageFilter.EDGE_ENHANCE_MORE)
        tensor_image = pil_to_tensor(edge_enhanced_image)
        enhanced_images.append(tensor_image)

    # Stack the list of tensors along a new dimension, effectively creating a batch
    enhanced_images_batch = torch.stack(enhanced_images) / 255.0

    return enhanced_images_batch

def is_multi_nested(lst):
    # Check if the input is a list
    if not isinstance(lst, list):
        return False
    # Iterate through the list to check for nested lists
    for item in lst:
        if isinstance(item, list):
            return True  # Found a nested list
    return False  # No nested lists found