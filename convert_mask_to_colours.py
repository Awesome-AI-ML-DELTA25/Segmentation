import os
from PIL import Image
import numpy as np

# Define a color map for 3 classes (0, 1, 2)
COLORS = {
    0: (0, 0, 255),    # blue for background
    1: (255, 0, 0),    # red for class 1
    2: (0, 255, 0),    # green for class 2
}

def mask_to_color(mask):
    # mask: 2D numpy array of class indices (uint8)
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_idx, color in COLORS.items():
        color_mask[mask == class_idx] = color
    return color_mask

def print_mask_classes(mask, fname):
    unique_classes = np.unique(mask)
    print(f"{fname}: unique classes in mask: {unique_classes}")
    for class_idx in unique_classes:
        if class_idx in COLORS:
            print(f"Class {class_idx}: Color {COLORS[class_idx]}")
        else:
            print(f"Class {class_idx}: No color defined")

input_dir = 'segmentation/predictions/'
output_dir = 'segmentation/predictions/colored/'
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    print(f"Processing file: {fname}")
    if fname.endswith('.png'):
        mask = np.array(Image.open(os.path.join(input_dir, fname)))
        print_mask_classes(mask, fname)
        color_mask = mask_to_color(mask)
        out_path = os.path.join(output_dir, fname)
        Image.fromarray(color_mask).save(out_path)
        # Optionally display the image
        # Image.fromarray(color_mask).show()
