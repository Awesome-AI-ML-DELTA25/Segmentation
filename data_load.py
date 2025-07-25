import kagglehub
import os
import pandas as pd
import numpy as np
from PIL import Image

# Download latest version
path = kagglehub.dataset_download("lucasiturriago/breast-cancer-multi-annotators")

print("Path to dataset files:", path)

# /Users/saicharan/.cache/kagglehub/datasets/lucasiturriago/breast-cancer-multi-annotators/versions/6
'''
# List all files in the downloaded dataset directory
first_level = os.listdir(path)
print(first_level)

for dir in first_level:
    dir_path = os.path.join(path, dir)
    if os.path.isdir(dir_path):
        print(f"Directory: {dir}")
        files = os.listdir(dir_path)
        print(f"  Files: {files}")
'''

class ImagePaths_from_dir():
    def __init__(self, path, num_classes=3):
        self.path = path
        self.image_paths = []
        self.combined_masks = []
        self.num_classes = num_classes
        self._load_pairs()

    def _load_pairs(self):
        patches_dir = os.path.join(self.path, 'patches')
        masks_root = os.path.join(self.path, 'masks/ground_truth')
        if not os.path.exists(patches_dir) or not os.path.exists(masks_root):
            print(patches_dir, masks_root)
            raise ValueError("Missing patches or masks directory.")

        # Get all image paths (order matters)
        self.image_paths = sorted([os.path.join(patches_dir, f) for f in os.listdir(patches_dir)])

        # Get all mask paths for each class (order matters)
        class_mask_lists = []
        for class_idx in range(self.num_classes):
            class_dir = os.path.join(masks_root, f'class_{class_idx}')
            mask_list = sorted([os.path.join(class_dir, f) for f in os.listdir(class_dir)])
            class_mask_lists.append(mask_list)

        # For each image, combine the corresponding masks from each class
        for i, img_path in enumerate(self.image_paths):
            mask_shape = None
            combined_mask = None
            for class_idx, mask_list in enumerate(class_mask_lists):
                mask_path = mask_list[i]
                mask = np.array(Image.open(mask_path))
                if mask_shape is None:
                    mask_shape = mask.shape
                    combined_mask = np.zeros(mask_shape, dtype=np.uint8)
                combined_mask[mask == 1] = class_idx
            self.combined_masks.append(combined_mask)

    def get_image_mask_pairs(self,num_samples=None):
        """
        return self.image_paths, self.combined_masks
        """
        if num_samples is not None:
            return self.image_paths[:num_samples], self.combined_masks[:num_samples]
        return self.image_paths, self.combined_masks
