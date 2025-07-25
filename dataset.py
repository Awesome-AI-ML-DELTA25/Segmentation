from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torch
import numpy as np

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_arrays, transform=None):
        self.image_paths = image_paths
        self.mask_arrays = mask_arrays
        self.transform = transform or T.Compose([
            T.Resize((352, 352)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = self.mask_arrays[idx]
        img = self.transform(img)
        # Resize mask if needed
        mask_pil = Image.fromarray(mask)
        mask_pil = T.Resize((352, 352), interpolation=T.InterpolationMode.NEAREST)(mask_pil)
        mask_tensor = torch.from_numpy(np.array(mask_pil)).long()
        return img, mask_tensor

