import torch
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from dataset import SegmentationDataset
from model import FCN
from train import train_model
from util import visualize_prediction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import kagglehub
import os
import pandas as pd
from data_load import ImagePaths_from_dir
# Download latest version
path = kagglehub.dataset_download("lucasiturriago/breast-cancer-multi-annotators")

print("Path to dataset files:", path)
#Something like this /Users/user_name/.cache/kagglehub/datasets/lucasiturriago/breast-cancer-multi-annotators/versions/6


training_data = ImagePaths_from_dir(os.path.join(path, 'Train'))

image_paths, mask_paths = training_data.get_image_mask_pairs(200)

# Use a subset for quick testing
dataset = SegmentationDataset(image_paths, mask_paths)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Initialize model, loss, optimizer
num_classes = 3  #  3 classes for breast cancer segmentation
model = FCN(num_classes=num_classes)
criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Train
train_model(model, dataloader, criterion, optimizer, device, num_classes, dataset)
torch.save(model.state_dict(), "segmentation/model.pth")
# Visualize one prediction
model.eval()

img, mask = dataset[0]
with torch.no_grad():
    pred = model(img.unsqueeze(0).to(device))
pred_mask = torch.argmax(pred, dim=1).squeeze().cpu()
visualize_prediction(img, pred_mask)
