import torch
from tqdm import tqdm
from util import compute_iou
import torchvision.utils as vutils
import os
from PIL import Image
import numpy as np

def train_model(model, dataloader, criterion, optimizer, device, num_classes, dataset=None):
    model.to(device)
    pred_dir = "segmentation/predictions/"
    os.makedirs(pred_dir, exist_ok=True)
    for epoch in range(50):
        model.train()
        total_loss = 0
        total_iou = 0
        for images, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            iou = compute_iou(preds, masks, num_classes)
            total_iou += iou

        avg_loss = total_loss / len(dataloader)
        avg_iou = total_iou / len(dataloader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, mIoU = {avg_iou:.4f}")

        # Save prediction for dataset[0] after each epoch
        model.eval()
        with torch.no_grad():
            if dataset is not None:
                sample_img, _ = dataset[0]
                sample_img = sample_img.unsqueeze(0).to(device)
                pred = model(sample_img)
                pred_mask = torch.argmax(pred, dim=1).squeeze().cpu().numpy().astype(np.uint8)
                out_path = os.path.join(pred_dir, f"epoch_{epoch+1}_prediction.png")
                Image.fromarray(pred_mask).save(out_path)
        model.train()
