import numpy as np
import matplotlib.pyplot as plt

def compute_iou(pred, target, num_classes):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        iou = intersection / union if union != 0 else 0
        ious.append(iou)
    return np.mean(ious)

def visualize_prediction(image, pred_mask):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image.permute(1, 2, 0))
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(pred_mask, cmap="jet")
    plt.title("Predicted Mask")
    plt.axis("off")
    plt.savefig("segmentation/segmentation_result.png", bbox_inches='tight', dpi=300)
    plt.close()
