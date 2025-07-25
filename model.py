import torch.nn as nn
import torchvision.models as models

class FCN(nn.Module):
    def __init__(self, num_classes):
        super(FCN, self).__init__()
        self.backbone = models.resnet18(weights="DEFAULT")
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        # For input 352x352, ResNet18 backbone outputs 11x11 feature map
        # To upsample 11x11 to 352x352: scale factor = 32
        # Use kernel_size=64, stride=32, padding=16, output_padding=0 for exact match
        self.head = nn.Sequential(
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, stride=32, padding=16, output_padding=0)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
