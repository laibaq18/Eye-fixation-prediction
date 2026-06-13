import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import fcn_resnet50


"""
create a 1D Gaussian filter with 25 values (window_size = 25)
"""

def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    dtype = torch.float32
    device = None

    # create a list of 25 values from 0...24
    # then subtract from each val 12 -> window_size//2
    # this would make the values range from -12 to 12
    x = torch.arange(window_size, device=device,
                     dtype=dtype) - window_size // 2
    if window_size % 2 == 0:
        x = x + 0.5

    # gaussian formula
    gauss = torch.exp(-x.pow(2.0) / (2 * sigma ** 2))

    # normalise values .. sum of values = 1
    g = gauss / gauss.sum()
    return g


class EyeFixationFCN(nn.Module):
    def __init__(
        self,
        center_bias_path,
        freeze_backbone=True,  # the ResNet-50 feature extractor will not be trained. Only the decoder part will learn
        image_size=(224, 224),
    ):
        super().__init__()

        # ResNet-50 backbone pretrained on ImageNet.
        # Decoder is NOT COCO pretrained because weights=None.
        self.fcn = fcn_resnet50(
            weights=None,
            weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
            num_classes=1,
            aux_loss=False,
        )

        if freeze_backbone:
            for param in self.fcn.backbone.parameters():
                param.requires_grad = False

        # Gaussian smoothing kernel: window size 25, sigma 11.2
        g = gaussian(window_size=25, sigma=11.2)

        """
        create a 2d kernel from 1D gaussian kernel returned
        via matrix multiplication
        size : [25, 25] 
        """
        kernel_2d = torch.matmul(g.unsqueeze(1), g.unsqueeze(0))

        # conv2d expects kernel shape:
        # [out_channels, in_channels, kernel_height, kernel_width]
        # [1, 1, 25, 25]
        kernel_4d = kernel_2d.unsqueeze(0).unsqueeze(0)

        """ Storing smoothing kernel as a non-trainable parameter """
        self.smoothing_kernel = nn.Parameter(
            kernel_4d,
            requires_grad=False,
        )

        # Load center bias density
        center_bias = np.load(center_bias_path).astype("float32")
        center_bias = torch.from_numpy(center_bias)

        # Remove unnecessary dimensions, [1, 224, 224] -> [224, 224]
        center_bias = center_bias.squeeze()

        # Safety check
        if center_bias.ndim != 2:
            raise ValueError(
                f"Expected center bias to be 2D [H, W], but got shape {center_bias.shape}"
            )

        # Normalize again just to be safe
        center_bias = center_bias / center_bias.sum()

        # Avoid log(0)
        center_bias = torch.clamp(center_bias, min=1e-12)
        log_center_bias = torch.log(center_bias)

        # Shape: [1, 1, H, W], so it broadcasts over batch -> [1, 1, 224, 224]
        log_center_bias = log_center_bias.unsqueeze(0).unsqueeze(0)

        """ Also Storing center bias as a non-trainable parameter """
        self.log_center_bias = nn.Parameter(
            log_center_bias,
            requires_grad=False,
        )

    def forward(self, x):
        # x: [B, 3, 224, 224]

        output = self.fcn(x)

        # torchvision segmentation models return a dictionary
        raw_logits = output["out"]  # [B, 1, H, W]

        smoothed_logits = F.conv2d(
            raw_logits,
            self.smoothing_kernel,
            padding="same", #keeps height and width unchanged
        )
        # smoothed_logits -> # [B, 1, H, W]

        center_bias = self.log_center_bias

        # print("raw_logits:", raw_logits.shape)
        # print("smoothed_logits:", smoothed_logits.shape)
        # print("center_bias:", center_bias.shape)

        # If center bias size differs from model output size, resize it
        if center_bias.shape[-2:] != smoothed_logits.shape[-2:]:
            print("Resizing center bias as it differs from model output size")
            center_bias = F.interpolate(
                center_bias,
                size=smoothed_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        final_logits = smoothed_logits + center_bias

        return final_logits # [B, 1, 224, 224]
