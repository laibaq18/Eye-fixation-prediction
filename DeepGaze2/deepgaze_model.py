import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import VGG19_Weights, vgg19


def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    dtype = torch.float32
    x = torch.arange(window_size, dtype=dtype) - window_size // 2

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma ** 2))
    g = gauss / gauss.sum()
    return g


class DeepGaze2(nn.Module):
    """
    DeepGaze-II-style model.

    It uses:
    - frozen ImageNet-pretrained VGG-19 feature extractor
    - intermediate VGG feature maps
    - trainable readout layers
    - Gaussian smoothing
    - log center bias
    """

    def __init__(
        self,
        center_bias_path,
        freeze_backbone=True,
        feature_indices=(28, 29, 31, 32, 35),
        reduced_channels=32,
        use_smoothing=True,
    ):
        super().__init__()

        self.use_smoothing = use_smoothing
        self.feature_indices = set(feature_indices)
        self.max_feature_index = max(feature_indices)

        # ImageNet-pretrained VGG-19.
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)

        # We only need convolutional feature extractor, not the classifier.
        self.vgg_features = vgg.features

        for module in self.vgg_features.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False

        if freeze_backbone:
            for param in self.vgg_features.parameters():
                param.requires_grad = False

        selected_channels = {
            28: 512,  # conv5_1 -> 512 channels
            29: 512,  # relu5_1 -> 512 channels
            31: 512,  # relu5_2 -> 512 channels
            32: 512,  # conv5_3 -> 512 channels
            35: 512,  # relu5_4 -> 512 channels
        }
        # total: 2560 channels

        self.reductions = nn.ModuleList([
            nn.Conv2d(selected_channels[idx], reduced_channels, kernel_size=1)
            for idx in feature_indices
        ])

        total_channels = reduced_channels * len(feature_indices)


        self.readout = nn.Sequential(
            nn.Conv2d(total_channels, 16, kernel_size=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 2, kernel_size=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(2, 1, kernel_size=1),
        )

        # Gaussian smoothing kernel
        g = gaussian(window_size=25, sigma=11.2)
        kernel_2d = torch.matmul(g.unsqueeze(1), g.unsqueeze(0))
        kernel_4d = kernel_2d.unsqueeze(0).unsqueeze(0)

        self.smoothing_kernel = nn.Parameter(
            kernel_4d,
            requires_grad=False,
        )

        # Load center bias density
        center_bias = np.load(center_bias_path).astype("float32")
        center_bias = torch.from_numpy(center_bias)

        # Remove unnecessary dimensions, e.g. [1, 224, 224] -> [224, 224]
        center_bias = center_bias.squeeze()

        if center_bias.ndim != 2:
            raise ValueError(
                f"Expected center bias to be 2D [H, W], but got shape {center_bias.shape}"
            )

        # Normalize just to be safe
        center_bias = center_bias / center_bias.sum()

        # Avoid log(0)
        center_bias = torch.clamp(center_bias, min=1e-12)

        # Use log center bias because model outputs are logits/log-probability-like values.
        log_center_bias = torch.log(center_bias)

        # Shape: [1, 1, H, W], so it broadcasts across batch.
        log_center_bias = log_center_bias.unsqueeze(0).unsqueeze(0)

        self.log_center_bias = nn.Parameter(
            log_center_bias,
            requires_grad=False,
        )

    def forward(self, x):
        input_size = x.shape[-2:]

        features = []
        reduction_idx = 0

        h = x

        for layer_idx, layer in enumerate(self.vgg_features):
            h = layer(h)

            if layer_idx in self.feature_indices:
                h_selected = h.clone()
                reduced = self.reductions[reduction_idx](h_selected)

                #reduced = self.reductions[reduction_idx](h)

                # Upsample each VGG feature map to original image size.
                reduced = F.interpolate(
                    reduced,
                    size=input_size,
                    mode="bilinear",
                    align_corners=False,
                )

                features.append(reduced)
                reduction_idx += 1

            if layer_idx >= self.max_feature_index:
                break

        # Shape: [B, reduced_channels * number_of_layers, H, W]
        features = torch.cat(features, dim=1)

        # Shape: [B, 1, H, W]
        raw_logits = self.readout(features)

        if self.use_smoothing:
            smoothed_logits = F.conv2d(
                raw_logits,
                self.smoothing_kernel,
                padding="same",
            )
        else:
            smoothed_logits = raw_logits

        center_bias = self.log_center_bias

        if center_bias.shape[-2:] != smoothed_logits.shape[-2:]:
            center_bias = F.interpolate(
                center_bias,
                size=smoothed_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        final_logits = smoothed_logits + center_bias

        return final_logits
