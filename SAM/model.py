import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights

from SAM.attentive_convlstm import AttentiveConvLSTM
from SAM.learned_priors import LearnedPriors


class EyeFixationSAMResNet(nn.Module):

    def __init__(self):

        super().__init__()

        backbone = resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V1,
            replace_stride_with_dilation=[False, True, True]
        )

        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,

            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        self.reduce = nn.Sequential(
            nn.Conv2d(
                2048,
                512,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True)
        )

        self.attentive_lstm = AttentiveConvLSTM(
            channels=512
        )

        # PRIOR BLOCK 1
        self.priors1 = LearnedPriors(16)

        self.prior_conv1 = nn.Sequential(
            nn.Conv2d(
                512 + 16,
                512,
                kernel_size=5,
                dilation=4,
                padding=8
            ),
            nn.ReLU(inplace=True)
        )

        # PRIOR BLOCK 2
        self.priors2 = LearnedPriors(16)

        self.prior_conv2 = nn.Sequential(
            nn.Conv2d(
                512 + 16,
                512,
                kernel_size=5,
                dilation=4,
                padding=8
            ),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(
                512,
                1,
                kernel_size=1
            ),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):

        input_h, input_w = x.shape[-2:]

        x = self.encoder(x)

        x = self.reduce(x)

        x = self.attentive_lstm(
            x,
            steps=4
        )

        # PRIOR BLOCK 1
        x = self.priors1(x)
        x = self.prior_conv1(x)

        # PRIOR BLOCK 2
        x = self.priors2(x)
        x = self.prior_conv2(x)

        x = self.final_conv(x)

        x = F.interpolate(
            x,
            size=(input_h, input_w),
            mode="bilinear",
            align_corners=False
        )

        return x
