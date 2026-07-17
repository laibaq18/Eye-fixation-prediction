import torch
import torch.nn as nn
import math


class LearnedPriors(nn.Module):

    def __init__(self, n_priors=16):

        super().__init__()

        self.n_priors = n_priors

        # paper initialization
        self.mu_x = nn.Parameter(
            torch.empty(n_priors).uniform_(0.3, 0.7)
        )

        self.mu_y = nn.Parameter(
            torch.empty(n_priors).uniform_(0.3, 0.7)
        )

        self.sigma_x = nn.Parameter(
            torch.empty(n_priors).uniform_(0.05, 0.3)
        )

        self.sigma_y = nn.Parameter(
            torch.empty(n_priors).uniform_(0.05, 0.3)
        )

    def forward(self, feature):

        B, C, H, W = feature.shape

        mu_x = torch.clamp(
            self.mu_x,
            min=0.25,
            max=0.75
        )

        mu_y = torch.clamp(
            self.mu_y,
            min=0.35,
            max=0.65
        )

        sigma_x = torch.clamp(
            self.sigma_x,
            min=0.1,
            max=0.9
        )

        sigma_y = torch.clamp(
            self.sigma_y,
            min=0.2,
            max=0.8
        )

        aspect = H / W


        y_min = (1.0 - aspect) / 2.0
        y_max = y_min + aspect

        y = torch.linspace(
            y_min,
            y_max,
            H,
            device=feature.device
        )

        x = torch.linspace(
            0.0,
            1.0,
            W,
            device=feature.device
        )
        yy, xx = torch.meshgrid(
            y,
            x,
            indexing="ij"
        )

        priors = []

        for i in range(self.n_priors):

            gaussian = (
                1.0 /
                (
                    2.0
                    * math.pi
                    * sigma_x[i]
                    * sigma_y[i]
                    + 1e-8
                )
            ) * torch.exp(
                -(
                    (
                        (xx - mu_x[i]) ** 2
                        /
                        (
                            2 * sigma_x[i] ** 2
                            + 1e-8
                        )
                    )
                    +
                    (
                        (yy - mu_y[i]) ** 2
                        /
                        (
                            2 * sigma_y[i] ** 2
                            + 1e-8
                        )
                    )
                )
            )

            gaussian = gaussian / (
                gaussian.max() + 1e-8
            )

            priors.append(gaussian)

        priors = torch.stack(
            priors,
            dim=0
        )

        priors = priors.unsqueeze(0)

        priors = priors.repeat(
            B,
            1,
            1,
            1
        )

        return torch.cat(
            [feature, priors],
            dim=1
        )