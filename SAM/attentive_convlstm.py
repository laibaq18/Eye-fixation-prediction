import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentiveConvLSTM(nn.Module):

    def __init__(self, channels=512):

        super().__init__()

        self.channels = channels

        #
        # Attention Module
        #

        self.Wa = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=True
        )

        self.Ua = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=True
        )

        self.Va = nn.Conv2d(
            channels,
            1,
            kernel_size=3,
            padding=1,
            bias=False
        )

        #
        # ConvLSTM Gates
        #

        self.W_i = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.U_i = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.W_f = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.U_f = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.W_c = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.U_c = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.W_o = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.U_o = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.initialize_weights()

    def initialize_weights(self):

        #
        # attentive_init='zero'
        #

        nn.init.zeros_(self.Va.weight)

        #
        # inner_init='orthogonal'
        #

        for layer in [
            self.U_i,
            self.U_f,
            self.U_c,
            self.U_o
        ]:
            nn.init.orthogonal_(
                layer.weight
            )

            if layer.bias is not None:
                nn.init.zeros_(
                    layer.bias
                )

    def forward(
        self,
        X,
        steps=4
    ):

        B, C, H, W = X.shape

        Ht = torch.zeros_like(X)
        Ct = torch.zeros_like(X)

        for _ in range(steps):

            #
            # Attention
            #

            e = self.Va(
                torch.tanh(
                    self.Wa(Ht)
                    +
                    self.Ua(X)
                )
            )

            a = F.softmax(
                e.view(B, -1),
                dim=1
            )

            a = a.view(
                B,
                1,
                H,
                W
            )

            X_tilde = X * a

            #
            # ConvLSTM
            #

            i = torch.sigmoid(
                self.W_i(X_tilde)
                +
                self.U_i(Ht)
            )

            f = torch.sigmoid(
                self.W_f(X_tilde)
                +
                self.U_f(Ht)
            )

            g = torch.tanh(
                self.W_c(X_tilde)
                +
                self.U_c(Ht)
            )

            o = torch.sigmoid(
                self.W_o(X_tilde)
                +
                self.U_o(Ht)
            )

            Ct = (
                f * Ct
                +
                i * g
            )

            Ht = (
                o
                *
                torch.tanh(Ct)
            )

        return Ht