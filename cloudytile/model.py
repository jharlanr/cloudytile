"""
CNN model for binary classification of satellite tiles based on visual utility.
"""
import torch
import torch.nn as nn


class CloudyTileCNN(nn.Module):
    """
    Classifies whether a satellite tile is "useful" (1) or "not useful" (0)
    based on cloud coverage and/or presence of no-data-pixels.

    Args:
        img_size: Input image size [height, width]. Default: (512, 512)
        channels: List of channel sizes for conv layers. Default: [16, 32, 64]
        fc_layers: List of hidden layer sizes for classifier. Default: [128]
        in_channels: Number of input channels. Default: 6
            - 3 for RGB only (legacy JPG mode)
            - 6 for RGB + NIR + SWIR1 + SWIR2 (multi-spectral NC mode)

    Input:
        x: Tensor of shape [B, in_channels, H, W] with values normalized to [0, 1]

    Output:
        Tensor of shape [B] with logits (use sigmoid for probabilities)
    """

    def __init__(
        self,
        img_size: tuple[int, int] = (512, 512),
        channels: list[int] = None,
        fc_layers: list[int] = None,
        in_channels: int = 6,
    ):
        super().__init__()

        if channels is None:
            channels = [16, 32, 64]
        if fc_layers is None:
            fc_layers = [128]

        self.img_size = img_size
        self.channels = channels
        self.fc_layers = fc_layers
        self.in_channels = in_channels

        # build convolutional blocks
        conv_layers = []
        prev_channels = in_channels
        for out_channels in channels:
            conv_layers.extend([
                nn.Conv2d(prev_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ])
            prev_channels = out_channels

        self.features = nn.Sequential(*conv_layers)

        # after len(channels) MaxPool2d(2), spatial dimensions are reduced by 2^len(channels)
        reduction = 2 ** len(channels)
        flat_size = channels[-1] * (img_size[0] // reduction) * (img_size[1] // reduction)

        # build classifier head
        fc = [nn.Flatten()]
        prev_size = flat_size
        for hidden_size in fc_layers:
            fc.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
            ])
            prev_size = hidden_size
        fc.append(nn.Linear(prev_size, 1))

        self.classifier = nn.Sequential(*fc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x.squeeze(1)


        