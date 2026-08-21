"""
CNN model for binary classification of satellite tiles based on visual utility.
"""
import torch
import torch.nn as nn


class CloudyTileCNN(nn.Module):
    """
    Classifies whether a satellite tile is "useful" (1) or "not useful" (0)
    based on cloud coverage and/or presence of no-data pixels.

    Args:
        img_size: Input size [height, width]; only used by the legacy flatten
            head. The default GAP head is input-size agnostic.
        channels: Channel sizes for conv layers. Default: [16, 32, 64]
        fc_layers: Hidden layer sizes for the classifier head. Default: [128]
        in_channels: Number of input channels. Default: 6 (full SDR band set);
            3 for RGB-only.
        head: "gap" (default) pools each feature channel to its global average
            before the dense head, so head size is independent of image area.
            "flatten" is the legacy head: Linear on the flattened feature map,
            which at 512x512 puts 33.5M of the model's 33.6M parameters in a
            single layer (262,144 x 128). Kept only to load old checkpoints.
        batch_norm: BatchNorm2d after each conv. Must be False to load
            checkpoints from before August 2026.
        dropout: Dropout probability before the first dense layer (0 disables,
            and keeps the flatten head's module indices identical to legacy
            checkpoints).

    Input:
        x: Tensor of shape [B, in_channels, H, W], normalized per band
            (NaN pixels set to exactly 0.0 after normalization).

    Output:
        Tensor of shape [B] with logits (use sigmoid for probabilities)
    """

    def __init__(
        self,
        img_size: tuple[int, int] = (512, 512),
        channels: list[int] = None,
        fc_layers: list[int] = None,
        in_channels: int = 6,
        head: str = "gap",
        head_reduce: int = None,
        batch_norm: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()

        if channels is None:
            channels = [16, 32, 64]
        if fc_layers is None:
            fc_layers = [128]
        # "gap"      -> AdaptiveAvgPool2d(1): one number per channel. Discards
        #               all spatial layout, and is resolution-independent.
        # "pool<N>"  -> AdaptiveAvgPool2d(N): an NxN grid per channel, so coarse
        #               spatial structure survives into the classifier (where in
        #               the tile the cloud sits, not just how much). Still
        #               resolution-independent, because adaptive pooling fixes
        #               the output size regardless of input size. Pair it with a
        #               narrow fc_layers -- 64x16x16 = 16,384 inputs means an
        #               fc width of 8 costs 131k weights and a width of 128
        #               would cost 2.1M.
        # "flatten"  -> no pooling; first dense layer scales with image AREA.
        #               Resolution-dependent; kept only for old checkpoints.
        pool_n = None
        if head.startswith("pool"):
            try:
                pool_n = int(head[4:])
            except ValueError:
                raise ValueError(f"head 'pool<N>' needs an integer N, got {head!r}")
            if pool_n < 1:
                raise ValueError(f"pool<N> needs N >= 1, got {pool_n}")
        elif head not in ("gap", "flatten"):
            raise ValueError(
                f"head must be 'gap', 'flatten', or 'pool<N>', got {head!r}"
            )

        self.img_size = img_size
        self.channels = channels
        self.fc_layers = fc_layers
        self.in_channels = in_channels
        self.head = head

        # build convolutional blocks
        conv_layers = []
        prev_channels = in_channels
        for out_channels in channels:
            conv_layers.append(
                nn.Conv2d(prev_channels, out_channels, kernel_size=3, padding=1)
            )
            if batch_norm:
                conv_layers.append(nn.BatchNorm2d(out_channels))
            conv_layers.extend([nn.ReLU(), nn.MaxPool2d(2)])
            prev_channels = out_channels

        self.features = nn.Sequential(*conv_layers)
        self.reduce = None

        if head == "gap":
            self.pool = nn.AdaptiveAvgPool2d(1)
            flat_size = channels[-1]
        elif pool_n is not None:
            # head_reduce collapses the channel axis with a 1x1 conv BEFORE
            # pooling. Without it a 16x16 grid over 64 channels flattens to
            # 16,384 values and the first dense layer dominates the model;
            # reducing to a handful of channels first keeps the spatial
            # information while making the flatten cheap. head_reduce=1 yields a
            # single 16x16 "usability map" that the MLP then reads -- the
            # segmentation-style formulation: score every region, then learn how
            # to aggregate, rather than averaging uniformly as GAP does.
            if head_reduce is not None:
                if head_reduce < 1:
                    raise ValueError(f"head_reduce must be >= 1, got {head_reduce}")
                reduce_layers = [nn.Conv2d(channels[-1], head_reduce, kernel_size=1)]
                if batch_norm:
                    reduce_layers.append(nn.BatchNorm2d(head_reduce))
                reduce_layers.append(nn.ReLU())
                self.reduce = nn.Sequential(*reduce_layers)
                pooled_channels = head_reduce
            else:
                pooled_channels = channels[-1]
            self.pool = nn.AdaptiveAvgPool2d(pool_n)
            flat_size = pooled_channels * pool_n * pool_n
        else:
            self.pool = None
            # after len(channels) MaxPool2d(2), spatial dims shrink by 2^len
            reduction = 2 ** len(channels)
            flat_size = channels[-1] * (img_size[0] // reduction) * (img_size[1] // reduction)

        # build classifier head
        fc = [nn.Flatten()]
        if dropout > 0:
            fc.append(nn.Dropout(dropout))
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
        if self.reduce is not None:
            x = self.reduce(x)
        if self.pool is not None:
            x = self.pool(x)
        x = self.classifier(x)
        return x.squeeze(1)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
