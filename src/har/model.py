import torch
import torch.nn as nn


class TemporalBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()

        padding = (
            kernel_size - 1
        ) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )

        self.bn1 = nn.BatchNorm1d(
            out_channels
        )

        self.relu1 = nn.ReLU()

        self.dropout1 = nn.Dropout(
            dropout
        )

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )

        self.bn2 = nn.BatchNorm1d(
            out_channels
        )

        self.relu2 = nn.ReLU()

        self.dropout2 = nn.Dropout(
            dropout
        )

        if in_channels != out_channels:

            self.residual = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
            )

        else:

            self.residual = nn.Identity()

    def forward(self, x):

        residual = self.residual(x)

        out = self.conv1(x)

        # Remove extra right-side padding
        # introduced by causal-style convolution.
        out = out[:, :, :x.size(2)]

        out = self.bn1(out)

        out = self.relu1(out)

        out = self.dropout1(out)

        out = self.conv2(out)

        out = out[:, :, :x.size(2)]

        out = self.bn2(out)

        out = self.relu2(out)

        out = self.dropout2(out)

        return torch.relu(
            out + residual
        )


class TCN(nn.Module):

    def __init__(
        self,
        input_features: int,
        num_classes: int,
        channels=(32, 64, 64),
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        layers = []

        in_channels = input_features

        for index, out_channels in enumerate(
            channels
        ):

            dilation = 2 ** index

            layers.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )

            in_channels = out_channels

        self.network = nn.Sequential(
            *layers
        )

        self.classifier = nn.Linear(
            in_channels,
            num_classes
        )

    def forward(self, x):
        """
        Input:
            [batch, sequence_length, features]

        Output:
            [batch, num_classes]
        """

        # Conv1d expects:
        #
        # [batch, channels, sequence]
        x = x.transpose(
            1,
            2
        )

        x = self.network(x)

        # Use the final temporal position.
        x = x[:, :, -1]

        return self.classifier(x)