import torch
from torch import nn
import torch.nn.functional as F


class CNNFeatureExtractor(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        conv1_size: int = 6,
        conv2_size: int = 16,
        conv3_size: int = 16,
        output_size: int = 10,
    ) -> None:
        """Initialize a `SimpleDenseNet` module.

        :param input_size: The number of input features.
        :param lin1_size: The number of output features of the first linear layer.
        :param lin2_size: The number of output features of the second linear layer.
        :param lin3_size: The number of output features of the third linear layer.
        :param output_size: The number of output features of the final linear layer.
        """
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=conv1_size, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(conv1_size)
        self.conv2 = nn.Conv2d(in_channels=conv1_size, out_channels=conv2_size, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(conv2_size)
        self.conv3 = nn.Conv2d(in_channels=conv2_size, out_channels=conv3_size, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(conv3_size)
        self.dropout = nn.Dropout(p=0.2) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        batch_size, channels, width, height = x.size()

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = x.flatten(start_dim=1)

        return x


if __name__ == "__main__":
    _ = CNNFeatureExtractor()
