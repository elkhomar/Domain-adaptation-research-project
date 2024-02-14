import torch
from torch import nn
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        conv1_size: int = 32,
        conv2_size: int = 64,
    ) -> None:
        """Initialize a `SimpleDenseNet` module.

        :param input_size: The number of input features.
        :param lin1_size: The number of output features of the first linear layer.
        :param lin2_size: The number of output features of the second linear layer.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(1, conv1_size, 3, 1)
        self.conv2 = nn.Conv2d(conv1_size, conv2_size, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)

        return x

class Classifier(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        input_size: int = 9216,
        hidden_size: int = 128,
    ) -> None:
        """Initialize a `SimpleDenseNet` module.

        :param input_size: The number of input features.
        :param lin1_size: The number of output features of the first linear layer.
        :param lin2_size: The number of output features of the second linear layer.
        :param lin3_size: The number of output features of the third linear layer.
        :param output_size: The number of output features of the final linear layer.
        """
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        logits = self.fc2(x)
        return F.softmax(logits)

class Skada_net(nn.Module):
    """A simple fully-connected neural net for computing predictions."""
    def __init__(
        self,
    ) -> None:
        """Initialize a `SimpleDenseNet` module.

        :param input_size: The number of input features.
        :param lin1_size: The number of output features of the first linear layer.
        :param lin2_size: The number of output features of the second linear layer.
        :param lin3_size: The number of output features of the third linear layer.
        :param output_size: The number of output features of the final linear layer.
        """
        super().__init__()

        self._classifier = Classifier()
        self._feature_extractor = FeatureExtractor()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        features = self._feature_extractor(x)
        logits = self._classifier(features)
        return logits


if __name__ == "__main__":
    _ = FeatureExtractor()
