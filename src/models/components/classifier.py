import torch
from torch import nn
import torch.nn.functional as F

class SimpleDenseNetDropout(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        input_size: int = 16,
        hidden_size: int = 1024,
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

        self.fc1 = nn.Linear(3 * 3 * input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size) 
        self.dropout = nn.Dropout(p=0.2) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """

        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.softmax(x)

        return x


if __name__ == "__main__":
    _ = SimpleDenseNetDropout()
