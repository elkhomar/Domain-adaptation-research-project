from ot import solve_sample, dist
from ot.lp import emd2
import torch
import torch.nn as nn

# def wasserstein(source, target):
#     a = torch.ones((source.shape[0]) / source.shape[0])
#     b = torch.ones((target.shape[0]) / target.shape[0])
#     M =  dist(source, target, metric='euclidean')
#     loss = emd2(a=a, b=b, M=M)
#     return loss

class MMDLoss(nn.Module):
    def __init__(self):
        super(MMDLoss, self).__init__()

    def gram_RBF(self, x, y, sigma):
        x_size = x.shape[0]
        y_size = y.shape[0]
        dim = x.shape[1]
        x = x.unsqueeze(1)  # Shape: [x_size, 1, dim]
        y = y.unsqueeze(0)  # Shape: [1, y_size, dim]
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        squared_diff = (tiled_x - tiled_y) ** 2
        squared_dist = torch.sum(squared_diff, -1)  # Sum over the feature dimension
        return torch.exp(-squared_dist / sigma)

    def forward(self, input, target, sigma):
        XX = self.gram_RBF(input, input, sigma)
        YY = self.gram_RBF(target, target, sigma)
        XY = self.gram_RBF(input, target, sigma)
        loss = torch.mean(XX) + torch.mean(YY) - 2 * torch.mean(XY)
        return loss

def wasserstein(source_features, target_features):
    result = solve_sample(source_features, target_features)
    loss = result.value
    return loss

class WassersteinDistanceCalculator:
    def __init__(self, metric='euclidean'):
        self.metric = metric

    def wasserstein(self, source, target):
        """
        Compute the Wasserstein distance between source and target distributions.
        
        Parameters:
        - source: torch.Tensor, shape (n_source_samples, n_features)
            Feature tensor for the source distribution.
        - target: torch.Tensor, shape (n_target_samples, n_features)
            Feature tensor for the target distribution.
            
        Returns:
        - loss: float
            The computed Wasserstein distance.
        """
        a = torch.empty(source.shape[0])
        b = torch.empty(target.shape[0])

        M = dist(source, target, metric='euclidean')

        loss = emd2(a, b, M)
        return loss
