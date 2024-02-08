from ot import solve_sample, dist
from ot.lp import emd2
import torch
import torch.nn as nn

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

# def wasserstein(source_features, target_features):
#     result = solve_sample(source_features, target_features, metric='sqeuclidean', reg=0.1, method='sinkhorn', max_iter=2000)
#     if hasattr(result, 'value'): 
#         loss_value = result.value
#         loss = torch.tensor(loss_value, dtype=torch.float32, requires_grad=True).to(source_features.device)
#     return loss

class WassersteinDistanceCalculator:
    def __init__(self, metric='euclidean'):
        self.metric = metric

    def compute_cost_matrix(self, source, target):
        """
        Compute the cost matrix using the specified metric.
        
        Parameters:
        - source: torch.Tensor, shape (n_source_samples, n_features)
            Feature tensor for the source distribution.
        - target: torch.Tensor, shape (n_target_samples, n_features)
            Feature tensor for the target distribution.
            
        Returns:
        - cost_matrix: np.ndarray, shape (n_source_samples, n_target_samples)
            The cost matrix representing the pairwise distances.
        """
        # Convert tensors to numpy for compatibility with the ot library
        source_np = source.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        # Compute the cost matrix
        cost_matrix = dist(source_np, target_np, metric=self.metric)
        return cost_matrix

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
        a = torch.ones(source.shape[0], dtype=torch.float32) / source.shape[0]
        b = torch.ones(target.shape[0], dtype=torch.float32) / target.shape[0]
        
        # Convert histograms to numpy for compatibility with the ot library
        a_np = a.detach().cpu().numpy()
        b_np = b.detach().cpu().numpy()

        # Compute the cost matrix
        M = self.compute_cost_matrix(source, target)

        # Compute the EMD
        loss = emd2(a_np, b_np, M)
        return loss
