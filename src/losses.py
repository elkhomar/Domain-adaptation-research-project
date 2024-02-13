from ot import solve_sample, dist
import torch
import torch.functional as F
import torch.nn as nn
from ot.lp import emd2

def calculate_mean_distance(mnist_dataloader, usps_dataloader):
    total_distance = 0
    count = 0

    # Assurez-vous que les DataLoader retournent des lots de la même taille
    # ou gérez les cas où les derniers lots pourraient être de tailles différentes.
    for (mnist_images, _), (usps_images, _) in zip(mnist_dataloader, usps_dataloader):
        # Aplatir les images
        mnist_flat = mnist_images.view(mnist_images.shape[0], -1)
        usps_flat = usps_images.view(usps_images.shape[0], -1)
        # Calculer les distances euclidiennes
        distances = torch.cdist(mnist_flat, usps_flat, p=2)

        # Ajouter la distance moyenne de ce lot
        total_distance += distances.mean()
        count += 1

    # Calculer la distance moyenne globale
    mean_distance = total_distance / count
    return mean_distance.item()

class MMDLoss(nn.Module):
    def __init__(self, sigma=32):
        super(MMDLoss, self).__init__()
        self.sigma = sigma

    def calculate_mean_distance(self, mnist_dataloader, usps_dataloader):
        total_distance = 0
        count = 0

        # Assurez-vous que les DataLoader retournent des lots de la même taille
        # ou gérez les cas où les derniers lots pourraient être de tailles différentes.
        for (mnist_images, _), (usps_images, _) in zip(mnist_dataloader, usps_dataloader):
            # Aplatir les images
            mnist_flat = mnist_images.view(mnist_images.shape[0], -1)
            usps_flat = usps_images.view(usps_images.shape[0], -1)

            # Calculer les distances euclidiennes
            distances = torch.cdist(mnist_flat, usps_flat, p=2)

            # Ajouter la distance moyenne de ce lot
            total_distance += distances.mean()
            count += 1
            # Calculer la distance moyenne globale
            mean_distance = total_distance / count
            return mean_distance.item()

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

    def forward(self, input, target, sigma=32):
        XX = self.gram_RBF(input, input, sigma)
        YY = self.gram_RBF(target, target, sigma)
        XY = self.gram_RBF(input, target, sigma)
        loss = torch.mean(XX) + torch.mean(YY) - 2 * torch.mean(XY)
        return loss

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

class CoralLoss(nn.Module):
    def __init__(self):
        super(CoralLoss, self).__init__()

    def forward(self, source, target):
        d = source.size(1)  # dim vector

        source_c = self.compute_covariance(source)
        target_c = self.compute_covariance(target)

        loss = torch.sum(torch.mul((source_c - target_c), (source_c - target_c)))

        loss = loss / (4 * d * d)
        return loss

    def compute_covariance(self, input_data):
        """
        Compute Covariance matrix of the input data
        """
        n = input_data.size(0)  # batch_size

        # Check if using gpu or cpu
        if input_data.is_cuda:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        id_row = torch.ones(n).resize(1, n).to(device=device)
        sum_column = torch.mm(id_row, input_data)
        mean_column = torch.div(sum_column, n)
        term_mul_2 = torch.mm(mean_column.t(), mean_column)
        d_t_d = torch.mm(input_data.t(), input_data)
        c = torch.add(d_t_d, (-1 * term_mul_2)) * 1 / (n - 1)

        return c
