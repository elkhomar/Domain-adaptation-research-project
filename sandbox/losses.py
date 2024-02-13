from ot import solve_sample
import torch
import torch.nn as nn
from deeptime.kernels import GaussianKernel

def wasserstein(source_features, target_features):
    result = solve_sample(source_features, target_features, metric='sqeuclidean', reg=0.1, method='sinkhorn', max_iter=2000)
    if hasattr(result, 'value'): 
        discrepancy_loss_value = result.value
        discrepancy_loss = torch.tensor(discrepancy_loss_value, dtype=torch.float32, requires_grad=True).to(source_features.device)
    return discrepancy_loss

def coral(source, target):

    d = source.size(1)  # dim vector

    source_c = compute_covariance(source)
    target_c = compute_covariance(target)

    loss = torch.sum(torch.mul((source_c - target_c), (source_c - target_c)))

    loss = loss / (4 * d * d)
    return loss


def compute_covariance(input_data):
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

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mmd(input, target, sigma):
    def gram_RBF(x, y, sigma):
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

    XX = gram_RBF(input, input, sigma)
    YY = gram_RBF(target, target, sigma)
    XY = gram_RBF(input, target, sigma)
    loss = torch.mean(XX) + torch.mean(YY) - 2 * torch.mean(XY)
    return loss

    
