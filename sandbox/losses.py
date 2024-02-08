from ot import solve_sample
import torch

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
