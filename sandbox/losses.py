from ot import solve_sample, dist
from ot.lp import emd2
import torch
import torch.nn as nn

def calculate_mean_distance(mnist_dataloader, usps_dataloader):
    total_distance = 0
    count = 0

    # Assurez-vous que les DataLoader retournent des lots de la même taille
    # ou gérez les cas où les derniers lots pourraient être de tailles différentes.
    for (mnist_images, _), (usps_images, _) in zip(mnist_dataloader, usps_dataloader):
        # Aplatir les images
        mnist_flat = mnist_images.view(mnist_images.shape[0], -1)
        usps_flat = usps_images.view(usps_images.shape[0], -1)

        # Adapter les tailles si nécessaire (exemple : interpoler USPS pour correspondre à MNIST)
        # Cette étape dépend de la taille des images MNIST et USPS dans vos DataLoader
        if mnist_flat.shape[1] != usps_flat.shape[1]:
            # Supposons MNIST est 28x28 et USPS est 16x16, ajustez USPS à 28x28
            usps_resized = F.interpolate(usps_images.unsqueeze(1), size=(28, 28), mode='bilinear', align_corners=False)
            usps_flat = usps_resized.view(usps_resized.shape[0], -1)
        
        # Calculer les distances euclidiennes
        distances = torch.cdist(mnist_flat, usps_flat, p=2)

        # Ajouter la distance moyenne de ce lot
        total_distance += distances.mean()
        count += 1

    # Calculer la distance moyenne globale
    mean_distance = total_distance / count
    return mean_distance.item()

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

# def wasserstein(source_features, target_features):
#     result = solve_sample(source_features, target_features, metric='sqeuclidean', reg=0.1, method='sinkhorn', max_iter=2000)
#     if hasattr(result, 'value'): 
#         loss_value = result.value
#         loss = torch.tensor(loss_value, dtype=torch.float32, requires_grad=True).to(source_features.device)
#     return loss

def wasserstein(source, target):
    a = torch.ones(source.shape[0]) / source.shape[0] 
    b = torch.ones(target.shape[0]) / target.shape[0]
    M =  dist(source, target, metric='euclidean')
    loss = emd2(a, b, M=M)
    return loss

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
