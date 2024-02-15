import torch
import torch.functional as F
import torch.nn as nn
import ot

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
    def __init__(self):
        super(MMDLoss, self).__init__()

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

class WassersteinLoss(nn.Module):
    def __init__(self):
        super(WassersteinLoss, self).__init__()

    def forward(self, source, target, reg=0):
        return ot.solve_sample(source, target, reg=reg).value
    
class WassersteinLoss(nn.Module):
    def __init__(self, reg=1, unbiased=False):
        super(WassersteinLoss, self).__init__()
        self.reg = reg
        self.unbiased = unbiased

    def forward(self, source, target):
        if self.unbiased:
            s1 = source[:len(source)//2]
            s2 = source[len(source)//2:]
            t1 = target[:len(target)//2]
            t2 = target[len(target)//2:]
            
            # 
            ls1s2 = torch.sqrt(ot.solve_sample(s1, s2, reg=self.reg).value)
            lt1t2 = torch.sqrt(ot.solve_sample(t1, t2, reg=self.reg).value)

            ls1t1 = torch.sqrt(ot.solve_sample(s1, t1, reg=self.reg).value)
            # ls1t2 = torch.sqrt(ot.solve_sample(s1, t2, reg=self.reg).value)
            # ls2t1 = torch.sqrt(ot.solve_sample(s2, t1, reg=self.reg).value)
            # ls2t2 = torch.sqrt(ot.solve_sample(s2, t2, reg=self.reg).value)
            loss = ls1t1 - 0.5 * (ls1s2 + lt1t2)
        else:
            loss = torch.sqrt(ot.solve_sample(source, target, reg=self.reg).value)
        return loss

class SlicedWassersteinLoss(nn.Module):
    def __init__(self, unbiased=False, n_proj=1000):
        super(SlicedWassersteinLoss, self).__init__()
        self.unbiased = unbiased
        self.n_proj = n_proj
        self.seed = torch.initial_seed()

    def forward(self, source, target):
        torch.use_deterministic_algorithms(True, warn_only=True)
        if self.unbiased:
            s1 = source[:len(source)//2]
            s2 = source[len(source)//2:]
            t1 = target[:len(target)//2]
            t2 = target[len(target)//2:]
            
            # 
            ls1s2 = ot.sliced_wasserstein_distance(s1, s2, n_projections=self.n_proj)
            lt1t2 = ot.sliced_wasserstein_distance(t1, t2, n_projections=self.n_proj)

            ls1t1 = ot.sliced_wasserstein_distance(s1, t1, n_projections=self.n_proj)
            # ls1t2 = ot.sliced_wasserstein_distance(s1, t2, n_projections=self.n_proj)
            # ls2t1 = ot.sliced_wasserstein_distance(s2, t1, n_projections=self.n_proj)
            # ls2t2 = ot.sliced_wasserstein_distance(s2, t2, n_projections=self.n_proj)
            loss = ls1t1 - 0.5 * (ls1s2 + lt1t2)
        else:
            loss = ot.sliced_wasserstein_distance(source, target, n_projections=self.n_proj)
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
