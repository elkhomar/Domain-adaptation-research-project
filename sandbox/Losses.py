import torch
import torch.nn as nn
import torch
import torch.nn.functional as F

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
