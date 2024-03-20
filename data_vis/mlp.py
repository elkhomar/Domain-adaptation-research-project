import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.distributions.multivariate_normal import MultivariateNormal
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
from losses import MMDLossBandwith
from losses import RBF
from losses import RQ
from itertools import zip_longest
from tqdm import tqdm
import imageio
import glob


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



class MultivariateNormalDataset(Dataset):
    def __init__(self, size, params_1, params_2, A=None, B=None):
        self.size = size

        # Assurez-vous que les paramètres sont de type Float
        mean_1 = torch.tensor(params_1['mean'], dtype=torch.float)
        cov_1 = torch.tensor(params_1['cov'], dtype=torch.float)
        self.class_0_distribution = torch.distributions.MultivariateNormal(mean_1, cov_1)
        class_0_samples = self.class_0_distribution.sample((size,))

        mean_2 = torch.tensor(params_2['mean'], dtype=torch.float)
        cov_2 = torch.tensor(params_2['cov'], dtype=torch.float)
        self.class_1_distribution = torch.distributions.MultivariateNormal(mean_2, cov_2)
        class_1_samples = self.class_1_distribution.sample((size,))

        # Convertit A et B en Float si nécessaire
        A = A.float() if A is not None else None
        B = B.float() if B is not None else None

        # Apply linear transformation if A and B are provided
        if A is not None and B is not None:
            class_0_samples = F.linear(class_0_samples, A, B)
            class_1_samples = F.linear(class_1_samples, A, B)

        # Concatenate samples and labels
        self.samples = torch.cat((class_0_samples, class_1_samples), 0)
        self.labels = torch.cat((torch.zeros(size, dtype=torch.long), torch.ones(size, dtype=torch.long)), 0)

    def __len__(self):
        return 2 * self.size

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]





def visualize_samples(preds_source, label_source, preds_target, label_target, filename):
    if isinstance(preds_source, torch.Tensor):
        preds_source = preds_source.cpu().numpy()
    if isinstance(label_source, torch.Tensor):
        label_source = label_source.cpu().numpy()
    if isinstance(preds_target, torch.Tensor):
        preds_target = preds_target.cpu().numpy()
    if isinstance(label_target, torch.Tensor):
        label_target = label_target.cpu().numpy()

    # Assuming preds_* are [N, 2] (N samples, 2D predictions)
    source_0 = preds_source[label_source == 0]
    source_1 = preds_source[label_source == 1]
    target_0 = preds_target[label_target == 0]
    target_1 = preds_target[label_target == 1]

    plt.figure(figsize=(4,4))
    plt.scatter(source_0[:,0], source_0[:,1], color="blue", edgecolor="#333", label="Class 0 Source")
    plt.scatter(source_1[:,0], source_1[:,1], color="red", edgecolor="#333", label="Class 1 Source")
    plt.scatter(target_0[:,0], target_0[:,1], color="cyan", edgecolor="#333", label="Class 0 Target", alpha=0.5)
    plt.scatter(target_1[:,0], target_1[:,1], color="magenta", edgecolor="#333", label="Class 1 Target", alpha=0.5)
    plt.title("Predictions vs. Targets")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.legend()
    plt.savefig("data_vis/runs/"+filename)
    plt.close()

class CustomLoss(nn.Module):
    def __init__(self, discrepancyloss=MMDLossBandwith(kernel=RBF())):
        super(CustomLoss, self).__init__()
        self.discrepancyloss=discrepancyloss
    def forward(self, output_source, output_g_source, labels,output_g_targets, lambd=1.0):
        bce_loss=nn.BCEWithLogitsLoss()
        loss =  bce_loss(output_source, labels.float()) + 0.01*self.discrepancyloss(output_g_source, output_g_targets)
        return loss

class BinaryClassificationMLP(nn.Module):
    def __init__(self):
        super(BinaryClassificationMLP, self).__init__()
        # First hidden layer
        self.hidden1 = nn.Linear(2, 64)
        # Second hidden layer
        self.hidden2 = nn.Linear(64, 32)
        # Penultimate layer
        self.penultimate = nn.Linear(32, 2)
        # Output layer, note that we do not apply sigmoid here
        # This is because PyTorch's BCEWithLogitsLoss combines a sigmoid layer
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        # Apply ReLU activation function to the first hidden layer
        x = F.relu(self.hidden1(x))
        # Apply ReLU activation function to the second hidden layer
        x = F.relu(self.hidden2(x))
        # Apply ReLU activation function to the penultimate layer
        x = F.relu(self.penultimate(x))
        y = x.clone()
        # Output layer
        x = self.output(x)
        return x , y
    
def train_model(model, optimizer, dataloader1, dataloader2,loss_module, num_epochs=100, device='cpu'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.train()
    for epoch in tqdm(range(num_epochs)):
        for ((data1, target1), (data2, target2)) in zip_longest(dataloader1, dataloader2, fillvalue=(None, None)):
            if data1 is not None and data2 is not None:
                data1, target1 = data1.to(device), target1.to(device)
                data2, target2 = data2.to(device), target2.to(device)

                optimizer.zero_grad()

                # Assuming your model outputs predictions directly
                output_g_source = model(data1)[1]
                output_source = model(data1)[0]
                output_g_target = model(data2)[1]
                output_source = output_source.squeeze(dim=1) 
                output_g_target = output_g_target.squeeze(dim=1) 
                loss = loss_module(output_source= output_source,output_g_source= output_g_source, labels=target1 ,output_g_targets= output_g_target)
                loss.backward()
                optimizer.step()

                # Assuming your predictions and targets are 2D for visualization
                if epoch % 1 == 0:  # For example, save plots every 10 epochs
                    visualize_samples(output_g_source.detach(), target1.detach(), output_g_target.detach(), target2.detach(), f'epoch_{epoch}.png')

import glob 
import imageio
def create_gif(image_path_pattern, output_filename, fps=5):
    """
    Create a GIF from a set of images.

    Args:
        image_path_pattern (str): Pattern to match the images to include in the GIF.
        output_filename (str): The filename for the output GIF.
        fps (int): Frames per second in the output GIF.
    """
    image_files = sorted(glob.glob(image_path_pattern))
    print(f"Found {len(image_files)} files: {image_files}")  # Pour débogage
    
    # Read the images and append them to a list
    images = [imageio.imread(image_file) for image_file in image_files]
    
    if images:
        # Save the images as a GIF
        imageio.mimsave(output_filename, images, fps=fps)
    else:
        print("Aucune image trouvée. Aucun GIF créé.")

# Example usage
# create_gif('epoch_*.png', 'training_progress.gif', fps=2)
if __name__ == "__main__":    
    size = 200  # Number of points per class
    params_1 = {'mean': [4, 4], 'cov': [[1, -0.75], [-0.75, 2]]}  # Class 0 parameters
    params_2 = {'mean': [-4, -4], 'cov': [[1, 0], [0, 1]]}  # Class 1 parameters
    theta = 1.5*torch.tensor(np.pi / 3) 
    R = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                  [torch.sin(theta), torch.cos(theta)]])
    A = torch.tensor([[1.0, 1.5], [1.5, 1.0]],dtype=float)  # Matrice 2x2
    B = torch.tensor([-3, 0.5],dtype=float)
    dataset = MultivariateNormalDataset(size, params_1, params_2)
    dataset2 = MultivariateNormalDataset(size, params_1, params_2, A=R, B=B)
    visualize_samples(dataset.samples, dataset.labels, dataset2.samples, dataset2.labels, 'avant_epoch_0.png')
    dataloader1 = DataLoader(dataset, batch_size=200, shuffle=True)
    dataloader2 = DataLoader(dataset2, batch_size=200, shuffle=True)
    # Create the MLP model
    model = BinaryClassificationMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_module=CustomLoss()
    train_model(model, optimizer, dataloader1, dataloader2,loss_module, num_epochs=100, device='gpu')


create_gif(image_path_pattern="./*.png", output_filename='third_gif.gif', fps=2)





# %%
