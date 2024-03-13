import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.distributions.multivariate_normal import MultivariateNormal
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
from src.losses import MMDLossBandwith
from src.losses import RBF
from src.losses import RQ
from itertools import zip_longest
from tqdm.notebook import tqdm
import imageio
import glob
import matplotlib.pyplot as plt
import torch
import numpy as np

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class MultivariateNormalDataset(Dataset):
    def __init__(self, size, params_1, params_2):
        """
        Args:
            size (int): Number of points per class.
            params_1 (dict): Parameters {'mean': [x1, x2], 'cov': [[var1, cov], [cov, var2]]} for class 0.
            params_2 (dict): Parameters {'mean': [x1, x2], 'cov': [[var1, cov], [cov, var2]]} for class 1.
        """
        self.size = size

        # Generate points for class 0
        self.class_0_distribution = torch.distributions.MultivariateNormal(
            torch.tensor(params_1['mean']), 
            torch.tensor(params_1['cov'])
        )
        self.class_0_samples = self.class_0_distribution.sample((size,))

        # Generate points for class 1
        self.class_1_distribution = torch.distributions.MultivariateNormal(
            torch.tensor(params_2['mean']), 
            torch.tensor(params_2['cov'])
        )
        self.class_1_samples = self.class_1_distribution.sample((size,))

        # Concatenate samples and labels
        self.samples = torch.cat((self.class_0_samples, self.class_1_samples), 0)
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
    plt.savefig(filename)
    plt.close()

class CustomLoss(nn.Module):
    def __init__(self, lambd, discrepancyloss=MMDLossBandwith()):
        super(CustomLoss, self).__init__()
        self.lambd=lambd
        self.discrepancyloss=discrepancyloss
    def forward(self, inputs_source, labels,targets):
        loss = nn.BCEWithLogitsLoss(inputs_source, labels) + self.lambd*self.discrepancyloss(inputs_source, targets)
        return loss

class BinaryClassificationMLP(nn.Module):
    def __init__(self):
        super(BinaryClassificationMLP, self).__init__()
        # First hidden layer
        self.hidden1 = nn.Linear(2, 5)
        # Second hidden layer
        self.hidden2 = nn.Linear(5, 5)
        # Penultimate layer
        self.penultimate = nn.Linear(5, 2)
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
        # Output layer
        x = self.output(x)
        # Apply sigmoid activation function
        return x
    
def train_model(model, optimizer, dataloader1, dataloader2, loss_module, num_epochs=100, device='cpu'):
    model.train()
    for epoch in tqdm(range(num_epochs)):
        for ((data1, target1), (data2, target2)) in zip_longest(dataloader1, dataloader2, fillvalue=(None, None)):
            if data1 is not None and data2 is not None:
                data1, target1 = data1.to(device), target1.to(device)
                data2, target2 = data2.to(device), target2.to(device)

                optimizer.zero_grad()

                # Assuming your model outputs predictions directly
                preds_source = model(data1)
                preds_target = model(data2)

                loss_source = loss_module(preds_source, target1.float())
                loss_target = loss_module(preds_target, target2.float())
                loss = loss_source + loss_target  # Or however you want to combine these losses

                loss.backward()
                optimizer.step()

                # Assuming your predictions and targets are 2D for visualization
                if epoch % 3 == 0:  # For example, save plots every 10 epochs
                    visualize_samples(preds_source.detach(), target1.detach(), preds_target.detach(), target2.detach(), f'epoch_{epoch}.png')


def create_gif(image_path_pattern, output_filename, fps=10):
    """
    Create a GIF from a set of images.

    Args:
        image_path_pattern (str): Pattern to match the images to include in the GIF.
        output_filename (str): The filename for the output GIF.
        fps (int): Frames per second in the output GIF.
    """
    # Get all the image files matching the pattern
    image_files = sorted(glob.glob(image_path_pattern))
    
    # Read the images and append them to a list
    images = [imageio.imread(image_file) for image_file in image_files]
    
    # Save the images as a GIF
    imageio.mimsave(output_filename, images, fps=fps)

# Example usage
create_gif('epoch_*.png', 'training_progress.gif', fps=2)

dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
# Create the MLP model
model = BinaryClassificationMLP()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Example usage
size = 50  # Number of points per class
params_1 = {'mean': [1, 1], 'cov': [[1, 0], [0, 1]]}  # Class 0 parameters
params_2 = {'mean': [3, 3], 'cov': [[1, 0], [0, 1]]}  # Class 1 parameters

dataset = MultivariateNormalDataset(size, params_1, params_2)

# Example: Access the 10th item
print(dataset[10])  # Each item is a tuple (sample, label)

# You can use this dataset with a DataLoader to create batches for training.
from torch.utils.data import DataLoader




