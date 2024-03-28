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
from sklearn.manifold import TSNE
from matplotlib.animation import FuncAnimation
import os
import matplotlib.pyplot as plt

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
from sklearn.decomposition import PCA
class CustomLoss(nn.Module):
    def __init__(self, discrepancyloss=MMDLossBandwith(kernel=RBF())):
        super(CustomLoss, self).__init__()
        self.discrepancyloss=discrepancyloss
    def forward(self, output_source, output_g_source, labels,output_g_targets, lambd=1.0):
        bce_loss=nn.BCEWithLogitsLoss()
        loss =  bce_loss(output_source, labels.float()) + self.discrepancyloss(output_g_source, output_g_targets)
        return loss

class BinaryClassificationMLP(nn.Module):
    def __init__(self):
        super(BinaryClassificationMLP, self).__init__()
        # First hidden layer
        self.hidden1 = nn.Linear(2, 32)
        # Second hidden layer
        self.hidden2 = nn.Linear(32, 16)
        # Penultimate layer
        self.penultimate = nn.Linear(16, 5)
        # Output layer, note that we do not apply sigmoid here
        # This is because PyTorch's BCEWithLogitsLoss combines a sigmoid layer
        self.output = nn.Linear(5, 1)

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
    
def train_model(model, optimizer, dataloader1, dataloader2, loss_module, num_epochs=80, device='cpu'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    plot_files = []  # List to track plot files for potential GIF creation
    batch_counter = 0  # Counter to track batch number across epochs
    for epoch in tqdm(range(num_epochs)):
        for ((data1, target1), (data2, target2)) in zip_longest(dataloader1, dataloader2, fillvalue=(None, None)):
            if data1 is not None and data2 is not None:
                data1, target1 = data1.to(device), target1.to(device)
                data2, target2 = data2.to(device), target2.to(device)

            optimizer.zero_grad()

            # Process data through the model
            output1, embeddings1 = model(data1)
            output2, embeddings2 = model(data2)
            output1 = output1.squeeze(dim=1)
            output2 = output2.squeeze(dim=1)
            loss = loss_module(output_source=output1, output_g_source=embeddings1, labels=target1, output_g_targets=embeddings2)
            loss.backward()
            optimizer.step()

            # Prepare data for plotting
            target1_np = target1.cpu().detach().numpy()
            target2_np = target2.cpu().detach().numpy()
            colors = ['blue', 'red']  # Adjust this based on the number of your classes
            colors_for_plot1 = [colors[int(label)] for label in target1_np]
            colors_for_plot2 = [colors[int(label)] for label in target2_np]

            embeddings = torch.cat((embeddings1, embeddings2), dim=0).cpu().detach().numpy()
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(embeddings)
            pca_embeddings1 = pca_result[:len(embeddings1)]
            pca_embeddings2 = pca_result[len(embeddings1):]

            plt.figure(figsize=(8, 6))
            plt.scatter(pca_embeddings1[:, 0], pca_embeddings1[:, 1], c=colors_for_plot1, marker='o', label='Output1', alpha=0.5)
            plt.scatter(pca_embeddings2[:, 0], pca_embeddings2[:, 1], c=colors_for_plot2, marker='x', label='Output2', alpha=0.5)
            plt.title(f'PCA of Embeddings at Epoch {epoch+1}, Batch {batch_counter+1}')
            plt.legend()
            plot_file = f'pca_epoch_{epoch+1}_final_{batch_counter+1}.png'
            plt.savefig(plot_file)
            plt.close()
            plot_files.append(plot_file)

            batch_counter += 1 
    return plot_files

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

def create_tsne_plots(embeddings_list, labels_list, num_points_output1):
    """
    Create t-SNE plots with different markers for output1 and output2.
    
    Args:
    - embeddings_list: List of numpy arrays containing embeddings from both outputs.
    - labels_list: List of numpy arrays containing labels for both outputs.
    - num_points_output1: Number of points in each array belonging to output1.
    """
    for i, (embeddings, labels) in enumerate(zip(embeddings_list, labels_list)):
        tsne_results = TSNE(n_components=2, random_state=42).fit_transform(embeddings)
        
        plt.figure(figsize=(8, 6))
        
        # Define colors and markers
        colors = ['blue', 'red']  # Define more colors if you have more classes
        markers = ['o', 'x']  # 'o' for dots (output1), 'x' for crosses (output2)
        
        # Split embeddings and labels by output source
        embeddings1, embeddings2 = tsne_results[:num_points_output1[i]], tsne_results[num_points_output1[i]:]
        labels1, labels2 = labels[:num_points_output1[i]], labels[num_points_output1[i]:]
        
        # Plot points for output1
        for label in np.unique(labels1):
            indices = labels1 == label
            plt.scatter(embeddings1[indices, 0], embeddings1[indices, 1], c=colors[label], label=f'Class {label} (Output1)', alpha=0.5, marker=markers[0])
        
        # Plot points for output2 with different marker
        for label in np.unique(labels2):
            indices = labels2 == label
            plt.scatter(embeddings2[indices, 0], embeddings2[indices, 1], c=colors[label], label=f'Class {label} (Output2)', alpha=0.5, marker=markers[1])
        
        plt.legend()
        plt.title(f't-SNE at Epoch {i}')
        plt.savefig(f'epoch_{i}.png')
        plt.close()

import imageio

def create_gif_from_images(image_filenames, output_filename='training_visualization.gif'):
    """
    Creates a GIF from a list of image files.

    Parameters:
    - image_filenames: List of strings, where each string is a path to an image file.
    - output_filename: String, the name of the output GIF file.

    The function saves the resulting GIF in the current working directory or the path included in output_filename.
    """
    with imageio.get_writer(output_filename, mode='I') as writer:
        for filename in image_filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
            # Optional: Remove the file after adding it to the GIF if you want to clean up
            # os.remove(filename)

    print(f"GIF saved as {output_filename}")

# Example usage:
# plot_files = ['path/to/image1.png', 'path/to/image2.png', ...]
# create_gif_from_images(plot_files, 'my_training_visualization.gif')





# Example usage
# create_gif('epoch_*.png', 'training_progress.gif', fps=2)
if __name__ == "__main__":    
    torch.manual_seed(43)
    size = 200  # Number of points per class
    params_1 = {'mean': [4, 4], 'cov': [[1, -0.75], [-0.75, 2]]}  # Class 0 parameters
    params_2 = {'mean': [-10, -4], 'cov': [[1, 0], [0, 1]]}  # Class 1 parameters
    theta = 1.5*torch.tensor(np.pi / 3) 
    R = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                  [torch.sin(theta), torch.cos(theta)]])
    A = torch.tensor([[1.0, 1.5], [1.5, 1.0]],dtype=float)  # Matrice 2x2
    B = torch.tensor([-3, 0.5],dtype=float)
    dataset = MultivariateNormalDataset(size, params_1, params_2)
    dataset2 = MultivariateNormalDataset(size, params_1, params_2, A=R, B=B)
    #visualize_samples(dataset.samples, dataset.labels, dataset2.samples, dataset2.labels, 'avant_epoch_0.png')
    samples_1 = dataset.samples
    samples_2 = dataset2.samples
    labels_1 = dataset.labels
    labels_2 = dataset2.labels
    source_0 = samples_1[labels_1 == 0]
    source_1 = samples_1[labels_1 == 1]
    target_0 = samples_2[labels_2 == 0]
    target_1 = samples_2[labels_2 == 1]
    plt.figure(figsize=(10,10))
    plt.scatter(source_0[:,0], source_0[:,1], color="blue", marker='o',label="Class 0 Source", alpha=0.5)
    plt.scatter(source_1[:,0], source_1[:,1], color="red", marker='o', label="Class 1 Source", alpha=0.5)
    plt.scatter(target_0[:,0], target_0[:,1], color="blue", marker='x',label="Class 0 Target", alpha=0.5)
    plt.scatter(target_1[:,0], target_1[:,1], color="red", marker='x', label="Class 1 Target", alpha=0.5)
    plt.legend()
    plt.title('Scatter plot of source and target data')
    plt.savefig('scatter_plot.png')
    plt.close()
    dataloader1 = DataLoader(dataset, batch_size=400, shuffle=True)
    dataloader2 = DataLoader(dataset2, batch_size=400, shuffle=True)
    # Create the MLP model
    model = BinaryClassificationMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_module=CustomLoss()
   # Update this line to accept three outputs
    imag = train_model(model, optimizer, dataloader1, dataloader2, loss_module, num_epochs=200)
    create_gif_from_images(imag, 'training_progress5.gif')
    # Make sure to pass num_points_output1_list to the function
    # create_gif (imag, 'training_progress.gif', interval=500)


