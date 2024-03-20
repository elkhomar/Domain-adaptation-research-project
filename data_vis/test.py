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
import matplotlib.pyplot as plt
from mlp import MultivariateNormalDataset

size = 200  # Number of points per class
params_1 = {'mean': [4, 4], 'cov': [[1, -0.75], [-0.75, 2]]}  # Class 0 parameters
params_2 = {'mean': [-4, -4], 'cov': [[1, 0], [0, 1]]}  # Class 1 parameters
theta = 1.5*torch.tensor(np.pi / 3) 
R = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                  [torch.sin(theta), torch.cos(theta)]])
A = torch.tensor([[1.0, 1.5], [1.5, 1.0]],dtype=float)  # Matrice 2x2
B = torch.tensor([0, 0.5],dtype=float)
dataset = MultivariateNormalDataset(size, params_1, params_2)
dataset2 = MultivariateNormalDataset(size, params_1, params_2, A=R, B=B)


samples_1 = dataset.samples
samples_2 = dataset2.samples
labels_1 = dataset.labels
labels_2 = dataset2.labels
source_0 = samples_1[labels_1 == 0]
source_1 = samples_1[labels_1 == 1]
target_0 = samples_2[labels_2 == 0]
target_1 = samples_2[labels_2 == 1]
plt.figure(figsize=(4,4))
plt.scatter(source_0[:,0], source_0[:,1], color="blue", edgecolor="#333", label="Class 0 Source")
plt.scatter(source_1[:,0], source_1[:,1], color="red", edgecolor="#333", label="Class 1 Source")
plt.scatter(target_0[:,0], target_0[:,1], color="cyan", edgecolor="#333", label="Class 0 Target", alpha=0.5)
plt.scatter(target_1[:,0], target_1[:,1], color="magenta", edgecolor="#333", label="Class 1 Target", alpha=0.5)
plt.legend()
plt.title('Scatter plot 1')
plt.savefig('scatter_plot1.png')
plt.close()