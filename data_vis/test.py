import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.distributions.multivariate_normal import MultivariateNormal
import torch
import matplotlib.pyplot as plt
import numpy as np
from losses import MMDLossBandwith
from losses import RBF
from losses import RQ
import matplotlib.pyplot as plt
from mlp import MultivariateNormalDataset
from losses import MMDLoss

torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('hello')
size = 400  # Number of points per class
params_1 = {'mean': [4, 4], 'cov': [[1, -0.75], [-0.75, 2]]}  # Class 0 parameters
params_2 = {'mean': [-10, -4], 'cov': [[1, 0], [0, 1]]}  # Class 1 parameters
theta = 1.5*torch.tensor(np.pi / 3) 
R = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                [torch.sin(theta), torch.cos(theta)]])
A = torch.tensor([[1.0, 1.5], [1.5, 1.0]],dtype=float)  # Matrice 2x2
B = torch.tensor([-3, 0.5],dtype=float)
dataset = MultivariateNormalDataset(size, params_1, params_2)
dataset2 = MultivariateNormalDataset(size, params_1, params_2, A=R, B=B)
samples_1 = dataset.samples
samples_1.requires_grad_(True)
samples_2 = dataset2.samples
mmd_norm=[]
sigmas=[]
mmd_loss = []
for sigma in np.arange(0.01,15,0.05):
    sigmas.append(sigma)
    print(sigma)
    mmd = MMDLoss(sigma)
    loss=mmd(samples_1, samples_2,sigma)
    loss.backward()
    # mmd_loss.append(loss.item())
    mmd_norm.append(samples_1.grad.norm().item())
    samples_1.grad.zero_()
    mmd.zero_grad()
plt.plot(sigmas,mmd_norm)
plt.title("MMD's norm w.r.t. sigma")
plt.savefig('mmd_loss_norm.png')
plt.show()  

# plt.plot(sigmas,mmd_loss)
# plt.savefig('mmd_loss.png')
# plt.show()

# labels_1 = dataset.labels
# labels_2 = dataset2.labels
# source_0 = samples_1[labels_1 == 0]
# source_1 = samples_1[labels_1 == 1]
# target_0 = samples_2[labels_2 == 0]
# target_1 = samples_2[labels_2 == 1]
# plt.figure(figsize=(10,10))
# plt.scatter(source_0[:,0], source_0[:,1], color="blue", marker='o',label="Class 0 Source", alpha=0.5)
# plt.scatter(source_1[:,0], source_1[:,1], color="red", marker='o', label="Class 1 Source", alpha=0.5)
# plt.scatter(target_0[:,0], target_0[:,1], color="blue", marker='x',label="Class 0 Target", alpha=0.5)
# plt.scatter(target_1[:,0], target_1[:,1], color="red", marker='x', label="Class 1 Target", alpha=0.5)
# plt.legend()
# plt.title('Scatter plot of source and target data')
# plt.savefig('scatter_plot.png')
# plt.close()