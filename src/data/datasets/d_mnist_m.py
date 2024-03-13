import torch
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Grayscale, Resize
import os
from PIL import Image

class MnistMDataset(Dataset):

    def __init__(self):
        self.all_outputs, self.train_outputs, self.val_outputs = self.load_mnistm()

    def load_mnistm(self, root='data/MNIST-M'):
        """
        Loads the MNIST-M dataset, applies transformations, and returns images and labels in grayscale.
        """
        mnistm_file = os.path.join(root, 'mnistm.tensor')
        train_file = os.path.join(root, 'train_mnistm.tensor')
        val_file = os.path.join(root, "val_mnistm.tensor")

        all_outputs = torch.load(mnistm_file)
        train_outputs = torch.load(train_file)
        val_outputs = torch.load(val_file)

        return all_outputs, train_outputs, val_outputs