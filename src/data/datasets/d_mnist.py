import torch
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from torchvision import datasets


class MnistDataset(Dataset):

    def __init__(self):
        self.all_outputs, self.train_outputs, self.val_outputs = self.load_mnist()

    def load_mnist(self, path="data"):
        """
        Loads the MNIST dataset and returns the images and labels
        """

        train_dataset = datasets.MNIST(root=path, train=True, download=True)
        val_dataset = datasets.MNIST(root=path, train=False)

        # transform the images and labels (normalize and one hot)

        train_images = (train_dataset.data.float()).unsqueeze(1)
        val_images = (val_dataset.data.float()).unsqueeze(1)

        train_images = (train_images - train_images.mean()) / (
            train_images.std() + 1e-6
        )
        val_images = (val_images - val_images.mean()) / (val_images.std() + 1e-6)

        train_labels = one_hot(train_dataset.targets, num_classes=10).float()
        val_labels = one_hot(val_dataset.targets, num_classes=10).float()

        # merge train and val data
        images = torch.cat([train_images, val_images], 0)
        labels = torch.cat([train_labels, val_labels], 0)

        train_outputs = (train_images, train_labels)
        val_outputs = (val_images, val_labels)
        all_outputs = (images, labels)

        return all_outputs, train_outputs, val_outputs
