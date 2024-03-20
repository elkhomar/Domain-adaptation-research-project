import torch
from torch.nn.functional import one_hot
import pickle
import gzip
from torch.utils.data import Dataset


class UspsDataset(Dataset):

    def __init__(self):
        self.all_outputs, self.train_outputs, self.val_outputs = self.load_usps()

    def load_usps(self, path="data/USPS/usps_28x28.pkl"):
        """
        Loads the USPS dataset from a pickle file, processes it, and returns images and labels
        """

        with gzip.open(path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        train_images = data[0][0]
        train_labels = data[0][1]
        val_images = data[1][0]
        val_labels = data[1][1]

        train_images = torch.tensor(train_images, dtype=torch.float32)
        val_images = torch.tensor(val_images, dtype=torch.float32)

        train_images = (train_images - train_images.mean()) / (
            train_images.std() + 1e-6
        )
        val_images = (val_images - val_images.mean()) / (val_images.std() + 1e-6)

        train_labels = one_hot(
            torch.tensor(train_labels).long(), num_classes=10
        ).float()
        val_labels = one_hot(torch.tensor(val_labels).long(), num_classes=10).float()

        # merge train and val data
        images = torch.cat([train_images, val_images], 0)
        labels = torch.cat([train_labels, val_labels], 0)
        train_outputs = (train_images, train_labels)
        val_outputs = (val_images, val_labels)
        all_outputs = (images, labels)
        return all_outputs, train_outputs, val_outputs
