
from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision import datasets, transforms
import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
from torch.nn.functional import one_hot
import gzip
import pickle


def group_id_2_label(labels, num_classes):
    labels = torch.tensor(labels, dtype=torch.long)
    return torch.nn.functional.one_hot(labels, num_classes=num_classes)

class mnist_usps_Dataset(Dataset):

    def __init__(self, data_dir):

        self.prepare_data(data_dir)
    
    def prepare_data(self, data_dir):
        """
        Downloads the MNIST and USPS datasets and stores them in the `data_dir` directory.
        """
        mnist_train_images, mnist_train_labels, mnist_test_images, mnist_test_labels = self.load_mnist()
        usps_train_images, usps_train_labels, usps_test_images, usps_test_labels = self.load_usps()

        self.source_images = torch.cat([mnist_train_images, mnist_test_images], dim=0)
        self.source_labels = torch.cat([mnist_train_labels, mnist_test_labels], dim=0)

        self.target_images = torch.cat([usps_train_images, usps_test_images], dim=0).unsqueeze(1)
        self.target_labels = torch.cat([usps_train_labels, usps_test_labels], dim=0).unsqueeze(1)

        # We chose the proportion of source vs target data to be 1:1 by only taking a small subset of mnist

        self.source_images = self.source_images[:self.target_images.shape[0]].unsqueeze(1)
        self.source_labels = self.source_labels[:self.target_labels.shape[0]].unsqueeze(1)
        self.images = torch.cat([self.source_images, self.target_images], 1)
        self.labels = torch.cat([self.source_labels, self.target_labels], 1)

    def load_mnist(self, path='data'):
        """
        Loads the MNIST dataset and returns DataLoader objects for the train and test sets.

        Parameters:
        - download (bool, optional): If True, downloads the dataset from the internet if it's not available at `path`. Default is True.
        - path (str, optional): The path where the MNIST dataset is stored or will be downloaded. Default is './data'.
        - batch_size (int, optional): The size of each batch returned by the DataLoader. Default is 64.

        Returns:
        - tuple: A tuple containing two DataLoader instances:
            - train_loader (DataLoader): DataLoader for the training set.
            - test_loader (DataLoader): DataLoader for the test set.
        """

        train_dataset = datasets.MNIST(root=path, train=True, download=True)
        test_dataset = datasets.MNIST(root=path, train=False)

        train_images = (train_dataset.data.float()).unsqueeze(1)
        test_images = (test_dataset.data.float()).unsqueeze(1)

        train_images = (train_images - train_images.mean())/(train_images.std()+1e-6)
        test_images = (test_images - test_images.mean())/(test_images.std()+1e-6)



        return train_images, one_hot(train_dataset.targets, num_classes=10).float(), test_images, one_hot(test_dataset.targets, num_classes=10).float()
    
    def load_usps(self, path='data/USPS/usps_28x28.pkl'):
        """
        Loads the USPS dataset from a pickle file, processes it, and returns DataLoader objects for the train and test sets.

        Parameters:
        - data_dir (str): The directory path where the USPS dataset pickle file is stored.
        Returns:
        - tuple: A tuple containing two DataLoader instances:
            - train_loader (DataLoader): DataLoader for the training set.
            - test_loader (DataLoader): DataLoader for the test set.
        """
            
        with gzip.open(path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        train_images = data[0][0]
        train_labels = data[0][1]
        test_images = data[1][0]
        test_labels = data[1][1]
        
        train_images = torch.tensor(train_images, dtype=torch.float32)
        test_images = torch.tensor(test_images, dtype=torch.float32)
        
        train_labels = group_id_2_label(train_labels, 10)
        test_labels = group_id_2_label(test_labels, 10)

        train_images = (train_images - train_images.mean())/(train_images.std()+1e-6)
        test_images = (test_images - test_images.mean())/(test_images.std()+1e-6)

        
        return train_images, train_labels, test_images, test_labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return (self.images[index], self.labels[index])
    
class DataModule(LightningDataModule):
    """Example preprocessing and batching poses

    A DataModule implements 6 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_split: Tuple[int, int] = (0.8, 0.2),
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.data = mnist_usps_Dataset(data_dir)
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

        self.train_val_split = train_val_split

    @property
    def n_features(self):
        return 4

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """

        if not self.data_train and not self.data_val:
            # Training and Val set
            self.data_train, self.data_val = random_split(
                dataset=self.data,
                lengths=[int(self.train_val_split[0]*len(self.data)) + 1, int(self.train_val_split[1]*len(self.data))],
                generator=torch.Generator().manual_seed(42),
            )
        
    def train_dataloader(self, shuffle=True):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=shuffle,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass

    def get_pose_from_model(self, index):
        return self.data_train.model_output_to_pose(index)


if __name__ == "__main__":
    _ = DataModule()
