
from typing import Any, Dict, Optional, Tuple

import torch
from skimage.transform import resize
import cv2
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
import math
import gzip
import pickle

class MNIST_SVHN_Dataset(Dataset):

    def __init__(self, data_dir, flip_domain=False, is_train=True):
        self.is_train = is_train
        self.flip_domain = flip_domain
        self.prepare_data(data_dir, is_train)

    def load_mnist(self, path='data', is_train=True, is_target=False):
        """
        Loads the MNIST dataset and returns the images and labels
        """
        
        train_dataset = datasets.MNIST(root=path, train=True, download=True)
        test_dataset = datasets.MNIST(root=path, train=False)

        # transform the images and labels (normalize and one hot)

        train_images = (train_dataset.data.float()).unsqueeze(1)
        test_images = (test_dataset.data.float()).unsqueeze(1)

        train_images = (train_images - train_images.mean())/(train_images.std()+1e-6)
        test_images = (test_images - test_images.mean())/(test_images.std()+1e-6)

        train_labels = one_hot(train_dataset.targets, num_classes=10).float()
        test_labels = one_hot(test_dataset.targets, num_classes=10).float()

        # merge train and test data
        images = torch.cat([train_images, test_images], 0)
        labels = torch.cat([train_labels, test_labels], 0)
        
        output = (train_images, train_labels) if is_train else (test_images, test_labels)
        if is_target:
            output = (images, labels)
        
        return output

    def load_svhn(self, path='data/SVHN', is_train=True, is_target=False):
        """
        Loads the SVHN dataset and applies grayscale and resize transformations using OpenCV.
        """
        # Load SVHN dataset
        train_dataset = datasets.SVHN(root=path, split='train', download=True)
        test_dataset = datasets.SVHN(root=path, split='test', download=True)

        # Function to apply transformations using OpenCV
        def transform_images(images):
            transformed_images = []
            for image in images:
                # OpenCV expects images in BGR format
                if image.shape[-1] == 3:  # Convert RGB to BGR
                    image = image[..., ::-1]
                resized_image = cv2.resize(image, (28, 28))  # Resize image
                grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                transformed_images.append(grayscale_image[..., np.newaxis])  # Add channel dimension back
            return np.array(transformed_images)

        # Apply transformations
        train_images = transform_images(train_dataset.data.transpose((0,2,3,1)))  # Ensure images are in HWC format
        test_images = transform_images(test_dataset.data.transpose((0,2,3,1)))  # Ensure images are in HWC format

        # Normalize images
        train_images = (train_images - train_images.mean()) / (train_images.std() + 1e-6)
        test_images = (test_images - test_images.mean()) / (test_images.std() + 1e-6)

        # Convert numpy arrays to PyTorch tensors
        train_images = torch.tensor(train_images, dtype=torch.float32).permute(0, 3, 1, 2)  # Ensure CHW format for PyTorch
        test_images = torch.tensor(test_images, dtype=torch.float32).permute(0, 3, 1, 2)  # Ensure CHW format for PyTorch

        train_labels = torch.tensor(train_dataset.labels)
        test_labels = torch.tensor(test_dataset.labels)

        # Apply one-hot encoding
        train_labels = one_hot(train_labels, num_classes=10).float()
        test_labels = one_hot(test_labels, num_classes=10).float()

        # Convert labels to tensors
        train_labels = torch.tensor(train_labels)
        test_labels = torch.tensor(test_labels)

        # Merge train and test data if needed
        images = torch.cat([train_images, test_images], 0)
        labels = torch.cat([train_labels, test_labels], 0)

        output = (train_images, train_labels) if is_train else (test_images, test_labels)
        if is_target:
            output = (images, labels)
        
        return output

    def prepare_data(self, data_dir, is_train):
        """
        Load the MNIST and SVHN datasets and assign source/target.
        The default is to use MNIST as source and SVHN as target. But can be changed by setting the flip_domain flag to True
        The source will be split into train and val while the target will use everything, we get the following datasets:
        if is_train: (source_train, target_all)
        else: (source_val, target_all)
        """

        if not self.flip_domain:
            self.source_images, self.source_labels = self.load_mnist(is_train = self.is_train, is_target=False)
            self.target_images, self.target_labels = self.load_svhn(is_train = self.is_train, is_target=True)
        else:
            self.source_images, self.source_labels = self.load_svhn(is_train = self.is_train, is_target=False)
            self.target_images, self.target_labels = self.load_mnist(is_train = self.is_train, is_target=True)

   
    def __len__(self):
        return max(len(self.target_images), len(self.source_images))

    def __getitem__(self, index):
        
        # We pack the data by creating a channel for each domain
        index_source = index %(len(self.source_images) - 1)
        index_target = index %(len(self.target_images) - 1)
        image = torch.cat([self.source_images[index_source].unsqueeze(0), self.target_images[index_target].unsqueeze(0)], 0)
        label = torch.cat([self.source_labels[index_source].unsqueeze(0), self.target_labels[index_target].unsqueeze(0)], 0)

        return (image, label)

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
        flip_domain: bool = False,
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.data_train = MNIST_SVHN_Dataset(data_dir, flip_domain, is_train=True)
        self.data_val = MNIST_SVHN_Dataset(data_dir, flip_domain, is_train=False)
        self.train_val_split = train_val_split

    @property
    def n_features(self):
        return 4

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """

        # if not self.data_train and not self.data_val:
        #     # Training and Val set
        #     self.data_train, self.data_val = random_split(
        #         dataset=self.data,
        #         lengths=[int(self.train_val_split[0]*len(self.data)), int(self.train_val_split[1]*len(self.data))],
        #         generator=torch.Generator().manual_seed(42),
        #     )
        
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

if __name__ == "__main__":
    _ = DataModule()
