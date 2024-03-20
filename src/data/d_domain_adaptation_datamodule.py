from typing import Any, Dict, Optional
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import torch
from torch.utils.data import Dataset


class FusedDataset(Dataset):
    """
    A Dataset class that merges the sources and targets per batch in two different channels
    """

    def __init__(self, source_dataset, target_dataset, is_train=True):
        self.is_train = is_train
        self.target_images, self.target_labels = target_dataset.all_outputs
        self.source_images, self.source_labels = (
            source_dataset.train_outputs if is_train else source_dataset.val_outputs
        )

    def __len__(self):
        return max(len(self.target_images), len(self.source_images))

    def __getitem__(self, index):

        # We pack the data by creating a channel for each domain
        index_source = index % (len(self.source_images) - 1)
        index_target = index % (len(self.target_images) - 1)
        image = torch.cat(
            [
                self.source_images[index_source].unsqueeze(0),
                self.target_images[index_target].unsqueeze(0),
            ],
            0,
        )
        label = torch.cat(
            [
                self.source_labels[index_source].unsqueeze(0),
                self.target_labels[index_target].unsqueeze(0),
            ],
            0,
        )

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
        source_dataset,
        target_dataset,
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.data_train = FusedDataset(source_dataset, target_dataset, is_train=True)
        self.data_val = FusedDataset(source_dataset, target_dataset, is_train=False)

    @property
    def n_features(self):
        return None

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
