from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
import torch
from src.visualisation import visualize_tsne_domain, visualize_tsne_labels
import os


class PlotEmbedding(Callback):

    def plot_embedding(self, trainer, pl_module, epoch):

        max_samples = 3000

        # Get the source and target images
        source_train_images = pl_module.transfer_batch_to_device(trainer.datamodule.data_train.source_images, pl_module.device, 0)
        source_val_images = pl_module.transfer_batch_to_device(trainer.datamodule.data_val.source_images, pl_module.device, 0)
        target_images = pl_module.transfer_batch_to_device(trainer.datamodule.data_train.target_images, pl_module.device, 0)

        # Sample a subset of the data
        source_train_samples = torch.arange(len(source_train_images))[:min(max_samples, len(source_train_images))]
        source_val_samples = torch.arange(len(source_val_images))[:min(max_samples, len(source_val_images))]
        target_samples = torch.arange(len(target_images))[:min(max_samples, len(target_images))]

        # Get the embeddings through the feature extractor
        source_train_embedding = pl_module.f(source_train_images[source_train_samples])
        source_val_embedding = pl_module.f(source_val_images[source_val_samples])
        target_embedding = pl_module.f(target_images[target_samples])

        source_train_labels = trainer.datamodule.data_train.source_labels[source_train_samples]
        source_val_labels = trainer.datamodule.data_val.source_labels[source_val_samples]
        target_labels = trainer.datamodule.data_train.target_labels[target_samples]

        visualize_tsne_labels(source_train_embedding, target_embedding, source_train_labels, target_labels, filename=trainer.default_root_dir + f"/visualisations/labels_tsne/epoch{epoch}_tsne_labels_srctrain_target")
        visualize_tsne_domain(source_train_embedding, target_embedding, filename=trainer.default_root_dir + f"/visualisations/domain_tsne/epoch{epoch}_tsne_domain_srctrain_target")

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.plot_embedding(trainer, pl_module, -1)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % 5 == 0 or trainer.current_epoch in (1, 2, 3) :
            self.plot_embedding(trainer, pl_module, trainer.current_epoch)


