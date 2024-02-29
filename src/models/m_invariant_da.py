from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy


class InvariantDAModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """
    def __init__(
        self,
        f: torch.nn.Module,
        g:torch.nn.Module,
        lambd,
        loss,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.f = f
        self.g = g
        self.lambd = lambd
        self.loss = loss
        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc_source = Accuracy(task="multiclass", num_classes=10)
        self.train_acc_target = Accuracy(task="multiclass", num_classes=10)
        self.val_acc_source = Accuracy(task="multiclass", num_classes=10)
        self.val_acc_target = Accuracy(task="multiclass", num_classes=10)
        self.test_acc = Accuracy(task="multiclass", num_classes=10)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.train_discrepancy_loss = MeanMetric()
        self.train_classification_loss_source = MeanMetric()
        self.train_classification_loss_target = MeanMetric()
        self.train_loss = MeanMetric()
        self.val_discrepancy_loss = MeanMetric()
        self.val_classification_loss_source = MeanMetric()
        self.val_classification_loss_target = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

        torch.autograd.set_detect_anomaly(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `g(f(.))`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.g(self.f(x))

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_discrepancy_loss.reset()
        self.val_classification_loss_source.reset()
        self.val_classification_loss_target.reset()
        self.val_loss.reset()
        self.test_loss.reset()

        # for tracking best so far validation accuracy
        self.val_acc_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels for source and target.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch
        x_source, x_target = x[:, 0], x[:, 1]
        y_source, y_target = y[:, 0], y[:, 1]

        z_source = self.f(x_source)
        logits_source = self.g(z_source)
        classification_loss_source = self.criterion(logits_source, y_source)

        z_target = self.f(x_target)
        #z_target = torch.zeros_like(z_source)
        logits_target = self.g(z_target)
        #logits_target = torch.zeros_like(logits_source)
        classification_loss_target = self.criterion(logits_target, y_target)

        preds_source = torch.argmax(logits_source, dim=1)
        preds_target = torch.argmax(logits_target, dim=1)

        discrepancy_loss = self.embedding_distance(z_source, z_target, y_source = y_source, preds_target = logits_target)

        return classification_loss_source, classification_loss_target, discrepancy_loss, (preds_source, preds_target), (logits_source, logits_target), (y_source, y_target)
    
    def embedding_distance(
            self, source, target, **kwargs
    ) -> torch.Tensor:
        """Calculate the distance between source and target through the invariant.

        :param source_embedding: A tensor of source embeddings.
        :param target_embedding: A tensor of target embeddings.
        :return: A single number quantifying source and target embedding descrepency.
        """
        return self.loss(source, target, **kwargs)
    
    def compute_covariance(self, input_data):
        """
        Compute Covariance matrix of the input data
        """
        n = input_data.size(0)  # batch_size

        # Check if using gpu or cpu
        if input_data.is_cuda:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        id_row = torch.ones(n).resize(1, n).to(device=device)
        sum_column = torch.mm(id_row, input_data)
        mean_column = torch.div(sum_column, n)
        term_mul_2 = torch.mm(mean_column.t(), mean_column)
        d_t_d = torch.mm(input_data.t(), input_data)
        c = torch.add(d_t_d, (-1 * term_mul_2)) * 1 / (n - 1)

        return c

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.
    def embedding_distance(

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels for source and target
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        classification_loss_source, classification_loss_target, discrepancy_loss, preds, logits, labels = self.model_step(batch)
        loss = classification_loss_source + self.lambd * discrepancy_loss

        # update and log metrics
        self.train_loss(loss)
        self.train_classification_loss_source(classification_loss_source)
        self.train_classification_loss_target(classification_loss_target)
        self.train_discrepancy_loss(discrepancy_loss)
        self.train_acc_source(logits[0], torch.argmax(labels[0], 1))
        self.train_acc_target(logits[1], torch.argmax(labels[1], 1))
        
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/classification_loss_source", self.train_classification_loss_source, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/oracle_classification_loss_target", self.train_classification_loss_target, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/discrepancy_loss", self.train_discrepancy_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc_source", self.train_acc_source, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/oracle_acc_target", self.train_acc_target, on_step=True, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        classification_loss_source, classification_loss_target, discrepancy_loss, preds, logits, labels = self.model_step(batch)
        loss = classification_loss_source + self.lambd * discrepancy_loss
        # update and log metrics
        self.val_loss(loss)
        self.val_classification_loss_source(classification_loss_source)
        self.val_classification_loss_target(classification_loss_target)
        self.val_discrepancy_loss(discrepancy_loss)
        self.val_acc_source(logits[0], torch.argmax(labels[0], 1))
        self.val_acc_target(logits[1], torch.argmax(labels[1], 1))

        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/classification_loss_source", self.val_classification_loss_source, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/oracle_classification_loss_target", self.val_classification_loss_target, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/discrepancy_loss", self.val_discrepancy_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/acc_source", self.val_acc_source, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/oracle_acc_target", self.val_acc_target, on_step=True, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc_target.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)


    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        classification_loss_source, classification_loss_target, discrepancy_loss, preds, logits, labels = self.model_step(batch)
        loss = classification_loss_source + self.lambd * discrepancy_loss
        # update and log metrics
        self.test_loss(loss)
        self.test_acc(logits[1], torch.argmax(labels[1], 1))
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = InvariantDAModule(None, None, None, None, None, None, None)
