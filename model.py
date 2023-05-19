import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.optim import SGD, Adam, AdamW, RAdam
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import ImageFolder
from cfg import Config


class TimmBasedClassifierModel(L.LightningModule):
    optimizers = {"adam": Adam, "sgd": SGD, "adamw": AdamW, "radam": RAdam}

    def __init__(
        self,
        num_classes,
        model_name,
        optimizer="adam",
        lr=1e-3,
        batch_size=16,
        transfer=Config.transfer,
        tune_fc_only=Config.tuneFcOnly,
        exportable=Config.exportable,
        scriptable=Config.scriptable,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.lr = lr
        self.batch_size = batch_size

        self.optimizer = self.optimizers[optimizer]
        # instantiate loss criterion
        self.loss_fn = (
            nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()
        )
        # create accuracy metric
        self.acc = Accuracy(
            task="binary" if num_classes == 1 else "multiclass", num_classes=num_classes
        )

        self.model = timm.create_model(
            model_name,
            pretrained=transfer,
            num_classes=num_classes,
            exportable=exportable,
            scriptable=scriptable,
        )

        if tune_fc_only:  # option to only tune the fully-connected layers
            for child in list(self.model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, X):
        return self.model(X)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)

    def _step(self, batch):
        x, y = batch
        preds = self(x)
        result = torch.argmax(preds, -1)

        if self.num_classes == 1:
            preds = preds.flatten()
            y = y.float()

        loss = self.loss_fn(preds, y)
        acc = self.acc(result, y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        # perform logging
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        # perform logging
        self.log("val_loss", loss, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)
