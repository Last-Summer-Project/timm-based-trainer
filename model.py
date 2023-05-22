from typing import Any
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.optim import SGD, Adam, AdamW, RAdam
from torch.utils.data import DataLoader
from torchmetrics import (
    Accuracy,
    F1Score,
    Precision,
    Recall,
    AUROC,
    ConfusionMatrix,
    MetricCollection,
)
from torchvision import transforms
from torchvision.datasets import ImageFolder
from cfg import Config


class TimmBasedClassifierModel(L.LightningModule):
    optimizers = {"adam": Adam, "sgd": SGD, "adamw": AdamW, "radam": RAdam}

    def __init__(
        self,
        num_classes=Config.numClasses,
        model_name=Config.modelName,
        optimizer=Config.optimizer,
        lr=Config.learningRate,
        batch_size=Config.batchSize,
        transfer=Config.transfer,
        tune_fc_only=Config.tuneFcOnly,
        exportable=Config.exportable,
        scriptable=Config.scriptable,
        save_hyperpram=Config.saveHyperParam,
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
        # create metrics
        task = "binary" if num_classes == 1 else "multiclass"
        default_args = {"task": task, "num_classes": num_classes}
        self.acc = Accuracy(**default_args)

        metrics = MetricCollection(
            {
                "f1score": F1Score(**default_args),
                "precision": Precision(**default_args),
                "recall": Recall(**default_args),
                "confusion_matrix": ConfusionMatrix(**default_args),
                "auroc": AUROC(**default_args),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")

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

        if save_hyperpram:
            self.save_hyperparameters()

    def forward(self, X):
        return self.model(X)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)

    def _log(self, computed, **kwargs):
        scalar_metrics = {k: v for k, v in computed.items() if len(v.shape) == 0}
        self.log_dict(scalar_metrics, **kwargs)

    def _step(self, batch, metrics_target):
        x, y = batch
        preds = self(x)

        if self.num_classes == 1:
            preds = preds.flatten()
            y = y.float()

        loss = self.loss_fn(preds, y)
        acc = self.acc(preds, y)
        metrics = metrics_target(preds, y)
        return loss, acc, metrics

    def training_step(self, batch, batch_idx):
        loss, acc, metrics = self._step(batch, self.train_metrics)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self._log(
            metrics,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        return loss

    def on_train_epoch_end(self) -> None:
        self.confusion_matrix(self.train_metrics.compute())
        self.train_metrics.reset()
        self.acc.reset()
        return super().on_train_epoch_end()

    def validation_step(self, batch, batch_idx):
        loss, acc, metrics = self._step(batch, self.val_metrics)
        # perform logging
        self.log("val_loss", loss, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)
        self._log(metrics, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        self.confusion_matrix(self.val_metrics.compute())
        self.val_metrics.reset()
        self.acc.reset()
        return super().on_validation_epoch_end()

    def confusion_matrix(self, metrics):
        key = [s for s in metrics.keys() if "confusion_matrix" in s.lower()][0]
        cm = metrics[key]

        df_cm = pd.DataFrame(
            cm.detach().cpu().numpy().astype(int),
            index=range(self.num_classes),
            columns=range(self.num_classes),
        )
        plt.figure(figsize=(10, 7))
        sns.set(font_scale=1.2)
        fig_ = sns.heatmap(
            df_cm, annot=True, annot_kws={"size": 16}, fmt="d", cmap="Spectral"
        ).get_figure()
        plt.close(fig_)
        self.logger.experiment.add_figure(key, fig_, self.current_epoch)

    def predict_step(self, batch, batch_idx):
        x = batch
        preds = self(x)
        result = torch.argmax(preds, -1)

        return preds, result
