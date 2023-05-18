from lightning import Trainer
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import ModelCheckpoint
from datamodule import SangchuDataModule
from model import ResNetClassifierModel
from cfg import Config
import torch

def main():
    #torch.set_float32_matmul_precision(precision)
    dm = SangchuDataModule()
    model = ResNetClassifierModel(Config.numClasses, Config.resnetVersion)

    checkpoint_last = ModelCheckpoint(filename="last", save_last=True)
    checkpoint_acc = ModelCheckpoint(filename="acc_best-{epoch}-{val_acc:.2f}", monitor="val_acc", mode="max", save_top_k=1)
    checkpoint_loss = ModelCheckpoint(filename="loss_best-{epoch}-{val_loss:.2f}", monitor="val_loss", mode="min", save_top_k=1)

    trainer = Trainer(max_epochs=Config.epochs, default_root_dir=Config.rootDir, callbacks= [
        checkpoint_last,
        checkpoint_acc,
        checkpoint_loss
    ])

    # autobatch
    tuner = Tuner(trainer)
    tuner.scale_batch_size(model, datamodule=dm, mode="power")

    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    main()
