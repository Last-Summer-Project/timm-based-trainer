from lightning import Trainer
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import ModelCheckpoint
from datasets import SimpleDataset
from model import TimmBasedClassifierModel
from cfg import Config
from torch.utils.data import DataLoader
import torch
import argparse


def main():
    parser = argparse.ArgumentParser(description='Predict an image by model.')
    parser.add_argument("--model", type=str, help="Model", required=True)
    parser.add_argument("--input", type=str, help="Input Image", required=True)
    parser.add_argument("--output", type=str, help="Output Folder", default="./output")

    args = parser.parse_args()

    # Tensor core test and set.
    if (
        torch.cuda.is_available()
        and torch.backends.cuda.matmul.allow_tf32 is False
        and torch.cuda.get_device_capability() >= (8, 0)
    ):
        torch.set_float32_matmul_precision("high")

    ds = SimpleDataset(args.input)
    dl = DataLoader(
        ds,
        batch_size=Config.batchSize,
        num_workers=min(len(ds), Config.numWorkers)
    )

    model = TimmBasedClassifierModel().load_from_checkpoint(args.model, model_name=Config.modelName, num_classes=Config.numClasses)
    trainer = Trainer()
    pred = trainer.predict(model, dl)
    print("Result is : ", pred)

if __name__ == "__main__":
    main()
