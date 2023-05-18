import lightning as L
from torch.utils.data import DataLoader
from cfg import Config
from datasets import SangchuDataset
from torchvision import transforms
from typing import Any, Callable, Optional, Tuple
from multiprocessing import cpu_count

default_transformer = transforms.Compose(
    [
        transforms.Resize((500, 500)),
        transforms.ToTensor(),
        transforms.Normalize((0.48232,), (0.23051,)),
    ]
)


class SangchuDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = Config.dataDir,
        batch_size: int = Config.batchSize,
        img_size: int = 500,
        transform: Optional[Callable] = default_transformer,
        target_transform: Optional[Callable] = None,
        shuffle: bool = Config.shuffle,
        num_workers: int = Config.numWorkers,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.transform = transform
        self.target_transform = target_transform
        self.shuffle = shuffle
        self.num_workers = num_workers

    def prepare_data(self):
        # download
        SangchuDataset(self.data_dir, Config.dataName, download=True)

    def setup(self, stage: str):
        self.data_train = SangchuDataset(
            self.data_dir,
            Config.dataName,
            img_size=self.img_size,
            transform=self.transform,
            target_transform=self.target_transform,
        )
        self.data_val = SangchuDataset(
            self.data_dir,
            Config.dataName,
            img_size=self.img_size,
            img_type="val",
            transform=self.transform,
            target_transform=self.target_transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val, batch_size=self.batch_size, num_workers=self.num_workers
        )
