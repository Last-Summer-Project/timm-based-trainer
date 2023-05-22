import os
import lightning as L
import json
from pathlib import Path
import torch.nn.functional as F
from typing import Any, Callable, Optional, Tuple
from glob import glob

from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive
from cfg import Config
from torchvision import transforms

default_transformer = transforms.Compose(
    [
        transforms.Resize((500, 500)),
        transforms.ToTensor(),
        transforms.Normalize((0.48232,), (0.23051,)),
    ]
)


class SangchuDataset(Dataset):
    classes = ["NORMAL", "DISEASE-sclerotinia_minor", "DISEASE-bremia-lactucae"]

    def __init__(
        self,
        root: str = Config.dataDir,
        data_name: str = Config.dataName,
        img_type: str = "train",
        img_size: int = 500,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        self.root = root
        self.data_name = data_name
        self.img_type = img_type
        self.img_size = img_size
        self.transform = transform
        self.target_transform = target_transform
        if download:
            self.download()

        self.data, self.targets = self._load_data()

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def images_path(self):
        return os.path.join(self.raw_folder, self.data_name, "images", self.img_type)

    @property
    def labels_path(self):
        return os.path.join(self.raw_folder, self.data_name, "labels", self.img_type)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.raw_folder, Config.dataName))

    def download(self) -> None:
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        try:
            print(f"Downloading {Config.downloadURL}")
            download_and_extract_archive(
                Config.downloadURL, download_root=self.raw_folder
            )
        except Exception as e:
            raise RuntimeError(f"Error downloading {Config.downloadURL} : {e}")

    def _load_data(self):
        images_list = list(glob("**/*.*", root_dir=self.images_path, recursive=True))
        labels_list = list(map(lambda x: x + ".json", images_list))
        labels_list = list(map(lambda x: self._read_label(x), labels_list))

        return images_list, labels_list

    def _label_path(self, label: str):
        return os.path.join(self.labels_path, label)

    def _read_label(self, label: str):
        with open(self._label_path(label), "r") as f:
            j = json.loads(f.read())
            item = Config.remapClass.get(int(j["annotations"]["disease"]), -1)
            if item == -1:
                raise RuntimeError(f"Invalid value specified on {label}")
            return item

    def _image_path(self, img: str):
        return os.path.join(self.images_path, img)

    def _read_image(self, img: str):
        image = Image.open(self._image_path(img))
        # image.verify()
        image = letterbox_image(image, (self.img_size, self.img_size))
        return image

    def __getitem__(self, index: int) -> Any:
        img, target = self.data[index], self.targets[index]

        img = self._read_image(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


class SimpleDataset:
    def __init__(
        self,
        images: list[str] | str,
        img_size: int = 500,
        transform: Optional[Callable] = default_transformer,
        target_transform: Optional[Callable] = None,
    ) -> None:
        if isinstance(images, str):
            images = [images]
        self.images = images
        self.img_size = img_size
        self.transform = transform
        self.target_transform = target_transform

    def _read_image(self, img: str):
        image = Image.open(img)
        image = letterbox_image(image, (self.img_size, self.img_size))
        return image

    def __getitem__(self, index: int) -> Any:
        img = self.images[index]
        img = self._read_image(img)
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self) -> int:
        return len(self.images)


def letterbox_image(image: Image, size: Tuple[int, int]) -> Image:
    """resize image with unchanged aspect ratio using padding"""
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new("RGB", size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image
