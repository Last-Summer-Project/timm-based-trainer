from pathlib import Path
from typing import Any, Callable, Optional, Tuple
from glob import glob

from cfg import Config
from PIL import Image
from datasets import letterbox_image
from datamodule import default_transformer
from torch.utils.data import DataLoader
