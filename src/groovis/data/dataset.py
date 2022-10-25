from pathlib import Path
from typing import Literal

import albumentations as A
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset

from groovis.utils import image_path_to_array

from .augmentation import SIMCLR_AUG

Splits = Literal["train", "validation"]
IMG_EXTENSIONS = [".webp", ".jpg", ".jpeg", ".png"]


class Animals(Dataset):
    def __init__(self, root: str, transforms: A.Compose = SIMCLR_AUG):
        self.paths = [
            path for path in Path(root).iterdir() if path.suffix in IMG_EXTENSIONS
        ]
        self.transforms = transforms

    def __getitem__(self, index) -> list[torch.Tensor]:
        # return image_path_to_tensor(self.paths[index])
        image = image_path_to_array(self.paths[index])

        return [self.transforms(image=image)["image"] / 255.0 for _ in range(2)]

    def __len__(self):
        return len(self.paths)


class Imagenette(Dataset):
    def __init__(self, transforms: A.Compose = SIMCLR_AUG, split: Splits = "train"):
        self.dataset = load_dataset(path="frgfm/imagenette", name="320px", split=split)
        self.transforms = transforms

    def __getitem__(self, index) -> list[torch.Tensor]:
        image: Image.Image = self.dataset[index]["image"]  # 라벨 정보를 제외하고 이미지 정보만 가져옴.
        image = image.convert("RGB")  # 흑백 이미지도 존재할 수 있으므로, batch 단위 전처리를 위하여 채널을 통일함.
        image = np.array(image)

        return [self.transforms(image=image)["image"] / 255.0 for _ in range(2)]

    def __len__(self):
        return self.dataset.num_rows
