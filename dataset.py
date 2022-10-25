from dotenv import load_dotenv

load_dotenv()

import albumentations as A
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from src.groovis.data.dataset import SIMCLR_AUG

# dataset for training
# dataset = load_dataset(path="frgfm/imagenette", name='320px', split='train')
# print(dataset)
# print(dataset[0])
# print(dataset[0]['image'])


class Imagenette(Dataset):
    def __init__(self, transforms: A.Compose = SIMCLR_AUG):
        self.dataset = load_dataset(
            path="frgfm/imagenette", name="320px", split="train"
        )

        self.transforms = transforms

    def __getitem__(self, index) -> list[torch.Tensor]:
        image = self.dataset[index]["image"]  # 라벨 정보는 제외하고 이미지 정보만 가져옴.
        image = np.array(image)

        return [self.transforms(image=image)["image"] / 255.0 for _ in range(2)]

    def __len__(self):
        return len(self.paths)
