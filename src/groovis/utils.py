import numpy as np
import torch
from einops import rearrange
from PIL import Image

IMAGE_SIZE = 224


def image_path_to_tensor(path: str) -> torch.Tensor:
    image = Image.open(path).resize((IMAGE_SIZE, IMAGE_SIZE))
    image = np.array(image)
    image = torch.tensor(image, dtype=torch.float32) / 255.0

    # print(image)
    # print(image.shape)  # [Height, Width, channel]

    image = rearrange(
        image, "h w c -> c h w"
    )  # [Height, Width, channel] -> [channel, Height, Width]

    return image


def image_path_to_array(path: str) -> np.ndarray:
    image = Image.open(path)
    image = np.array(image)

    return image
