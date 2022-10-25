import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(image)
    plt.savefig("augmented.png", bbox_inches="tight", pad_inches=0)


image = np.array(Image.open("/workspaces/vision/data/train/dog.jpg"))

# transform = A.HorizontalFlip(p=1)
transform = A.Compose(
    [
        A.RandomResizedCrop(
            height=224,
            width=224,
            scale=(0.08, 1),
            ratio=(0.75, 1.3333333333333333),
            always_apply=True,
        ),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=0.8),
        A.ToGray(p=0.2),
        A.GaussianBlur(blur_limit=(21, 23), sigma_limit=(0.1, 2), always_apply=True),
    ]
)

augmented_image = transform(image=image)["image"]

visualize(augmented_image)
