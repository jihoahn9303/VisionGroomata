import torch
import torch.nn.functional as F

from groovis.utils import image_path_to_tensor


def test_invariance():
    # vision = Vision()
    vision = torch.load("build/vision.pth")

    image_tiger_1 = image_path_to_tensor("/workspaces/vision/data/test/tiger_1.jpg")
    image_tiger_2 = image_path_to_tensor("/workspaces/vision/data/test/tiger_2.jpg")
    image_dog = image_path_to_tensor("/workspaces/vision/data/test/dog.jpg")

    tiger_1 = vision(image_tiger_1)  # shape: (d, )
    tiger_2 = vision(image_tiger_2)  # shape: (d, )
    dog = vision(image_dog)  # shape: (d, )

    diff_tiger_tiger = F.l1_loss(tiger_2, tiger_1)
    diff_tiger_dog_1 = F.l1_loss(tiger_1, dog)
    diff_tiger_dog_2 = F.l1_loss(tiger_2, dog)

    quality = (diff_tiger_dog_1 + diff_tiger_dog_2) / 2 - diff_tiger_tiger
    print(quality)

    assert quality > 0
