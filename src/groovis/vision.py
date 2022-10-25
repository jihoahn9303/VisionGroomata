import torch
from einops.layers.torch import Rearrange, Reduce
from torch import nn

PATCH_SIZE = 16


class PatchEmbed(nn.Module):
    def __init__(
        self, patch_size: int = 16, channels: int = 3, embed_dim: int = 1024
    ) -> None:
        super().__init__()

        # module for rearrange
        self.split_images = Rearrange(
            "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size
        )

        # module for weight averaging
        self.projection = nn.Linear(
            in_features=channels * patch_size**2, out_features=embed_dim, bias=True
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.split_images(images)

        representation = self.projection(patches)

        # Below code is equivalent to self.projection(patches)
        # representation = torch.einsum(
        #     "b n p, d p -> b n d",
        #     patches,
        #     self.weight
        # )

        # representation += repeat(
        #     self.bias,
        #     "d -> b n d",
        #     b=representation.shape[0],
        #     n=representation.shape[1]
        # )

        return representation


class Architecture(nn.Module):
    def __init__(
        self, patch_size: int = 16, channels: int = 3, embed_dim: int = 1024
    ) -> None:
        super().__init__()

        # module for patch embedding
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, channels=channels, embed_dim=embed_dim
        )

        # module for pooling
        self.pool = Reduce("b n d -> b d", "mean")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        representation = self.patch_embed(images)
        # representation = reduce(representation, "b n d-> b d", "mean")   # pooling
        representation = self.pool(representation)

        return representation


# self-contained class (architecture + instruction for architecture)
class Vision(nn.Module):
    def __init__(self, architecture: nn.Module) -> None:
        super().__init__()

        self.architecture = architecture

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.architecture(images)


# vision = Vision()
# print(vision._parameters)
# print(vision._modules)
# print(vision._modules['patch_embed']._parameters)
# print(vision.parameters(recurse=False))
# print(vision.parameters(recurse=True))

# for name, parameter in vision.named_parameters(recurse=True):
#     print(name)
