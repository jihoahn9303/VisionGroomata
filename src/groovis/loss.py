import math
from typing import Union

import torch
import torch.nn.functional as F
from einops import rearrange


class SimCLRLoss:
    def __init__(self, temperature: float = 0.1):
        """Implements SimCLR's NT-Xent loss.

        Args:
            temperature (float): scaling parameter for softmax.
        """

        self.temperature = temperature

    def __call__(
        self, representations_1: torch.Tensor, representations_2: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluate the loss of given pairs of representations
        """

        representations = self.combine_representations(
            representations_1, representations_2
        )
        similarity = self.compare_representations(representations)
        loss = self.evaluate_similarity_alt(similarity)

        return loss

    def combine_representations(
        self, representations_1: torch.Tensor, representations_2: torch.Tensor
    ) -> torch.Tensor:
        # first methodlogy
        # B, D = representations_1.shape
        # representations = torch.empty(2*B, D, dtype=torch.float)
        # representations[::2] = representations_1
        # representations[1::2] = representations_2

        # second metholody
        representations = rearrange(
            [representations_1, representations_2], "g b d -> (b g) d"
        )
        return representations

    def compare_representations(self, representations: torch.Tensor) -> torch.Tensor:
        representations = F.normalize(representations, dim=1)
        similarity = torch.einsum("i d, j d -> i j", representations, representations)

        return similarity

    @staticmethod
    def generate_simclr_positive_indices(
        batch_size: int, device: Union[torch.device, str, None] = None
    ) -> torch.Tensor:
        base = torch.arange(batch_size, device=device)
        odds = base * 2 + 1
        even = base * 2

        return rearrange([odds, even], "g b -> (b g)")

    def evaluate_similarity_alt(
        self,
        similarity: torch.Tensor,
    ) -> torch.Tensor:
        N, _ = similarity.shape
        B = N // 2  # batch size

        similarity.div_(self.temperature)
        similarity.fill_diagonal_(torch.finfo(torch.float).min)

        similarity = similarity.log_softmax(dim=1)

        loss = F.nll_loss(
            similarity, self.generate_simclr_positive_indices(B, similarity.device)
        )

        return loss

    def combine_representations_slow(
        self, representations_1: torch.Tensor, representations_2: torch.Tensor
    ) -> torch.Tensor:
        B, D = representations_1.shape

        representations = torch.empty(2 * B, D, dtype=torch.float)

        for idx in range(B):
            representations[2 * idx] = representations_1[idx]
            representations[2 * idx + 1] = representations_2[idx]

        return representations

    def compare_representations_slow(
        self, representations: torch.Tensor
    ) -> torch.Tensor:
        N, D = representations.shape  # N = 2* batch_size

        similarity = torch.empty(N, N, dtype=torch.float)

        for i in range(N):
            representations[i] = representations[i] / torch.norm(representations[i])

        for idx_1 in range(N):
            for idx_2 in range(N):
                similarity[idx_1][idx_2] = torch.dot(
                    representations[idx_1], representations[idx_2]
                )

        return similarity

    def evaluate_similarity_alt_slow(
        self, similarity: torch.Tensor, temparature: float = 0.1
    ) -> torch.Tensor:
        N, _ = similarity.shape
        similarity /= temparature

        loss = torch.tensor(0.0)

        for idx in range(N):
            base_row = similarity[idx]
            if idx % 2 == 0:
                positive = base_row[idx + 1]
            else:
                positive = base_row[idx - 1]
            base_row[idx] = torch.finfo(torch.float).min

            positive_probs = positive.exp() / base_row.exp().sum()
            # max_similarity = base_row.exp().sum().log()
            loss += -math.log(positive_probs)

        return loss / N


def evaluate_similarity(similarity: torch.Tensor) -> torch.Tensor:
    N, _ = similarity.shape

    loss = torch.tensor(0.0)

    for idx in range(N):
        base_row = similarity[idx]
        if idx % 2 == 0:
            positive = base_row[idx + 1]
        else:
            positive = base_row[idx - 1]
        base_row[idx] = torch.finfo(torch.float).min
        # neutral = base_row[idx]
        # averaged_negatives = (base_row.sum() - (neutral + positive)) / (N-2)
        # loss += (averaged_negatives - positive)
        max_similarity = base_row.exp().sum().log()
        loss += max_similarity - positive

    return loss / N
