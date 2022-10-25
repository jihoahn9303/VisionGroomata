import pytest
import torch

from groovis.loss import SimCLRLoss


@pytest.fixture
def loss_fn():
    return SimCLRLoss(temperature=0.1)


def test_combine_representations(
    loss_fn: SimCLRLoss,
):
    representations_1 = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
    )

    representations_2 = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 1.0, 2.0], [2.0, 2.0, 3.0]]
    )

    assert torch.allclose(
        loss_fn.combine_representations(
            representations_1.clone(), representations_2.clone()
        ),
        loss_fn.combine_representations_slow(
            representations_1.clone(), representations_2.clone()
        ),
    )


def test_compare_representations(loss_fn: SimCLRLoss):
    representations_1 = torch.tensor(
        [[1.3, 2.0, 3.0], [0.4, 0.0, 1.0], [1.5, 1.0, 1.0], [1.6, 1.0, 2.0]]
    )

    assert torch.allclose(
        loss_fn.compare_representations(representations_1.clone()),
        loss_fn.compare_representations_slow(representations_1.clone()),
    )


def test_evaluate_similarity_alt(loss_fn: SimCLRLoss):
    similarity = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 2.0, 3.0],
            [2.0, 2.0, 0.0, 3.0],
            [3.0, 3.0, 3.0, 0.0],
        ]
    )

    assert torch.allclose(
        loss_fn.evaluate_similarity_alt(similarity.clone()),
        loss_fn.evaluate_similarity_alt_slow(similarity.clone()),
    )
