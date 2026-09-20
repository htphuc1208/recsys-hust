import torch

from src.models.matrix_factorization import MatrixFactorization


def test_matrix_factorization_forward_shape():
    model = MatrixFactorization(
        num_users=4,
        num_items=6,
        embedding_dim=3,
        use_bias=True,
    )

    user_ids = torch.tensor([0, 1, 2])
    item_ids = torch.tensor([1, 3, 5])

    scores = model(user_ids, item_ids)

    assert scores.shape == (3,)


def test_matrix_factorization_without_bias():
    model = MatrixFactorization(
        num_users=2,
        num_items=2,
        embedding_dim=2,
        use_bias=False,
    )

    with torch.no_grad():
        model.user_embedding.weight.copy_(
            torch.tensor(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ]
            )
        )
        model.item_embedding.weight.copy_(
            torch.tensor(
                [
                    [5.0, 6.0],
                    [7.0, 8.0],
                ]
            )
        )

    score = model(
        torch.tensor([0]),
        torch.tensor([1]),
    )

    assert torch.allclose(
        score,
        torch.tensor([23.0]),
    )
