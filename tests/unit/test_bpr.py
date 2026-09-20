from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from src.models.bpr import BPR
from src.models.bpr.data import (
    build_bpr_pairs,
    iter_bpr_batches,
)
from src.models.bpr.train import calculate_bpr_loss


def test_bpr_forward_shape():
    model = BPR(
        num_users=4,
        num_items=6,
        embedding_dim=3,
        use_item_bias=True,
    )

    user_ids = torch.tensor([0, 1, 2])
    item_ids = torch.tensor([1, 3, 5])

    scores = model(user_ids, item_ids)

    assert scores.shape == (3,)


def test_bpr_without_item_bias():
    model = BPR(
        num_users=2,
        num_items=2,
        embedding_dim=2,
        use_item_bias=False,
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


def test_bpr_item_bias_is_added():
    model = BPR(
        num_users=1,
        num_items=2,
        embedding_dim=2,
        use_item_bias=True,
    )

    with torch.no_grad():
        model.user_embedding.weight.zero_()
        model.item_embedding.weight.zero_()
        model.item_bias.weight.copy_(torch.tensor([[0.5], [1.5]]))

    scores = model(
        torch.tensor([0, 0]),
        torch.tensor([0, 1]),
    )

    assert torch.allclose(
        scores,
        torch.tensor([0.5, 1.5]),
    )


def test_bpr_loss_is_lower_when_positive_score_is_higher():
    good_loss = calculate_bpr_loss(
        torch.tensor([3.0]),
        torch.tensor([1.0]),
    )
    bad_loss = calculate_bpr_loss(
        torch.tensor([1.0]),
        torch.tensor([3.0]),
    )

    assert good_loss < bad_loss


def test_build_bpr_pairs_one_positive_four_negatives():
    pairs = build_bpr_pairs(
        user_idx=7,
        item_ids=[10, 11, 12, 13, 14],
        labels=[1, 0, 0, 0, 0],
    )

    assert pairs == [
        (7, 10, 11),
        (7, 10, 12),
        (7, 10, 13),
        (7, 10, 14),
    ]


def test_build_bpr_pairs_distributes_negatives_across_positives():
    pairs = build_bpr_pairs(
        user_idx=3,
        item_ids=[10, 20, 30, 31, 32, 33, 34, 35, 36],
        labels=[1, 1, 0, 0, 0, 0, 0, 0, 0],
    )

    assert len(pairs) == 7
    assert [pair[2] for pair in pairs] == [
        30,
        31,
        32,
        33,
        34,
        35,
        36,
    ]

    positive_counts = {
        10: 0,
        20: 0,
    }
    for _, positive_item, _ in pairs:
        positive_counts[positive_item] += 1

    assert positive_counts == {
        10: 4,
        20: 3,
    }


def _write_pointwise_part(
    path: Path,
    rows: list[dict],
) -> None:
    """Ghi dữ liệu pointwise nhỏ phục vụ unit test."""
    table = pa.Table.from_pylist(
        rows,
        schema=pa.schema(
            [
                ("impression_id", pa.int64()),
                ("user_idx", pa.int64()),
                ("item_idx", pa.int64()),
                ("label", pa.int8()),
            ]
        ),
    )
    pq.write_table(table, path)


def test_iter_bpr_batches_keeps_impression_across_files(
    tmp_path: Path,
):
    _write_pointwise_part(
        tmp_path / "part-00000.parquet",
        [
            {
                "impression_id": 100,
                "user_idx": 0,
                "item_idx": 10,
                "label": 1,
            },
            {
                "impression_id": 100,
                "user_idx": 0,
                "item_idx": 11,
                "label": 0,
            },
        ],
    )
    _write_pointwise_part(
        tmp_path / "part-00001.parquet",
        [
            {
                "impression_id": 100,
                "user_idx": 0,
                "item_idx": 12,
                "label": 0,
            },
            {
                "impression_id": 100,
                "user_idx": 0,
                "item_idx": 13,
                "label": 0,
            },
            {
                "impression_id": 101,
                "user_idx": 1,
                "item_idx": 20,
                "label": 1,
            },
            {
                "impression_id": 101,
                "user_idx": 1,
                "item_idx": 21,
                "label": 0,
            },
        ],
    )

    batches = list(
        iter_bpr_batches(
            tmp_path,
            batch_size=10,
            shuffle=False,
            shuffle_buffer_size=10,
            read_batch_size=2,
        )
    )

    assert len(batches) == 1

    user_ids, positive_ids, negative_ids = batches[0]

    assert user_ids.tolist() == [0, 0, 0, 1]
    assert positive_ids.tolist() == [10, 10, 10, 20]
    assert negative_ids.tolist() == [11, 12, 13, 21]
