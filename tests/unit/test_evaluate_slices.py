from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.evaluate import evaluate_impressions
from src.models.baselines.popularity import PopularityRecommender


def test_evaluate_impressions_multi_slice(tmp_path: Path) -> None:
    impressions_dir = tmp_path / "dev_impressions"
    impressions_dir.mkdir(parents=True)

    # 2 impressions:
    # 1. Warm user (history > 5), contains cold item (seen_in_train_mask has False)
    # 2. Cold user (history <= 5), all warm items (seen_in_train_mask all True)
    table = pa.Table.from_pylist(
        [
            {
                "impression_id": 1,
                "user_idx": 101,
                "history_item_idxs": [1, 2, 3, 4, 5, 6],  # > 5 items -> warm user
                "candidate_item_idxs": [10, 20, 99],
                "labels": [1, 0, 0],
                "candidate_seen_in_train_mask": [True, True, False],  # has cold candidate
            },
            {
                "impression_id": 2,
                "user_idx": 102,
                "history_item_idxs": [1, 2],  # <= 5 items -> cold user
                "candidate_item_idxs": [10, 20],
                "labels": [0, 1],
                "candidate_seen_in_train_mask": [True, True],  # all warm candidates
            },
        ]
    )
    pq.write_table(table, impressions_dir / "part-00000.parquet")

    # Fit a simple popularity model where item 10 has score 5 and item 20 has score 2
    train_df = pd.DataFrame(
        {
            "user_idx": [1, 1, 2, 2, 3, 3, 4],
            "item_idx": [10, 10, 10, 10, 10, 20, 20],
        }
    )
    model = PopularityRecommender(user_col="user_idx", item_col="item_idx")
    model.fit(train_df)

    all_items = {10, 20, 99}
    results = evaluate_impressions(
        model=model,
        impressions_path=impressions_dir,
        k_list=[5, 10],
        all_catalog_items=all_items,
    )

    assert results["total_evaluated_impressions"] == 2
    slices = results["slices"]
    assert slices["overall"]["count"] == 2
    assert slices["cold_item"]["count"] == 1
    assert slices["warm_item"]["count"] == 1
    assert slices["cold_user"]["count"] == 1
    assert slices["warm_user"]["count"] == 1

    # In imp 1: scores are 10 (5.0), 20 (2.0), 99 (0.0).
    # Ranked order: [10, 20, 99]. True item is 10 (click at rank 1).
    # NDCG@10 = 1.0, Recall@10 = 1.0, MRR@10 = 1.0
    assert slices["cold_item"]["metrics"]["NDCG@10"] == 1.0

    # In imp 2: scores are 10 (5.0), 20 (2.0).
    # Ranked order: [10, 20]. True item is 20 (click at rank 2).
    # MRR@10 = 0.5
    assert slices["warm_item"]["metrics"]["MRR@10"] == 0.5

    # Overall NDCG@10 should be mean of 1.0 and 1/log2(3) = (1.0 + 0.6309) / 2
    assert slices["overall"]["metrics"]["NDCG@10"] > 0.0

    # Beyond accuracy
    beyond = results["beyond_accuracy"]
    assert beyond["total_catalog_items"] == 3
    assert beyond["catalog_coverage@10"] > 0.0
