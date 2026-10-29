# RecSys HUST — Tổng hợp 3 nhiệm vụ: News Features, Evaluation & MIND Submission

> **Phạm vi:** Nhánh `feature/data-evaluation-pipeline` — phần việc Data Engineering, Evaluation & Analysis (Thành viên 3).  
> **Dữ liệu:** Microsoft MIND Large.  
> **Trạng thái:** Nhiệm vụ 1 và 2 đã có kết quả thực nghiệm Full Dev; Nhiệm vụ 3 đã xuất và kiểm định thành công **5.000 impressions Test**. **Chưa chạy Full Test / chưa nộp lên máy chủ đánh giá**, chờ mô hình Hybrid cuối cùng.

README này tổng hợp **cả ba nhiệm vụ trong một tài liệu**, dùng để bàn giao code, tái lập kết quả và hướng dẫn các thành viên khác sử dụng pipeline. Đây là tài liệu bổ sung cho `README.md` chung của dự án; không thay thế phần hướng dẫn cài đặt, phục vụ API hoặc huấn luyện các kiến trúc khác.

## Mục lục

1. [Tổng quan và trạng thái](#1-tổng-quan-và-trạng-thái)
2. [Chuẩn bị môi trường và dữ liệu](#2-chuẩn-bị-môi-trường-và-dữ-liệu)
3. [Nhiệm vụ 1 — News Features](#3-nhiệm-vụ-1--news-features)
4. [Nhiệm vụ 2 — Evaluation & Bootstrap CI](#4-nhiệm-vụ-2--evaluation--bootstrap-ci)
5. [Nhiệm vụ 3 — MIND Test Submission Pipeline](#5-nhiệm-vụ-3--mind-test-submission-pipeline)
6. [Quy trình dùng lại và mở rộng cho mô hình mới](#6-quy-trình-dùng-lại-và-mở-rộng-cho-mô-hình-mới)
7. [Kiểm thử, Git và bàn giao](#7-kiểm-thử-git-và-bàn-giao)

---

## 1. Tổng quan và trạng thái

| Nhiệm vụ | Mục tiêu | Kết quả hiện tại |
|---|---|---|
| **1. News Features** | Đo độ bao phủ entity, thử bổ sung `abstract_entity`, tối ưu lưu/tải sparse TF-IDF | Đã triển khai, đánh giá trên MIND Large, giữ cấu hình baseline tốt hơn cho NDCG tổng thể |
| **2. Evaluation & Bootstrap CI** | Đánh giá nhiều lát cắt, CI 95%, so sánh trước/sau P3, tổng hợp bảng và biểu đồ | Đã hoàn thiện cho **BPR-MF, Content-Based và hai biến thể P3**: **120/120 dòng metric–slice có CI**, Full Dev **376.471 impressions** |
| **3. Test Submission** | Suy luận không nhãn và xuất `prediction.txt` đúng định dạng Microsoft MIND | Đã chạy thử **5.000 Test impressions / 199.107 candidates**; TXT, ZIP và đối chiếu thứ tự với raw TSV đều hợp lệ |

**Quyết định về phạm vi:** Hiện chỉ dùng hai mô hình nền **BPR-MF** (`exp04`) và **Content-Based** (`exp05`), cùng **BPR + P3** (`exp07`) và **Content-Based + P3** (`exp06`). MF (BCE), Random, Popularity, ItemKNN, SASRec và Hybrid **không phải huấn luyện/chạy lại** để hoàn tất phạm vi hiện tại. Pipeline vẫn hỗ trợ thêm các mô hình đó sau khi có checkpoint và kết quả tương thích.

### Luồng công việc

```text
MIND Large (train / dev / test)
        |
        v
Ingestion + prepare_mind (Parquet + ID mappings)
        |
        +--> [1] News Features
        |      Entity coverage -> Content features -> Feature Store
        |      -> BPR/Content-Based checkpoint (checkpoint đã được huấn luyện)
        |
        +--> [2] Offline Evaluation (Dev có labels)
        |      NDCG / Recall / MRR / Beyond-Accuracy
        |      -> Bootstrap CI theo slice -> P3 paired CI -> CSV/PNG/MD
        |
        +--> [3] Submission Inference (Test không labels)
               Checkpoint -> scores -> ranking -> prediction.txt
               -> prediction.zip -> validate với behaviors.tsv gốc
```

**Nguyên tắc dữ liệu:** Train dùng để học mô hình, từ vựng TF-IDF và encoder; Dev dùng để đánh giá/chọn cấu hình; Test **không có nhãn click**, chỉ dùng suy luận và xuất thứ hạng. Không được suy luận chất lượng Test từ dữ liệu nhãn không tồn tại.

## 2. Chuẩn bị môi trường và dữ liệu

Chạy PowerShell tại thư mục gốc `recsys-hust` (Windows + VS Code):

```powershell
python --version
python -m pip install -e ".[dev]"
python -m pytest --version
```

Cấu trúc cần có trên máy triển khai:

```text
data/
├── raw/mind_large/
│   ├── MINDlarge_train/
│   │   ├── behaviors.tsv
│   │   ├── news.tsv
│   │   └── entity_embedding.vec
│   ├── MINDlarge_dev/
│   │   ├── behaviors.tsv
│   │   └── news.tsv
│   └── MINDlarge_test/
│       ├── behaviors.tsv
│       └── news.tsv
└── processed/mind_large/
    ├── mappings/{users,items}/
    ├── train/{news,impressions,positive_interactions,...}/
    ├── dev/{news,impressions}/
    ├── test/{news,impressions}/
    └── features/b4*/

artifacts/checkpoints/
├── exp04_mind_bpr_bpr.pkl
└── exp05_mind_content_content_based.pkl
```

Kiểm tra nhanh:

```powershell
Test-Path data/processed/mind_large/mappings/items
Test-Path data/processed/mind_large/dev/impressions
Test-Path data/processed/mind_large/test/impressions
Test-Path data/raw/mind_large/MINDlarge_test/behaviors.tsv
Test-Path artifacts/checkpoints/exp04_mind_bpr_bpr.pkl
Test-Path artifacts/checkpoints/exp05_mind_content_content_based.pkl
```

Nếu chưa có dữ liệu đã xử lý, thực hiện theo `README.md` gốc của repository, từ tải/giải nén MIND Large rồi chạy:

```powershell
python -m src.ingest.mind --split all
python -m src.features.prepare_mind --split all
```

**Không đưa dataset, checkpoint hoặc các file dự đoán hàng triệu dòng lên Git.** Các thành phần này được giữ cục bộ hoặc lưu trên nơi chia sẻ artifact của nhóm.

---

## 3. Nhiệm vụ 1 — News Features

### 3.1. Mục tiêu

- Kiểm tra độ bao phủ của `title_entities` và `abstract_entities` trong `news.tsv` bằng các vector Wikidata từ `entity_embedding.vec`.
- Đánh giá việc thêm `abstract_entity_embeddings` có thật sự giúp cải thiện chất lượng gợi ý hay không, thay vì mặc định bật tính năng mới.
- Tối ưu cách lưu/tải các ma trận TF-IDF thưa ở định dạng CSR `.npz` mà **không thay đổi kết quả dự đoán**.

### 3.2. Module chính

| Thành phần | Vai trò |
|---|---|
| `src/features/entity_coverage.py` | Thống kê entity presence, embedding coverage, chồng lấp và bổ sung giữa title/abstract theo nhóm item |
| `src/features/news_content.py` | Tạo TF-IDF, category/subcategory one-hot, embedding từ title entity; tùy chọn thêm abstract entity |
| `src/features/feature_store_tools.py` | Chuyển nén/không nén sparse `.npz`, benchmark thời gian tải feature |
| `src/models/content_based/recommender.py` | Hỗ trợ đặc trưng abstract entity tùy chọn; tương thích bộ feature cũ |
| `src/models/content_based/weight_sweep.py` | Quét 5 hoặc 6 trọng số feature |
| `configs/experiment/exp11_mind_content_baseline_new_features.yaml` | Cấu hình baseline có đủ feature mới nhưng trọng số abstract entity bằng 0 |
| `configs/experiment/exp12_mind_content_abstract_entity.yaml` | Cấu hình thử thêm 0,05 trọng số abstract entity |

### 3.3. Cách chạy

**A. Đo entity coverage** (không huấn luyện mô hình):

```powershell
python -m src.features.entity_coverage
```

Đầu ra:

```text
reports/tables/news_entity_coverage.csv
reports/tables/news_entity_complementarity.csv
reports/metrics/news_entity_coverage.json
```

**B. Tạo Feature Store có abstract entity để thí nghiệm** (chỉ chạy khi cần dựng lại đặc trưng):

```powershell
python -m src.features.news_content `
  --include-abstract-entity `
  --output-dir data/processed/mind_large/features/b4_abstract_experiment
```

Để tạo bản `.npz` **không nén** ở thư mục riêng, dùng:

```powershell
python -m src.features.feature_store_tools convert `
  --source data/processed/mind_large/features/b4_abstract_experiment `
  --output data/processed/mind_large/features/b4_abstract_uncompressed
```

Đo thời gian tải feature bằng checkpoint tương ứng (cần checkpoint Exp11 đã được huấn luyện trước đó):

```powershell
python -m src.features.feature_store_tools benchmark `
  --checkpoint artifacts/checkpoints/exp11_mind_content_baseline_new_features_content_based.pkl `
  --feature-dir data/processed/mind_large/features/b4_abstract_experiment `
  --feature-dir data/processed/mind_large/features/b4_abstract_uncompressed
```

**C. Đánh giá lại hai cấu hình khi cần tái lập thí nghiệm:**

```powershell
python -m src.evaluate --config configs/experiment/exp11_mind_content_baseline_new_features.yaml
python -m src.evaluate --config configs/experiment/exp12_mind_content_abstract_entity.yaml
```

Hai lệnh `evaluate` **không tự huấn luyện**; cần các checkpoint Exp11/Exp12 tương ứng. Không cần chạy lại nếu chỉ muốn xem kết quả đã lưu trong `reports/metrics/`.

### 3.4. Kết quả thực nghiệm đã lưu

**Coverage của 130.379 news items** trong catalog:

| Chỉ số | Kết quả |
|---|---:|
| Title entity embedding coverage | **72,60%** |
| Abstract entity embedding coverage | **73,84%** |
| Coverage khi có title **hoặc** abstract embedding | **87,21%** |
| Độ phủ tăng so với chỉ title | **+14,61 điểm phần trăm** |
| Có embedding ở cả hai | 77.227 items |
| Chỉ title | 17.429 items |
| Chỉ abstract | 19.042 items |
| Không có cả hai | 16.681 items |

Các nhóm Train/Dev/Test ở bảng coverage là các nhóm **item xuất hiện lần đầu** trong từng split, không phải số impressions. Nguồn: `reports/metrics/news_entity_coverage.json` và `reports/tables/news_entity_complementarity.csv`.

**Thử nghiệm độ chính xác (Full Dev, 376.471 impressions):**

| Cấu hình | NDCG@10 Overall | NDCG@10 Catalog Cold Item |
|---|---:|---:|
| Exp11 — baseline, abstract entity = 0 | **0,39147** | 0,49944 |
| Exp12 — abstract entity = 0,05 | 0,38646 | **0,51254** |

**Kết luận:** Thêm abstract entity cải thiện coverage và nhóm item mới trong thí nghiệm này, nhưng **giảm NDCG@10 tổng thể**. Vì vậy, **giữ trọng số abstract entity = 0 làm mặc định**, không loại bỏ khả năng bật lại khi có cấu hình/Hybrid mới.

**Benchmark Feature Store** (warm-cache, chỉ đo thời gian nạp feature, không tính unpickle checkpoint):

| Kiểu lưu | Median thời gian load |
|---|---:|
| Sparse `.npz` có nén | 51,964 ms |
| Sparse `.npz` không nén | **19,820 ms** |

Hai bản có dự đoán giống nhau trong phép thử đã ghi. Bản không nén tải nhanh hơn khoảng **2,62 lần**, đổi lại dung lượng lớn hơn. Đây là kết quả trên môi trường đã đo, không phải bảo đảm tốc độ cho mọi máy.

---

## 4. Nhiệm vụ 2 — Evaluation & Bootstrap CI

### 4.1. Mục tiêu

- Đánh giá đầy đủ **NDCG@10, Recall@10, MRR@10** trên các lát cắt người dùng/bài viết.
- Tính **Bootstrap confidence interval 95%**, lấy **user làm đơn vị cluster bootstrap** để xét sự phụ thuộc giữa các impressions của cùng người dùng.
- Với P3, tính **paired Bootstrap** cho chênh lệch `after - before` trên cùng tập impressions.
- Tự xuất bảng CSV, báo cáo Markdown, biểu đồ PNG từ các JSON kết quả.
- Cho phép sau này thêm model mới **không phải sửa hardcode tên model trong báo cáo**.

### 4.2. Các lát cắt được hỗ trợ

`overall`, `cold_user`, `warm_user`, `cold_item`, `warm_item`, `catalog_cold_item`, `catalog_warm_item`, `head_item`, `long_tail_item`, `zero_click_item`.

Lưu ý: một số item slices được tính theo clicked/relevant items và **có thể chồng nhau**; không cộng số lượng các slice để suy ra tổng số impressions.

### 4.3. Thành phần và output

| Thành phần | Vai trò |
|---|---|
| `src/evaluate.py` | Chấm điểm Dev và lưu bootstrap CI theo slice vào JSON |
| `src/evaluation/bootstrap.py` | Phương pháp lấy mẫu Bootstrap được evaluator sử dụng |
| `src/evaluation/p3_evaluation.py` | Đánh giá trước/sau P3 và paired CI |
| `src/evaluation/run_all.py` | Tổng hợp JSON, tùy chọn đánh giá lại cấu hình còn thiếu CI |
| `src/evaluation/final_report.py` | Lọc model, xuất bảng/biểu đồ/báo cáo cuối |

**Chạy báo cáo cuối từ JSON đã có** — **không train, không load checkpoint, không suy luận lại**:

```powershell
python -m src.evaluation.final_report
```

Đầu ra:

```text
reports/tables/task2_final/
├── task2_model_comparison.csv
├── task2_slice_ci.csv
├── task2_p3_paired_delta_ci.csv
├── task2_p3_tradeoff.csv
├── task2_report_manifest.json
└── task2_summary.md

reports/figures/task2_final/
├── task2_model_accuracy.png
├── task2_slice_ndcg.png
├── task2_p3_paired_delta.png
└── task2_p3_tradeoff.png
```

`final_report` đọc `reports/metrics/*_dev.json`. Nếu một JSON không có CI, công cụ ghi `not_recorded` thay vì tự sinh CI từ giá trị trung bình. Nếu các mô hình có số impressions khác nhau, công cụ mặc định không trộn vào cùng bảng so sánh chính.

### 4.4. Kết quả Full Dev

**Bộ dữ liệu:** 376.471 impressions; **4 cấu hình** trong phạm vi hiện tại; **1.000 bootstrap resamples** với confidence level 95% và đơn vị `user` trong các kết quả đã lưu.

| Cấu hình | NDCG@10 | Recall@10 | MRR@10 |
|---|---:|---:|---:|
| `exp04_mind_bpr` | 0,35718 | 0,61520 | 0,29466 |
| **`exp05_mind_content`** | **0,39147** | **0,63358** | **0,33733** |
| `exp06_mind_content_p3` | 0,38594 | 0,62491 | 0,33281 |
| `exp07_mind_bpr_p3` | 0,35565 | 0,60814 | 0,29572 |

**Ví dụ CI 95% của NDCG@10 overall:**

| Mô hình | 95% CI |
|---|---|
| BPR | [0,356046; 0,358201] |
| Content-Based | [0,390398; 0,392551] |

**Trạng thái báo cáo:** `task2_report_manifest.json` ghi nhận **120/120 dòng CI** (4 cấu hình × 10 lát cắt × 3 metrics), không có model nào bị loại vì lệch kích thước Full Dev.

### 4.5. Trade-off của P3

**Content-Based + P3**, so với Content-Based:

- NDCG@10: `0,39147 → 0,38594` (paired delta ≈ **−0,005524**, CI 95% `[-0,005865; -0,005181]`).
- Category Diversity@10: `0,65586 → 0,70414`.
- Long-tail Exposure@10: `0,68641 → 0,71767`.
- Zero-click Exposure@10: `0,38162 → 0,41311`.

**BPR + P3**, so với BPR:

- NDCG@10: `0,35718 → 0,35565` (paired delta ≈ **−0,001525**, CI 95% `[-0,001821; -0,001234]`).
- Category Diversity@10: `0,87215 → 0,90902`.
- Long-tail Exposure@10: `0,43790 → 0,50460`.
- Zero-click Exposure@10: `0,21136 → 0,26537`.

**Diễn giải:** P3 mang lại độ đa dạng và tăng tiếp xúc bài long-tail/zero-click, nhưng giảm NDCG@10 tổng thể ở hai cấu hình đã thử. Đây là **đánh đổi giữa relevance và diversity**; không nên kết luận P3 cải thiện tất cả chỉ số. Không coi CI của hai mô hình độc lập là phép thử paired cho sự khác biệt giữa hai mô hình.

### 4.6. Khi muốn bổ sung mô hình hoặc đánh giá lại CI

Chạy Full Dev cho checkpoint đã có:

```powershell
python -m src.evaluate --config configs/experiment/exp04_mind_bpr.yaml
python -m src.evaluate --config configs/experiment/exp05_mind_content.yaml
```

Tổng hợp lại các file JSON:

```powershell
python -m src.evaluation.run_all
python -m src.evaluation.final_report
```

Để tự nhận mọi kết quả `*_dev.json`, kể cả Hybrid thêm sau:

```powershell
python -m src.evaluation.final_report --profile all `
  --output-dir reports/tables/task2_all_models `
  --figures-dir reports/figures/task2_all_models
```

Hoặc chọn riêng một tập model:

```powershell
python -m src.evaluation.final_report `
  --model exp04_mind_bpr `
  --model exp05_mind_content `
  --model exp10_mind_hybrid
```

**Điều kiện:** Model mới phải được evaluator hỗ trợ và có file JSON theo schema hiện tại; cần dùng cùng tập Full Dev để so sánh công bằng. Chỉ thêm tên model trong lệnh **không tự huấn luyện** và **không tự tạo Bootstrap CI nếu JSON chưa chứa**.

---

## 5. Nhiệm vụ 3 — MIND Test Submission Pipeline

### 5.1. Mục tiêu và định dạng bắt buộc

Tập MIND Test **không có nhãn click**. Nhiệm vụ 3 dùng checkpoint đã huấn luyện để dự đoán và tạo file xếp hạng **`prediction.txt` đúng định dạng Microsoft MIND**, sau đó đóng thành **`prediction.zip`**.

Mỗi dòng:

```text
24481 [4,1,3,2]
24482 [2,1]
24483 [3,1,2]
```

Quy định:

1. Không có header; mỗi dòng là `impression_id [rank_1,rank_2,...,rank_N]`.
2. Rank là **hoán vị của `1..N`**, trong đó **1 là ưu tiên cao nhất**.
3. **Rank phải nằm ở vị trí tương ứng candidate ban đầu** trong `behaviors.tsv`, không phải danh sách article IDs đã sort.
4. Giữ **đúng thứ tự toàn bộ impression**; không bỏ candidate, không chỉ xuất Top-K.
5. `prediction.zip` có đúng một file `prediction.txt` ngay ở thư mục gốc của ZIP.
6. Test có candidate **không gắn `-0/-1`**, khác với Train/Dev có nhãn.

**Ví dụ:** Candidates ban đầu `[N1, N2, N3, N4]`, độ ưu tiên mô hình là `N2 > N4 > N3 > N1` → xuất `[4,1,3,2]`.

### 5.2. Các module

```text
src/submission/
├── __init__.py
├── __main__.py    # CLI --config, --split, --limit, --validate-only
├── io.py          # Duyệt Parquet và đối chiếu behaviors.tsv gốc
├── inference.py   # Gọi model.predict(), tùy chọn P3, xử lý item/user lạ
├── ranking.py     # Xếp rank ổn định, serialize chuẩn MIND
├── runner.py      # Điều phối inference, TXT/ZIP, manifest
└── validate.py    # Kiểm tra format, số lượng, ZIP, thứ tự gốc

tests/unit/test_submission.py
```

Pipeline **không huấn luyện lại**. Nó nạp model bằng `BaseRecommender.load()` và sử dụng `predict()` tương thích, với `history_item_idxs` khi cần. Với config `reranking.enabled=true`, pipeline nối thêm P3. `--config` là cách đổi cấu hình/model, miễn checkpoint tồn tại và giao diện đầu vào tương thích.

### 5.3. Chạy thử Dev

```powershell
python -m src.submission `
  --config configs/experiment/exp05_mind_content.yaml `
  --split dev `
  --limit 20 `
  --require-raw
```

Kết quả đã chạy: **20 impressions / 719 candidates**. `submission_ready=false` là đúng vì đây là Dev và có giới hạn mẫu.

### 5.4. Chạy thử 5.000 mẫu Test — chế độ hiện tại

```powershell
python -m src.submission `
  --config configs/experiment/exp05_mind_content.yaml `
  --split test `
  --limit 5000 `
  --require-raw
```

Đầu ra:

```text
reports/submissions/mind_large/test_samples/exp05_mind_content/first_5000/
├── prediction.txt
├── prediction.zip
└── submission_manifest.json
```

Kiểm tra lại **không cần load checkpoint**:

```powershell
python -m src.submission `
  --config configs/experiment/exp05_mind_content.yaml `
  --split test `
  --limit 5000 `
  --require-raw `
  --validate-only
```

Kiểm tra số dòng với PowerShell:

```powershell
(Get-Content reports/submissions/mind_large/test_samples/exp05_mind_content/first_5000/prediction.txt | Measure-Object -Line).Lines
```

**Kết quả thực tế đã xác nhận trên máy triển khai:**

| Kiểm tra | Kết quả |
|---|---:|
| Impressions | **5.000** |
| Tổng candidates đã xếp hạng | **199.107** |
| Số dòng trong `prediction.txt` | **5.000** |
| `validation.zip_valid` | `true` |
| `validation.original_order_verified` | `true` |
| `validation.partial` | `true` |
| `submission_ready` | **`false` (đúng thiết kế)** |

Bản 5.000 mẫu **không được nộp**: nó chỉ dùng smoke test và luôn được lưu trong `test_samples/`, tách khỏi đường dẫn Full Test.

### 5.5. Full Test — **chỉ chạy sau khi chốt mô hình Hybrid**

Ví dụ với Content-Based; khi có Hybrid đã được tích hợp, thay bằng YAML của Hybrid:

```powershell
python -m src.submission `
  --config configs/experiment/exp05_mind_content.yaml `
  --split test `
  --require-raw
```

Đường dẫn tương ứng:

```text
reports/submissions/mind_large/test/exp05_mind_content/
├── prediction.txt
├── prediction.zip
└── submission_manifest.json
```

Validate lại:

```powershell
python -m src.submission `
  --config configs/experiment/exp05_mind_content.yaml `
  --split test `
  --require-raw `
  --validate-only
```

Chỉ xem file **đủ điều kiện kiểm định cục bộ để nộp** khi:

- `submission_ready = true` và `complete_dataset = true`.
- `validation.zip_valid = true`.
- `validation.original_order_verified = true` với `behaviors.tsv` **Test gốc**.
- Số dòng, số candidate và hoán vị rank của từng impression được kiểm tra đầy đủ.
- Không có lỗi load checkpoint, mismatch raw hoặc giá trị điểm không hữu hạn.

File gửi lên hệ thống đánh giá là **`prediction.zip`**, không phải file manifest. Kiểm định cục bộ không đồng nghĩa máy chủ sẽ chấp nhận bài nộp; việc nộp thực tế phụ thuộc hệ thống đánh giá còn hoạt động và quy tắc tại thời điểm nộp.

**Lưu ý:** Nếu raw Test ở nơi khác, dùng `--raw-behaviors "D:\MINDlarge_test\behaviors.tsv" --require-raw`. Nếu chạy lại cùng output, thêm `--overwrite` khi chủ động muốn thay file. Không dùng `--limit` cho bài nộp chính thức.

### 5.6. Trường hợp đặc biệt đã xử lý

- `cold_user` (`user_idx=-1`) hoặc item không được mapping (`item_idx=-1`): vẫn giữ impression/candidate; có fallback theo logic mô hình, không xóa vị trí xếp hạng.
- Điểm bằng nhau: rank ổn định theo thứ tự candidate ban đầu.
- P3 chỉ tái xếp hạng Top-K theo cấu hình nhưng **đầu ra vẫn chứa mọi candidate** của impression.
- Điểm `NaN/Inf`, số điểm không khớp, impression ID trùng, thứ tự raw khác, ZIP hỏng hoặc Test có nhãn: báo lỗi.
- Manifest ghi các trường để truy vết như `unknown_candidates`, `p3_fallback_impressions`, `model_checkpoint`, `validation`.

---

## 6. Quy trình dùng lại và mở rộng cho mô hình mới

### 6.1. Đổi mô hình hiện tại

Thay file YAML ở `--config`:

| Trường hợp | YAML |
|---|---|
| BPR | `configs/experiment/exp04_mind_bpr.yaml` |
| Content-Based | `configs/experiment/exp05_mind_content.yaml` |
| Content-Based + P3 | `configs/experiment/exp06_mind_content_p3.yaml` |
| BPR + P3 | `configs/experiment/exp07_mind_bpr_p3.yaml` |

Ví dụ thử 5.000 impressions bằng BPR + P3:

```powershell
python -m src.submission `
  --config configs/experiment/exp07_mind_bpr_p3.yaml `
  --split test `
  --limit 5000 `
  --require-raw
```

Output của mỗi experiment được tách thư mục theo `experiment_name` trong YAML, nên không ghi đè kết quả của model khác. Muốn đánh giá so sánh Dev, dùng `src.evaluate` trước, rồi chạy lại `src.evaluation.final_report`.

### 6.2. Thêm Hybrid/SASRec/MF trong tương lai

Để **chỉ cần đổi `--config`**, mô hình mới phải đáp ứng đủ:

1. Có checkpoint đã huấn luyện, lưu bằng cách mà `BaseRecommender.load()` đọc được.
2. Đã được đăng ký/khởi tạo trong hệ thống mô hình hiện tại và hỗ trợ giao diện `predict(user_idx, candidate_item_idxs, history_item_idxs=...)`, hoặc có adapter tương ứng nếu cần context khác.
3. YAML trỏ đúng checkpoint; với P3, cấu hình `reranking.enabled` và `reranking.base_checkpoint` phải chính xác.
4. Có khả năng dự đoán **toàn bộ candidate** và xử lý hợp lý các trường hợp cold user/item.
5. Đánh giá trên Full Dev và lưu JSON tương thích nếu muốn xuất báo cáo so sánh bằng `final_report --profile all`.

Khi các điều kiện trên đạt, **không cần viết lại** phần rank/TXT/ZIP/validator. Nếu kiến trúc mô hình mới yêu cầu thêm dữ liệu ngoài history/candidate, cần mở rộng adapter, không thể cam kết chỉ đổi YAML trong mọi trường hợp.

### 6.3. Quy trình khuyến nghị khi Hybrid hoàn thiện

```text
Nhận checkpoint Hybrid + config + xác nhận mapping phù hợp
   ↓
Đánh giá Full Dev (bao gồm các lát cắt / CI nếu cần)
   ↓
Xuất lại báo cáo Task 2 bằng --profile all
   ↓
Chạy Hybrid trên 5.000 Test impressions; validate TXT/ZIP/raw
   ↓
Chạy Hybrid trên toàn bộ MIND Large Test (không --limit)
   ↓
Validate Full Test + kiểm tra manifest
   ↓
Nộp prediction.zip lên hệ thống đánh giá khi cần
```

---

## 7. Kiểm thử, Git và bàn giao

### 7.1. Chạy unit tests theo từng nhiệm vụ

**Task 1:**

```powershell
python -m pytest `
  tests/unit/test_entity_coverage.py `
  tests/unit/test_abstract_entity_feature.py `
  tests/unit/test_abstract_entity_sweep.py `
  tests/unit/test_feature_store_tools.py `
  -v
```

**Task 2:**

```powershell
python -m pytest `
  tests/unit/test_task2_final_report.py `
  tests/unit/test_multi_model_reports.py `
  tests/unit/test_p3_delta_export.py `
  tests/unit/test_bootstrap_item_slices.py `
  tests/unit/test_slice_bootstrap_ci.py `
  -v
```

**Task 3:**

```powershell
python -m pytest `
  tests/unit/test_submission.py `
  tests/unit/test_p3_reranking.py `
  -v --basetemp=.\tmp_pytest_submission_final
```

> Nếu một unit test dùng dữ liệu giả lập mà đọc nhầm `behaviors.tsv` thật, cần dùng bản sửa `test_test_sample_rejects_insufficient_impressions` đã bổ sung đường dẫn TSV giả lập. Sau bản sửa này, **chạy lại và xác nhận test pass trên máy triển khai trước khi commit**; kết quả chạy được 5.000 Test mẫu không thay thế cho unit test.

### 7.2. Những gì nên và không nên commit

**Nên commit:** source dưới `src/`, file `tests/`, YAML cấu hình, tài liệu này và các bảng/biểu đồ báo cáo nhỏ nếu nhóm thống nhất lưu để tái lập kết quả.

**Không commit:** `data/raw/`, `data/processed/`, checkpoint `.pkl`/trọng số, `reports/submissions/`, `tmp_pytest_*/`, file nén dataset.

Lệnh Git sau khi chép file README này vào thư mục gốc:

```powershell
git branch --show-current
git status --short

git add .gitignore src/submission/ tests/unit/test_submission.py README_THREE_TASKS.md
# Chỉ stage các file còn thiếu sau khi kiểm tra git status.
git diff --cached --check
git diff --cached --stat
git status --short

git commit -m "feat: add MIND submission pipeline and document three tasks"
git push -u origin HEAD
```

Nếu có các thay đổi ở nhiệm vụ 1/2 chưa commit, cần kiểm tra và stage riêng các file đó; không dùng `git add .` khi chưa rà soát dataset/checkpoint/log. Nếu tài liệu README cũ đã xóa khỏi repository và Git đang theo dõi, xem `git status` để quyết định đưa thay đổi xóa file vào cùng commit hay không.

### 7.3. Checklist bàn giao

- [x] News Entity Coverage và Feature Store benchmark có báo cáo.
- [x] Kết quả Full Dev Exp11/Exp12 được lưu; không tự bật abstract entity theo mặc định.
- [x] Báo cáo Task 2 có **4 cấu hình / 120 trong 120 dòng CI**.
- [x] Submission Pipeline xuất được TXT/ZIP từ 5.000 MIND Test impressions.
- [x] `prediction.txt` có 5.000 dòng; ZIP và raw order đã kiểm định.
- [ ] Xác nhận toàn bộ unit tests sau bản sửa cuối trên máy triển khai, sau đó commit/push.
- [ ] Đánh giá/huấn luyện mô hình Hybrid cuối cùng (phần việc tiếp theo của nhóm).
- [ ] Xuất **Full Test** và nộp `prediction.zip` khi cần.

## Tài liệu liên quan

- `README.md`: Cài đặt và kiến trúc chung của RecSys HUST.
- `reports/metrics/news_entity_coverage.json`: Kết quả Nhiệm vụ 1.
- `reports/tables/task2_final/task2_summary.md`: Tổng hợp Nhiệm vụ 2.
- `reports/submissions/mind_large/test_samples/.../submission_manifest.json`: Kiểm định Nhiệm vụ 3 trên máy đã chạy (không push Git).
- [MIND dataset](https://msnews.github.io/) và [mô tả dữ liệu MIND](https://github.com/msnews/msnews.github.io/blob/master/assets/doc/introduction.md).
