# Model Card — Hệ gợi ý MIND Large

## 1. Phạm vi

Bộ Model Card này mô tả bốn hệ thống được đánh giá trên cùng **376,471** impression của tập dev:

- BPR.
- Content-based.
- BPR + P3 re-ranker.
- Content-based + P3 re-ranker.

Mục tiêu là đánh giá đồng thời độ liên quan (relevance), cold-start, diversity và exposure của long-tail item.

## 2. Mô hình nền

### BPR

BPR học latent representation từ interaction train. Hyperparameter cuối được lấy từ `configs/experiment/exp04_mind_bpr.yaml`.

### Content-based

Content-based dùng metadata item và user profile từ lịch sử tương tác. Feature weights cuối:

```text
{
  "title": 0.7,
  "abstract": 0.0,
  "category": 0.15,
  "subcategory": 0.15,
  "title_entity": 0.0
}
```

`cold_user_strategy = zero`.

## 3. P3 re-ranking

P3 tái xếp hạng candidate bằng ba thành phần: relevance, category diversity và long-tail exposure. Hai nhánh được tuning độc lập.

BPR + P3 weights:

```text
{
  "relevance": 0.75,
  "diversity": 0.05,
  "long_tail": 0.2
}
```

Content-based + P3 weights:

```text
{
  "relevance": 0.8,
  "diversity": 0.15,
  "long_tail": 0.05
}
```

## 4. Kết quả chính

| Model | NDCG@10 | Recall@10 | MRR@10 | Catalog cold NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| BPR | 0.35718 | 0.61520 | 0.29466 | 0.29342 |
| Content-based | 0.39147 | 0.63358 | 0.33733 | 0.49944 |

Bootstrap CI được tính theo user với cấu hình trong experiment YAML. BPR NDCG@10 CI = [0.356046, 0.358201]; Content-based NDCG@10 CI = [0.390398, 0.392551].

## 5. Trade-off khi dùng P3

- BPR + P3: NDCG@10 delta = -0.001530 (-0.43%); Category Diversity@10 +4.23%; Long-tail Exposure@10 +15.23%.
- Content-based + P3: NDCG@10 delta = -0.005530 (-1.41%); Category Diversity@10 +7.36%; Long-tail Exposure@10 +4.55%.

P3 được thiết kế để chấp nhận một mức giảm relevance có kiểm soát nhằm tăng diversity và long-tail exposure. Chi tiết nằm trong `reports/tables/p3_bpr_comparison.csv` và `reports/tables/p3_content_comparison.csv`.

## 6. Cold-start và slicing

`cold_item` của project là **impression-level slice**: impression được xếp vào cold item nếu có ít nhất một candidate chưa xuất hiện trong train. `catalog_cold_item` là phân tích bổ sung, chỉ xét relevant/clicked item có thực sự chưa thấy trong train hay không.

`cold_user` theo evaluator là user có history length <= 5 hoặc `user_idx == -1`.

Head item được định nghĩa là top 20% các item có click trong train theo click count; long-tail là 80% còn lại trong nhóm đã có click; zero-click được báo cáo riêng.

## 7. Data Minimization / GDPR-motivated experiment

MIND Large train interactions có timestamp thực tế chỉ phủ khoảng sáu ngày, vì vậy experiment dùng 1d/3d/5d/all thay vì 7d/30d/90d/all. Individual click trong `history` không có timestamp riêng nên không được gán timestamp giả.

Riêng Content-based trong experiment này đặt `use_impression_history = false` để user profile chỉ dựa trên retained train history; điều này tránh sử dụng full dev history làm đường tắt.

Experiment này đánh giá trade-off theo nguyên tắc data minimization và **không** được diễn giải như bằng chứng rằng toàn bộ hệ thống tuân thủ GDPR.

## 8. Intended use

Hệ thống phục vụ nghiên cứu/đánh giá recommender trên MIND Large, đặc biệt cho cold-start item, re-ranking responsible exposure và data minimization. Không nên diễn giải metric offline như bằng chứng trực tiếp cho chất lượng sản phẩm online hoặc trải nghiệm người dùng thực tế.

## 9. Hạn chế

- Kết quả là offline evaluation trên MIND Large dev.
- Khoảng thời gian train có timestamp ngắn nên retention window bị giới hạn ở 1/3/5/all ngày.
- P3 tối ưu trade-off theo metric proxy; diversity/exposure cao hơn không tự động đồng nghĩa với mức hài lòng người dùng cao hơn.
- Gini rất gần 1 do catalog lớn và recommendation exposure tập trung; cần diễn giải cùng catalog coverage và các exposure metric khác.
- Kết quả Content-based trong Data Minimization không trực tiếp so sánh tuyệt đối với exp05 thông thường vì experiment data minimization cố ý tắt full dev impression history.
- Content feature weights và P3 re-ranking weights đều được lựa chọn dựa trên dev data; project hiện không dùng validation/holdout riêng cho hyperparameter selection. Vì vậy các metric cuối trên dev có thể có optimistic bias và nên được xem là development estimates, không phải unbiased test estimates.
