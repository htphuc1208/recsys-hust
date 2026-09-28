# Data Sheet — MIND Large trong project

## 1. Dataset được sử dụng

Project sử dụng MIND Large đã được preprocess vào `data/processed/mind_large/`. Các experiment chính dùng train interactions, dev impressions, item mapping và metadata item.

## 2. Các trường evaluator sử dụng

Dev impression evaluator đọc các trường:

- `impression_id`
- `user_idx`
- `history_item_idxs`
- `candidate_item_idxs`
- `labels`
- `candidate_seen_in_train_mask`

Train interaction cho Data Minimization sử dụng `timestamp`; BPR dùng pointwise interactions còn Content-based dùng positive interactions.

## 3. Phạm vi thời gian có timestamp

Dữ liệu timestamp dùng cho experiment retention bắt đầu từ **2019-11-09 00:00:00** và kết thúc ở **2019-11-14 23:59:59**; cutoff là **2019-11-15 00:00:00**. Do release hiện tại chỉ cho interaction timestamp trong khoảng gần sáu ngày, experiment không thể tái tạo trung thực retention 30 hoặc 90 ngày.

## 4. Retention datasets

| Window | Pointwise rows | Positive rows | Pointwise retained (%) | Positive retained (%) | Pointwise MB | Positive MB |
| --- | --- | --- | --- | --- | --- | --- |
| 1d | 3258011 | 671044 | 20.17 | 19.83 | 41.23 | 16.97 |
| 3d | 10082218 | 2098695 | 62.42 | 62.02 | 116.93 | 51.66 |
| 5d | 14754905 | 3093602 | 91.35 | 91.43 | 177.76 | 77.44 |
| all | 16152221 | 3383656 | 100.00 | 100.00 | 194.35 | 85.60 |

Các variant 1d/3d/5d/all được materialize bằng cùng Parquet writer/compression để phép đo storage có thể so sánh công bằng.

## 5. Slicing convention

- `cold_item`: impression có ít nhất một candidate chưa thấy trong train.
- `warm_item`: tất cả candidate trong impression đã thấy trong train.
- `cold_user`: history length <= 5 hoặc `user_idx == -1`.
- `head_item`: top 20% item đã có click theo train click count.
- `long_tail_item`: 80% item đã có click còn lại.
- `zero_click_item`: item không có click trong train.
- `catalog_cold_item`: relevant item chưa thấy trong train.
- `catalog_warm_item`: relevant item đã thấy trong train.

## 6. Feature dùng cho Content-based

Content-based sử dụng feature đã chuẩn bị trong `data/processed/mind_large/features/b4`, gồm title TF-IDF, abstract TF-IDF, category one-hot, subcategory one-hot và average pretrained title entity embedding. Final feature weights:

```text
{
  "title": 0.7,
  "abstract": 0.0,
  "category": 0.15,
  "subcategory": 0.15,
  "title_entity": 0.0
}
```

Các trọng số này được lưu trong `configs/experiment/exp05_mind_content.yaml`.

## 7. Data minimization và privacy framing

Experiment chỉ giảm lượng interaction được giữ lại để train/profile user. Không tạo timestamp giả cho item trong historical click list. Với Content-based, full dev impression history được tắt riêng trong experiment retention để tránh làm mất ý nghĩa data minimization.

Experiment không thu thập thêm dữ liệu cá nhân và không chứng minh GDPR compliance; nó chỉ đánh giá trade-off của nguyên tắc data minimization trong thiết lập offline hiện tại.

## 8. Hạn chế dữ liệu

- Timestamp riêng của từng historical click không có trong representation đang dùng.
- Offline impression logs không phản ánh đầy đủ preference, satisfaction hoặc causal effect của recommendation.
- Popularity group phụ thuộc vào train click frequency nên có thể thay đổi nếu preprocessing hoặc retention window thay đổi.
