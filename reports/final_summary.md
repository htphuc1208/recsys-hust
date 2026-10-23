# Final Experimental Summary

## 1. Experimental setup

Bốn hệ thống chính được đánh giá trên cùng **376,471** MIND Large dev impressions: BPR, Content-based, BPR + P3 và Content-based + P3. Accuracy dùng NDCG/Recall/MRR; beyond-accuracy dùng catalog coverage, Gini, category diversity, long-tail exposure và zero-click exposure. Bootstrap CI dùng resampling theo user theo cấu hình experiment.

## 2. B4 — Content-based Recommender

Content-based đạt NDCG@10 = **0.39147**, so với BPR = **0.35718**, tương ứng chênh lệch tương đối **9.60%** trên full dev.

Khác biệt nổi bật ở `catalog_cold_item`: BPR = **0.29342**, Content-based = **0.49944**. Slice này xét relevant item thực sự chưa thấy trong train và bổ sung cho convention `cold_item` ở mức impression.

Final feature weights:

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

## 3. P3 — Popularity-Aware & Diversity Re-ranking

### BPR + P3

Final P3 weights:

```text
{
  "relevance": 0.75,
  "diversity": 0.05,
  "long_tail": 0.2
}
```

- NDCG@10: 0.35718 → 0.35565 (-0.43%).
- Category Diversity@10: 0.87215 → 0.90902 (4.23%).
- Long-tail Exposure@10: 0.43790 → 0.50460 (15.23%).

### Content-based + P3

Final P3 weights:

```text
{
  "relevance": 0.8,
  "diversity": 0.15,
  "long_tail": 0.05
}
```

- NDCG@10: 0.39147 → 0.38594 (-1.41%).
- Category Diversity@10: 0.65586 → 0.70414 (7.36%).
- Long-tail Exposure@10: 0.68641 → 0.71767 (4.55%).

Kết quả cho thấy P3 tạo trade-off có chủ đích: relevance giảm ở mức nhất định trong khi diversity và exposure của non-head item tăng. Paired bootstrap CI được lưu trong hai JSON exp06/exp07 và bảng CSV tương ứng.

## 4. Evaluation rigor

Evaluator báo cáo:

- project `cold_item` / `warm_item` convention ở mức impression;
- `cold_user` / `warm_user`;
- head / long-tail / zero-click item;
- catalog cold / warm relevant item;
- Bootstrap 95% CI cho model thường;
- paired Bootstrap CI cho P3 delta.

Việc giữ cả `cold_item` và `catalog_cold_item` giúp vừa tương thích convention chung của project vừa phân tích đúng cold-start của relevant item.

## 5. Data Minimization / GDPR-motivated experiment

MIND train interaction có timestamp chỉ phủ khoảng sáu ngày nên experiment dùng **1d / 3d / 5d / all**, thay vì 7/30/90/all. Không gán timestamp giả cho historical clicks.

### BPR

- 1d giữ 20.17% dữ liệu, giảm storage 78.79%, NDCG@10 = 0.35197, utility retained = 98.54%.
- 3d giữ 62.42% dữ liệu, giảm storage 39.83%, NDCG@10 = 0.35370, utility retained = 99.03%.

### Content-based

Lưu ý: Content-based `all` trong Data Minimization không trực tiếp so sánh với exp05 thông thường. Experiment này cố ý đặt `use_impression_history = false` để user profile chỉ sử dụng retained training interactions, tránh full dev history làm che mất tác động của retention window.

- 1d giữ 19.83% dữ liệu, giảm storage 80.17%, NDCG@10 = 0.31769, utility retained = 88.54%.
- 3d giữ 62.02% dữ liệu, giảm storage 39.64%, NDCG@10 = 0.34640, utility retained = 96.54%.

BPR giữ utility tốt hơn khi retention window ngắn. Content-based nhạy hơn vì user content profile phụ thuộc trực tiếp vào lượng retained history. Training time của Content-based chỉ vài giây nên không nên diễn giải quá mạnh các khác biệt nhỏ về thời gian giữa windows.

## 6. Responsible AI discussion

P3 cải thiện category diversity và long-tail exposure, qua đó giảm xu hướng recommendation exposure tập trung vào các item phổ biến. Tuy nhiên, thay đổi của Gini index tương đối nhỏ; vì vậy kết quả không nên được diễn giải rằng concentration bias đã được loại bỏ.

Các beyond-accuracy metrics là offline proxies và không trực tiếp chứng minh user satisfaction, fairness hay chất lượng trải nghiệm trong triển khai thực tế.

Data Minimization experiment cho thấy trong một số setting có thể giảm đáng kể lượng interaction được lưu trữ và sử dụng để train trong khi vẫn giữ phần lớn offline recommendation utility. Đây là một thí nghiệm được thúc đẩy bởi nguyên tắc data minimization; nó không tự thân chứng minh hệ thống tuân thủ đầy đủ GDPR.

Mọi kết luận về giảm dữ liệu cần được trình bày cùng utility cost thay vì chỉ báo cáo lợi ích về privacy/storage.

## 7. Limitations

- Offline MIND Large evaluation không thay thế online A/B testing.
- Train timestamp span ngắn giới hạn retention experiment.
- Individual historical clicks không có timestamp riêng trong representation sử dụng.
- Data Minimization Content baseline không bằng exp05 thông thường vì experiment cố ý không dùng full dev impression history.
- Head/long-tail phụ thuộc train click frequency và cutoff cụ thể.
- Content feature weights và P3 re-ranking weights đều được lựa chọn dựa trên dev data; project hiện không dùng validation/holdout riêng cho hyperparameter selection. Vì vậy các final dev estimates có thể có optimistic bias và nên được xem là offline development results, không phải unbiased test estimates.

## 8. Generated artifacts

- `reports/tables/model_accuracy_comparison.csv`
- `reports/tables/p3_bpr_comparison.csv`
- `reports/tables/p3_content_comparison.csv`
- `reports/tables/item_slice_comparison.csv`
- `reports/tables/data_minimization_summary.csv`
- `reports/figures/`
- `reports/model_card.md`
- `reports/datasheet.md`

Các bảng/biểu đồ được sinh trực tiếp từ JSON/CSV experiment để giảm lỗi copy số liệu thủ công.
