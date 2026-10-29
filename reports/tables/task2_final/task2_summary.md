# Báo cáo đánh giá hệ gợi ý – RecSys HUST

**Chế độ:** `core` | **Dev impressions:** 376,471 | **Kết quả mô hình:** 4

Báo cáo tái tạo trực tiếp từ JSON đánh giá có sẵn, không chạy huấn luyện hay suy luận. Các khoảng tin cậy chỉ được hiển thị khi đã được lưu bởi evaluator.

## 1. Bảng tổng hợp độ chính xác

| Mô hình | NDCG@10 | 95% CI | Recall@10 | MRR@10 |
|---|---:|---|---:|---:|
| `exp04_mind_bpr` | 0.35718 | [0.35605, 0.35820] | 0.61520 | 0.29466 |
| `exp05_mind_content` | 0.39147 | [0.39040, 0.39255] | 0.63358 | 0.33733 |
| `exp06_mind_content_p3` | 0.38594 | [0.38490, 0.38703] | 0.62491 | 0.33281 |
| `exp07_mind_bpr_p3` | 0.35565 | [0.35453, 0.35672] | 0.60814 | 0.29572 |

## 2. Bootstrap CI theo lát cắt

Có **120 / 120** dòng metric–lát cắt có CI 95%; **0** dòng chưa ghi CI và **0** dòng không có dữ liệu.

- `available`: khoảng tin cậy lấy từ JSON đã đánh giá.
- `not_recorded`: có metric nhưng chưa lưu Bootstrap CI; **không được nội suy CI từ mean**.
- `no_data`: không có metric/lát cắt hợp lệ trong báo cáo nguồn.

## 3. P3: Paired Bootstrap trên cùng impression

| Mô hình | Lát cắt | ΔNDCG@10 | Paired CI 95% | Nhận xét |
|---|---|---:|---|---|
| `exp06_mind_content_p3` | `overall` | -0.00552 | [-0.00587, -0.00518] | CI < 0 |
| `exp06_mind_content_p3` | `catalog_cold_item` | 0.01180 | [0.01102, 0.01253] | CI > 0 |
| `exp06_mind_content_p3` | `head_item` | -0.02118 | [-0.02171, -0.02068] | CI < 0 |
| `exp06_mind_content_p3` | `long_tail_item` | -0.00692 | [-0.00750, -0.00638] | CI < 0 |
| `exp06_mind_content_p3` | `zero_click_item` | 0.01842 | [0.01794, 0.01890] | CI > 0 |
| `exp07_mind_bpr_p3` | `overall` | -0.00153 | [-0.00182, -0.00123] | CI < 0 |
| `exp07_mind_bpr_p3` | `catalog_cold_item` | -0.00344 | [-0.00422, -0.00265] | CI < 0 |
| `exp07_mind_bpr_p3` | `head_item` | -0.01520 | [-0.01569, -0.01473] | CI < 0 |
| `exp07_mind_bpr_p3` | `long_tail_item` | -0.00115 | [-0.00167, -0.00065] | CI < 0 |
| `exp07_mind_bpr_p3` | `zero_click_item` | 0.01537 | [0.01491, 0.01586] | CI > 0 |

## 4. Beyond-accuracy và các đánh đổi

| Mô hình P3 | Chỉ số | Trước P3 | Sau P3 | Chênh lệch |
|---|---|---:|---:|---:|
| `exp06_mind_content_p3` | `catalog_coverage@10` | 0.03771 | 0.03826 | 0.00055 |
| `exp06_mind_content_p3` | `gini_index@10` | 0.99800 | 0.99795 | -0.00005 |
| `exp06_mind_content_p3` | `category_diversity@10` | 0.65586 | 0.70414 | 0.04828 |
| `exp06_mind_content_p3` | `long_tail_exposure@10` | 0.68641 | 0.71767 | 0.03126 |
| `exp06_mind_content_p3` | `zero_click_exposure@10` | 0.38162 | 0.41311 | 0.03149 |
| `exp07_mind_bpr_p3` | `catalog_coverage@10` | 0.02680 | 0.03005 | 0.00325 |
| `exp07_mind_bpr_p3` | `gini_index@10` | 0.99910 | 0.99895 | -0.00015 |
| `exp07_mind_bpr_p3` | `category_diversity@10` | 0.87215 | 0.90902 | 0.03687 |
| `exp07_mind_bpr_p3` | `long_tail_exposure@10` | 0.43790 | 0.50460 | 0.06670 |
| `exp07_mind_bpr_p3` | `zero_click_exposure@10` | 0.21136 | 0.26537 | 0.05401 |

Giá trị ở bảng này là chênh lệch quan sát được; không suy diễn độ tin cậy thống kê cho các chỉ số chưa có Bootstrap CI.

## 5. Phạm vi và giới hạn

- Chỉ đối chiếu các báo cáo cùng số impression Full Dev; mẫu khác được loại khỏi so sánh mặc định.
- Các lát cắt theo item có thể chồng nhau; không cộng số lượng slice để suy ra tổng số impression.
- Không suy diễn ý nghĩa thống kê cho chênh lệch **hai model khác nhau** từ hai CI riêng. Muốn kết luận cần paired Bootstrap của hiệu số theo cùng impression.
- Kết quả Dev đã được dùng trong quá trình lựa chọn cấu hình; không coi là Test độc lập.

## 6. Khả năng mở rộng

- `--profile core`: chỉ chọn BPR, Content-Based và hai kết quả P3 hiện có.
- `--profile all`: tự đọc tất cả JSON `*_dev.json` hợp lệ được thêm vào trong tương lai.
- `--model <experiment_name>` (lặp lại): chọn một tập mô hình tùy ý, không sửa mã nguồn.
- Khi thêm một mô hình, chỉ cần chạy evaluator để có `reports/metrics/<experiment_name>_dev.json` cùng schema hiện hành rồi chạy lại script; không cần checkpoint khi chỉ xuất báo cáo.
