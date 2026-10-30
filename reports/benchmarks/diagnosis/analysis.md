# Chẩn đoán độ trễ serving — 30/10/2026

## Kết luận

Nút thắt chính của bản hiện tại là khả năng xử lý đồng thời trong một Python
process: tạo/duyệt nhiều object Python để hợp nhất ứng viên, nhiều tầng thread
pool và việc xếp lịch giữa Python, native inference và event loop. Không có bằng
chứng cho thấy chỉ thay tham số FAISS sẽ giải quyết được vấn đề.

Tải đơn đã đạt mục tiêu; tải đồng thời chưa đạt. Các timeout hiện có giới hạn
việc chờ từng future, không bảo đảm deadline của toàn request HTTP.

Không thay code production, checkpoint hay `reports/benchmarks/serving.json`.
Các thay đổi đối chứng chỉ tồn tại trong process chẩn đoán.

## Phạm vi và cách đo

- Windows, Python 3.13.8, 16 logical CPU / 12 physical cores được psutil báo cáo.
- Bundle đang có: `exp10-hybrid`, `AdaptiveGatingHybrid`, inference trên CPU.
- Dùng 512 impressions Dev có user được mapping và có history; 511 history khác
  nhau. Độ dài history p50 = 22, p95 = 112,7, tối đa = 458.
- Làm nóng toàn bộ 512 request trước khi đo: đây là trường hợp thuận lợi cho
  cache SASRec 4.096 entries. Không đại diện cho tỷ lệ cache miss ở production.
- Đo trực tiếp có instrumentation: 3 giây/mức concurrency. Phép cô lập fusion:
  4 giây/mức. HTTP: client/server khác process, localhost, một Uvicorn worker,
  tắt access log; lượt đầu 6 giây/mức, lượt có instrumentation 8 giây/mức.
- Đây là phép chẩn đoán ngắn trên máy đang sử dụng, không phải kết quả nghiệm
  thu SLO hay kiểm thử tải bền vững. Các lần chạy có dao động; không cộng hoặc
  trừ p95 của các phân phối khác nhau để suy ra một p95 thành phần.

## 1. Đã tái hiện được bằng code hiện tại

HTTP không instrumentation (`http_baseline.json`):

| Concurrency | HTTP p95 (ms) | Pipeline p95 (ms) | RPS | Ranker timeout |
|---:|---:|---:|---:|---:|
| 1 | 13,328 | 10,331 | 89,46 | 0/537 |
| 5 | 62,305 | 49,937 | 101,47 | 0/613 |
| 10 | 143,984 | 108,031 | 94,50 | 17/572 |
| 20 | 285,687 | 147,681 | 84,97 | 345/517 |

Lượt HTTP có instrumentation (`http_instrumented_baseline.json`) cho p95
20,962 / 85,797 / 245,688 / 349,970 ms. Hình dạng suy giảm giống nhau: tăng
concurrency làm latency tăng mạnh nhưng thông lượng không tăng tương ứng.
Tất cả response trong hai lượt đều HTTP 200; HTTP 200 không đồng nghĩa ranker
đã phục vụ thành công thay vì fallback.

Báo cáo `serving.json` gốc được sửa lần cuối tối 29/10; `pipeline.py` và
`resources.py` hiện tại được sửa sáng 30/10. Kết quả gốc không đủ để đánh giá
những thay đổi mới, dù vấn đề tải đồng thời vẫn được tái hiện bằng lượt đo mới.

## 2. Có nút thắt ngay trong phần Python của retrieval

Code liên quan:

- `src/indexing/retrievers.py:59`: chuyển từng kết quả vector thành `Candidate`
  với set/dict provenance.
- `src/indexing/multi_source.py:55`: tạo thêm object hợp nhất, gọi `add_source`,
  cập nhật dict, tính RRF, đếm nguồn cá nhân hóa và sort trong Python.
- Cấu hình lấy tối đa 400 + 300 + 300 + 500 = 1.500 kết quả nguồn/request,
  hợp nhất rồi giữ 800. Ranker chỉ chấm lại 100.

Phép đo cô lập (`fusion.json`): cache sẵn toàn bộ kết quả của 4 nguồn cho 32
query, bỏ chi phí encoder và FAISS; vẫn dùng dispatcher và fusion thật. Cache
được GC collect/freeze trước khi đo để tránh scan khối object do phép đối chứng
tạo ra.

| Concurrency | p95 (ms) | RPS | CPU process |
|---:|---:|---:|---:|
| 1 | 2,767 | 469,50 | 98,4% |
| 20 | 72,317 | 351,65 | 97,8% |

Ngay cả khi nguồn trả kết quả có sẵn, phần còn lại vẫn vượt ngân sách retrieval
30 ms ở concurrency 20. CPU khoảng một lõi và thông lượng không tăng là bằng
chứng mạnh cho giới hạn thực thi Python/GIL và tranh chấp dispatcher. Không thể
quy toàn bộ 72 ms cho phép cộng RRF: số này gồm dispatch, xếp lịch và fusion.

Thử thực thi các nguồn cache ngay trong luồng gọi thay vì pool không giải quyết
được tail latency: p95 ở concurrency 20 vẫn 120,374 ms. Chỉ bỏ pool cũng chưa
đủ khi nhiều request vẫn tranh thực thi phần Python.

Tài liệu CPython xác nhận một thread thực thi Python bytecode tại một thời điểm
trên build có GIL: https://docs.python.org/3.13/library/threading.html

## 3. FAISS cần ngân sách CPU, nhưng không phải toàn bộ nguyên nhân

Index hiện tại đều là `IndexIDMap2(IndexFlatIP)`:

- collaborative: 9.998 item, dimension 65;
- sequential: 1.000 item, dimension 64;
- content: 130.379 item, dimension 64.

Phép đo content FAISS tách riêng trong process mới, query vector đã tính sẵn,
không tạo Candidate hay chạy RRF được lưu trong `faiss_only_clean.json`:

| Concurrency | Search p95 (ms) | QPS | CPU process |
|---:|---:|---:|---:|
| 1 | 4,257 | 329,19 | 96,5% |
| 20 | 13,159 | 2.018,07 | 1.454,7% |

Đây là chi phí search, không phải toàn bộ retrieval. FAISS dùng được nhiều lõi
và đạt thông lượng cao hơn nhiều mức 80–100 RPS của toàn pipeline. Kết quả
2.018 QPS thuộc lượt process mới, không trộn với lượt 1.188 QPS ở suite overhead
có các phép stress trước đó.

FAISS CPU search hỗ trợ concurrent read và Python binding nhả GIL; phần object
và fusion Python xung quanh không có đặc tính đó:
https://github.com/facebookresearch/faiss/wiki/Threads-and-asynchronous-calls

Không lấy kết quả này làm cam kết khi index CF/SASRec tăng tới toàn catalog.
Các index hiện tại còn nhỏ hơn nhiều so với catalog content.

## 4. Ba tầng luồng tạo thêm chỗ chờ

Luồng thực thi:

```text
HTTP / event loop
  -> AnyIO worker chạy RecommendationPipeline.recommend
      -> retrieval pool 16 workers, submit 4 task/request
      -> ranker pool 8 workers, submit 1 task/request
  -> quay lại event loop, serialize và trả HTTP
```

Ở concurrency 20, lượt HTTP có instrumentation đo được:

- p95 từ lúc offload tới lúc worker bắt đầu chạy: **64,649 ms**;
- p95 thời gian chạy hàm trong worker: **161,207 ms**;
- p95 từ lúc worker hoàn thành tới lúc coroutine tiếp tục: **65,513 ms**;
- p95 phần ngoài pipeline, tính chênh lệch trên từng request: **213,799 ms**.

Phần ngoài pipeline còn gồm HTTP parsing/serialization, networking và client
scheduling; không được gán toàn bộ cho một queue. Tuy vậy, hai độ trễ server
đã được đo riêng đều đủ lớn để phá ngân sách 50 ms.

Ở phép đo trực tiếp ban đầu (`direct.json`), ranker queue p95 khoảng 104 ms
và retrieval queue p95 khoảng 66 ms tại concurrency 20. Các nguồn dùng chung
pool; popularity được submit cuối, timeout chỉ 20 ms nên có thể bị hủy khi còn
đang chờ, dù chính phép lấy popularity khá rẻ.

## 5. Content và gating bị phình thời gian khi cạnh tranh

Trong phép đo trực tiếp đã làm nóng cache:

| Thành phần | p95 ở C=1 (ms) | p95 ở C=20 (ms) |
|---|---:|---:|
| Content predict | 4,721 | 57,926 |
| Gating forward | 1,051 | 124,465 |
| Ranker tổng | 6,533 | 193,751 |
| Retrieval tổng | 10,452 | 222,562 |

Các số là thời gian wall-clock, gồm bị tạm dừng/chờ được chạy, không phải chỉ
thời gian tính toán trên CPU. Không nên diễn giải gating cần 124 ms phép toán.

`ContentBasedRecommender._build_profiles()` tạo lại profile từ lịch sử, sparse
mean, chuẩn hóa, rồi sparse multiplication ở mỗi lần dự đoán. Gating chạy ba
MLP nhỏ và nhiều thao tác tensor, đồng thời `.eval()` được gọi lại mỗi request.
Các chi phí nhỏ này bị khuếch đại khi nhiều thread tranh chạy Python/native.

Các đối chứng trong `ablation.json` chưa giải quyết được toàn pipeline:

- Giảm rank workers từ 8 xuống 1: vẫn timeout nhiều, p95 C=20 khoảng 226 ms.
- Dùng MLP NumPy tương đương trong process thử nghiệm (sai lệch trọng số tối đa
  1,79e-7), hoặc thêm cache profile đã làm nóng: vẫn không đạt SLO.
- Hạ GIL switch interval xuống 0,1 ms làm CPU và latency xấu đi rõ rệt
  (`overhead.json`). Không đề xuất đây là cách sửa.

Kết quả này ủng hộ việc sửa lượng công việc Python và kiến trúc phục vụ trước,
thay vì chỉ đổi một tham số hoặc thay riêng runtime gating.

Đã đo thêm SASRec trực tiếp trên 200 histories trong process mới: p95 encode
không qua cache **2,855 ms**, cache hit **0,005 ms** ở tải đơn. Vì vậy chưa có
căn cứ quy độ trễ hàng trăm ms cho riêng forward SASRec; phép đo này chưa đánh
giá miss đồng thời hoặc tỷ lệ cache hit thực tế.

## 6. Timeout hiện tại không phải deadline HTTP

- Nguồn chính timeout 50 ms; ranker timeout 100 ms. Bản thân ngân sách cấu hình
  đã cho phép vượt HTTP p95 50 ms khi phải chờ gần hết timeout.
- Chờ vào AnyIO worker, submit/lock, fusion, sort, quay lại event loop và HTTP
  response đều nằm ngoài giới hạn của `future.result(timeout=...)`.
- `future.cancel()` không dừng task đã chạy. Đã kiểm tra trực tiếp: cancel trả
  `False` khi task đang chạy; task tiếp tục chiếm tài nguyên sau khi request đã
  chuyển sang fallback. Đây là cơ chế khuếch đại quá tải, không phải bằng chứng
  mọi timeout trong phép đo đều có nguyên nhân giống nhau.
- Bản code cũ dùng `with ThreadPoolExecutor(...)` cho từng request và có thể
  chờ task ở bước shutdown. Bản working tree hiện tại đã bỏ cách đó; không coi
  đây là lỗi còn tồn tại trong code hiện tại.

Tài liệu Future/cancellation:
https://docs.python.org/3.13/library/concurrent.futures.html

## 7. Cách đo hiện có dễ dẫn tới tối ưu nhầm

`source_latencies_ms` trong `multi_source.py:53` được ghi khi vòng lặp đọc tới
future, không phải khi nguồn hoàn thành. Nguồn nhanh nằm sau nguồn chậm có thể
bị báo chậm theo nguồn trước.

Đã tái hiện bằng nguồn giả: collaborative thực thi 25,23 ms, content thực thi
0,0005 ms nhưng báo cáo content là 25,325 ms và popularity 25,033 ms. Không nên
dùng trường này để kết luận nguồn nào tốn CPU; cần tách queue, execution và
completion/consumption time.

Ngoài ra, `benchmark.py` mặc định gửi `user_id=None`, history rỗng nếu không có
payload file. Khi đó pipeline bỏ ranker; kết quả không đại diện cho Hybrid.
Warmup chỉ lặp payload đầu tiên. JSON chính thức chưa lưu payload fingerprint,
cấu hình serving đầy đủ và model version, gây khó đối chiếu giữa các lần tối ưu.

## Thứ tự xử lý đề xuất

1. Sửa phép đo: stage timers thật, thời gian chờ queue, deadline, model/config/
   payload fingerprint; phân biệt cold-user fallback với timeout.
2. Giảm object Python trong hot path: giữ item IDs/scores/ranks trong arrays,
   fusion/top-k theo mảng, chỉ tạo provenance/object tại ranh giới cần sử dụng.
3. Quản lý admission/backpressure và deadline toàn request; giữ fallback rẻ
   ngoài hàng đợi chung đang quá tải. Không xem giảm timeout là tối ưu năng lực.
4. Tách/batch inference và đánh giá nhiều process theo ngân sách RAM. Cần đo
   lại thực tế; không cam kết chỉ tăng Uvicorn workers sẽ đạt 50 ms.
5. Tối ưu profile content/gating và cache có giới hạn, phiên bản model và invalidation
   rõ ràng. Chỉ chọn HNSW/IVF hoặc giảm candidates sau khi đo recall/quality.

Điểm cấu hình phụ cần sửa: `_limit_intra_op_threads()` gọi `threadpool_limits`
trước khi FAISS được import. Trong startup thực tế đã thấy OpenBLAS DLL của
FAISS còn 16 threads dù config intra-op là 1. Tuy nhiên singleton search có thể
không dùng đường BLAS này; chưa chứng minh đây là nút thắt của các lượt đo trên.

## Tái lập

```powershell
.\.venv\Scripts\python.exe scripts/diagnose_serving_latency.py --suite baseline --duration 3
.\.venv\Scripts\python.exe scripts/diagnose_serving_latency.py --suite fusion --duration 4
.\.venv\Scripts\python.exe scripts/diagnose_serving_http.py --duration 8
```

HTTP script mở server tạm tại `127.0.0.1:18081` và dừng process đó khi hoàn tất.
Endpoint timings chỉ thuộc server chẩn đoán, không được thêm vào app production.
Các prototype đối chứng và instrumentation không được xem là bản sửa production.
