# Báo cáo khám phá dữ liệu Diginetica

## 1. Thông tin báo cáo

- **Đề tài:** BTL-K03 — Thiên lệch độ phổ biến và hình học không gian embedding trong mô hình SBR
- **Notebook:** [`01_data_exploration.ipynb`](../../notebooks/01_data_exploration.ipynb)
- **Môi trường chạy:** `D:\DL\.venv\Scripts\python.exe`, Python 3.12.6
- **Thời điểm chạy:** 23/10/2026
- **Trạng thái:** 12/12 code cell chạy thành công, không có exception
- **Phạm vi:** khám phá dữ liệu raw, kiểm tra chất lượng và mô phỏng tác động của tiền xử lý

Notebook này chưa chia train/validation/test, chưa tạo cold item, chưa huấn luyện mô hình và chưa tính các chỉ số đầu ra như Coverage@20, recommendation Gini hay ARP.

---

## 2. Dữ liệu đã sử dụng

Notebook sử dụng ba file trong thư mục `data/raw/diginetica/`.

| Vai trò | File | Kích thước | Số dòng | Nội dung được sử dụng |
|---|---|---:|---:|---|
| Tương tác | `train-item-views.csv` | 43.899.457 byte | 1.235.380 | Session, item, ngày và thứ tự/thời điểm click |
| Category | `product-categories.csv` | 2.163.623 byte | 184.047 | Ánh xạ một item sang một category |
| Metadata | `products.csv` | 7.520.465 byte | 184.047 | Kiểm tra catalog và tính toàn vẹn của item |

SHA-256 của ba file:

```text
train-item-views.csv
5a27dbd1a7d47867ef64aff378dbed971131f0f6fcd86fee8bc1c8cbed6f413f

product-categories.csv
e5b7d2cda115f0ec3f5d5d571d247147a23a5ec1bf62492a2feb6a90b68fafa2

products.csv
cfe9d17ebd07dba16879a04fc556f223f758001f99a2e3bccb71d8edfffa6cac
```

Ba hash trên không thay đổi sau khi chạy notebook. Vì vậy EDA không chỉnh sửa các file raw.

### 2.1. Vai trò thực tế của từng file

#### `train-item-views.csv`

Đây là nguồn dữ liệu chính. Các cột được sử dụng:

| Cột | Vai trò |
|---|---|
| `sessionId` | Xác định các interaction thuộc cùng một session |
| `userId` | Chỉ kiểm tra missing; không dùng để xây dựng mô hình SBR |
| `itemId` | Xác định sản phẩm được xem |
| `timeframe` | Tạo thứ tự và thời gian tương đối bên trong session |
| `eventdate` | Xác định ngày xảy ra interaction |

#### `product-categories.csv`

File này được nối với interaction qua `itemId` để:

- Đếm category trong mỗi session.
- Đo tỷ lệ chuyển tiếp cùng category.
- Kiểm tra category có đủ ý nghĩa cho phân tích cosine embedding sau này không.

#### `products.csv`

Trong EDA hiện tại, file này chỉ được dùng để:

- Kiểm tra `itemId` có tồn tại trong catalog metadata không.
- Kiểm tra duplicate và missing.
- Phân biệt catalog metadata với catalog đã có interaction.

`pricelog2` và `product.name.tokens` không được phân tích sâu vì GRU4Rec, NARM và SASRec trong phạm vi đề tài là các baseline dựa trên ID, không sử dụng hai trường này làm đầu vào.

### 2.2. Dữ liệu không sử dụng

Notebook không sử dụng các file phụ như `train-clicks.csv`, `train-purchases.csv`, `train-queries.pbz2` hay `valid.solution.csv`. Chúng không cần thiết cho pipeline đã chốt và không thuộc ba file cốt lõi mà RecBole dùng để chuyển đổi Diginetica.

---

## 3. Kết quả theo từng cell thực thi

Notebook có 21 cell, gồm 11 markdown cell giải thích và 10 code cell thực thi. Phần dưới đây trình bày lần lượt 10 code cell.

## 3.1. Code cell 1 — Thiết lập môi trường và đường dẫn

### Dữ liệu/đầu vào

- Thư mục làm việc hiện tại.
- File đánh dấu gốc dự án `BTL-K03-tong-quan-de-tai.md`.

### Xử lý

- Tự động tìm `PROJECT_ROOT` thay vì phụ thuộc notebook được mở từ thư mục nào.
- Khai báo đường dẫn raw, bảng kết quả và hình kết quả.
- Thiết lập pandas, seaborn và matplotlib.

### Kết quả

```text
Python      : D:\DL\.venv\Scripts\python.exe
Project root: D:\DA-3
Raw data    : D:\DA-3\data\raw\diginetica
```

### Kết luận

Notebook đã chạy đúng virtual environment `D:\DL\.venv`. Đường dẫn không bị phụ thuộc vào việc mở notebook từ thư mục gốc hay thư mục `notebooks/`.

---

## 3.2. Code cell 2 — Đọc dữ liệu và lập manifest

### Dữ liệu/đầu vào

Ba file raw đã nêu trong Mục 2.

### Xử lý

- Kiểm tra file có tồn tại.
- Đọc CSV với dấu phân cách `;`.
- Xem `NA` là missing value.
- Tính số dòng, dung lượng và SHA-256.
- Hiển thị năm dòng đầu mỗi bảng.

### Kết quả

| Bảng | Số dòng | Nhận xét |
|---|---:|---|
| Views | 1.235.380 | Nguồn interaction chính |
| Categories | 184.047 | Một dòng cho mỗi item metadata |
| Products | 184.047 | Một dòng cho mỗi item metadata |

### Kết luận

Ba file cần thiết đều tồn tại, đọc được và có thể định danh chính xác bằng hash. Manifest được lưu tại [`eda_raw_file_manifest.csv`](../../results/tables/eda_raw_file_manifest.csv).

---

## 3.3. Code cell 3 — Schema, missing, duplicate và liên kết

### Dữ liệu/đầu vào

Ba DataFrame vừa đọc.

### Xử lý

- So sánh tên và thứ tự cột với schema mong đợi.
- Đếm missing và số giá trị khác nhau.
- Tìm dòng trùng hoàn toàn.
- Tìm khóa event trùng `(sessionId, itemId, timeframe, eventdate)`.
- Kiểm tra `eventdate` có parse được không.
- Kiểm tra `timeframe` âm.
- Kiểm tra item interaction có category và product metadata không.

### Kết quả schema

| Bảng/cột | Số giá trị khác nhau | Missing |
|---|---:|---:|
| `views.sessionId` | 310.324 | 0 |
| `views.userId` | 87.934 | 862.389, tương đương 69,81% |
| `views.itemId` | 122.993 | 0 |
| `views.eventdate` | 152 ngày | 0 |
| `categories.itemId` | 184.047 | 0 |
| `categories.categoryId` | 1.217 | 0 |
| `products.itemId` | 184.047 | 0 |

### Kết quả chất lượng

- Cả ba schema đều đúng.
- Không có dòng trùng hoàn toàn.
- Không có khóa event trùng.
- Không có `itemId` trùng trong bảng category hoặc product.
- Không có lỗi parse ngày.
- Không có `timeframe` âm.
- Không có item interaction thiếu category.
- Không có item interaction thiếu product metadata.
- Có 61.054 item metadata chưa từng xuất hiện trong interaction.

### Kết luận

Dữ liệu có chất lượng liên kết tốt và không cần xử lý missing cho trường chính. `userId` thiếu gần 70%, nhưng đây không phải lỗi cản trở vì đề tài dùng session-based recommendation. Không nên mở rộng sang phân tích theo user từ cột này.

Bảng chi tiết:

- [`eda_schema_summary.csv`](../../results/tables/eda_schema_summary.csv)
- [`eda_data_quality_summary.csv`](../../results/tables/eda_data_quality_summary.csv)

---

## 3.4. Code cell 4 — Chuẩn hóa thứ tự sự kiện và trục thời gian

### Dữ liệu/đầu vào

Các cột `sessionId`, `eventdate`, `timeframe` và thứ tự dòng gốc.

### Xử lý

`timeframe` là số mili giây tương đối kể từ đầu session, nên được dùng làm khóa chính để sắp thứ tự event bên trong session. `eventdate` được giữ làm ngày lịch để thống kê và chia tập theo thời gian.

Dữ liệu được sắp xếp ổn định theo:

```text
sessionId → timeframe → thứ tự dòng gốc
```

Không dùng `eventdate + timeframe` để đo duration. Nếu cộng trực tiếp như vậy, session ngắn đi qua nửa đêm sẽ bị diễn giải sai thành dài hơn 24 giờ.

### Kết quả

| Chỉ tiêu | Giá trị |
|---|---:|
| Ngày đầu | 01/01/2016 |
| Ngày cuối | 01/06/2016 |
| Số ngày | 152 |
| `timeframe` nhỏ nhất | 12 ms |
| `timeframe` lớn nhất | 1.199.992 ms |
| Session không tăng thời gian trong thứ tự raw | 180.643 |
| Tỷ lệ session sai thứ tự raw | 58,21% |
| Session đi qua nhiều `eventdate` | 260 |
| Nhóm trùng `(sessionId, timeframe)` | 4 |

### Kết luận

Không được dùng thứ tự dòng trong CSV làm thứ tự chuỗi. Có tới 58,21% session không tăng theo `timeframe` trong thứ tự raw. Mọi bước tiền xử lý và sinh prefix-target phải sắp theo `timeframe`. Với bốn nhóm thời gian bị trùng, stable sort giữ nguyên `_row_order` làm quy tắc phá hòa xác định.

Bảng kết quả: [`eda_time_quality_summary.csv`](../../results/tables/eda_time_quality_summary.csv).

---

## 3.5. Code cell 5 — Cấu trúc session và click lặp

### Dữ liệu/đầu vào

Interaction đã sắp thời gian và đã nối category.

### Xử lý

Với mỗi session, notebook tính:

- Số interaction.
- Số item khác nhau.
- Số category khác nhau.
- Thời lượng session.
- Số event lặp item.

### Phân vị session

| Phân vị | Session length | Unique item | Unique category | Duration (giây) |
|---:|---:|---:|---:|---:|
| Min | 1 | 1 | 1 | 0,00 |
| P25 | 1 | 1 | 1 | 0,00 |
| Median | 3 | 2 | 1 | 171,79 |
| P75 | 5 | 4 | 1 | 555,87 |
| P90 | 9 | 7 | 2 | 947,95 |
| P95 | 12 | 10 | 3 | 1.065,91 |
| P99 | 19 | 16 | 5 | 1.152,72 |
| Max | 87 | 77 | 18 | 1.196,75 |

### Nhóm độ dài

| Nhóm | Session | Tỷ lệ session | Event | Tỷ lệ event |
|---|---:|---:|---:|---:|
| 1 | 90.694 | 29,23% | 90.694 | 7,34% |
| 2–5 | 148.480 | 47,85% | 460.235 | 37,25% |
| 6–10 | 50.708 | 16,34% | 379.061 | 30,68% |
| >10 | 20.442 | 6,59% | 305.390 | 24,72% |

### Click lặp

- 86.439 session có item được click lặp, tương đương 27,85%.
- 168.313 event là lượt lặp trong session, tương đương 13,62%.
- 72.265 lượt là lặp liên tiếp.
- 96.048 lượt là lặp không liên tiếp.

### Kết luận

Phần lớn session ngắn: median bằng 3 và 77,07% session có độ dài không quá 5. Tuy nhiên session dài hơn 10 chỉ chiếm 6,59% số session nhưng tạo 24,72% số event. Điều này cần được chú ý vì sequence splitting làm session dài sinh nhiều mẫu hơn.

Session length 1 chiếm 29,23% và không thể tạo cặp input-target, vì vậy cần loại ở preprocessing. Click lặp là hiện tượng đáng kể và không nên tự động xoá vì có thể phản ánh hành vi quay lại sản phẩm.

Hình: [`eda_session_length_histogram_ecdf.png`](../../results/figures/eda_session_length_histogram_ecdf.png).

Bảng:

- [`eda_session_quantiles.csv`](../../results/tables/eda_session_quantiles.csv)
- [`eda_session_length_groups.csv`](../../results/tables/eda_session_length_groups.csv)
- [`eda_repeat_click_summary.csv`](../../results/tables/eda_repeat_click_summary.csv)

---

## 3.6. Code cell 6 — Audit session qua ngày và thời gian bất thường

### Mục tiêu

Phân biệt session thực sự kéo dài bất thường với session ngắn đi qua nửa đêm; đồng thời kiểm tra các trường hợp trùng `timeframe` để chốt quy tắc sắp thứ tự.

### Kết quả

| Chỉ tiêu | Giá trị |
|---|---:|
| Tổng session | 310.324 |
| Session xuất hiện trên nhiều ngày | 260 |
| Tất cả session nhiều ngày đều nằm trên đúng hai ngày liên tiếp | Có |
| Session nhiều ngày bị đảo `eventdate` sau khi sắp theo `timeframe` | 0 |
| Nhóm trùng `(sessionId, timeframe)` | 4 |
| Số dòng thuộc bốn nhóm trùng | 8 |
| Duration P99 | 1.152,72 giây |
| Duration P99,9 | 1.182,02 giây |
| Duration lớn nhất | 1.196,75 giây |
| Session dài ít nhất 30 phút | 0 |

Bốn nhóm trùng thời gian đều gồm hai item khác nhau tại cùng một `timeframe`. Không có bằng chứng để tự ý xóa một trong hai event; `_row_order` được dùng để phá hòa ổn định.

### Kết luận

260 session nhiều ngày không phải các session kéo dài hơn 24 giờ. Chúng đều đi qua hai ngày liên tiếp, `timeframe` vẫn tăng liên tục và duration tối đa chỉ khoảng 19 phút 57 giây. Đây phù hợp với hiện tượng session đi qua nửa đêm, vì vậy giữ nguyên `sessionId`, không tách theo `eventdate`. Khi chia tập theo thời gian, toàn bộ session phải nằm trong cùng một split và có thể gán theo ngày của event cuối.

Hình: [`eda_session_time_anomalies.png`](../../results/figures/eda_session_time_anomalies.png).

Bảng:

- [`eda_session_time_audit_summary.csv`](../../results/tables/eda_session_time_audit_summary.csv)
- [`eda_session_time_anomalies.csv`](../../results/tables/eda_session_time_anomalies.csv)
- [`eda_duplicate_timeframes.csv`](../../results/tables/eda_duplicate_timeframes.csv)

---

## 3.7. Code cell 7 — Biến động thời gian và vòng đời item

### Dữ liệu/đầu vào

Toàn bộ interaction đã có thứ tự theo `timeframe` và ngày lịch.

### Xử lý

Theo từng ngày, notebook tính:

- Số interaction.
- Số session.
- Số item hoạt động.
- Số item xuất hiện lần đầu.

Theo từng item, notebook tính:

- Ngày xuất hiện đầu tiên.
- Ngày xuất hiện cuối.
- Số ngày hoạt động.
- Tuổi item tính tới cuối dữ liệu.

### Kết quả theo ngày

| Chỉ tiêu | Trung bình | Median | Lớn nhất |
|---|---:|---:|---:|
| Interaction/ngày | 8.127,50 | 9.591 | 17.320 |
| Session/ngày | 2.043,32 | 2.394 | 4.338 |
| Active item/ngày | 5.217,80 | 6.287 | 10.143 |
| New item/ngày | 809,16 | 747 | 3.587 |

Tương quan Spearman giữa tần suất raw và tuổi item:

```text
rho = 0,504
```

### Kết luận

Item tồn tại lâu hơn có xu hướng tích lũy nhiều interaction hơn. Vì vậy, quan hệ `train_frequency ↔ embedding_norm` sau này có thể chịu ảnh hưởng của tuổi item. Kết quả embedding chỉ nên được diễn giải là quan hệ liên kết, không phải bằng chứng frequency trực tiếp gây ra norm lớn.

Hình: [`eda_daily_activity_and_new_items.png`](../../results/figures/eda_daily_activity_and_new_items.png).

Bảng:

- [`eda_daily_activity.csv`](../../results/tables/eda_daily_activity.csv)
- [`eda_item_lifecycle_raw.csv`](../../results/tables/eda_item_lifecycle_raw.csv)

---

## 3.8. Code cell 8 — Popularity và long tail

### Dữ liệu/đầu vào

Hai định nghĩa popularity mô tả trên raw:

```text
interaction_frequency = số interaction của item
session_frequency     = số session khác nhau chứa item
```

Các đại lượng này chưa phải popularity chính thức của thí nghiệm. Sau khi chia tập, chúng phải được tính lại chỉ từ train.

### Phân vị popularity

| Phân vị | Interaction frequency | Session frequency |
|---:|---:|---:|
| Min | 1 | 1 |
| P25 | 1 | 1 |
| Median | 3 | 2 |
| P75 | 8 | 7 |
| P90 | 22 | 19 |
| P95 | 41 | 35 |
| P99 | 121 | 102 |
| Max | 1.270 | 1.030 |

Spearman giữa interaction frequency và session frequency:

```text
rho = 0,9812
```

Hai cách đếm popularity gần như đồng biến hoàn toàn, nhưng interaction frequency vẫn được giữ làm định nghĩa chính theo đề bài.

### Mức tập trung interaction

| Nhóm item phổ biến | Số item | Tỷ lệ interaction nắm giữ |
|---|---:|---:|
| Top 1% | 1.230 | 21,65% |
| Top 5% | 6.150 | 47,77% |
| Top 10% | 12.300 | 62,53% |
| Top 20% | 24.599 | 77,54% |

Input-data Gini:

```text
Gini = 0,726
```

### Các dải tần suất

| Tần suất | Số item | Tỷ lệ item | Tỷ lệ interaction |
|---|---:|---:|---:|
| 1 | 42.060 | 34,20% | 3,40% |
| 2–4 | 36.404 | 29,60% | 7,95% |
| 5 | 5.335 | 4,34% | 2,16% |
| 6–10 | 14.906 | 12,12% | 9,19% |
| 11–20 | 11.067 | 9,00% | 13,17% |
| 21–50 | 8.551 | 6,95% | 21,78% |
| 51–100 | 2.997 | 2,44% | 16,77% |
| >100 | 1.673 | 1,36% | 25,57% |

### Kết luận

Diginetica có long tail rất rõ:

- 63,80% item xuất hiện dưới 5 lần.
- 1,36% item có trên 100 interaction nhưng chiếm 25,57% toàn bộ interaction.
- Top 20% item chiếm 77,54% interaction.
- Gini đầu vào đạt 0,726.

Do đó dataset phù hợp trực tiếp để nghiên cứu popularity bias. Tuy nhiên, Gini này là Gini của dữ liệu đầu vào, không phải recommendation Gini của đầu ra mô hình.

Hình:

- [`eda_item_rank_frequency_loglog.png`](../../results/figures/eda_item_rank_frequency_loglog.png)
- [`eda_interaction_lorenz_curve.png`](../../results/figures/eda_interaction_lorenz_curve.png)
- [`eda_items_and_interactions_by_frequency_band.png`](../../results/figures/eda_items_and_interactions_by_frequency_band.png)

---

## 3.9. Code cell 9 — Cấu trúc category

### Dữ liệu/đầu vào

Interaction đã sắp thời gian và mapping `itemId → categoryId`.

### Xử lý

- Đếm item, interaction và session theo category.
- Đếm số category trong mỗi session.
- Đo tỷ lệ hai event liên tiếp cùng category.
- Đo lại sau khi loại chuyển tiếp self-repeat như `A → A`.
- So sánh với baseline lấy ngẫu nhiên độc lập theo phân bố category.

### Kết quả

| Chỉ tiêu | Giá trị |
|---|---:|
| Tổng category metadata | 1.217 |
| Category chỉ có một item | 95 |
| Session chỉ có một category | 251.222 |
| Tỷ lệ session chỉ có một category | 80,95% |
| Chuyển tiếp cùng category | 88,43% |
| Chuyển tiếp cùng category, bỏ self-repeat | 87,45% |
| Baseline ngẫu nhiên | 1,08% |

Quy mô category không đồng đều:

- Median là 48 item metadata/category.
- P75 là 187 item/category.
- Category lớn nhất có 3.851 item metadata.
- Trong dữ liệu interaction, category hoạt động lớn nhất có 2.771 item và 57.362 interaction.

### Kết luận

Category phản ánh cấu trúc hành vi rất mạnh. Tỷ lệ chuyển tiếp cùng category vẫn đạt 87,45% sau khi loại self-repeat, cao hơn rất nhiều baseline 1,08%. Vì vậy phân tích cosine embedding giữa item cùng category và khác category là có cơ sở dữ liệu.

Tuy nhiên, category có kích thước rất không đồng đều. Khi lấy cặp cosine sau này phải kiểm soát sampling, tránh để category lớn sinh quá nhiều cặp và thống trị trung bình.

Hình: [`eda_category_structure_and_transition.png`](../../results/figures/eda_category_structure_and_transition.png).

Bảng:

- [`eda_category_statistics.csv`](../../results/tables/eda_category_statistics.csv)
- [`eda_category_summary.csv`](../../results/tables/eda_category_summary.csv)

---

## 3.10. Code cell 10 — Mô phỏng pipeline lọc và target frequency

### Dữ liệu/đầu vào

Interaction đã sắp xếp và có category.

### Pipeline tham chiếu

```text
Raw
→ bỏ session length 1
→ bỏ item có support dưới 5
→ bỏ lại session còn dưới 2 event
```

Notebook chỉ mô phỏng trong bộ nhớ, không ghi dữ liệu processed.

### Attrition table

| Giai đoạn | Event | Session | Item | Category | Prefix-target example |
|---|---:|---:|---:|---:|---:|
| Raw | 1.235.380 | 310.324 | 122.993 | 1.208 hoạt động | 925.056 |
| Session length ≥ 2 | 1.144.686 | 219.630 | 121.468 | 1.201 | 925.056 |
| Sau item support ≥ 5 | 1.004.834 | 216.140 | 43.136 | 995 | 788.694 |
| Kiểm tra session length ≥ 2 lần nữa | 993.483 | 204.789 | 43.136 | 995 | 788.694 |
| Fixed-point, chỉ để đo độ nhạy | 989.204 | 204.061 | 42.171 | 974 | 785.143 |

Pipeline một lượt cuối cùng giữ lại:

- 80,42% event raw.
- 65,99% session raw.
- 35,07% item raw.
- 82,37% category đang hoạt động.
- 85,26% số prefix-target example có thể tạo từ raw.

### Target frequency

Sau pipeline tham chiếu, notebook xem mọi vị trí sau item đầu tiên trong session là một target.

```text
Spearman(interaction frequency, target frequency) = 0,9709
Số item chưa từng làm target = 8
```

Interaction frequency và target frequency rất tương đồng nhưng không giống hệt nhau. Khi phân tích cơ chế embedding, target frequency là một biến chẩn đoán hữu ích vì target quyết định số lần item xuất hiện như lớp đúng trong loss.

### Mức đóng góp mẫu theo session length

| Nhóm session | Số session | Prefix-target example | Tỷ lệ example |
|---|---:|---:|---:|
| 2–5 | 145.365 | 300.702 | 38,13% |
| 6–10 | 44.517 | 286.307 | 36,30% |
| >10 | 14.907 | 201.685 | 25,57% |

Session dài hơn 10 chỉ chiếm khoảng 7,28% session sau lọc nhưng tạo 25,57% mẫu huấn luyện. Vì vậy sequence splitting làm session dài có trọng số lớn hơn số lượng session ban đầu.

### Kết luận

Quy tắc lọc ảnh hưởng mạnh đến catalog: từ 122.993 item raw xuống 43.136 item trong pipeline một lượt. Không được báo cáo kết quả mà thiếu attrition table.

Không nên mặc định lọc đến fixed point. Fixed-point làm giảm thêm 965 item so với pipeline tham chiếu. Lựa chọn cuối phải được cố định trong cấu hình preprocessing và áp dụng giống nhau cho cả ba mô hình.

Bảng:

- [`eda_preprocessing_attrition.csv`](../../results/tables/eda_preprocessing_attrition.csv)
- [`eda_target_frequency_diagnostic.csv`](../../results/tables/eda_target_frequency_diagnostic.csv)
- [`eda_prefix_target_contribution.csv`](../../results/tables/eda_prefix_target_contribution.csv)

---

## 3.11. Code cell 11 — Độ nhạy của ngưỡng hỗ trợ item

### Mục tiêu và quy tắc

Cùng một fixed-point pipeline được chạy với `minimum item frequency ∈ {1, 2, 3, 5, 10}`. Ngưỡng 1 là mốc không lọc item theo support nhưng vẫn loại session không thể tạo target. `Raw-tail` được định nghĩa cố định là item có frequency dưới 5 trong raw để so sánh mức bảo toàn cùng một phần đuôi.

### Kết quả trade-off

| Ngưỡng | Event | Session | Item | Mẫu next-item | Event giữ lại | Item giữ lại | Raw-tail giữ lại | Gini sau lọc |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.144.686 | 219.630 | 121.468 | 925.056 | 92,66% | 98,76% | 98,06% | 0,7170 |
| 2 | 1.098.507 | 215.544 | 78.989 | 882.963 | 88,92% | 64,22% | 43,93% | 0,6471 |
| 3 | 1.057.976 | 211.595 | 60.454 | 846.381 | 85,64% | 49,15% | 20,41% | 0,6048 |
| 5 | 989.204 | 204.061 | 42.171 | 785.143 | 80,07% | 34,29% | 0,00% | 0,5515 |
| 10 | 853.942 | 186.341 | 23.920 | 667.601 | 69,12% | 19,45% | 0,00% | 0,4801 |

Ngưỡng 5 giữ 80,07% interaction và 84,88% mẫu next-item, nhưng chỉ giữ 34,29% item. Theo định nghĩa raw-tail cố định (`raw frequency < 5`), ngưỡng này loại toàn bộ nhóm đó. Ngưỡng 2 giữ nhiều tail hơn đáng kể nhưng vocabulary tăng từ 42.171 lên 78.989 item.

Gini giảm khi ngưỡng tăng không có nghĩa mô hình đã giảm thiên lệch. Đây là hệ quả cơ học của việc cắt bỏ phần lớn item tần suất thấp khỏi catalog, làm phân phối trên các item còn lại đồng đều hơn.

### Kết luận

Không có một ngưỡng “tốt nhất” chỉ dựa trên EDA raw. Ngưỡng 5 được giữ làm cấu hình tham chiếu; ngưỡng 2 hoặc 3 là ứng viên bảo toàn long-tail để kiểm tra bằng pilot training. Ngưỡng chính thức phải dùng cùng một cách cho cả ba mô hình và frequency phải được tính lại từ train để tránh leakage.

Hình: [`eda_frequency_threshold_tradeoff.png`](../../results/figures/eda_frequency_threshold_tradeoff.png).

Bảng:

- [`eda_frequency_threshold_sensitivity.csv`](../../results/tables/eda_frequency_threshold_sensitivity.csv)
- [`eda_threshold_frequency_bands.csv`](../../results/tables/eda_threshold_frequency_bands.csv)

---

## 3.12. Code cell 12 — Kết luận tự động và bàn giao

### Xử lý

Cell cuối biến các phát hiện quan trọng thành mười câu hỏi kiểm tra có bằng chứng đi kèm.

### Kết quả

| Câu hỏi | Kết quả | Bằng chứng |
|---|---|---|
| Ba file có liên kết đầy đủ? | Có | Không thiếu category hoặc product cho item đã tương tác |
| Có cần sắp xếp lại chuỗi? | Có | 180.643 session sai thứ tự raw |
| Session length 1 có đáng kể? | Có | 29,23% session |
| Click lặp có đáng kể? | Có | 13,62% event |
| Dữ liệu có long tail? | Có | 63,80% item dưới 5 lần; Gini 0,726 |
| Category có phản ánh session? | Có | 87,45% so với baseline 1,08% |
| Item age có thể gây nhiễu? | Có | Spearman frequency-age 0,504 |
| Session nhiều ngày đều dưới 30 phút? | Có | 260 session qua hai ngày liên tiếp; duration tối đa 1.196,75 giây |
| Ngưỡng support có trade-off lớn? | Có | Item giữ lại giảm từ 98,76% ở ngưỡng 1 xuống 19,45% ở ngưỡng 10 |
| Fixed-point có làm đổi catalog? | Có | 43.136 so với 42.171 item |

### Kết luận

Dữ liệu đủ chất lượng và phù hợp để tiếp tục xây dựng thí nghiệm. Các điều kiện bắt buộc chuyển sang preprocessing là:

1. Sắp đúng thứ tự theo `timeframe`; dùng `_row_order` để phá hòa.
2. Loại session length 1 theo pipeline đã chốt.
3. Ghi attrition sau mọi bước lọc.
4. Giữ nguyên session qua nửa đêm và chia theo ngày event cuối, không tách một session sang nhiều split.
5. Tính popularity chỉ từ train.
6. Kiểm tra ties trước khi chia popular/medium/rare.
7. Audit cold/warm item và candidate catalog.
8. Dùng ngưỡng 5 làm tham chiếu; cân nhắc ngưỡng 2/3 qua pilot training.

Bảng kết luận: [`eda_conclusions.csv`](../../results/tables/eda_conclusions.csv).

---

## 4. Tổng hợp các kết luận EDA

### 4.1. Về chất lượng dữ liệu

- Không có duplicate chính xác hoặc lỗi mapping item.
- Ngày và timeframe hợp lệ.
- `userId` thiếu nhiều nhưng không ảnh hưởng bài toán SBR.
- Dữ liệu raw không được sắp theo `timeframe`, nên sorting là bắt buộc.
- Bốn nhóm trùng `timeframe` được phá hòa ổn định bằng thứ tự dòng gốc, không tự ý xóa event.

### 4.2. Về cấu trúc session

- Session nhìn chung ngắn, median bằng 3.
- Gần một phần ba session không tạo được mẫu next-item.
- Click lặp xuất hiện đáng kể và cần được giữ trừ khi giao thức yêu cầu khác.
- Session dài ít về số lượng nhưng tạo nhiều training example.
- 260 session qua nửa đêm được giữ nguyên; không có session nào dài từ 30 phút trở lên khi duration được tính đúng theo `timeframe`.

### 4.3. Về popularity bias

- Dữ liệu có phân bố long-tail rất mạnh.
- Tương tác tập trung vào một nhóm item nhỏ.
- Dataset phù hợp để đo popularity bias và sự khác biệt popular/rare.
- Tuổi item có quan hệ với frequency, nên không được diễn giải tương quan embedding theo hướng nhân quả.

### 4.4. Về category

- Category mapping đầy đủ cho item đã tương tác.
- Session có tính đồng nhất category rất cao.
- Category đủ ý nghĩa để so sánh cosine cùng/khác category.
- Sampling cặp cosine phải cân bằng để category lớn không chi phối kết quả.

### 4.5. Về preprocessing

- Lọc item support ảnh hưởng rất mạnh tới kích thước catalog.
- Pipeline một lượt và fixed-point không tương đương.
- Cần dùng một pipeline duy nhất cho GRU4Rec, NARM và SASRec.
- Các con số sau lọc phải được báo cáo thay vì chỉ nêu thống kê raw.
- Ngưỡng 5 giữ 80,07% event nhưng chỉ giữ 34,29% item; đây là trade-off đáng kể với long-tail.
- Gini sau lọc giảm khi ngưỡng tăng chủ yếu do catalog tail bị cắt, không phải bằng chứng mô hình đã giảm popularity bias.
- EDA giữ ngưỡng 5 làm tham chiếu và đề xuất 2/3 làm ứng viên pilot; chưa tuyên bố ngưỡng tối ưu chỉ từ raw.

---

## 5. Những điều chưa thể kết luận từ EDA

EDA hiện tại chưa thể trả lời:

- Mô hình nào có popularity bias lớn nhất.
- Coverage@20, recommendation Gini và ARP của từng mô hình.
- Popular item có embedding norm lớn hơn rare item không.
- Norm embedding có liên quan tới số lần item được gợi ý không.
- Item cold nằm ở đâu trong không gian embedding.
- GRU4Rec, NARM hay SASRec dự đoán tốt hơn.

Các câu hỏi này cần train/validation/test chính thức, checkpoint mô hình, top-20 recommendation và item embedding.

---

## 6. Giới hạn diễn giải

1. Popularity hiện tại được tính trên toàn bộ raw chỉ để mô tả. Không dùng các giá trị này làm feature hoặc group trong đánh giá chính.
2. Tỷ lệ cùng category cao không chứng minh category gây ra hành vi click; đây là quan hệ quan sát.
3. Baseline category ngẫu nhiên giả định hai lần lấy item độc lập theo phân bố category toàn cục.
4. `timeframe` được dùng để sắp event trong session; `eventdate` là ngày lịch. Không có giờ bắt đầu tuyệt đối nên không diễn giải `eventdate + timeframe` như duration thực của session qua nửa đêm.
5. Kết quả độ nhạy ngưỡng được tính trên raw để mô tả trade-off; lựa chọn chính thức phải được kiểm tra lại trên train và bằng pilot training.
6. Gini sau lọc không được so sánh máy móc với recommendation Gini vì catalog và cơ chế sinh phân phối khác nhau.
7. Fixed-point chỉ là sensitivity check, không phải quyết định preprocessing.
8. Cold-start 20% chưa được triển khai vì cần chốt chính xác Mục 1.6/Option B.

---

## 7. File đầu ra

### Hình

1. [`eda_session_length_histogram_ecdf.png`](../../results/figures/eda_session_length_histogram_ecdf.png)
2. [`eda_daily_activity_and_new_items.png`](../../results/figures/eda_daily_activity_and_new_items.png)
3. [`eda_item_rank_frequency_loglog.png`](../../results/figures/eda_item_rank_frequency_loglog.png)
4. [`eda_interaction_lorenz_curve.png`](../../results/figures/eda_interaction_lorenz_curve.png)
5. [`eda_items_and_interactions_by_frequency_band.png`](../../results/figures/eda_items_and_interactions_by_frequency_band.png)
6. [`eda_category_structure_and_transition.png`](../../results/figures/eda_category_structure_and_transition.png)
7. [`eda_session_time_anomalies.png`](../../results/figures/eda_session_time_anomalies.png)
8. [`eda_frequency_threshold_tradeoff.png`](../../results/figures/eda_frequency_threshold_tradeoff.png)

### Bảng

Toàn bộ 24 bảng EDA được lưu trong [`results/tables`](../../results/tables/). Các bảng quan trọng nhất là:

- `eda_data_quality_summary.csv`
- `eda_time_quality_summary.csv`
- `eda_session_quantiles.csv`
- `eda_session_time_audit_summary.csv`
- `eda_session_time_anomalies.csv`
- `eda_duplicate_timeframes.csv`
- `eda_popularity_concentration.csv`
- `eda_frequency_bands.csv`
- `eda_frequency_threshold_sensitivity.csv`
- `eda_threshold_frequency_bands.csv`
- `eda_category_summary.csv`
- `eda_preprocessing_attrition.csv`
- `eda_target_frequency_diagnostic.csv`
- `eda_conclusions.csv`

---

## 8. Bước tiếp theo

Sau khi báo cáo EDA được xác nhận, `02_preprocessing.ipynb` cần thực hiện:

1. Chốt giao thức cold-start 20% từ Mục 1.6/Option B.
2. Dùng ngưỡng 5 làm cấu hình tham chiếu; kiểm tra ngưỡng 2 hoặc 3 bằng pilot nếu tài nguyên cho phép.
3. Áp dụng pipeline lọc chính thức và lưu attrition.
4. Chia train/validation/test theo ngày event cuối, không tách session qua nửa đêm và không làm rò rỉ thời gian.
5. Tính popularity từ train.
6. Chia nhóm cold/rare/medium/popular và kiểm tra số mẫu từng nhóm.
7. Xuất dữ liệu đúng định dạng để ba mô hình sử dụng chung.

---

## 9. Tài liệu tham chiếu cho quy trình

- [RecBole RecSysDatasets — Diginetica](https://github.com/RUCAIBox/RecSysDatasets/blob/master/conversion_tools/usage/DIGINETICA.md)
- [Neural Attentive Session-based Recommendation — NARM](https://arxiv.org/pdf/1711.04725)
- [SR-GNN preprocessing source](https://raw.githubusercontent.com/CRIPAC-DIG/SR-GNN/master/datasets/preprocess.py)
- [SR-GNN issue về thứ tự `timeframe` của Diginetica](https://github.com/CRIPAC-DIG/SR-GNN/issues/3)
- [Session-based recommendation preprocessing cho Diginetica](https://github.com/rn5l/session-rec/blob/master/preprocessing/session_based/preprocess_diginetica.py)
- [NISER: Normalized Item and Session Representations to Handle Popularity Bias](https://arxiv.org/abs/1909.04276)
- [Exploring Popularity Bias in Session-based Recommendation](https://arxiv.org/abs/2312.07855)
