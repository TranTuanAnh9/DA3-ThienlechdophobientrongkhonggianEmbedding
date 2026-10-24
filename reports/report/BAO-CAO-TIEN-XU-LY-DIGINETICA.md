# Báo cáo tiền xử lý dữ liệu Diginetica

## 1. Thông tin báo cáo

- **Đề tài:** BTL-K03 — Thiên lệch độ phổ biến và hình học không gian embedding trong mô hình SBR
- **Notebook:** [`02_data_preprocessing.ipynb`](../../notebooks/02_data_preprocessing.ipynb)
- **Notebook EDA đầu vào:** [`01_data_exploration.ipynb`](../../notebooks/01_data_exploration.ipynb)
- **Cấu hình:** [`data.yaml`](../../configs/data.yaml)
- **Môi trường chạy:** `D:\DL\.venv\Scripts\python.exe`, Python 3.12.6
- **Thời điểm hoàn thành:** 24/10/2026
- **Trạng thái:** 13/13 code cell chạy thành công, không có exception hoặc warning trong output
- **Kiểm tra:** 23/23 audit nội bộ và 11/11 pytest đều đạt

Mục tiêu của notebook 02 là chuyển dữ liệu raw đã được khảo sát trong notebook 01 thành các tập dữ liệu có thể dùng chung để huấn luyện và đánh giá GRU4Rec, NARM và SASRec. Notebook này chưa huấn luyện mô hình và chưa tính HR@K, MRR@K, NDCG@K, Coverage@K, ARP hay recommendation Gini.

---

## 2. Từ vấn đề phát hiện trong EDA đến cách xử lý trong notebook 02

Đây là phần quan trọng nhất của quá trình tiền xử lý. Mỗi quyết định trong notebook 02 đều xuất phát từ một vấn đề hoặc yêu cầu đã được xác định ở notebook 01.

| Phát hiện từ EDA 01 | Rủi ro nếu không xử lý | Cách xử lý trong notebook 02 | Kết quả/kiểm tra |
|---|---|---|---|
| 58,21% session không tăng theo `timeframe` trong thứ tự dòng raw | Chuỗi click bị sai thứ tự, làm sai dữ liệu đầu vào của mô hình tuần tự | Sắp xếp ổn định theo `session_id → timeframe → row_order` | Train, validation và test đều vượt kiểm tra thứ tự không giảm |
| Có 4 nhóm event trùng `(sessionId, timeframe)` | Không có thứ tự duy nhất cho các event bằng thời gian | Dùng thứ tự dòng gốc `row_order` để phá hòa; không tự ý xóa event | Chuỗi có thứ tự xác định và tái lập được |
| 29,23% session có đúng một interaction | Không thể tạo cặp lịch sử → next item | Loại session có độ dài dưới 2 và lọc lặp sau mỗi thay đổi catalog | Mọi session cuối cùng đều có ít nhất 2 event |
| 13,62% event là click lặp item trong session | Xóa click lặp có thể làm mất hành vi quay lại sản phẩm | Giữ nguyên click lặp; chỉ chuẩn hóa thứ tự | Dữ liệu đầu vào vẫn bảo toàn hành vi lặp |
| Có 260 session đi qua nửa đêm nhưng duration tối đa dưới 20 phút | Tách theo ngày có thể chia một session thành hai tập | Giữ nguyên session; gán split theo ngày của event cuối cùng | Không session nào xuất hiện trong nhiều split |
| Dữ liệu có trục thời gian từ 01/01/2016 đến 01/06/2016 | Chia ngẫu nhiên có thể đưa hành vi tương lai vào train | Chia temporal split: train quá khứ, validation 7 ngày, test 7 ngày cuối | Mọi mốc thời gian vượt audit |
| 63,80% item raw xuất hiện dưới 5 lần; Gini đầu vào 0,726 | Catalog rất thưa; embedding của item cực hiếm khó học ổn định | Dùng `minimum item frequency = 5` làm cấu hình tham chiếu và lọc đến fixed point | Train trước cold còn 38.237 item đủ support |
| Ngưỡng 5 loại phần lớn catalog và ảnh hưởng long-tail | Kết quả có thể phụ thuộc mạnh vào lựa chọn ngưỡng | Ghi đầy đủ attrition; cố định ngưỡng 5 cho cả ba mô hình; giữ ngưỡng 2/3 làm thí nghiệm mở rộng | Không tuyên bố ngưỡng 5 là tối ưu chỉ dựa trên EDA |
| Popularity trên raw có thể chứa thông tin validation/test | Rò rỉ dữ liệu khi chia rare/medium/popular | Tính lại toàn bộ frequency và nhóm popularity chỉ từ final train | Tổng train frequency đúng bằng số train event |
| Category mapping đầy đủ và 87,45% chuyển tiếp khác self-repeat vẫn cùng category | Category có ý nghĩa, nhưng kích thước category không đều | Dùng category để phân tầng khi chọn controlled cold và giữ category trong item mapping | 629/629 category có cold vẫn còn warm peer |
| EDA chưa tạo cold item và tài liệu chỉ nhắc “20% cold-start/Option B” | Không có giao thức cold rõ ràng để phân tích embedding chưa huấn luyện | Tạo controlled item holdout 20%, seed cố định, tách riêng natural temporal cold | 7.647 controlled cold và 6.700 natural cold |
| Các mô hình phải dùng cùng một không gian item | Nếu mỗi mô hình tự ánh xạ hoặc tự chia dữ liệu thì kết quả không còn so sánh công bằng | Xuất một mapping, một split và một bộ dữ liệu chung cho cả ba mô hình | ID nội bộ duy nhất, bắt đầu từ 1; ID 0 dành cho padding |

Tóm lại, notebook 01 trả lời **dữ liệu đang có vấn đề hoặc đặc điểm gì**, còn notebook 02 biến các phát hiện đó thành **quy tắc dữ liệu chính thức và có thể tái lập**.

---

## 3. Dữ liệu đầu vào

Notebook 02 sử dụng lại ba file raw đã được kiểm tra trong EDA:

| File | Vai trò trong preprocessing |
|---|---|
| `train-item-views.csv` | Nguồn interaction, session, item, ngày và thứ tự click |
| `product-categories.csv` | Ánh xạ item sang category |
| `products.csv` | Kiểm tra catalog và tính toàn vẹn metadata |

Trước khi xử lý, notebook tính lại SHA-256 và so sánh với manifest của EDA. Cả ba hash đều khớp, chứng minh notebook 02 dùng đúng phiên bản dữ liệu đã được phân tích trong notebook 01 và không chỉnh sửa dữ liệu raw.

Sau khi đọc, schema được chuẩn hóa về các tên chính:

```text
session_id
item_id
timeframe
event_date
row_order
category_id
```

`userId` không được dùng làm đầu vào mô hình vì đây là bài toán session-based recommendation và trường này thiếu gần 70% trong dữ liệu raw.

---

## 4. Quy trình tiền xử lý chính thức

Pipeline tổng quát:

```text
Ba file raw
    ↓
Kiểm tra hash và schema
    ↓
Sắp thứ tự event trong session
    ↓
Loại session không tạo được next-item target
    ↓
Temporal split theo ngày cuối session
    ↓
Lọc fixed-point trên train với item frequency ≥ 5
    ↓
Chọn controlled cold bằng 20% catalog train đã lọc
    ↓
Xóa controlled cold khỏi train và lọc fixed-point lại
    ↓
Tạo warm validation/test
    ↓
Tạo mapping và nhóm cold/rare/medium/popular
    ↓
Tạo sequence, prefix-target và file RecBole
    ↓
Audit và xuất dữ liệu
```

### 4.1. Chuẩn hóa thứ tự event

Các event được sắp theo:

```text
session_id → timeframe → row_order
```

- `timeframe` là khóa thời gian chính bên trong session.
- `row_order` là khóa phá hòa khi hai event có cùng `timeframe`.
- Click lặp được giữ nguyên.
- 260 session qua nửa đêm được giữ nguyên, không tách thành session mới.

Sau bước loại session length 1 ban đầu:

| Chỉ tiêu | Giá trị |
|---|---:|
| Event raw | 1.235.380 |
| Event còn lại | 1.144.686 |
| Session còn lại | 219.630 |
| Session qua nửa đêm được giữ | 260 |

### 4.2. Chia dữ liệu theo thời gian

Notebook dùng ngày của event cuối cùng trong mỗi session để gán toàn bộ session vào một split:

| Tập | Khoảng thời gian | Event trước lọc train/cold | Session | Item |
|---|---|---:|---:|---:|
| Train | 01/01/2016–18/05/2016 | 990.708 | 189.774 | 114.768 |
| Validation | 19/05/2016–25/05/2016 | 65.533 | 12.701 | 25.611 |
| Test | 26/05/2016–01/06/2016 | 88.445 | 17.155 | 30.290 |

Cách chia này đảm bảo:

- Không có session bị cắt giữa các tập.
- Train luôn xảy ra trước validation.
- Validation luôn xảy ra trước test.
- Không dùng dữ liệu tương lai để quyết định popularity hoặc học tham số mô hình.

#### Vì sao chọn cửa sổ 7 ngày cho validation và test?

Đây là **lựa chọn thiết kế của pipeline**, không phải yêu cầu bắt buộc của Diginetica và cũng chưa được chứng minh là cửa sổ tối ưu. Lựa chọn này dựa trên các căn cứ sau:

1. **Hai cửa sổ có cùng độ dài và cùng thành phần thứ trong tuần.** Validation chạy từ thứ Năm 19/05 đến thứ Tư 25/05/2016; test chạy từ thứ Năm 26/05 đến thứ Tư 01/06/2016. Mỗi tập vì vậy bao phủ đủ bảy thứ trong tuần và có thể so sánh trên cùng cấu trúc lịch. Tuy nhiên, EDA chưa thực hiện kiểm định seasonality theo thứ trong tuần, nên không được diễn giải rằng dữ liệu đã chứng minh một chu kỳ tuần mạnh.
2. **Quy mô đánh giá vẫn đủ lớn để chạy các metric của đề tài.** Sau toàn bộ lọc warm, validation còn 39.450 event, 9.622 session và 29.828 prefix-target example; test còn 52.792 event, 12.791 session và 40.001 prefix-target example. Đây là bằng chứng trực tiếp rằng cửa sổ 7 ngày không tạo ra tập đánh giá quá nhỏ trong pipeline hiện tại. Từ “đủ lớn” ở đây mang nghĩa thực dụng về số lượng mẫu, không phải một kết luận power analysis thống kê.
3. **Phần lớn lịch sử được dành cho huấn luyện.** Khoảng train từ 01/01 đến 18/05/2016 gồm 139 ngày lịch và 138 ngày thực sự có quan sát; ngày 22/04/2016 không có event. Toàn bộ dữ liệu có 152 ngày quan sát trên khoảng lịch 153 ngày từ 01/01 đến 01/06/2016. Vì vậy không nên gọi chính xác train là “138 ngày lịch”.
4. **Temporal split mô phỏng đúng hướng sử dụng.** Mô hình học từ quá khứ, dùng tuần kế tiếp để chọn cấu hình và dùng tuần cuối để đánh giá. Lập luận chống temporal leakage đến từ thứ tự thời gian này, không phụ thuộc riêng vào con số 7.

Notebook hiện chưa so sánh cửa sổ 7 ngày với 14 hoặc 30 ngày. Nếu cần kiểm tra độ nhạy theo thời gian, có thể chạy thêm một temporal-window sensitivity study; cho đến lúc đó, 7 ngày được xem là **baseline hợp lý và có đủ dữ liệu**, không phải lựa chọn tối ưu đã được chứng minh.

### 4.3. Lọc fixed-point trên train trước cold holdout

Chỉ train được dùng để quyết định catalog có thể học. Hai điều kiện được áp dụng lặp lại đến khi không còn thay đổi:

```text
session length ≥ 2
item frequency ≥ 5
```

Phải lọc lặp vì khi xóa item hiếm, một số session có thể chỉ còn một event; khi tiếp tục xóa session đó, tần suất của item khác có thể giảm xuống dưới 5.

Kết quả trước khi tạo controlled cold:

| Giai đoạn | Event | Session | Item | Category | Prefix-target |
|---|---:|---:|---:|---:|---:|
| Train ngay sau temporal split | 990.708 | 189.774 | 114.768 | 1.192 | 800.934 |
| Train sau fixed-point | 841.402 | 174.691 | 38.237 | 948 | 666.711 |

Ngưỡng 5 được dùng làm cấu hình tham chiếu, không phải kết luận rằng đây là ngưỡng tối ưu tuyệt đối. Thí nghiệm ngưỡng 2 hoặc 3 có thể được thực hiện sau như một sensitivity/pilot study nếu đủ tài nguyên.

---

## 5. Giao thức cold-start 20%

### 5.1. Controlled cold

Controlled cold được tạo theo **item-level holdout**:

1. Lấy catalog train sau fixed-point gồm 38.237 item làm mẫu số.
2. Chọn `round(20% × 38.237) = 7.647` item.
3. Chọn có phân tầng theo category để không tập trung cold vào một vài category lớn.
4. Dùng seed `2024` để kết quả có thể tái lập.
5. Chỉ chọn trong category có ít nhất 5 item trước holdout.
6. Bảo đảm mỗi category có cold vẫn giữ ít nhất một warm item.
7. Xóa toàn bộ interaction của controlled cold khỏi train.
8. Vẫn giữ ID và category của chúng trong item mapping để có thể quan sát embedding chưa huấn luyện.

Kết quả:

| Chỉ tiêu | Giá trị |
|---|---:|
| Catalog train trước holdout | 38.237 item |
| Controlled cold | 7.647 item |
| Tỷ lệ thực tế | 19,99895% |
| Controlled cold có event trong tương lai | 4.630 item |
| Category chứa controlled cold | 629 |
| Category vẫn có warm peer | 629/629 |

Tỷ lệ không đúng chính xác 20,00000% vì số item phải là số nguyên; 7.647 là số nguyên gần nhất của 20% × 38.237.

### 5.2. Lọc lại sau khi loại controlled cold

Việc xóa controlled cold làm giảm tần suất của các item còn lại và có thể làm session ngắn đi. Vì vậy train phải được lọc fixed-point lần nữa.

| Giai đoạn | Event | Session | Item | Category | Prefix-target |
|---|---:|---:|---:|---:|---:|
| Ngay sau khi xóa controlled cold | 671.642 | 170.305 | 30.590 | 948 | 501.337 |
| Final train sau fixed-point | 649.632 | 150.783 | 29.936 | 945 | 498.849 |

Có thêm 654 warm item bị loại do không còn đạt điều kiện sau holdout. Đây là **post-holdout attrition**, không phải controlled cold.

### 5.3. Natural temporal cold

Ngoài controlled cold, temporal split tự nhiên tạo ra **6.700 item** xuất hiện trong validation/test nhưng chưa từng xuất hiện trong train gốc. Chúng được đánh dấu là `natural_temporal` và theo dõi riêng.

Hai loại cold không được trộn lẫn:

- **Controlled cold:** từng đủ điều kiện trong train nhưng bị chủ động holdout để tạo nhóm kiểm soát.
- **Natural temporal cold:** chỉ xuất hiện sau mốc train do diễn tiến thời gian tự nhiên.

Với mô hình chỉ dùng item ID, natural cold không có embedding đã học và không thể được đánh giá next-item công bằng như warm item nếu không bổ sung side information hoặc một giao thức cold-start riêng.

### 5.4. Vai trò của cold trong đề tài

Controlled cold trong pipeline hiện tại dùng cho **minh họa hình học embedding**, không dùng làm tập đánh giá accuracy chính. Khi huấn luyện ba mô hình, phải bảo đảm:

- Controlled cold không xuất hiện như positive target.
- Controlled cold không được đưa vào negative sampling.
- Embedding của controlled cold không nhận gradient.
- Lưu embedding ngay sau khởi tạo để xác nhận embedding cold vẫn chưa được huấn luyện.
- HR/NDCG/MRR chính chỉ đánh giá trên warm target.

---

## 6. Tạo warm validation và test

Sau khi final warm catalog được xác định từ train:

- Chỉ giữ event thuộc final warm item trong validation/test chính.
- Loại lại session còn dưới 2 event.
- Giữ bản `*_events_full.parquet` để audit cold và mức attrition, không dùng nhầm làm tập accuracy chính.

Kết quả cuối:

| Tập | Event | Session | Item | Category | All-prefix example |
|---|---:|---:|---:|---:|---:|
| Train | 649.632 | 150.783 | 29.936 | 945 | 498.849 |
| Validation warm | 39.450 | 9.622 | 12.227 | 704 | 29.828 |
| Test warm | 52.792 | 12.791 | 13.847 | 738 | 40.001 |

Việc validation/test giảm so với temporal split ban đầu là có chủ đích: tập chính chỉ đo chất lượng gợi ý đối với item mà mô hình ID-based có khả năng học từ train.

---

## 7. Item mapping và nhóm popularity

### 7.1. Item mapping

Một mapping chung được tạo cho cả ba mô hình:

- ID nội bộ bắt đầu từ 1.
- ID 0 được dành cho padding.
- `item_id` raw và `internal_item_id` đều duy nhất.
- Mỗi item có category.
- Mapping chứa final warm item và controlled cold item.
- Natural temporal cold không thuộc model catalog chính vì không xuất hiện trong catalog train trước holdout.

Điều này ngăn tình trạng GRU4Rec, NARM và SASRec sử dụng ba cách đánh số hoặc ba catalog khác nhau.

### 7.2. Định nghĩa popularity

Popularity được tính lại **chỉ từ final train**:

```text
train_frequency(item) = số lần item xuất hiện trong final train
```

Nhóm được tạo theo phân vị có xử lý ties:

- Rare: `train_frequency ≤ 6`.
- Medium: `7 ≤ train_frequency ≤ 26`.
- Popular: `train_frequency ≥ 27`.
- Cold: `train_frequency = 0` đối với controlled cold.

Kết quả trên model catalog:

| Nhóm | Item | Train interaction | Frequency nhỏ nhất | Median | Frequency lớn nhất |
|---|---:|---:|---:|---:|---:|
| Cold | 7.647 | 0 | 0 | 0 | 0 |
| Rare | 6.884 | 37.525 | 5 | 5 | 6 |
| Medium | 16.940 | 218.091 | 7 | 11 | 26 |
| Popular | 6.112 | 394.016 | 27 | 45 | 860 |

Số item của từng nhóm không bắt buộc đúng chính xác 20% vì nhiều item có cùng frequency tại ranh giới phân vị. Tách các item bằng nhau một cách ngẫu nhiên chỉ để đạt tỷ lệ 20% sẽ làm định nghĩa nhóm thiếu nhất quán.

---

## 8. Tạo sequence và mẫu next-item

Notebook xuất hai dạng mẫu:

### 8.1. All-prefix examples

Với một session:

```text
[A, B, C, D]
```

Các mẫu được tạo là:

```text
[A]       → B
[A, B]    → C
[A, B, C] → D
```

Số mẫu:

| Tập | All-prefix examples |
|---|---:|
| Train | 498.849 |
| Validation | 29.828 |
| Test | 40.001 |

### 8.2. Last-item examples

Mỗi session chỉ tạo một mẫu, dùng toàn bộ phần trước để dự đoán item cuối:

```text
[A, B, C] → D
```

| Tập | Last-item examples |
|---|---:|
| Train | 150.783 |
| Validation | 9.622 |
| Test | 12.791 |

Việc lưu cả hai dạng cho phép notebook huấn luyện chọn giao thức phù hợp mà không phải chia dữ liệu lại.

---

## 9. Dữ liệu được xuất

### 9.1. Dữ liệu canonical và processed

Thư mục [`data/interim/diginetica`](../../data/interim/diginetica) chứa dữ liệu canonical sau khi chuẩn hóa thứ tự và session.

Thư mục [`data/processed/diginetica`](../../data/processed/diginetica) chứa:

- `train_events.parquet`, `validation_events.parquet`, `test_events.parquet`.
- `validation_events_full.parquet`, `test_events_full.parquet` phục vụ audit.
- `train_sessions.parquet`, `validation_sessions.parquet`, `test_sessions.parquet`.
- `train_examples.parquet`, `validation_examples.parquet`, `test_examples.parquet`.
- `train_last_item_examples.parquet`, `validation_last_item_examples.parquet`, `test_last_item_examples.parquet`.
- `item_mapping.csv`.
- `category_mapping.csv`.
- `popularity_groups.csv`.
- `cold_items.csv`.
- `preprocessing_config.json`.
- `dataset_statistics.json`.

Parquet được dùng vì giữ kiểu dữ liệu tốt, đọc nhanh và có kích thước nhỏ hơn CSV đối với các bảng lớn.

### 9.2. Dữ liệu RecBole

Thư mục [`data/processed/recbole/diginetica`](../../data/processed/recbole/diginetica) chứa:

```text
diginetica.train.inter
diginetica.valid.inter
diginetica.test.inter
diginetica.item
```

Đây không phải dataset khác. Đây là cùng dữ liệu Diginetica đã xử lý, được chuyển sang atomic format để RecBole có thể đọc:

- `.inter`: session, item và timestamp của từng interaction.
- `.item`: ID nội bộ, ID raw, category và cờ controlled cold.

Khi huấn luyện, RecBole phải sử dụng đúng ba file benchmark đã tạo; không được tự chia ngẫu nhiên dữ liệu lần nữa.

### 9.3. Bảng và hình bàn giao

Các bảng chính trong [`results/tables`](../../results/tables):

- `preprocessing_base_split_statistics.csv`.
- `preprocessing_attrition_official.csv`.
- `preprocessing_cold_start_summary.csv`.
- `preprocessing_final_split_statistics.csv`.
- `preprocessing_popularity_group_summary.csv`.
- `preprocessing_example_summary.csv`.
- `preprocessing_checks.csv`.
- `cold_category_allocation.csv`.

Các hình chính trong [`results/figures`](../../results/figures):

- `preprocessing_temporal_split.png`.
- `preprocessing_attrition_and_groups.png`.

---

## 10. Kiểm tra tính đúng đắn

Notebook chạy 23 điều kiện audit và tất cả đều đạt:

### Nhóm kiểm tra dữ liệu đầu vào

- Ba SHA-256 khớp manifest EDA.
- Thứ tự `timeframe` không giảm trong từng session.
- Mọi session cuối cùng có độ dài ít nhất 2.

### Nhóm kiểm tra temporal split

- Train, validation và test không trùng session.
- Train kết thúc trước validation.
- Validation kết thúc trước test.

### Nhóm kiểm tra cold-start

- Controlled cold có đúng 7.647 item.
- Không controlled cold nào xuất hiện trong train.
- Mọi category chứa cold đều còn warm peer.
- Validation/test chính chỉ chứa warm item.

### Nhóm kiểm tra mapping và popularity

- Raw ID và internal ID đều duy nhất.
- Internal ID bắt đầu từ 1.
- Không item nào thiếu category.
- Mọi final warm item có train frequency ít nhất 5.
- Tổng train frequency đúng bằng 649.632 train event.

Ba file kiểm thử độc lập:

- [`test_data_split.py`](../../tests/test_data_split.py)
- [`test_item_mapping.py`](../../tests/test_item_mapping.py)
- [`test_no_cold_leakage.py`](../../tests/test_no_cold_leakage.py)

Kết quả: **11/11 pytest passed**.

---

## 11. Những quyết định đã chốt và những giới hạn còn lại

### 11.1. Đã chốt

1. Temporal split theo ngày cuối session.
2. Validation và test mỗi tập dài 7 ngày.
3. Giữ click lặp và session qua nửa đêm.
4. Session length tối thiểu bằng 2.
5. Item frequency tối thiểu bằng 5.
6. Lọc fixed-point trên train.
7. Controlled cold bằng 20% catalog train sau lọc, phân tầng category, seed 2024.
8. Popularity chỉ được tính từ final train.
9. Main accuracy evaluation chỉ dùng warm item.
10. Ba mô hình dùng chung split, mapping và catalog.

### 11.2. Giới hạn và lưu ý

- Tài liệu đề tài nhắc đến “Mục 1.6/Option B” nhưng phần định nghĩa chi tiết đó không có trong tài liệu được cung cấp. Giao thức controlled cold hiện tại là giả định minh bạch dựa trên mô tả “20% cold-start” và yêu cầu quan sát embedding chưa huấn luyện. Nếu nhận được quy định chính thức khác, notebook phải được cập nhật và chạy lại.
- Ngưỡng 5 là baseline đã chốt cho pipeline chính, chưa được chứng minh là tối ưu. Ngưỡng 2 hoặc 3 có thể được đánh giá bằng pilot training như thí nghiệm mở rộng.
- Natural cold không thể được mô hình item-ID-only xử lý công bằng nếu không có side information hoặc cơ chế cold-start riêng.
- Controlled cold chỉ có ý nghĩa là embedding chưa huấn luyện nếu notebook huấn luyện ngăn chúng đi vào negative sampling và không cho chúng nhận gradient.
- Các ngưỡng rare/popular là kết quả của final train hiện tại; nếu thay đổi split, seed cold hoặc ngưỡng support thì phải tính lại.

---

## 12. Kết luận và bàn giao cho bước huấn luyện

Notebook 02 đã giải quyết các vấn đề quan trọng được phát hiện trong EDA:

- Sửa thứ tự chuỗi nhưng không làm mất click lặp.
- Loại session không thể tạo next-item target.
- Chia dữ liệu đúng theo thời gian và không cắt session.
- Kiểm soát tác động của item cực hiếm bằng ngưỡng support và attrition table.
- Ngăn leakage bằng cách tính popularity chỉ từ train.
- Tạo controlled cold 20% có thể tái lập và tách riêng natural cold.
- Tạo một không gian ID và một bộ split chung cho cả ba mô hình.
- Xuất đầy đủ sequence, next-item examples và RecBole atomic files.
- Xác nhận pipeline bằng 23 audit và 11 kiểm thử độc lập.

Dữ liệu hiện đã sẵn sàng cho bước tiếp theo: huấn luyện GRU4Rec, NARM và SASRec trên cùng split và cùng cấu hình đánh giá. Trong các notebook huấn luyện, bắt buộc phải dùng đúng item mapping, không chia lại dữ liệu, không cho controlled cold tham gia học, và lưu checkpoint cùng embedding để phục vụ phân tích popularity bias và hình học embedding.
