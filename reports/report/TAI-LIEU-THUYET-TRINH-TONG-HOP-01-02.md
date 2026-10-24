# Tài liệu thuyết trình tổng hợp: từ bài toán đến EDA và tiền xử lý Diginetica

## 1. Mục đích của tài liệu

Tài liệu này giúp trình bày mạch nghiên cứu từ đầu đến trạng thái hiện tại của đề tài:

> **BTL-K03 — Thiên lệch độ phổ biến và hình học không gian embedding trong mô hình Session-Based Recommendation**

Nội dung tập trung vào:

1. Đề tài đang giải quyết vấn đề gì.
2. Ý tưởng nghiên cứu và các câu hỏi cần trả lời.
3. Vì sao sử dụng Diginetica.
4. Notebook 01 đã khám phá dữ liệu như thế nào.
5. Notebook 01 phát hiện những vấn đề gì.
6. Notebook 02 xử lý từng vấn đề đó ra sao.
7. Dữ liệu cuối cùng hiện có đặc điểm gì.
8. Kết luận nào đã có thể đưa ra và kết luận nào chưa thể đưa ra.
9. Những bước tiếp theo để hoàn thành đề tài.

Phạm vi đã hoàn thành tại thời điểm viết tài liệu:

- [`01_data_exploration.ipynb`](../../notebooks/01_data_exploration.ipynb): đã hoàn thành EDA.
- [`02_data_preprocessing.ipynb`](../../notebooks/02_data_preprocessing.ipynb): đã hoàn thành preprocessing và chuẩn bị dữ liệu huấn luyện.
- Chưa huấn luyện GRU4Rec, NARM và SASRec.
- Chưa có kết quả Recall, MRR, NDCG, Coverage, recommendation Gini hoặc ARP.
- Chưa trích xuất embedding đã huấn luyện để phân tích norm, cosine hoặc t-SNE.

Vì vậy, khi thuyết trình phải phân biệt rõ:

```text
Kết quả EDA và preprocessing: đã có bằng chứng thực nghiệm.
Kết quả về ba mô hình và embedding: mới là kế hoạch nghiên cứu tiếp theo.
```

---

## 2. Đề tài đang nghiên cứu vấn đề gì?

### 2.1. Bài toán Session-Based Recommendation

Session-Based Recommendation, viết tắt là SBR, là bài toán gợi ý dựa trên chuỗi hành vi trong một phiên truy cập hiện tại.

Ví dụ:

```text
Session: điện thoại → ốp lưng → sạc dự phòng → ?
```

Mô hình nhận phần lịch sử:

```text
[điện thoại, ốp lưng, sạc dự phòng]
```

và dự đoán item tiếp theo mà người dùng có thể tương tác, chẳng hạn:

```text
tai nghe
```

Trong dữ liệu, một session được biểu diễn bằng chuỗi item có thứ tự:

```text
[A, B, C, D]
```

Chuỗi này có thể tạo các bài toán dự đoán next-item:

```text
[A]       → B
[A, B]    → C
[A, B, C] → D
```

### 2.2. Popularity bias là gì?

Trong dữ liệu thương mại điện tử, một số ít item thường nhận phần lớn tương tác, còn rất nhiều item chỉ xuất hiện vài lần. Đây là phân bố long-tail.

Popularity bias xảy ra khi mô hình tiếp tục ưu tiên các item vốn đã phổ biến trong danh sách gợi ý. Một vòng lặp có thể xuất hiện:

```text
Item đã phổ biến
      ↓
Được mô hình gợi ý nhiều
      ↓
Nhận thêm nhiều tương tác
      ↓
Càng trở nên phổ biến
```

Một mô hình có accuracy tốt chưa chắc phân phối cơ hội xuất hiện công bằng cho catalog. Ngược lại, một mô hình có danh sách gợi ý rất đa dạng nhưng dự đoán gần như ngẫu nhiên cũng không phải mô hình tốt. Vì vậy cần xem đồng thời:

- Chất lượng dự đoán.
- Mức tập trung của danh sách gợi ý.

### 2.3. Vì sao phân tích embedding?

Ba mô hình biểu diễn mỗi item bằng một vector embedding nhiều chiều:

```text
item A = [0,2; 0,8; -0,1; ...]
item B = [0,4; 0,7;  0,0; ...]
```

Các item xuất hiện nhiều trong train có thể nhận nhiều lần cập nhật gradient hơn item hiếm. Do đó, ngoài việc nhìn danh sách gợi ý đầu ra, đề tài còn muốn quan sát cấu trúc bên trong mô hình:

- Norm embedding có liên hệ với tần suất train không?
- Item cùng category có embedding gần nhau hơn item khác category không?
- Popular, rare và cold item nằm ở đâu trong không gian embedding?
- Đặc điểm embedding có đi kèm với số lần item được gợi ý không?

Đây là phân tích mối liên hệ, không tự động chứng minh quan hệ nhân quả.

### 2.4. Bản chất của đề tài

Đề tài hiện tại là đề tài **chẩn đoán và phân tích**, không phải đề tài bắt buộc phát minh một thuật toán khử bias mới.

Quy trình nghiên cứu tổng quát:

```text
Dữ liệu Diginetica
        ↓
EDA và kiểm tra chất lượng
        ↓
Tiền xử lý + cold-start 20%
        ↓
Huấn luyện GRU4Rec, NARM, SASRec
        ↓
Sinh danh sách Top-20
        ↓
Đo accuracy và popularity bias
        ↓
Trích xuất item embedding
        ↓
Phân tích norm, cosine và t-SNE
        ↓
Liên hệ embedding với lượt gợi ý
```

Hiện dự án đã hoàn thành hai khối đầu tiên: EDA và preprocessing.

---

## 3. Mục tiêu và câu hỏi nghiên cứu

### 3.1. Mục tiêu tổng quát

Định lượng popularity bias trong đầu ra của ba mô hình SBR và kiểm tra xem thiên lệch đó có đi kèm với khác biệt hình học trong item embedding hay không.

### 3.2. Ba mô hình dự kiến

- **GRU4Rec:** dùng GRU để mã hóa chuỗi session.
- **NARM:** kết hợp GRU với attention để nắm bắt ý định chính trong session.
- **SASRec:** dùng self-attention để mô hình hóa quan hệ giữa các vị trí trong chuỗi.

Ba mô hình phải sử dụng cùng:

- Train/validation/test split.
- Item mapping.
- Candidate catalog.
- Nhóm popularity.
- Top-k bằng 20.
- Quy tắc đánh giá.

### 3.3. Các câu hỏi nghiên cứu chính

1. Mô hình nào có popularity bias mạnh hơn ở danh sách Top-20?
2. Item phổ biến trong train có được gợi ý nhiều hơn không?
3. Tần suất train có tương quan với norm embedding không?
4. Norm embedding có tương quan với số lần item được gợi ý không?
5. Item cùng category có cosine similarity cao hơn item khác category không?
6. Popular, rare và cold item có phân bố khác nhau trong không gian embedding không?
7. Các kết luận có ổn định qua hai random seed không?

### 3.4. Các chỉ số sẽ dùng sau khi huấn luyện

Chất lượng dự đoán:

- Recall@20.
- MRR@20.
- NDCG@20.

Popularity bias ở đầu ra:

- Catalog Coverage@20.
- Recommendation Gini.
- Average Recommendation Popularity, viết tắt là ARP.
- Số lần từng item xuất hiện trong các danh sách Top-20.

Hình học embedding:

- Norm của từng item embedding.
- Spearman giữa train frequency và norm.
- Cosine similarity của cặp cùng category và khác category.
- t-SNE để minh họa popular/rare/cold.

Lưu ý quan trọng:

> Input-data Gini của dữ liệu không phải recommendation Gini của đầu ra mô hình. Hai chỉ số có đối tượng đo khác nhau và phải được gọi đúng tên.

---

## 4. Vì sao chọn Diginetica?

Diginetica là dữ liệu hành vi thương mại điện tử từ CIKM Cup 2016. Dataset phù hợp với đề tài vì:

1. Có chuỗi hành vi theo session, phù hợp trực tiếp với SBR.
2. Có phân bố popular–rare rõ ràng, phù hợp nghiên cứu popularity bias.
3. Có category cho item, cho phép phân tích cấu trúc embedding cùng/khác category.
4. Là benchmark được sử dụng phổ biến trong nghiên cứu recommendation.
5. Kích thước đủ lớn để thể hiện long-tail nhưng vẫn khả thi với tài nguyên của bài tập lớn.

### 4.1. Ba file raw được sử dụng

| File | Số dòng | Vai trò |
|---|---:|---|
| `train-item-views.csv` | 1.235.380 | Session, item, ngày và thứ tự click |
| `product-categories.csv` | 184.047 | Ánh xạ item sang category |
| `products.csv` | 184.047 | Kiểm tra catalog và metadata |

Ba file liên kết bằng `itemId`.

Các trường chính của interaction:

- `sessionId`: xác định session.
- `itemId`: xác định sản phẩm.
- `timeframe`: thời gian tương đối và thứ tự trong session.
- `eventdate`: ngày xảy ra interaction.
- `userId`: không được dùng làm đầu vào chính vì bài toán là SBR và trường này thiếu nhiều.

### 4.2. Quy mô raw

| Chỉ tiêu | Giá trị |
|---|---:|
| Interaction | 1.235.380 |
| Session | 310.324 |
| Item có tương tác | 122.993 |
| Tổng category metadata | 1.217 |
| Ngày quan sát | 152 |

Dữ liệu nằm trong khoảng lịch từ 01/01/2016 đến 01/06/2016. Khoảng này gồm 153 ngày lịch nhưng chỉ có 152 ngày có event vì ngày 22/04/2016 không có quan sát.

---

## 5. Notebook 01 — EDA được thực hiện để làm gì?

Notebook 01 không chỉ tạo biểu đồ mô tả. Mục tiêu của nó là trả lời các câu hỏi trước khi thiết kế preprocessing:

1. Ba file có đọc và liên kết đúng không?
2. Event trong session đã đúng thứ tự chưa?
3. Session có đủ dài để tạo next-item target không?
4. Click lặp có phổ biến không và có nên xóa không?
5. Dữ liệu có long-tail đủ rõ để nghiên cứu popularity bias không?
6. Category có mang tín hiệu hành vi không?
7. Ngưỡng lọc item ảnh hưởng đến long-tail ra sao?
8. Những quy tắc nào phải được chuyển sang notebook 02?

Notebook 01 gồm 25 cell, trong đó 12 code cell đã chạy thành công và không có error.

---

## 6. Kết quả EDA 01: chất lượng và tính toàn vẹn dữ liệu

### 6.1. Những kiểm tra đã làm

- Kiểm tra schema của ba file.
- Đếm missing value.
- Kiểm tra dòng trùng hoàn toàn.
- Kiểm tra event trùng khóa.
- Kiểm tra item trùng trong metadata.
- Parse ngày.
- Kiểm tra `timeframe` âm.
- Kiểm tra interaction item có category và product metadata hay không.
- Tính SHA-256 để định danh phiên bản dữ liệu.

### 6.2. Kết quả

- Không có dòng trùng hoàn toàn.
- Không có khóa event trùng.
- Không có `itemId` trùng trong bảng category hoặc product.
- Không có lỗi parse ngày.
- Không có `timeframe` âm.
- Không có item đã tương tác bị thiếu category.
- Không có item đã tương tác bị thiếu product metadata.
- Có 61.054 item trong metadata chưa từng xuất hiện trong interaction.
- `userId` thiếu 69,81%.

### 6.3. Ý nghĩa

Dữ liệu đủ sạch để tiếp tục. `userId` thiếu nhiều nhưng không cản trở vì mô hình dựa trên session, không dựa trên lịch sử dài hạn của user.

Không cần tự ý xóa hoặc điền `userId`. Việc cố phân tích theo user có thể làm sai phạm vi đề tài.

### Cách nói khi thuyết trình

> “Đầu tiên, nhóm kiểm tra tính toàn vẹn của ba file. Các khóa item liên kết đầy đủ, không có duplicate quan trọng và không có lỗi thời gian. User ID thiếu gần 70%, nhưng đây không phải vấn đề đối với session-based recommendation vì đơn vị phân tích của chúng tôi là session.”

---

## 7. Kết quả EDA 01: thứ tự thời gian trong session

### 7.1. Vấn đề phát hiện

Có 180.643 session, tương đương 58,21%, không tăng theo `timeframe` nếu giữ nguyên thứ tự dòng raw.

Điều này có nghĩa là thứ tự dòng CSV không phải thứ tự click đáng tin cậy.

Nếu đưa dữ liệu chưa sắp vào mô hình, chuỗi hành vi có thể bị đảo:

```text
Chuỗi thật: A → B → C
Chuỗi raw:  C → A → B
```

Mô hình tuần tự khi đó sẽ học một quan hệ chuyển tiếp sai.

### 7.2. Quy tắc rút ra

Phải sắp ổn định theo:

```text
sessionId → timeframe → row_order
```

EDA còn phát hiện 4 nhóm có cùng `(sessionId, timeframe)`. Không có bằng chứng để xóa một event trong các nhóm này, nên thứ tự dòng gốc được dùng làm khóa phá hòa.

### 7.3. Kết luận

Sắp xếp lại event là yêu cầu bắt buộc, không phải thao tác làm đẹp dữ liệu.

### Cách nói khi thuyết trình

> “Vấn đề chất lượng quan trọng nhất là gần 58,21% session không đúng thứ tự thời gian nếu đọc theo thứ tự dòng. Vì ba mô hình đều học chuỗi, chúng tôi bắt buộc sắp lại theo session, timeframe và thứ tự dòng để phá hòa.”

---

## 8. Kết quả EDA 01: cấu trúc session và click lặp

### 8.1. Độ dài session

| Phân vị | Session length |
|---:|---:|
| Nhỏ nhất | 1 |
| P25 | 1 |
| Median | 3 |
| P75 | 5 |
| P90 | 9 |
| Lớn nhất | 87 |

Phân nhóm session:

| Nhóm độ dài | Tỷ lệ session | Tỷ lệ event |
|---|---:|---:|
| 1 | 29,23% | 7,34% |
| 2–5 | 47,85% | 37,25% |
| 6–10 | 16,34% | 30,68% |
| Trên 10 | 6,59% | 24,72% |

### 8.2. Vì sao loại session length 1?

Session chỉ có một interaction không thể tạo cặp lịch sử–target:

```text
[A] → không có item tiếp theo
```

Vì vậy 29,23% session length 1 phải bị loại trước khi tạo next-item examples.

### 8.3. Click lặp

- 27,85% session có item được click lặp.
- 13,62% event là lượt lặp trong session.
- Có cả lặp liên tiếp và lặp không liên tiếp.

Click lặp có thể phản ánh người dùng quay lại xem một sản phẩm. Không có căn cứ để coi đó là dữ liệu lỗi, nên pipeline giữ nguyên.

### 8.4. Ảnh hưởng của session dài

Session dài hơn 10 chỉ chiếm một tỷ lệ nhỏ về số session nhưng tạo khoảng một phần tư số event và số lượng lớn prefix-target examples. Khi dùng all-prefix training, session dài sẽ đóng góp nhiều mẫu hơn session ngắn.

Đây là một đặc điểm cần ghi nhận, không nhất thiết là lỗi phải sửa.

### Cách nói khi thuyết trình

> “Session Diginetica nhìn chung ngắn, median chỉ bằng 3. Gần một phần ba session chỉ có một click nên không thể tạo target và phải loại. Ngược lại, click lặp được giữ vì nó có thể là tín hiệu người dùng quay lại sản phẩm, không phải duplicate kỹ thuật.”

---

## 9. Kết quả EDA 01: session qua ngày

EDA phát hiện 260 session xuất hiện trên hai ngày lịch khác nhau. Nếu chỉ nhìn `eventdate`, có thể hiểu nhầm đây là session dài hơn 24 giờ.

Kiểm tra bằng `timeframe` cho thấy:

- Tất cả chỉ đi qua hai ngày liên tiếp.
- Không session nào dài từ 30 phút trở lên.
- Duration lớn nhất khoảng 1.196,75 giây, tức gần 19 phút 57 giây.

Đây là các session ngắn đi qua nửa đêm.

Quyết định:

- Giữ nguyên `sessionId`.
- Không tách session theo `eventdate`.
- Khi chia dữ liệu, gán toàn bộ session theo ngày của event cuối cùng.

Nếu tách session theo ngày, một hành vi liên tục có thể bị chia thành hai chuỗi giả tạo.

---

## 10. Kết quả EDA 01: thời gian và vòng đời item

Dữ liệu có trung bình khoảng:

- 8.127,5 interaction/ngày.
- 2.043,32 session/ngày.
- 5.217,8 active item/ngày.
- 809,16 new item/ngày.

Tương quan Spearman giữa tần suất raw và tuổi quan sát của item:

```text
rho = 0,504
```

Điều này cho thấy item tồn tại lâu hơn có xu hướng tích lũy nhiều interaction hơn.

Ý nghĩa đối với phân tích embedding sau này:

- Nếu frequency tương quan với embedding norm, tuổi item có thể là biến gây nhiễu.
- Không nên kết luận đơn giản rằng frequency trực tiếp làm norm lớn.
- Kết luận phù hợp là “có mối liên hệ” hoặc “đi kèm với”, không phải kết luận nhân quả.

---

## 11. Kết quả EDA 01: popularity và long-tail

### 11.1. Phân phối tần suất

| Chỉ tiêu | Giá trị |
|---|---:|
| Item xuất hiện đúng 1 lần | 34,20% |
| Item xuất hiện dưới 5 lần | 63,80% |
| Top 1% item nắm giữ interaction | 21,65% |
| Top 10% item nắm giữ interaction | 62,53% |
| Top 20% item nắm giữ interaction | 77,54% |
| Input-data Gini | 0,726 |

Một nhóm rất nhỏ item nhận phần lớn interaction, trong khi đa số item chỉ xuất hiện vài lần. Đây là long-tail rõ ràng.

### 11.2. Gini 0,726 có ý nghĩa gì?

Gini đo mức độ bất bình đẳng của phân phối interaction giữa các item:

- Gần 0: interaction phân bố tương đối đều.
- Càng gần 1: interaction càng tập trung vào một nhóm nhỏ item.

Gini raw bằng 0,726 cho thấy đầu vào đã mất cân bằng mạnh.

Nhưng cần nói chính xác:

> “Diginetica có mức tập trung popularity cao ở dữ liệu đầu vào.”

Không được nói:

> “Mô hình đã có popularity bias bằng 0,726.”

Mô hình chưa được huấn luyện tại EDA. Recommendation Gini chỉ có thể tính sau khi sinh Top-20.

### 11.3. Kết luận

Diginetica phù hợp trực tiếp để nghiên cứu popularity bias vì có cả head item và long-tail item rõ ràng.

### Cách nói khi thuyết trình

> “Long-tail của dữ liệu rất mạnh: 63,8% item xuất hiện dưới 5 lần, trong khi top 20% item nắm hơn 77% interaction. Gini đầu vào bằng 0,726. Đây là điều kiện phù hợp để nghiên cứu bias, nhưng chưa phải bằng chứng rằng mô hình gây bias.”

---

## 12. Kết quả EDA 01: category có ý nghĩa không?

### 12.1. Kết quả

| Chỉ tiêu | Giá trị |
|---|---:|
| Tổng category metadata | 1.217 |
| Session chỉ có một category | 80,95% |
| Chuyển tiếp liên tiếp cùng category | 88,43% |
| Cùng category sau khi bỏ self-repeat | 87,45% |
| Baseline ngẫu nhiên | 1,08% |

Ngay cả khi bỏ các chuyển tiếp lặp cùng item, tỷ lệ hai item liên tiếp cùng category vẫn là 87,45%, cao hơn rất nhiều baseline ngẫu nhiên 1,08%.

### 12.2. Ý nghĩa

Category phản ánh cấu trúc hành vi thật trong session. Vì vậy, sau khi huấn luyện có cơ sở để hỏi:

> Item cùng category có cosine similarity cao hơn item khác category không?

### 12.3. Lưu ý sampling

Category có kích thước không đồng đều. Nếu sinh mọi cặp item, category lớn sẽ tạo số cặp nhiều hơn rất nhiều và chi phối kết quả trung bình.

Do đó, phân tích cosine sau này phải lấy mẫu cân bằng theo category.

---

## 13. Kết quả EDA 01: độ nhạy của ngưỡng lọc item

### 13.1. Vì sao cần ngưỡng tần suất?

Item chỉ xuất hiện một hoặc hai lần cung cấp rất ít tín hiệu để học embedding ổn định. Một vocabulary quá lớn và rất thưa còn làm tăng thời gian, bộ nhớ và độ khó huấn luyện.

Nhưng nếu lọc quá mạnh:

```text
Ngưỡng cao
→ nhiều rare item bị loại
→ long-tail bị cắt ngắn
→ kết luận popularity bias có thể thay đổi
```

Do đó EDA thử nhiều ngưỡng thay vì chấp nhận ngưỡng 5 một cách máy móc.

### 13.2. Kết quả sensitivity

| Ngưỡng | Event giữ lại | Item giữ lại | Số item cuối | Gini sau lọc |
|---:|---:|---:|---:|---:|
| 1 | 92,66% | 98,76% | 121.468 | 0,7170 |
| 2 | 88,92% | 64,22% | 78.989 | 0,6471 |
| 3 | 85,64% | 49,15% | 60.454 | 0,6048 |
| 5 | 80,07% | 34,29% | 42.171 | 0,5515 |
| 10 | 69,12% | 19,45% | 23.920 | 0,4801 |

Ngưỡng 5 giữ phần lớn event nhưng chỉ giữ khoảng một phần ba item raw. Đây là một trade-off lớn.

### 13.3. Tại sao Gini sau lọc giảm?

Khi loại nhiều item hiếm khỏi mẫu số, phân phối trên catalog còn lại có thể trông đồng đều hơn về mặt cơ học. Điều đó không có nghĩa mô hình đã giảm bias, vì chưa có mô hình nào được huấn luyện.

### 13.4. Quyết định

- Dùng ngưỡng 5 làm baseline chính.
- Ghi lại đầy đủ attrition.
- Nếu còn tài nguyên, thử thêm ngưỡng 2 hoặc 3 như sensitivity/pilot experiment.
- Không tuyên bố ngưỡng 5 là tối ưu chỉ dựa trên EDA.

---

## 14. Từ notebook 01 sang notebook 02

Mối liên hệ giữa hai notebook:

| Phát hiện trong notebook 01 | Rủi ro | Xử lý trong notebook 02 |
|---|---|---|
| Thứ tự dòng raw không đúng | Chuỗi đầu vào sai | Stable sort theo session, timeframe, row order |
| Session length 1 chiếm 29,23% | Không có target tiếp theo | Loại session dưới 2 event |
| Click lặp có ý nghĩa | Mất hành vi nếu xóa | Giữ click lặp |
| 260 session qua nửa đêm | Có thể bị cắt sai | Giữ nguyên, chia theo ngày cuối session |
| Long-tail mạnh | Vocabulary thưa | Lọc minimum frequency 5 |
| Lọc item làm session ngắn đi | Pipeline một lượt chưa ổn định | Lọc lặp đến fixed point |
| Popularity raw chứa tương lai | Leakage | Tính popularity chỉ từ final train |
| Category có tín hiệu mạnh | Có thể dùng để kiểm soát cold/cosine | Cold được phân tầng category |
| Chưa có giao thức cold cụ thể | Không phân tích được cold embedding | Tạo controlled item holdout 20% |
| Ba mô hình cần so sánh công bằng | Split/mapping khác nhau làm sai so sánh | Xuất một split và mapping chung |

Đây là câu chuyện trung tâm cần trình bày:

> “EDA không đứng riêng. Mỗi phát hiện quan trọng trong EDA được chuyển thành một quy tắc cụ thể và có kiểm tra trong preprocessing.”

---

## 15. Notebook 02 — quy trình xử lý tổng thể

Notebook 02 gồm 27 cell, trong đó 13 code cell đã chạy thành công, không có error hoặc warning trong output.

Pipeline:

```text
Ba file raw
    ↓
Kiểm tra SHA-256 và schema
    ↓
Chuẩn hóa cột và sắp thứ tự event
    ↓
Loại session không tạo được target
    ↓
Temporal split theo ngày cuối session
    ↓
Fixed-point filtering trên train
    ↓
Controlled cold holdout 20%
    ↓
Loại cold khỏi train và lọc lại
    ↓
Tạo warm validation/test
    ↓
Tạo item/category mapping
    ↓
Tính popularity từ final train
    ↓
Tạo sequence và next-item examples
    ↓
Xuất Parquet và RecBole
    ↓
Audit toàn bộ pipeline
```

---

## 16. Notebook 02: chia train, validation và test

### 16.1. Cách chia

Chia theo thời gian, dựa trên ngày của event cuối cùng trong session:

| Tập | Khoảng thời gian | Event ban đầu | Session | Item |
|---|---|---:|---:|---:|
| Train | 01/01–18/05/2016 | 990.708 | 189.774 | 114.768 |
| Validation | 19/05–25/05/2016 | 65.533 | 12.701 | 25.611 |
| Test | 26/05–01/06/2016 | 88.445 | 17.155 | 30.290 |

Toàn bộ session thuộc đúng một split. Không có session bị chia đôi.

### 16.2. Tại sao không chia ngẫu nhiên?

Nếu chia ngẫu nhiên, interaction xảy ra muộn có thể lọt vào train trong khi interaction sớm lại nằm ở test. Mô hình khi đó gián tiếp nhìn thấy tương lai.

Temporal split mô phỏng tình huống thực tế:

```text
Học từ lịch sử đến 18/05
→ chọn cấu hình bằng tuần 19–25/05
→ đánh giá bằng tuần 26/05–01/06
```

### 16.3. Vì sao validation và test là 7 ngày?

Đây là lựa chọn thiết kế, không phải Diginetica bắt buộc và chưa được chứng minh là tối ưu.

Căn cứ:

1. Hai cửa sổ có cùng độ dài và cùng thành phần thứ trong tuần: đều từ thứ Năm đến thứ Tư.
2. Sau toàn bộ xử lý, validation vẫn có 29.828 và test có 40.001 prefix-target examples.
3. Giữ phần lớn lịch sử cho train.
4. Cấu trúc quá khứ → tuần validation → tuần test dễ diễn giải.

Diễn đạt chính xác:

> “Bảy ngày là baseline hợp lý và có đủ dữ liệu trong pipeline hiện tại.”

Không nên nói:

> “Bảy ngày đã được chứng minh là tốt nhất.”

Notebook chưa chạy sensitivity cho cửa sổ 14 hoặc 30 ngày.

### 16.4. Số ngày train cần nói chính xác

Train từ 01/01 đến 18/05 gồm:

- 139 ngày lịch.
- 138 ngày thực sự có event.

Không nên gọi 138 là số ngày lịch vì ngày 22/04 không có quan sát.

---

## 17. Notebook 02: fixed-point filtering trên train

Hai điều kiện:

```text
session length ≥ 2
item frequency ≥ 5
```

### 17.1. Fixed point là gì?

Ví dụ session:

```text
[A, X]
```

Nếu `X` có tần suất dưới 5 và bị xóa:

```text
[A]
```

Session chỉ còn một item và cũng phải xóa. Khi xóa session này, tần suất của `A` có thể giảm. Vì vậy phải tiếp tục kiểm tra cho đến khi không còn thay đổi.

### 17.2. Kết quả trước cold holdout

| Giai đoạn | Event | Session | Item | Category | Prefix-target |
|---|---:|---:|---:|---:|---:|
| Train ngay sau split | 990.708 | 189.774 | 114.768 | 1.192 | 800.934 |
| Train sau fixed-point | 841.402 | 174.691 | 38.237 | 948 | 666.711 |

Con số này cho thấy filtering làm thay đổi catalog rất mạnh. Vì vậy báo cáo phải công khai attrition, không chỉ báo cáo kết quả cuối.

---

## 18. Notebook 02: controlled cold-start 20%

### 18.1. Cold và rare khác nhau thế nào?

- Rare item: vẫn xuất hiện trong train nhưng ít.
- Cold item: có frequency bằng 0 trong train.

Không nên gọi item xuất hiện một hoặc hai lần là cold. Chúng là extremely rare nhưng vẫn có tín hiệu train.

### 18.2. Cách tạo controlled cold

Sau fixed-point, train có 38.237 item đủ điều kiện. Pipeline:

1. Chọn 20% item ở cấp item.
2. Số lượng: `round(0,20 × 38.237) = 7.647`.
3. Phân tầng theo category.
4. Dùng seed 2024 để tái lập.
5. Chỉ lấy từ category có đủ item.
6. Bảo đảm category có cold vẫn còn ít nhất một warm item.
7. Xóa toàn bộ interaction của cold khỏi train.
8. Giữ ID và category của cold trong model catalog.

Kết quả:

| Chỉ tiêu | Giá trị |
|---|---:|
| Catalog trước holdout | 38.237 item |
| Controlled cold | 7.647 item |
| Tỷ lệ thực tế | 19,99895% |
| Controlled cold có future event | 4.630 item |
| Category chứa cold | 629 |
| Category vẫn có warm peer | 629/629 |

Tỷ lệ không thể đúng tuyệt đối 20,00000% vì số item phải là số nguyên.

### 18.3. Vì sao phân tầng theo category?

Nếu lấy ngẫu nhiên hoàn toàn, một số category có thể mất quá nhiều item hoặc không còn warm item để so sánh. Phân tầng giúp controlled cold phân bố theo cấu trúc category và giữ được warm peer trong cùng category.

Điều này cần thiết cho câu hỏi:

> “Cold item nằm ở đâu so với warm item cùng category?”

### 18.4. Lọc lại sau holdout

Xóa 7.647 item làm một số session ngắn đi và làm frequency của warm item khác giảm. Pipeline chạy fixed-point lần nữa:

| Giai đoạn | Event | Session | Item | Category | Prefix-target |
|---|---:|---:|---:|---:|---:|
| Ngay sau xóa controlled cold | 671.642 | 170.305 | 30.590 | 948 | 501.337 |
| Final train | 649.632 | 150.783 | 29.936 | 945 | 498.849 |

Có thêm 654 warm item bị loại do post-holdout attrition. Chúng không phải controlled cold.

### 18.5. Natural temporal cold

Temporal split tự nhiên tạo ra 6.700 item xuất hiện trong validation/test nhưng không xuất hiện trong train gốc.

Phân biệt:

| Loại | Nguồn hình thành | Vai trò hiện tại |
|---|---|---|
| Controlled cold | Chủ động holdout từ catalog train | Minh họa embedding chưa huấn luyện |
| Natural temporal cold | Chỉ xuất hiện sau mốc train | Audit hiện tượng cold tự nhiên |

Với mô hình chỉ dùng item ID, natural cold không có đủ thông tin để được gợi ý có ý nghĩa. Muốn giải quyết cold-start thực sự cần side information hoặc cơ chế mô hình khác.

### 18.6. Giới hạn quan trọng

Tài liệu đề tài có nhắc “Mục 1.6/Option B” nhưng phần định nghĩa chi tiết không có trong nội dung được cung cấp. Giao thức trên là giả định minh bạch dựa vào yêu cầu cold-start 20% và embedding cold chưa huấn luyện.

Nếu nhận được protocol chính thức khác, phải cập nhật notebook và chạy lại.

---

## 19. Notebook 02: warm evaluation splits

Validation/test chính chỉ giữ final warm item và các session còn ít nhất 2 event.

Kết quả cuối:

| Tập | Event | Session | Item | Category | Prefix-target |
|---|---:|---:|---:|---:|---:|
| Train | 649.632 | 150.783 | 29.936 | 945 | 498.849 |
| Validation warm | 39.450 | 9.622 | 12.227 | 704 | 29.828 |
| Test warm | 52.792 | 12.791 | 13.847 | 738 | 40.001 |

### Tại sao main evaluation chỉ dùng warm item?

GRU4Rec, NARM và SASRec baseline hiện tại là item-ID-based. Nếu item chưa xuất hiện trong train, mô hình không có dữ liệu để học embedding của nó.

Đưa cold target vào Recall/NDCG chính sẽ trộn hai câu hỏi:

1. Mô hình dự đoán next-item warm tốt đến đâu?
2. Mô hình giải quyết cold-start tốt đến đâu?

Ba baseline hiện tại chỉ được thiết kế hợp lệ cho câu hỏi thứ nhất. Vì vậy accuracy chính dùng warm target; cold được phân tích riêng.

Notebook vẫn giữ `validation_events_full.parquet` và `test_events_full.parquet` để audit, tránh làm mất dấu cold item tự nhiên.

---

## 20. Notebook 02: item mapping và nhóm popularity

### 20.1. Item mapping

Một mapping chung được tạo cho cả ba mô hình:

- Internal item ID bắt đầu từ 1.
- ID 0 dành cho padding.
- Raw item ID và internal ID đều duy nhất.
- Mỗi item có category.
- Mapping chứa final warm item và controlled cold.

Mục tiêu là bảo đảm khi so sánh embedding theo cùng item ID, ba mô hình thực sự đang nói về cùng một sản phẩm.

### 20.2. Popularity phải tính từ đâu?

Popularity chính thức được tính chỉ từ final train:

```text
train_frequency(item) = số lần item xuất hiện trong final train
```

Nếu dùng validation/test để tính nhóm, ta đã sử dụng thông tin tương lai và gây leakage.

### 20.3. Các nhóm cuối

| Nhóm | Điều kiện | Item | Train interaction | Median frequency |
|---|---|---:|---:|---:|
| Cold | frequency = 0 | 7.647 | 0 | 0 |
| Rare | frequency ≤ 6 | 6.884 | 37.525 | 5 |
| Medium | frequency 7–26 | 16.940 | 218.091 | 11 |
| Popular | frequency ≥ 27 | 6.112 | 394.016 | 45 |

Popular có ít item hơn medium nhưng nắm phần lớn train interaction. Long-tail vẫn tồn tại sau preprocessing.

### 20.4. Vì sao nhóm không đúng chính xác 20%?

Các item có thể có cùng frequency tại ranh giới phân vị. Pipeline không tách ngẫu nhiên những item bằng nhau chỉ để đạt đúng 20%, vì như vậy hai item có cùng popularity lại bị gán hai nhãn khác nhau.

Do đó dùng tie-aware quantiles. Ngưỡng cuối cùng là:

- Rare: frequency không quá 6.
- Popular: frequency từ 27 trở lên.

---

## 21. Notebook 02: tạo dữ liệu huấn luyện

### 21.1. Session sequence

Mỗi session được lưu dưới dạng chuỗi item đã sắp đúng thứ tự.

### 21.2. All-prefix examples

Với session:

```text
[A, B, C, D]
```

tạo:

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

### 21.3. Last-item examples

Mỗi session chỉ tạo một bài toán dự đoán item cuối:

```text
[A, B, C] → D
```

| Tập | Last-item examples |
|---|---:|
| Train | 150.783 |
| Validation | 9.622 |
| Test | 12.791 |

Lưu cả hai dạng giúp các notebook mô hình dùng đúng giao thức mà không cần chia dữ liệu lại.

---

## 22. Dữ liệu RecBole là gì?

RecBole là thư viện hỗ trợ huấn luyện và đánh giá recommender system.

Dữ liệu RecBole trong dự án không phải dataset mới. Đó là Diginetica đã xử lý, được chuyển sang định dạng atomic của RecBole.

Các file:

```text
diginetica.train.inter
diginetica.valid.inter
diginetica.test.inter
diginetica.item
```

Vai trò:

- `.inter`: session, item và timestamp của interaction.
- `.item`: internal ID, raw item ID, category và cờ cold.

Khi huấn luyện phải cấu hình RecBole sử dụng đúng ba file benchmark này. Không được để thư viện tự chia ngẫu nhiên dữ liệu lần nữa.

Ngoài RecBole, pipeline còn xuất Parquet để:

- Kiểm tra dữ liệu.
- Tính metric tùy chỉnh.
- Phân tích popularity.
- Ghép embedding với item/category/frequency.

---

## 23. Kiểm tra tính đúng đắn của notebook 02

Notebook chạy 23 audit và tất cả đều đạt.

### 23.1. Input và thứ tự

- Ba SHA-256 khớp manifest EDA.
- Event trong mỗi session đã đúng thứ tự.
- Mọi session cuối có ít nhất 2 event.

### 23.2. Temporal split

- Train và validation không trùng session.
- Train và test không trùng session.
- Validation và test không trùng session.
- Mọi train session nằm trước validation.
- Mọi validation session nằm trước test.

### 23.3. Cold-start

- Có đúng 7.647 controlled cold item.
- Controlled cold không xuất hiện trong train.
- Mọi cold category vẫn có warm peer.
- Validation/test chính chỉ chứa warm item.

### 23.4. Mapping và popularity

- Internal ID duy nhất.
- Raw ID duy nhất.
- Internal ID bắt đầu từ 1.
- Không item nào thiếu category.
- Mọi final warm item có frequency ít nhất 5.
- Tổng train frequency bằng đúng 649.632 train event.

Ngoài audit trong notebook, ba file kiểm thử độc lập kiểm tra split, mapping và cold leakage. Kết quả:

```text
11/11 pytest passed
```

Điều này không chứng minh mọi quyết định nghiên cứu là tối ưu, nhưng chứng minh implementation tuân thủ đúng giao thức đã chốt.

---

## 24. Kết quả tổng hợp sau notebook 01 và 02

### 24.1. Những gì đã biết chắc từ dữ liệu

1. Diginetica có long-tail mạnh.
2. Dữ liệu raw phải được sắp lại theo `timeframe`.
3. Session length 1 chiếm tỷ lệ lớn và không tạo được next-item target.
4. Click lặp là hiện tượng đáng kể và được giữ.
5. Session qua nửa đêm là session ngắn, không phải phiên dài bất thường.
6. Category có liên hệ mạnh với chuyển tiếp trong session.
7. Ngưỡng lọc ảnh hưởng mạnh đến catalog rare item.
8. Final train vẫn có phân hóa rõ rare–medium–popular.
9. Controlled cold 20% đã được tạo và kiểm tra không rò rỉ vào train.
10. Dữ liệu cuối đã sẵn sàng cho ba mô hình.

### 24.2. Những quyết định chính thức đã chốt

- Sort theo `session_id`, `timeframe`, `row_order`.
- Giữ click lặp.
- Giữ session qua nửa đêm.
- Minimum session length bằng 2.
- Minimum item frequency bằng 5.
- Lọc fixed-point.
- Temporal split 7 ngày validation và 7 ngày test.
- Controlled cold 20%, seed 2024, phân tầng category.
- Popularity chỉ lấy từ final train.
- Main accuracy evaluation chỉ dùng warm target.
- Cùng mapping và split cho GRU4Rec, NARM, SASRec.

### 24.3. Kết quả cuối đáng nhớ nhất

| Nội dung | Giá trị |
|---|---:|
| Raw interactions | 1.235.380 |
| Raw sessions | 310.324 |
| Raw active items | 122.993 |
| Input-data Gini | 0,726 |
| Item raw có frequency dưới 5 | 63,80% |
| Pre-holdout filtered train catalog | 38.237 |
| Controlled cold | 7.647 |
| Natural temporal cold | 6.700 |
| Final warm train items | 29.936 |
| Final train sessions | 150.783 |
| Final train interactions | 649.632 |
| Train prefix-target examples | 498.849 |
| Validation prefix-target examples | 29.828 |
| Test prefix-target examples | 40.001 |
| Audit | 23/23 passed |
| Pytest | 11/11 passed |

---

## 25. Những gì chưa thể kết luận

Cho đến khi ba mô hình được huấn luyện, chưa thể trả lời:

- Mô hình nào có Recall/NDCG/MRR cao nhất.
- Mô hình nào có Coverage thấp nhất hoặc recommendation Gini cao nhất.
- GRU4Rec, NARM hay SASRec thiên lệch popularity mạnh nhất.
- Tần suất train có tương quan với embedding norm không.
- Item cùng category có gần nhau trong embedding đã học không.
- Cold item nằm ở đâu trên t-SNE của từng mô hình.
- Norm embedding có đi kèm với rec_count không.
- Kết quả có ổn định qua hai seed không.

Không nên trình bày các câu hỏi này như kết quả đã có.

---

## 26. Bước tiếp theo của dự án

### 26.1. Huấn luyện ba mô hình

- GRU4Rec.
- NARM.
- SASRec.
- Hai seed cho mỗi mô hình.
- Lưu checkpoint tốt nhất theo validation.
- Lưu cấu hình và log để tái lập.

### 26.2. Ràng buộc cold trong huấn luyện

Controlled cold phải:

- Không xuất hiện như positive target.
- Không được lấy làm negative sample.
- Không nhận gradient.
- Có embedding khởi tạo được lưu để đối chiếu.

Nếu cold vẫn được negative sampling, embedding của nó đã được cập nhật và không còn là embedding “chưa huấn luyện”.

### 26.3. Đánh giá accuracy

- Recall@20.
- MRR@20.
- NDCG@20.

### 26.4. Đánh giá popularity bias

- Coverage@20.
- Recommendation Gini.
- ARP.
- Recommendation count theo item.
- Exposure theo rare/medium/popular.

### 26.5. Phân tích embedding

- Trích xuất embedding từ checkpoint tốt nhất.
- Tính norm từng item.
- Spearman giữa train frequency và norm.
- So sánh norm rare/medium/popular.
- So sánh cosine cùng category và khác category bằng sampling cân bằng.
- Dùng t-SNE để minh họa, không dùng làm bằng chứng định lượng chính.

### 26.6. Liên hệ embedding với đầu ra

Tạo bảng theo item:

```text
item_id
model
seed
train_frequency
popularity_group
embedding_norm
recommendation_count
exposure_share
category_id
```

Sau đó kiểm tra:

- `train_frequency ↔ recommendation_count`.
- `train_frequency ↔ embedding_norm`.
- `embedding_norm ↔ recommendation_count`.
- Quan hệ trong từng nhóm frequency.
- Khác biệt giữa ba mô hình và hai seed.

---

## 27. Mạch nói ngắn gọn khi thuyết trình

### Phần mở đầu

> “Đề tài của chúng tôi nghiên cứu popularity bias trong ba mô hình session-based recommendation. Không chỉ đo mô hình gợi ý đúng hay sai, chúng tôi còn muốn biết danh sách gợi ý có tập trung vào item phổ biến không và hiện tượng đó có đi kèm với khác biệt trong không gian embedding hay không.”

### Giới thiệu dữ liệu

> “Chúng tôi dùng Diginetica, gồm hơn 1,2 triệu interaction, hơn 310 nghìn session và gần 123 nghìn item có tương tác. Dataset có session, thời gian và category, phù hợp cho cả huấn luyện SBR lẫn phân tích embedding.”

### Kết quả EDA chính

> “EDA cho thấy dữ liệu có ba đặc điểm quan trọng. Thứ nhất, 58,21% session không đúng thứ tự nếu đọc theo dòng raw. Thứ hai, 29,23% session chỉ có một click và không tạo được target. Thứ ba, dữ liệu có long-tail mạnh: 63,8% item xuất hiện dưới 5 lần và Gini đầu vào bằng 0,726.”

### Category

> “Category có tín hiệu hành vi rõ: sau khi bỏ self-repeat, 87,45% chuyển tiếp liên tiếp vẫn cùng category, trong khi baseline ngẫu nhiên chỉ 1,08%. Đây là cơ sở để so sánh cosine embedding cùng và khác category sau khi huấn luyện.”

### Chuyển từ EDA sang preprocessing

> “Từ các phát hiện đó, notebook 02 sắp lại event, loại session không có target, giữ click lặp, giữ session qua nửa đêm, chia theo thời gian và tính popularity chỉ từ train để tránh leakage.”

### Temporal split

> “Train kết thúc ngày 18/05, validation là tuần 19–25/05 và test là tuần 26/05–01/06. Toàn bộ session được gán theo ngày của event cuối nên không session nào bị cắt qua nhiều tập.”

### Filtering

> “Chúng tôi dùng session length tối thiểu 2 và item frequency tối thiểu 5, lọc lặp đến điểm ổn định. Ngưỡng 5 là baseline chứ chưa được khẳng định tối ưu, vì EDA cho thấy nó đánh đổi mạnh với số rare item được giữ.”

### Cold-start

> “Sau lọc, train có 38.237 item. Chúng tôi chọn 7.647 item, tương đương gần 20%, làm controlled cold, phân tầng theo category và dùng seed 2024. Interaction của chúng bị xóa khỏi train nhưng ID và category vẫn được giữ để minh họa embedding chưa huấn luyện.”

### Kết quả preprocessing

> “Final train có 649.632 interaction, 150.783 session và 29.936 warm item. Validation và test chính chỉ chứa warm target để đánh giá công bằng các mô hình item-ID-based. Pipeline vượt 23 audit và 11 kiểm thử độc lập.”

### Kết thúc phần đã làm

> “Như vậy, hai notebook đầu đã tạo một dữ liệu chung, không leakage và có nhóm cold/rare/medium/popular rõ ràng. Bước tiếp theo là huấn luyện GRU4Rec, NARM và SASRec trên chính split này, sau đó đo accuracy, popularity bias và hình học embedding.”

---

## 28. Các câu hỏi phản biện có thể gặp

### Câu 1: Tại sao chọn Diginetica?

Vì Diginetica có chuỗi session, thời gian, category và long-tail rõ. Nó phù hợp đồng thời với dự đoán next-item, đo popularity bias và phân tích embedding theo category.

### Câu 2: Tại sao loại session length 1?

Vì một session chỉ có một item không thể tạo cặp input–target để dự đoán item tiếp theo.

### Câu 3: Tại sao không xóa click lặp?

Click lặp không phải duplicate kỹ thuật. Nó có thể thể hiện người dùng quay lại sản phẩm. EDA cho thấy click lặp đủ phổ biến nên tự ý xóa sẽ làm thay đổi hành vi thật.

### Câu 4: Tại sao item dưới 5 lần bị loại mà không gọi là cold?

Item xuất hiện 1–4 lần vẫn là warm vì đã có interaction trong train. Chúng chỉ là extremely rare. Cold được định nghĩa có frequency bằng 0 trong train. Lọc item cực hiếm nhằm giảm vocabulary quá thưa và giúp embedding có đủ tín hiệu tối thiểu.

### Câu 5: Tại sao chọn ngưỡng 5?

Ngưỡng 5 được dùng làm baseline phổ biến và cân bằng giữa quy mô dữ liệu với độ ổn định học. Tuy nhiên EDA cho thấy nó cắt nhiều tail item, nên dự án không tuyên bố đây là ngưỡng tối ưu; ngưỡng 2 hoặc 3 là thí nghiệm mở rộng hợp lý.

### Câu 6: Tại sao phải lọc fixed-point?

Xóa item hiếm có thể làm session còn một event. Xóa session đó lại làm frequency của item khác giảm. Phải lặp cho đến khi tất cả item và session đồng thời thỏa điều kiện.

### Câu 7: Tại sao chia theo thời gian?

Để mô phỏng học từ quá khứ và dự đoán tương lai. Chia ngẫu nhiên có thể làm thông tin tương lai lọt vào train.

### Câu 8: Tại sao validation/test chỉ có 7 ngày?

Đây là baseline thiết kế. Mỗi tập bao phủ đủ bảy thứ trong tuần và vẫn còn lần lượt 29.828 và 40.001 mẫu đánh giá sau lọc. Tuy nhiên dự án chưa chứng minh 7 ngày tối ưu; có thể bổ sung sensitivity 14 ngày.

### Câu 9: Tại sao gán split theo ngày cuối session?

Để toàn bộ session nằm trong cùng một split. Nếu chia từng event theo ngày, session qua nửa đêm có thể bị cắt làm hai và gây leakage hoặc mất ngữ cảnh.

### Câu 10: Tại sao controlled cold phải phân tầng category?

Để cold không tập trung bất thường vào một vài category và để mỗi category chứa cold vẫn có warm item làm đối chứng.

### Câu 11: Tại sao không đánh giá cold bằng Recall/NDCG chính?

Ba baseline chỉ dùng item ID. Item không xuất hiện trong train không có embedding đã học, nên đánh giá cold chung với warm sẽ không phản ánh công bằng chất lượng next-item recommendation. Cold hiện được dùng cho phân tích minh họa.

### Câu 12: Tại sao giữ controlled cold trong mapping?

Nếu loại hoàn toàn khỏi mapping thì không còn vector để quan sát. Giữ ID/category nhưng chặn interaction và gradient cho phép minh họa embedding khởi tạo chưa học.

### Câu 13: Gini 0,726 có phải bias của mô hình không?

Không. Đó là Gini của interaction raw. Bias đầu ra phải được tính lại từ tần suất item xuất hiện trong toàn bộ danh sách Top-20 của từng mô hình.

### Câu 14: Tại sao Gini giảm sau khi lọc?

Vì nhiều item ở phần tail bị xóa khỏi catalog, làm phân phối trên số item còn lại trông đồng đều hơn. Đây là hiệu ứng thay đổi mẫu số, không phải bằng chứng mô hình giảm bias.

### Câu 15: Dữ liệu RecBole có phải dataset khác không?

Không. Đó là cùng Diginetica đã được preprocessing, chỉ được chuyển sang định dạng mà RecBole có thể đọc.

### Câu 16: Vì sao category được dùng trong nghiên cứu?

EDA cho thấy 87,45% chuyển tiếp sau khi loại self-repeat vẫn cùng category, cao hơn rất nhiều baseline 1,08%. Vì vậy category có ý nghĩa hành vi và phù hợp làm tiêu chí kiểm tra cosine embedding.

### Câu 17: 23 audit và 11 test chứng minh điều gì?

Chúng chứng minh implementation tuân thủ giao thức: không trùng split, đúng thời gian, không cold trong train, mapping hợp lệ và popularity chỉ từ train. Chúng không chứng minh các hyperparameter hoặc cửa sổ thời gian là tối ưu.

### Câu 18: Đề tài có đề xuất phương pháp khử bias chưa?

Chưa. Trọng tâm hiện tại là chẩn đoán và so sánh. Chỉ sau khi xác định mô hình nào bias hơn và cơ chế embedding nào đi kèm bias mới có cơ sở đề xuất re-ranking, regularization, reweighting hoặc sampling cân bằng trong phần mở rộng.

---

## 29. Những lỗi diễn đạt cần tránh khi thuyết trình

Không nói:

> “Gini của mô hình là 0,726.”

Nên nói:

> “Gini của phân phối interaction đầu vào là 0,726.”

Không nói:

> “Ngưỡng 5 là tốt nhất.”

Nên nói:

> “Ngưỡng 5 là baseline đã chốt; EDA cho thấy có trade-off mạnh với long-tail.”

Không nói:

> “Cold item là item xuất hiện dưới 5 lần.”

Nên nói:

> “Rare item có frequency dương nhưng thấp; cold item có frequency bằng 0 trong train.”

Không nói:

> “t-SNE chứng minh các cụm item.”

Nên nói:

> “t-SNE được dùng để minh họa; kết luận định lượng dựa trên norm, Spearman và cosine.”

Không nói:

> “Norm embedding gây ra popularity bias.”

Nên nói:

> “Norm embedding có thể có mối liên hệ với train frequency và recommendation count.”

Không nói:

> “Bảy ngày là khoảng chia tối ưu.”

Nên nói:

> “Bảy ngày là baseline hợp lý, có cùng cấu trúc lịch và đủ số mẫu trong dữ liệu hiện tại.”

Không nói:

> “Chúng tôi đã so sánh ba mô hình.”

Nên nói:

> “Dữ liệu đã sẵn sàng để huấn luyện và so sánh ba mô hình ở bước tiếp theo.”

---

## 30. Kết luận cuối cùng ở trạng thái hiện tại

Hai notebook đầu đã hoàn thành phần nền tảng quan trọng nhất trước huấn luyện:

1. Xác minh ba file Diginetica đủ và liên kết đúng.
2. Phát hiện và sửa vấn đề thứ tự chuỗi.
3. Hiểu rõ cấu trúc session, click lặp và session qua ngày.
4. Chứng minh dữ liệu có long-tail mạnh và phù hợp nghiên cứu popularity bias.
5. Chứng minh category có tín hiệu hành vi rõ.
6. Phân tích trade-off của ngưỡng lọc item.
7. Chia dữ liệu theo thời gian để hạn chế leakage.
8. Tạo controlled cold 20% và nhận diện natural cold.
9. Tạo final warm train/validation/test.
10. Tính các nhóm popularity chỉ từ train.
11. Tạo sequence và next-item examples.
12. Xuất Parquet, mapping và dữ liệu RecBole.
13. Xác nhận pipeline bằng audit và test tự động.

Kết luận ngắn gọn nhất có thể dùng ở cuối phần thuyết trình:

> “Diginetica có long-tail mạnh, category có tín hiệu hành vi và đủ phù hợp để nghiên cứu popularity bias. Sau EDA, chúng tôi đã xây dựng một pipeline preprocessing theo thời gian, không trộn session, kiểm soát cold-start và tránh dùng thông tin validation/test để định nghĩa popularity. Dữ liệu cuối đã được kiểm tra và sẵn sàng để huấn luyện GRU4Rec, NARM và SASRec trên cùng một giao thức.”

---

## 31. Tài liệu và file kết quả liên quan

### Notebook

- [`01_data_exploration.ipynb`](../../notebooks/01_data_exploration.ipynb)
- [`02_data_preprocessing.ipynb`](../../notebooks/02_data_preprocessing.ipynb)

### Báo cáo kỹ thuật

- [`BAO-CAO-EDA-DIGINETICA.md`](../report/BAO-CAO-EDA-DIGINETICA.md)
- [`BAO-CAO-TIEN-XU-LY-DIGINETICA.md`](../report/BAO-CAO-TIEN-XU-LY-DIGINETICA.md)

### Bảng chính

- [`eda_conclusions.csv`](../../results/tables/eda_conclusions.csv)
- [`eda_frequency_threshold_sensitivity.csv`](../../results/tables/eda_frequency_threshold_sensitivity.csv)
- [`preprocessing_base_split_statistics.csv`](../../results/tables/preprocessing_base_split_statistics.csv)
- [`preprocessing_attrition_official.csv`](../../results/tables/preprocessing_attrition_official.csv)
- [`preprocessing_cold_start_summary.csv`](../../results/tables/preprocessing_cold_start_summary.csv)
- [`preprocessing_final_split_statistics.csv`](../../results/tables/preprocessing_final_split_statistics.csv)
- [`preprocessing_popularity_group_summary.csv`](../../results/tables/preprocessing_popularity_group_summary.csv)
- [`preprocessing_checks.csv`](../../results/tables/preprocessing_checks.csv)

### Dữ liệu đã xử lý

- [`data/processed/diginetica`](../../data/processed/diginetica)
- [`data/processed/recbole/diginetica`](../../data/processed/recbole/diginetica)
