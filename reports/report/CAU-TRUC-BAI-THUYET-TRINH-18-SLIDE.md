# Cấu trúc bài thuyết trình BTL-K03 — 18 slide

## Định hướng chung

- **Số slide:** 18, dưới giới hạn 20 slide.
- **Mạch nội dung:** mở đầu → vấn đề → phân tích dữ liệu → giải pháp preprocessing → kết quả → bước tiếp theo → kết luận.
- **Phần đã hoàn thành:** EDA và preprocessing Diginetica.
- **Phần chưa hoàn thành:** huấn luyện GRU4Rec/NARM/SASRec, đánh giá output bias và phân tích embedding.
- **Nguyên tắc trình bày:** mỗi slide chỉ có một thông điệp chính; số liệu lớn nên được thể hiện bằng biểu đồ, timeline hoặc sơ đồ thay vì đoạn văn.

---

## Slide 1 — Tiêu đề và câu hỏi trung tâm

### Tiêu đề

**Thiên lệch độ phổ biến và hình học không gian embedding trong mô hình SBR**

### Thông điệp chính

Đề tài nghiên cứu liệu các mô hình gợi ý theo session có ưu tiên item phổ biến và hiện tượng đó có đi kèm với khác biệt trong item embedding hay không.

### 3–5 ý quan trọng

- Dataset: Diginetica.
- Ba mô hình dự kiến: GRU4Rec, NARM và SASRec.
- Phân tích hai mặt: danh sách gợi ý đầu ra và embedding bên trong mô hình.
- Trạng thái hiện tại: đã hoàn thành EDA và preprocessing.

### Hình ảnh minh họa đề xuất

- Sơ đồ ngắn: `Session → Mô hình → Top-20 → Popularity bias / Embedding`.
- Một chuỗi sản phẩm đơn giản dẫn đến danh sách gợi ý.

---

## Slide 2 — Bài toán Session-Based Recommendation

### Tiêu đề

**Từ chuỗi hành vi đến dự đoán item tiếp theo**

### Thông điệp chính

SBR dự đoán item tiếp theo chỉ từ chuỗi hành vi trong session hiện tại.

### 3–5 ý quan trọng

- Một session là chuỗi item theo đúng thứ tự thời gian.
- Đầu vào ví dụ: `[A, B, C]`; target: `D`.
- Một session dài có thể sinh nhiều mẫu prefix–target.
- Thứ tự click sai sẽ làm mô hình học sai quan hệ chuyển tiếp.

### Hình ảnh minh họa đề xuất

- Minh họa chuỗi bốn sản phẩm: `A → B → C → D`.
- Ba mẫu: `[A]→B`, `[A,B]→C`, `[A,B,C]→D`.

---

## Slide 3 — Vấn đề popularity bias

### Tiêu đề

**Mô hình có đang khuếch đại lợi thế của item phổ biến?**

### Thông điệp chính

Accuracy cao chưa đủ; cần biết cơ hội xuất hiện trong danh sách gợi ý được phân phối cho catalog như thế nào.

### 3–5 ý quan trọng

- Một số ít item thường nhận phần lớn interaction.
- Mô hình có thể tiếp tục ưu tiên các item đã phổ biến.
- Popularity bias có thể tạo vòng lặp “phổ biến → được gợi ý → càng phổ biến”.
- Cần báo cáo đồng thời accuracy và mức tập trung của Top-20.

### Hình ảnh minh họa đề xuất

- Sơ đồ vòng lặp popularity feedback loop.
- Hình head–tail: ít popular item ở đầu, nhiều rare item ở đuôi.

---

## Slide 4 — Ý tưởng và câu hỏi nghiên cứu

### Tiêu đề

**Kết nối đầu ra gợi ý với cấu trúc embedding**

### Thông điệp chính

Đề tài không chỉ đo bias mà còn tìm dấu hiệu hình học có thể đi kèm với bias.

### 3–5 ý quan trọng

- Item phổ biến trong train có được gợi ý nhiều hơn không?
- Train frequency có tương quan với embedding norm không?
- Item cùng category có cosine similarity cao hơn item khác category không?
- Embedding norm có đi kèm với recommendation count không?
- Kết luận là mối liên hệ, không khẳng định quan hệ nhân quả.

### Hình ảnh minh họa đề xuất

- Sơ đồ tam giác: `Train frequency ↔ Embedding norm ↔ Recommendation count`.
- Các điểm embedding cùng màu category trong không gian hai chiều minh họa.

---

## Slide 5 — Dữ liệu Diginetica

### Tiêu đề

**Diginetica: dữ liệu phù hợp cho cả SBR và phân tích long-tail**

### Thông điệp chính

Diginetica có quy mô đủ lớn, có session, thời gian và category để trả lời các câu hỏi của đề tài.

### 3–5 ý quan trọng

- 1.235.380 interaction.
- 310.324 session và 122.993 item có tương tác.
- 1.217 category trong metadata.
- 152 ngày có quan sát, từ 01/01 đến 01/06/2016.
- Ba file chính: views, product categories và products.

### Hình ảnh minh họa đề xuất

- Ba khối file liên kết qua `itemId`.
- Bốn KPI lớn: interaction, session, item, category.

---

## Slide 6 — Mục tiêu của EDA

### Tiêu đề

**EDA để ra quyết định, không chỉ để mô tả dữ liệu**

### Thông điệp chính

Notebook 01 được dùng để xác định các quy tắc preprocessing có căn cứ dữ liệu.

### 3–5 ý quan trọng

- Kiểm tra schema, missing, duplicate và liên kết metadata.
- Kiểm tra thứ tự thời gian và cấu trúc session.
- Xác định long-tail và mức tập trung popularity.
- Kiểm tra category có mang tín hiệu hành vi không.
- Mô phỏng ảnh hưởng của các ngưỡng lọc item.

### Hình ảnh minh họa đề xuất

- Sơ đồ năm câu hỏi EDA dẫn vào một khối “Preprocessing decisions”.
- Có thể dùng checklist thay cho bảng số liệu.

---

## Slide 7 — Chất lượng dữ liệu và thứ tự click

### Tiêu đề

**Dữ liệu liên kết tốt nhưng thứ tự raw không đáng tin cậy**

### Thông điệp chính

Vấn đề quan trọng nhất không phải missing hay duplicate, mà là thứ tự event trong session.

### 3–5 ý quan trọng

- Không có duplicate chính xác hoặc item thiếu category/product.
- `userId` thiếu 69,81% nhưng không cản trở bài toán dựa trên session.
- 58,21% session không tăng theo `timeframe` trong thứ tự raw.
- Có 4 nhóm trùng `timeframe`; dùng thứ tự dòng gốc để phá hòa.
- Quy tắc: `sessionId → timeframe → row_order`.

### Hình ảnh minh họa đề xuất

- Hai chuỗi “trước sort” và “sau sort”.
- Callout lớn `58,21%` cùng mũi tên “bắt buộc stable sort”.

---

## Slide 8 — Cấu trúc session và click lặp

### Tiêu đề

**Session ngắn chiếm đa số, nhưng click lặp vẫn là tín hiệu hành vi**

### Thông điệp chính

Session length quyết định khả năng tạo target; click lặp không nên bị xóa tùy ý.

### 3–5 ý quan trọng

- Median session length bằng 3.
- 29,23% session chỉ có một click và không tạo được next-item target.
- 27,85% session có item được click lặp.
- 13,62% event là click lặp và được giữ lại.
- Session trên 10 event chỉ chiếm 6,59% nhưng tạo 24,72% event.

### Hình ảnh minh họa đề xuất

- Biểu đồ cột nhóm độ dài session.
- Hai callout: “bỏ length 1” và “giữ click lặp”.

---

## Slide 9 — Popularity và long-tail

### Tiêu đề

**Diginetica có long-tail và mức tập trung đầu vào rất mạnh**

### Thông điệp chính

Dataset cung cấp điều kiện phù hợp để nghiên cứu popularity bias, nhưng chưa chứng minh mô hình có bias.

### 3–5 ý quan trọng

- 34,20% item chỉ xuất hiện đúng một lần.
- 63,80% item xuất hiện dưới 5 lần.
- Top 20% item nắm 77,54% interaction.
- Input-data Gini bằng 0,726.
- Gini 0,726 là của dữ liệu đầu vào, không phải recommendation Gini.

### Hình ảnh minh họa đề xuất

- Lorenz curve hoặc biểu đồ rank–frequency log-log từ EDA.
- Khối so sánh `63,80% item` với `77,54% interaction của top 20%`.

---

## Slide 10 — Category và biến gây nhiễu

### Tiêu đề

**Category có tín hiệu mạnh, nhưng tuổi item cần được lưu ý**

### Thông điệp chính

Category phù hợp để phân tích cosine embedding, trong khi tuổi item làm giới hạn diễn giải quan hệ frequency–norm.

### 3–5 ý quan trọng

- 80,95% session chỉ có một category.
- 87,45% chuyển tiếp vẫn cùng category sau khi bỏ self-repeat.
- Baseline cùng category ngẫu nhiên chỉ 1,08%.
- Spearman giữa tuổi quan sát và tần suất item bằng 0,504.
- Phân tích embedding phải nói “liên hệ”, không kết luận frequency là nguyên nhân.

### Hình ảnh minh họa đề xuất

- Biểu đồ so sánh `87,45%` và `1,08%`.
- Sơ đồ `Item age ↔ Frequency ↔ Embedding norm?`.

---

## Slide 11 — Trade-off của ngưỡng lọc

### Tiêu đề

**Lọc item giúp học ổn định hơn nhưng cắt mạnh phần long-tail**

### Thông điệp chính

Ngưỡng 5 là baseline có đánh đổi rõ ràng, không phải giá trị tối ưu đã được chứng minh.

### 3–5 ý quan trọng

- Ngưỡng 2 giữ 78.989 item và 88,92% event.
- Ngưỡng 3 giữ 60.454 item và 85,64% event.
- Ngưỡng 5 giữ 42.171 item và 80,07% event.
- Ở ngưỡng 5 chỉ còn 34,29% item raw.
- Gini sau lọc giảm do cắt tail, không phải do mô hình giảm bias.

### Hình ảnh minh họa đề xuất

- Biểu đồ trade-off ngưỡng–event–item từ EDA.
- Một chiếc cân: “embedding ổn định” và “bảo toàn rare item”.

---

## Slide 12 — Từ EDA đến giải pháp preprocessing

### Tiêu đề

**Mỗi phát hiện EDA được chuyển thành một quyết định xử lý**

### Thông điệp chính

Notebook 02 là lời giải có hệ thống cho các vấn đề được phát hiện trong notebook 01.

### 3–5 ý quan trọng

- Sai thứ tự → stable sort theo `timeframe` và `row_order`.
- Session length 1 → loại; click lặp → giữ.
- Session qua nửa đêm → giữ nguyên và chia theo ngày cuối session.
- Long-tail quá thưa → minimum frequency 5 và lọc fixed-point.
- Nguy cơ leakage → popularity chỉ tính từ final train.

### Hình ảnh minh họa đề xuất

- Bảng hai cột `Vấn đề EDA → Quyết định preprocessing`.
- Flowchart màu từ đỏ/vấn đề sang xanh/giải pháp.

---

## Slide 13 — Temporal split

### Tiêu đề

**Học từ quá khứ, đánh giá trên tương lai**

### Thông điệp chính

Chia theo thời gian ở cấp session giúp mô phỏng sử dụng thực tế và tránh temporal leakage.

### 3–5 ý quan trọng

- Train: 01/01–18/05/2016.
- Validation: 19/05–25/05/2016.
- Test: 26/05–01/06/2016.
- Toàn bộ session được gán theo ngày của event cuối cùng.
- Cửa sổ 7 ngày là baseline hợp lý, chưa được chứng minh tối ưu.

### Hình ảnh minh họa đề xuất

- Timeline ba màu train/validation/test.
- Dưới timeline ghi số event ban đầu: 990.708 / 65.533 / 88.445.

---

## Slide 14 — Fixed-point filtering và attrition

### Tiêu đề

**Lọc lặp đến khi đồng thời thỏa điều kiện item và session**

### Thông điệp chính

Một lượt lọc là chưa đủ vì xóa item và xóa session tác động qua lại lẫn nhau.

### 3–5 ý quan trọng

- Điều kiện: session length ≥ 2 và item frequency ≥ 5.
- Xóa item hiếm có thể làm session chỉ còn một event.
- Xóa session lại có thể làm frequency của item khác giảm.
- Train giảm từ 114.768 xuống 38.237 item trước cold holdout.
- Event train còn 841.402 và session còn 174.691.

### Hình ảnh minh họa đề xuất

- Vòng lặp `lọc item → lọc session → kiểm tra lại`.
- Funnel chart từ raw train đến pre-cold filtered train.

---

## Slide 15 — Cold-start 20%

### Tiêu đề

**Controlled cold để quan sát embedding chưa được huấn luyện**

### Thông điệp chính

Cold được tạo có kiểm soát ở cấp item và được tách rõ khỏi natural temporal cold.

### 3–5 ý quan trọng

- Chọn 7.647/38.237 item, tương đương 19,99895%.
- Phân tầng theo category, seed 2024.
- Xóa toàn bộ interaction của controlled cold khỏi train nhưng giữ ID/category.
- 629/629 cold category vẫn còn warm peer.
- Có thêm 6.700 natural temporal cold item được theo dõi riêng.

### Hình ảnh minh họa đề xuất

- Catalog item chia 80% warm và 20% controlled cold.
- Hai hộp so sánh “Controlled cold” và “Natural cold”.
- Cảnh báo nhỏ: cold không được negative sampling hoặc nhận gradient.

---

## Slide 16 — Dữ liệu cuối và nhóm popularity

### Tiêu đề

**Một bộ dữ liệu chung đã sẵn sàng cho ba mô hình**

### Thông điệp chính

Final split vừa đủ lớn để huấn luyện/đánh giá, đồng thời giữ định nghĩa popularity không rò rỉ.

### 3–5 ý quan trọng

- Train: 649.632 event, 150.783 session, 29.936 warm item.
- Validation warm: 39.450 event, 9.622 session, 29.828 prefix-target.
- Test warm: 52.792 event, 12.791 session, 40.001 prefix-target.
- Rare: frequency ≤ 6; medium: 7–26; popular: ≥ 27.
- Nhóm item: 7.647 cold, 6.884 rare, 16.940 medium, 6.112 popular.

### Hình ảnh minh họa đề xuất

- Bảng final split ở bên trái.
- Biểu đồ cột số item và interaction theo nhóm ở bên phải.

---

## Slide 17 — Đầu ra, kiểm tra và bước tiếp theo

### Tiêu đề

**Pipeline đã được kiểm tra; bước tiếp theo là huấn luyện và phân tích**

### Thông điệp chính

Dữ liệu đã sẵn sàng và có thể tái lập, nhưng kết luận về mô hình vẫn phải chờ thí nghiệm huấn luyện.

### 3–5 ý quan trọng

- Đã xuất Parquet, sequence, next-item examples, mapping và RecBole atomic files.
- 23/23 audit và 11/11 pytest đều đạt.
- Tiếp theo: huấn luyện GRU4Rec, NARM và SASRec với hai seed.
- Đo Recall/MRR/NDCG cùng Coverage/Gini/ARP trên Top-20.
- Trích xuất embedding để phân tích norm, cosine, t-SNE và recommendation count.

### Hình ảnh minh họa đề xuất

- Pipeline với phần EDA/preprocessing được đánh dấu “Done”.
- Phần train/evaluation/embedding được đánh dấu “Next”.

---

## Slide 18 — Kết luận

### Tiêu đề

**Kết luận: dữ liệu đã đủ điều kiện cho thí nghiệm công bằng**

### Thông điệp chính

EDA đã xác định đúng rủi ro của dữ liệu và preprocessing đã biến chúng thành một giao thức chung, có kiểm tra, cho cả ba mô hình.

### 3–5 ý quan trọng

- Diginetica có long-tail mạnh và phù hợp nghiên cứu popularity bias.
- Category có tín hiệu hành vi đủ mạnh cho phân tích embedding.
- Temporal split và popularity-from-train giúp hạn chế leakage.
- Controlled cold, rare, medium và popular đã được định nghĩa rõ.
- Chưa kết luận mô hình nào bias hơn cho đến khi hoàn thành huấn luyện.

### Hình ảnh minh họa đề xuất

- Sơ đồ kết luận ba khối: `Dữ liệu phù hợp → Pipeline hợp lệ → Sẵn sàng huấn luyện`.
- Dòng cuối nổi bật: “Measure first, explain second, mitigate next.”

---

## Phân bổ nội dung theo logic bài thuyết trình

| Giai đoạn | Slide | Nội dung |
|---|---|---|
| Mở đầu | 1–2 | Đề tài và bài toán SBR |
| Vấn đề nghiên cứu | 3–4 | Popularity bias và ý tưởng phân tích embedding |
| Dữ liệu | 5–6 | Diginetica và mục tiêu EDA |
| Phân tích | 7–11 | Chất lượng, session, long-tail, category và threshold |
| Giải pháp | 12–16 | Preprocessing, split, filtering, cold-start và final data |
| Bàn giao | 17 | Kiểm tra và bước huấn luyện tiếp theo |
| Kết luận | 18 | Những gì đã chứng minh và giới hạn hiện tại |

## Các nguyên tắc cần giữ khi thiết kế slide

1. Không ghi rằng ba mô hình đã được huấn luyện.
2. Không gọi Gini 0,726 là bias của mô hình.
3. Không gọi item dưới 5 lần là cold item.
4. Không nói ngưỡng 5 hoặc cửa sổ 7 ngày là tối ưu.
5. Không dùng t-SNE như bằng chứng định lượng chính.
6. Mọi số liệu popularity chính thức phải được nói là tính từ final train.
7. Controlled cold chỉ minh họa embedding chưa huấn luyện trong giao thức hiện tại.
