# BÁO CÁO FILE 08 — MỐI LIÊN HỆ GIỮA HÌNH HỌC EMBEDDING, POPULARITY BIAS VÀ ACCURACY

## 1. Mục tiêu duy nhất của file 08

File 06 đã cho biết đầu ra của mô hình thiên lệch như thế nào. File 07 cho biết popular, medium, rare và cold item được biểu diễn khác nhau như thế nào trong không gian embedding.

File 08 trả lời câu hỏi nối hai phần trên:

> Những khác biệt trong hình học embedding có đi kèm với việc item được gợi ý nhiều hơn và được dự đoán chính xác hơn hay không?

File 08 không cố chứng minh quan hệ nhân quả. Kết quả chỉ cho biết các đặc điểm có liên hệ với nhau trong sáu thí nghiệm gồm ba mô hình và hai seed.

---

## 2. Dữ liệu được nối như thế nào?

Phân tích sử dụng:

- Ba mô hình: GRU4Rec, NARM và SASRec.
- Hai seed: 2024 và 2025.
- 40.001 truy vấn trong test warm cho mỗi lần chạy.
- Mỗi truy vấn có danh sách Top-20, tương ứng 800.020 vị trí recommendation cho mỗi lần chạy.
- Chỉ phân tích outcome của warm item vì đây là các item đã được học và thuộc candidate catalog của đánh giá chính.

Mỗi warm item được ghép các thông tin:

| Nhóm thông tin | Biến sử dụng |
| --- | --- |
| Popularity đầu vào | `train_frequency`, popular/medium/rare |
| Hình học embedding | Norm, khoảng cách tới tâm category, cosine với tâm category |
| Exposure đầu ra | Số lần item xuất hiện trong toàn bộ Top-20 |
| Accuracy | Số lần item là target và số lần target xuất hiện trong Top-20 |

Cold item không được đưa vào hồi quy exposure/accuracy chính vì embedding của chúng không được học từ interaction train và test chính là test warm.

---

## 3. Ba tầng phân tích

### Tầng 1 — Đặt geometry, exposure và accuracy cạnh nhau theo nhóm

Ta so sánh popular, medium và rare item để xem các đại lượng có thay đổi đồng thời hay không.

### Tầng 2 — Tương quan

Spearman được tính giữa từng đặc trưng geometry với:

- `recommendation_count`: item được gợi ý bao nhiêu lần.
- `item_recall@20`: tỷ lệ dự đoán đúng của từng item khi item đó làm target.

Tương quan được tính trên toàn warm catalog và trong từng nhóm popularity.

### Tầng 3 — Giữ train frequency cố định

Đây là tầng bằng chứng quan trọng nhất, vì frequency vừa liên hệ với embedding vừa liên hệ với recommendation count.

- Exposure: OLS với `log(1 + recommendation_count)`.
- Accuracy: logistic regression cho xác suất `hit@20` của từng truy vấn.
- Mô hình cơ sở chỉ dùng `log(1 + train_frequency)`.
- Sau đó lần lượt thêm norm, khoảng cách tới tâm hoặc cosine với tâm.

Nhờ vậy ta trả lời được:

> Với hai item có mức frequency tương đương, geometry có còn mang thêm thông tin về exposure và khả năng hit@20 không?

---

## 4. Kết quả mô tả theo nhóm popularity

Kết quả dưới đây là trung bình của hai seed.

| Mô hình | Nhóm | Norm | Khoảng cách tới tâm | Exposure Top-20 | Recall@20 |
| --- | --- | ---: | ---: | ---: | ---: |
| GRU4Rec | Popular | 0,998 | 0,700 | 63,96% | 64,70% |
| GRU4Rec | Medium | 1,186 | 0,710 | 31,77% | 45,36% |
| GRU4Rec | Rare | 1,357 | 0,802 | 4,27% | 32,76% |
| NARM | Popular | 1,019 | 0,692 | 64,15% | 65,34% |
| NARM | Medium | 1,110 | 0,718 | 31,69% | 45,86% |
| NARM | Rare | 1,218 | 0,792 | 4,16% | 32,74% |
| SASRec | Popular | 0,830 | 0,577 | 62,49% | 65,36% |
| SASRec | Medium | 0,961 | 0,612 | 32,79% | 48,10% |
| SASRec | Rare | 1,069 | 0,686 | 4,73% | 36,28% |

### Quan sát nhất quán ở cả ba mô hình

Khi đi từ rare đến popular:

- Norm trung bình giảm.
- Item nằm gần tâm category hơn.
- Tỷ trọng exposure tăng rất mạnh.
- Recall@20 tăng rõ rệt.

Như vậy, rare item vừa nằm ở vùng hình học khác popular item, vừa nhận ít exposure hơn và khó được dự đoán đúng hơn.

Tuy nhiên, bảng này mới cho thấy các hiện tượng xuất hiện đồng thời. Nó chưa chứng minh geometry tạo ra bias, vì popular item đồng thời có frequency cao hơn rất nhiều.

---

## 5. Vì sao không được kết luận chỉ từ tương quan thô?

Trên toàn warm catalog, norm và recommendation count có tương quan âm:

| Mô hình | Spearman norm ↔ recommendation count |
| --- | ---: |
| GRU4Rec | -0,488 |
| NARM | -0,398 |
| SASRec | -0,422 |

Nếu chỉ đọc bảng này, ta có thể kết luận nhầm rằng norm lớn khiến item ít được gợi ý.

Nguyên nhân là:

```text
Rare item → thường có norm lớn → nhưng frequency thấp → exposure thấp
Popular item → thường có norm nhỏ → nhưng frequency cao → exposure cao
```

Frequency đang gây nhiễu mối quan hệ norm–exposure. Sau khi giữ frequency cố định, hệ số của norm đổi từ âm sang dương ở cả ba mô hình.

Đây là bằng chứng cho thấy không được sử dụng tương quan tổng thể đơn thuần để giải thích cơ chế.

---

## 6. Geometry có liên hệ với exposure sau khi kiểm soát frequency không?

Các hệ số dưới đây là standardized beta của geometry. Dấu dương nghĩa là geometry tăng đi kèm exposure tăng; dấu âm nghĩa là geometry tăng đi kèm exposure giảm, trong khi frequency được giữ cố định.

| Mô hình | Beta norm | Beta khoảng cách tới tâm | Beta cosine với tâm | ΔR² lớn nhất do geometry |
| --- | ---: | ---: | ---: | ---: |
| GRU4Rec | +0,035 | -0,063 | +0,110 | 0,0100 |
| NARM | +0,070 | -0,056 | +0,110 | 0,0118 |
| SASRec | +0,100 | -0,019 | +0,083 | 0,0066 |

Kết quả nhất quán qua hai seed:

- Norm cao hơn một chút đi kèm exposure cao hơn sau khi giữ frequency cố định.
- Item càng xa tâm category thì exposure càng thấp.
- Item càng cùng hướng với tâm category thì exposure càng cao.
- Trong ba đặc trưng, cosine với tâm category bổ sung nhiều thông tin nhất cho exposure.

Tuy nhiên, mức tăng R² do geometry chỉ từ rất nhỏ đến khoảng 0,0118. Trong khi đó, chỉ riêng frequency đã giải thích khoảng 49%–51% biến thiên của `log(1 + recommendation_count)`.

Kết luận:

> Geometry có liên hệ độc lập với exposure, nhưng frequency vẫn là yếu tố giải thích chính. Geometry là tín hiệu bổ sung, không phải lời giải thích duy nhất của popularity bias.

---

## 7. Geometry có liên hệ với accuracy sau khi kiểm soát frequency không?

Logistic regression được chạy trên 40.001 truy vấn của mỗi lần thí nghiệm. Biến đích là target thật có xuất hiện trong Top-20 hay không.

| Mô hình | Log-odds beta norm | Log-odds beta khoảng cách | Log-odds beta cosine |
| --- | ---: | ---: | ---: |
| GRU4Rec | +0,124 | -0,188 | +0,320 |
| NARM | +0,160 | -0,225 | +0,316 |
| SASRec | +0,249 | -0,259 | +0,388 |

Diễn giải khi frequency được giữ cố định:

- Norm cao hơn đi kèm xác suất hit@20 cao hơn.
- Item xa tâm category hơn đi kèm xác suất hit@20 thấp hơn.
- Item cùng hướng hơn với tâm category đi kèm xác suất hit@20 cao hơn.
- Cosine với tâm category là tín hiệu geometry mạnh nhất trong ba biến đối với accuracy.

Các dấu hệ số giống nhau ở cả ba mô hình và hai seed. Tuy vậy, mức cải thiện log-loss vẫn nhỏ, nên geometry chỉ giải thích thêm một phần khả năng dự đoán đúng.

---

## 8. Geometry gap có giải thích thứ hạng bias giữa ba mô hình không?

| Mô hình | Centroid gap rare−popular | Exposure gap popular−rare | Recall gap popular−rare | Gini | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| GRU4Rec | 0,102 | 59,69 điểm % | 31,94 điểm % | 0,7036 | 88,51% |
| NARM | 0,100 | 59,98 điểm % | 32,60 điểm % | 0,7061 | 88,58% |
| SASRec | 0,109 | 57,76 điểm % | 29,08 điểm % | 0,6885 | 91,76% |

SASRec có:

- Gini thấp nhất.
- Coverage cao nhất.
- Exposure gap nhỏ nhất.
- Recall gap nhỏ nhất.
- Rare exposure và rare Recall@20 cao nhất.

Nhưng SASRec lại không có centroid gap nhỏ nhất; centroid gap của SASRec thậm chí lớn nhất trong ba mô hình.

Do đó:

> Geometry có liên hệ với exposure và accuracy ở cấp item, nhưng độ tách biệt geometry giữa popular và rare không quyết định một-một thứ hạng popularity bias ở cấp mô hình.

Nói cách khác, không thể chỉ nhìn centroid gap hoặc norm gap để kết luận mô hình nào thiên lệch hơn. Bias đầu ra còn chịu ảnh hưởng của kiến trúc mô hình, cách tạo session representation, hàm scoring và phân phối frequency.

---

## 9. Kết luận tổng hợp file 06–07–08

### File 06 — Hiện tượng đầu ra

- Cả ba mô hình đều ưu tiên popular item.
- Popular item nhận khoảng 62%–64% toàn bộ exposure; rare item chỉ nhận khoảng 4%–5%.
- Recall@20 của popular cao hơn rare khoảng 29–33 điểm phần trăm.
- SASRec có accuracy cao nhất và các chỉ số bias tốt nhất trong ba mô hình.

### File 07 — Cấu trúc bên trong embedding

- Frequency có liên hệ âm mạnh với norm.
- Rare item có norm lớn hơn và nằm xa tâm category hơn popular item.
- Warm item học được cấu trúc category; cold item không hình thành cấu trúc category có ý nghĩa.

### File 08 — Cầu nối giữa hai phần

- Sự khác biệt geometry đi kèm chênh lệch exposure và accuracy.
- Sau khi giữ frequency cố định, item gần tâm và cùng hướng với tâm category vẫn có lợi thế về exposure và hit@20.
- Norm thô bị frequency gây nhiễu; quan hệ đổi dấu sau kiểm soát.
- Geometry bổ sung thông tin nhưng frequency vẫn là yếu tố mạnh nhất.
- Geometry không một mình giải thích vì sao SASRec ít bias hơn GRU4Rec và NARM.

### Câu kết luận chính thức

> Trong ba mô hình SBR được khảo sát, popularity của item có liên hệ rõ với cả hình học embedding và đầu ra recommendation. Rare item thường có norm lớn hơn, nằm xa tâm category hơn, nhận ít exposure hơn và có Recall@20 thấp hơn popular item. Sau khi kiểm soát train frequency, các đặc trưng geometry vẫn mang thêm thông tin: item gần và cùng hướng với tâm category thường được gợi ý nhiều hơn và dễ dự đoán đúng hơn. Tuy nhiên, phần thông tin bổ sung của geometry nhỏ hơn đáng kể so với frequency và không giải thích hoàn toàn khác biệt bias giữa các kiến trúc. Vì vậy, kết quả ủng hộ mối liên hệ giữa popularity bias và embedding geometry, nhưng chưa chứng minh quan hệ nhân quả.

---

## 10. Hàm ý cải tiến

Kết quả tạo cơ sở cho các hướng tiếp theo:

1. **Giảm ảnh hưởng của frequency trong huấn luyện** bằng reweighting hoặc balanced sampling.
2. **Popularity-aware loss** để tăng trọng số cho rare target.
3. **Regularization hình học theo category** để kéo rare item về cấu trúc category phù hợp.
4. **Reranking sau mô hình** để giảm popular exposure và tăng rare exposure trong khi kiểm soát Recall@20.
5. **Feature-based cold-start** dùng category hoặc metadata để cold item có biểu diễn học được thay vì embedding ID ngẫu nhiên.

Các hướng trên mới là đề xuất từ kết quả quan sát. Muốn khẳng định hiệu quả phải triển khai từng phương pháp và so sánh lại accuracy, coverage, Gini, ARP và exposure theo nhóm.

---

## 11. Các tệp kết quả quan trọng

| Nội dung | Tệp |
| --- | --- |
| Geometry, exposure và recall theo nhóm | `results/tables/geometry_bias_group_seed_summary.csv` |
| Chênh lệch popular–rare | `results/tables/geometry_bias_gap_seed_summary.csv` |
| Spearman geometry–outcome | `results/statistical_tests/geometry_output_correlations.csv` |
| Hồi quy exposure | `results/statistical_tests/geometry_exposure_regression_seed_summary.csv` |
| Logistic regression accuracy | `results/statistical_tests/geometry_accuracy_logistic_seed_summary.csv` |
| Bảng nối kết cuối cùng | `results/tables/geometry_bias_final_link.csv` |
| Hình so sánh theo popularity | `results/figures/geometry_bias_group_link.png` |
| Hình hiệu ứng sau kiểm soát frequency | `results/figures/geometry_controlled_effects.png` |

---

## 12. Gợi ý trình bày trên slide

Có thể trình bày file 08 bằng ba slide:

### Slide 1 — Mục tiêu và phương pháp nối file 06–07

- Geometry: norm, khoảng cách và cosine với tâm category.
- Outcome: exposure và hit@20.
- Luôn kiểm soát train frequency.

### Slide 2 — Kết quả ở cấp item

- Rare xa tâm hơn, exposure thấp hơn và Recall@20 thấp hơn.
- Sau kiểm soát frequency: distance có hệ số âm, cosine có hệ số dương.
- Geometry có tín hiệu bổ sung nhưng ΔR² nhỏ.

### Slide 3 — Kết luận ở cấp mô hình

- SASRec ít bias và chính xác hơn.
- Nhưng SASRec không có centroid gap nhỏ nhất.
- Geometry liên hệ với bias nhưng không một mình quyết định bias giữa các mô hình.
