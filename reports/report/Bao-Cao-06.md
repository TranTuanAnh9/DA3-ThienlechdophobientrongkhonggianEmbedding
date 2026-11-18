# Báo cáo file 06 — Đánh giá chất lượng dự đoán và popularity bias

## 1. Mục tiêu của file 06

File 06 tổng hợp kết quả của ba mô hình GRU4Rec, NARM và SASRec để trả lời hai câu hỏi:

1. Mô hình nào dự đoán item tiếp theo tốt hơn?
2. Mô hình có ưu tiên popular item và làm giảm exposure của rare item hay không?

File 06 **không huấn luyện lại mô hình** và cũng **chưa phân tích hình học embedding**. Nó lấy kết quả đã sinh từ các notebook huấn luyện và phân tích đầu ra Top-20 recommendation.

```text
File 06: mô hình đề xuất item nào và popularity bias ở đầu ra lớn đến đâu?
File 07: embedding của các item được tổ chức như thế nào bên trong mô hình?
```

Notebook: [`06_popularity_bias.ipynb`](../../notebooks/06_popularity_bias.ipynb)

## Cách tra cứu kết quả nhanh

Nếu muốn đọc phần diễn giải, xem trực tiếp các mục tương ứng trong báo cáo:

| Muốn xem kết quả về | Xem mục trong báo cáo | Bảng số liệu gốc |
|---|---|---|
| Accuracy tổng thể | **Mục 5** | [`model_accuracy.csv`](../../results/tables/model_accuracy.csv) |
| Accuracy theo popular/medium/rare | **Mục 6** | `test_metrics.json` trong từng thư mục `runs/` |
| Last-item Recall@20 | **Mục 7** | [`model_accuracy.csv`](../../results/tables/model_accuracy.csv) |
| Cold-start diagnostic | **Mục 8** | `test_metrics.json` trong từng thư mục `runs/` |
| Recommendation count | **Mục 9** | Dùng để tạo các bảng popularity bias |
| Catalog Coverage@20 | **Mục 10** | [`popularity_bias.csv`](../../results/tables/popularity_bias.csv) |
| Recommendation Gini | **Mục 11** | [`popularity_bias.csv`](../../results/tables/popularity_bias.csv) |
| ARP — độ phổ biến trung bình | **Mục 12** | [`popularity_bias.csv`](../../results/tables/popularity_bias.csv) |
| Exposure popular/medium/rare/cold | **Mục 13** | [`exposure_by_group.csv`](../../results/tables/exposure_by_group.csv) |
| Biểu đồ Coverage | **Mục 15** | [`catalog_coverage.png`](../../results/figures/catalog_coverage.png) |
| Biểu đồ Gini và Lorenz | **Mục 15** | [`gini_comparison.png`](../../results/figures/gini_comparison.png) |
| Biểu đồ ARP | **Mục 15** | [`average_popularity.png`](../../results/figures/average_popularity.png) |
| Biểu đồ exposure | **Mục 15** | [`exposure_distribution.png`](../../results/figures/exposure_distribution.png) |

### Nếu chỉ cần xem ba bảng quan trọng nhất

1. [`model_accuracy.csv`](../../results/tables/model_accuracy.csv): kết quả dự đoán của từng mô hình và từng seed.
2. [`popularity_bias.csv`](../../results/tables/popularity_bias.csv): Coverage, Gini, ARP và popular/rare exposure tổng hợp.
3. [`exposure_by_group.csv`](../../results/tables/exposure_by_group.csv): chi tiết exposure của từng nhóm popularity.

### Cách đọc tên cột trong các bảng

- `model`: tên mô hình.
- `seed`: lần chạy thực nghiệm.
- `recall@20`, `mrr@20`, `ndcg@20`: các chỉ số accuracy.
- `coverage_warm@20`: tỷ lệ warm catalog được sử dụng.
- `gini_warm@20`: mức độ tập trung recommendation trên warm item.
- `arp@20`: độ phổ biến trung bình của item được đề xuất.
- `popular_group_exposure_share_pct`: tỷ lệ slot Top-20 thuộc popular item.
- `rare_group_exposure_share_pct`: tỷ lệ slot Top-20 thuộc rare item.

Các số trong phần diễn giải của báo cáo là trung bình qua hai seed; các file CSV giữ số liệu chi tiết của từng seed để kiểm tra lại.

---

## 2. Dữ liệu đầu vào

File 06 đọc kết quả từ sáu thí nghiệm:

| Mô hình | Seed |
|---|---:|
| GRU4Rec | 2024 |
| GRU4Rec | 2025 |
| NARM | 2024 |
| NARM | 2025 |
| SASRec | 2024 |
| SASRec | 2025 |

Với mỗi mô hình và seed, notebook đọc:

```text
runs/{model}/seed_{seed}/test_metrics.json
runs/{model}/seed_{seed}/top20.parquet
```

Trong đó:

- `test_metrics.json`: Recall, MRR, NDCG và chẩn đoán cold-start.
- `top20.parquet`: danh sách tối đa 20 item được đề xuất cho từng test query.

Quy mô đánh giá chính:

| Thành phần | Số lượng |
|---|---:|
| Test query warm | 40.001 |
| Top-K | 20 |
| Tổng recommendation slot mỗi mô hình/seed | 800.020 |
| Warm catalog | 29.936 item |
| Controlled cold catalog | 7.647 item |
| Tổng model catalog | 37.583 item |

Tổng số slot:

```text
40.001 query × 20 recommendation/query = 800.020 slot
```

---

## 3. Các bước đã thực hiện

### Bước 1 — Tải kết quả của sáu thí nghiệm

Notebook kiểm tra sự tồn tại của `test_metrics.json` và `top20.parquet` cho cả ba mô hình và hai seed. Nếu thiếu một kết quả, notebook dừng để tránh so sánh không đầy đủ.

Mục đích:

- Đảm bảo ba mô hình được so sánh trên cùng tập test.
- Đảm bảo mỗi mô hình có hai seed.
- Đánh giá độ ổn định của kết quả qua các lần khởi tạo khác nhau.

### Bước 2 — Tính và tổng hợp accuracy

Các chỉ số được lấy từ `test_metrics.json`, gồm:

- Accuracy trên toàn bộ warm target.
- Accuracy khi target thuộc popular, medium hoặc rare.
- Accuracy trên last-item target.
- Effective full-test accuracy có tính chẩn đoán cold target.

### Bước 3 — Đếm recommendation count

Từ `top20.parquet`, notebook đếm số lần mỗi item xuất hiện trong Top-20 trên toàn bộ test query.

Ví dụ:

```text
Item A xuất hiện trong 4.000 Top-20 list
→ recommendation count của A = 4.000
```

Từ recommendation count, notebook tính coverage, Gini, ARP và exposure theo nhóm popularity.

### Bước 4 — Phân tích theo nhóm popularity

Item được chia thành:

- Popular.
- Medium.
- Rare.
- Cold.

Các nhóm popular, medium và rare được xác định từ tần suất item trong final train. Cold item có train frequency bằng 0.

### Bước 5 — Tạo bảng và biểu đồ

Các bảng chính được lưu tại `results/tables/`:

- `model_accuracy.csv`
- `popularity_bias.csv`
- `exposure_by_group.csv`

Các biểu đồ chính:

- `catalog_coverage.png`
- `gini_comparison.png`
- `average_popularity.png`
- `exposure_distribution.png`

---

## 4. Các chỉ số accuracy

### 4.1. Recall@20

Recall@20 là tỷ lệ test query mà item đúng xuất hiện trong 20 item đầu tiên:

```text
Recall@20
= số query có target trong Top-20
  / tổng số query
```

Chỉ số này trả lời:

> Mô hình có tìm thấy item đúng trong danh sách Top-20 không?

Recall không quan tâm item đúng đứng thứ mấy trong Top-20.

### 4.2. MRR@20

MRR@20 quan tâm thứ hạng của item đúng:

- Đứng thứ 1: điểm cao nhất.
- Đứng thứ 2: điểm thấp hơn.
- Đứng càng thấp: điểm càng nhỏ.
- Không xuất hiện: điểm 0.

MRR trả lời:

> Item đúng có được đưa lên gần đầu danh sách không?

### 4.3. NDCG@20

NDCG@20 cũng đánh giá thứ hạng nhưng giảm điểm theo vị trí. Item đúng ở đầu danh sách quan trọng hơn item đúng ở cuối danh sách.

NDCG cao nghĩa là mô hình vừa tìm được item đúng vừa sắp xếp item đó ở vị trí tốt.

---

## 5. Kết quả accuracy tổng thể

Kết quả trung bình qua hai seed:

| Mô hình | Recall@20 | MRR@20 | NDCG@20 |
|---|---:|---:|---:|
| GRU4Rec | 0,5570 | 0,1989 | 0,2781 |
| NARM | 0,5625 | 0,1978 | 0,2786 |
| SASRec | **0,5729** | **0,2079** | **0,2885** |

### Nhận xét

- SASRec có Recall@20 cao nhất.
- SASRec có MRR@20 cao nhất.
- SASRec có NDCG@20 cao nhất.
- NARM nhỉnh hơn GRU4Rec về Recall và NDCG.
- Chênh lệch MRR giữa GRU4Rec và NARM rất nhỏ.

### Kết luận về accuracy

> SASRec cho chất lượng dự đoán tốt nhất trong ba mô hình theo cả Recall@20, MRR@20 và NDCG@20.

Accuracy cao chưa đồng nghĩa với popularity bias thấp. Vì vậy cần tiếp tục phân tích coverage, Gini, ARP và exposure.

---

## 6. Accuracy theo nhóm popularity

File 06 đánh giá riêng khi target thuộc popular, medium hoặc rare.

| Mô hình | Nhóm target | Recall@20 | MRR@20 | NDCG@20 |
|---|---|---:|---:|---:|
| GRU4Rec | Popular | 0,6470 | 0,2426 | 0,3328 |
| GRU4Rec | Medium | 0,4536 | 0,1434 | 0,2109 |
| GRU4Rec | Rare | 0,3276 | 0,1160 | 0,1621 |
| NARM | Popular | 0,6534 | 0,2424 | 0,3342 |
| NARM | Medium | 0,4586 | 0,1425 | 0,2114 |
| NARM | Rare | 0,3274 | 0,1060 | 0,1543 |
| SASRec | Popular | **0,6536** | 0,2420 | 0,3338 |
| SASRec | Medium | **0,4810** | **0,1658** | **0,2343** |
| SASRec | Rare | **0,3628** | **0,1362** | **0,1858** |

Ở cả ba mô hình:

```text
Accuracy popular > accuracy medium > accuracy rare
```

Ví dụ với SASRec:

- Recall popular: `0,6536`.
- Recall medium: `0,4810`.
- Recall rare: `0,3628`.

### Kết luận

> Popular item dễ dự đoán hơn nhiều so với rare item. Rare item vừa có accuracy thấp hơn vừa nhận được ít exposure hơn, cho thấy popularity bias xuất hiện ngay trong khả năng dự đoán chứ không chỉ trong số lần được đề xuất.

---

## 7. Last-item Recall@20

File 06 cũng đánh giá riêng item cuối cùng trong session.

| Mô hình | Last-item Recall@20 |
|---|---:|
| GRU4Rec | 0,5589 |
| NARM | 0,5663 |
| SASRec | **0,5792** |

Chỉ số này gần với tình huống thực tế:

> Sau khi người dùng kết thúc session, mô hình có đề xuất đúng item tiếp theo trong Top-20 không?

### Kết luận

> SASRec tiếp tục đạt kết quả tốt nhất khi đánh giá dự đoán item tiếp theo ở cuối session.

---

## 8. Chẩn đoán trên toàn bộ test có cold target

Test full bao gồm cả target warm và cold. File 06 thống kê:

| Loại target | Số lượng hoặc tỷ lệ |
|---|---:|
| Warm target | 40.001 |
| Controlled cold target | 11.614, khoảng 16,29% |
| Natural temporal cold target | 4.745, khoảng 6,66% |
| Unseen hoặc filtered target | 11.073 |

Nếu coi cold target là không dự đoán được bằng mô hình item-ID-only, effective full-test Recall giảm:

| Mô hình | Full-test Recall@20 có cold tính 0 |
|---|---:|
| GRU4Rec | 0,3125 |
| NARM | 0,3156 |
| SASRec | **0,3215** |

Con số này thấp hơn warm Recall vì cold target chưa có embedding được học.

Đây là chỉ số chẩn đoán giới hạn của cold-start, không nên so sánh trực tiếp với warm Recall như thể chúng là cùng một giao thức.

---

## 9. Recommendation count

Recommendation count là số lần một item xuất hiện trong toàn bộ các danh sách Top-20.

Với 40.001 query và Top-20:

```text
Tổng recommendation slot = 40.001 × 20 = 800.020
```

Recommendation count được dùng làm cơ sở để trả lời:

- Item nào được mô hình ưu tiên?
- Item nào nhận rất ít exposure?
- Recommendation có tập trung vào một nhóm nhỏ item không?
- Tần suất trong train có chuyển thành tần suất trong recommendation không?

---

## 10. Warm catalog và Coverage@20

Warm item là item đã xuất hiện trong train và có embedding được huấn luyện.

Warm Coverage@20 được tính:

```text
Warm Coverage@20
= số warm item được đề xuất ít nhất một lần
  / tổng số warm item
```

Kết quả:

| Mô hình | Warm Coverage@20 |
|---|---:|
| GRU4Rec | 0,8851 |
| NARM | 0,8858 |
| SASRec | **0,9176** |

### Kết luận

> SASRec sử dụng phạm vi catalog rộng nhất. Khoảng 91,76% warm item được đề xuất ít nhất một lần, cao hơn GRU4Rec và NARM.

Coverage cao cho thấy recommendation đa dạng hơn, nhưng chưa đủ để kết luận exposure công bằng. Một mô hình vẫn có thể dùng nhiều item nhưng dành phần lớn slot cho popular item.

---

## 11. Gini của recommendation

Gini đo mức độ tập trung của recommendation count giữa các item:

- Gini thấp: exposure phân bố tương đối đều.
- Gini cao: một số ít item nhận phần lớn recommendation.

Kết quả trên warm catalog:

| Mô hình | Warm Recommendation Gini |
|---|---:|
| GRU4Rec | 0,7036 |
| NARM | **0,7061** |
| SASRec | **0,6885** |

### Kết luận

- NARM có recommendation tập trung nhất.
- SASRec có mức độ tập trung thấp nhất.
- SASRec ít tập trung hơn nhưng vẫn chưa loại bỏ popularity bias.

> Lưu ý: Gini đầu vào của EDA và recommendation Gini là hai đại lượng khác nhau. Gini đầu vào mô tả phân phối interaction raw; recommendation Gini mô tả phân phối item trong output Top-20.

---

## 12. ARP — Average Recommendation Popularity

ARP đo độ phổ biến trung bình của các item được đề xuất.

ARP cao nghĩa là mô hình thường chọn item có tần suất train cao.

Kết quả:

| Mô hình | ARP@20 |
|---|---:|
| GRU4Rec | 77,52 |
| NARM | 76,57 |
| SASRec | **73,90** |

### Kết luận

> SASRec có ARP thấp nhất. Điều này cho thấy độ phổ biến trung bình của item được đề xuất thấp hơn tương đối so với GRU4Rec và NARM.

ARP không cho biết riêng rare item có được hỗ trợ tốt hay không, nên cần xem thêm exposure theo nhóm.

---

## 13. Exposure theo nhóm popularity
Exposure trong bài này nghĩa là số lần một item được đưa vào danh sách recommendation cho người dùng.

### 13.1. Tỷ lệ nhóm trong warm catalog

| Nhóm | Tỷ lệ warm catalog |
|---|---:|
| Popular | 20,42% |
| Medium | 56,59% |
| Rare | 23,00% |

### 13.2. Tỷ lệ interaction trong train

| Nhóm | Tỷ lệ train interaction |
|---|---:|
| Popular | 60,65% |
| Medium | 33,57% |
| Rare | 5,78% |

### 13.3. Tỷ lệ exposure trong Top-20

| Mô hình | Popular exposure | Medium exposure | Rare exposure |
|---|---:|---:|---:|
| GRU4Rec | 63,96% | khoảng 31,77% | 4,27% |
| NARM | 64,15% | khoảng 31,69% | 4,16% |
| SASRec | **62,49%** | **32,79%** | **4,73%** |

### Cách đọc

Popular item chỉ chiếm 20,42% warm catalog nhưng nhận khoảng 62–64% vị trí recommendation.

Rare item chiếm 23,00% warm catalog nhưng chỉ nhận khoảng 4–5% vị trí recommendation.

Đây là sự chênh lệch exposure rất lớn.

### 13.4. Exposure-to-train ratio

Chỉ số này so sánh:

```text
Tỷ lệ exposure recommendation
/
Tỷ lệ interaction trong train
```

Popular item có exposure-to-train ratio khoảng `1,03–1,06`, tức exposure ít nhất tương xứng với tỷ lệ interaction.

Rare item có exposure-to-train ratio khoảng `0,71–0,83`, tức nhận exposure thấp hơn tỷ lệ interaction trong train.

### Kết luận

> Cả ba mô hình đều ưu tiên popular item và làm giảm exposure của rare item. SASRec có popular exposure thấp nhất và rare exposure cao nhất, nhưng chênh lệch popularity vẫn còn rất rõ.

---

## 14. Within-group coverage

File 06 còn kiểm tra trong mỗi nhóm có bao nhiêu item được đề xuất ít nhất một lần.

Xu hướng:

- Popular group coverage: gần 100%.
- Medium group coverage: khoảng 90–94%.
- Rare group coverage:
  - GRU4Rec: khoảng 73–74%.
  - NARM: khoảng 72–76%.
  - SASRec: khoảng 80–82%.

### Kết luận

> Gần như toàn bộ popular item có cơ hội xuất hiện trong recommendation, trong khi một phần đáng kể rare item không bao giờ được đề xuất. SASRec cải thiện coverage trong nhóm rare tốt hơn hai mô hình còn lại.

---

## 15. Các biểu đồ trong file 06

### `catalog_coverage.png`

So sánh tỷ lệ warm catalog được mô hình sử dụng.

Thông điệp:

> SASRec sử dụng catalog rộng nhất.

### `gini_comparison.png`

Gồm biểu đồ cột Gini và đường cong Lorenz.

Thông điệp:

> NARM tập trung recommendation mạnh nhất; SASRec phân bố đều hơn tương đối.

### `average_popularity.png`

So sánh ARP@20.

Thông điệp:

> SASRec có độ phổ biến trung bình của item được đề xuất thấp nhất.

### `exposure_distribution.png`

So sánh tỷ lệ exposure của popular, medium và rare.

Thông điệp:

> Popular item nhận nhiều exposure hơn rất nhiều so với tỷ lệ catalog, còn rare item bị thiếu exposure.

---

## 16. Kết luận cuối cùng của file 06

### Kết luận về accuracy

> SASRec là mô hình có chất lượng dự đoán tốt nhất trong ba mô hình theo Recall@20, MRR@20 và NDCG@20.

### Kết luận về popularity bias

> Cả ba mô hình đều có popularity bias rõ ràng. Popular item được đề xuất nhiều hơn đáng kể, trong khi rare item có accuracy thấp hơn và nhận ít exposure hơn.

### Kết luận về coverage và Gini

> SASRec có coverage cao nhất và Gini thấp nhất, cho thấy mô hình sử dụng catalog rộng hơn và ít tập trung hơn. Tuy nhiên, popular item vẫn chiếm phần lớn exposure.

### Kết luận về rare item

Rare item chịu bất lợi ở cả hai phía:

```text
Rare item
→ khó dự đoán hơn
→ ít xuất hiện trong Top-20 hơn
→ một phần lớn không bao giờ được đề xuất
```

### Kết luận về cold item

> Cold item không được dùng làm ứng viên trong recommendation chính vì mô hình chỉ sử dụng item ID và chưa học được embedding của cold item. Cold-start được theo dõi riêng, không trộn với popularity bias của warm item.

---

## 17. Câu trình bày dùng trong thuyết trình

> Trong file 06, chúng tôi tổng hợp kết quả của GRU4Rec, NARM và SASRec trên cùng tập test, với hai seed cho mỗi mô hình. Kết quả cho thấy SASRec có chất lượng dự đoán tốt nhất, với Recall@20 bằng 0,5729, MRR@20 bằng 0,2079 và NDCG@20 bằng 0,2885. Tuy nhiên, cả ba mô hình đều thể hiện popularity bias. Popular item chỉ chiếm khoảng 20,42% warm catalog nhưng nhận khoảng 62–64% các vị trí recommendation, trong khi rare item chiếm khoảng 23% catalog nhưng chỉ nhận khoảng 4–5% exposure. SASRec có coverage cao nhất, Gini thấp nhất, popular exposure thấp nhất và rare exposure cao nhất trong ba mô hình, nên có trade-off tốt nhất giữa accuracy và độ đa dạng. Tuy vậy, bias vẫn còn rõ rệt. Đặc biệt, accuracy của popular target cao hơn nhiều so với rare target, cho thấy rare item vừa khó dự đoán vừa ít được đề xuất. Đây là cơ sở để file 07 tiếp tục kiểm tra xem sự thiên lệch này có liên quan đến cấu trúc embedding hay không.

---

## 18. Giới hạn diễn giải

File 06 cho biết **bias xuất hiện ở đầu ra recommendation**, nhưng chưa giải thích nguyên nhân bên trong embedding.

Không được kết luận từ file 06 rằng:

- Popularity là nguyên nhân duy nhất làm rare item khó dự đoán.
- Norm embedding gây ra recommendation bias.
- Gini thấp đồng nghĩa mô hình hoàn toàn công bằng.
- Cold exposure bằng 0 có nghĩa mô hình đã giải quyết cold-start.

Các câu hỏi về embedding, norm, cosine, centroid và vị trí cold item thuộc file 07.

