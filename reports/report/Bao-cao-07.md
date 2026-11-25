# Báo cáo file 07 — Phân tích hình học không gian embedding

## 1. Mục tiêu của file 07

File 07 đi vào bên trong ba mô hình GRU4Rec, NARM và SASRec để phân tích vector embedding của item sau khi huấn luyện.

File 06 trả lời:

```text
Mô hình đề xuất item nào và popularity bias ở đầu ra ra sao?
```

File 07 trả lời:

```text
Embedding của popular, medium, rare và cold item khác nhau như thế nào?
```

Hai câu hỏi chính của file 07:

1. Norm embedding có liên quan đến tần suất item trong train không?
2. Cold item nằm ở đâu trong không gian embedding so với warm item cùng category?

Notebook: [`07_embedding_geometry.ipynb`](../../notebooks/07_embedding_geometry.ipynb)

File 07 **không huấn luyện mô hình mới**. Nó đọc embedding đã lưu sau các notebook 03, 04 và 05.

---

## 2. Dữ liệu đầu vào

File 07 phân tích sáu ma trận embedding:

| Mô hình | Seed | Kích thước embedding |
|---|---:|---:|
| GRU4Rec | 2024 | 37.583 × 64 |
| GRU4Rec | 2025 | 37.583 × 64 |
| NARM | 2024 | 37.583 × 64 |
| NARM | 2025 | 37.583 × 64 |
| SASRec | 2024 | 37.583 × 64 |
| SASRec | 2025 | 37.583 × 64 |

Mỗi dòng embedding tương ứng với một item trong catalog:

```text
item_id → vector 64 chiều
```

Ngoài embedding, notebook dùng bảng item-level để biết:

- `item_id`.
- `category_id`.
- `train_freq`.
- `popularity_group`.
- Item là warm hay cold.
- Recommendation count.
- Khoảng cách tới centroid category.

File 07 kiểm tra tính nhất quán trước khi phân tích:

- Đúng 37.583 item ở mỗi mô hình và seed.
- Embedding đúng kích thước 64 chiều.
- Không có NaN hoặc Inf.
- Mapping item giữa embedding và bảng item-level khớp nhau.
- Cold item có `train_freq = 0`.

---

## Cách tra cứu kết quả nhanh

| Muốn xem kết quả về | Bảng hoặc hình cần mở |
|---|---|
| Norm theo nhóm popularity | [`norm_frequency.csv`](../../results/tables/norm_frequency.csv) |
| Norm trung bình qua hai seed | [`norm_frequency_seed_summary.csv`](../../results/tables/norm_frequency_seed_summary.csv) |
| Spearman frequency–norm | [`spearman_results.csv`](../../results/statistical_tests/spearman_results.csv) |
| Spearman tổng hợp qua hai seed | [`spearman_seed_summary.csv`](../../results/statistical_tests/spearman_seed_summary.csv) |
| Cosine cùng/khác category | [`cosine_category.csv`](../../results/tables/cosine_category.csv) |
| Cosine tổng hợp qua hai seed | [`cosine_seed_summary.csv`](../../results/tables/cosine_seed_summary.csv) |
| Kiểm định cosine | [`cosine_tests.csv`](../../results/statistical_tests/cosine_tests.csv) |
| Khoảng cách tới centroid | [`category_centroid.csv`](../../results/tables/category_centroid.csv) |
| Centroid tổng hợp qua hai seed | [`category_centroid_seed_summary.csv`](../../results/tables/category_centroid_seed_summary.csv) |
| Tọa độ t-SNE | [`tsne_coordinates.parquet`](../../results/tables/tsne_coordinates.parquet) |
| Boxplot norm | [`norm_boxplot.png`](../../results/figures/norm_boxplot.png) |
| Scatter frequency–norm | [`frequency_norm_scatter.png`](../../results/figures/frequency_norm_scatter.png) |
| t-SNE theo popularity | [`tsne_by_popularity.png`](../../results/figures/tsne_by_popularity.png) |
| t-SNE theo category | [`tsne_by_category.png`](../../results/figures/tsne_by_category.png) |

Các file có hậu tố `seed2024` hoặc `seed2025` là kết quả của từng lần chạy riêng.

---

## 3. Các bước đã thực hiện

### Bước 1 — Đọc embedding và kiểm tra mapping

Notebook đọc:

```text
runs/{model}/seed_{seed}/item_embeddings.npy
```

Sau đó nối embedding với bảng item-level và item mapping.

Mục đích:

> Đảm bảo vector đang được phân tích đúng item, đúng category, đúng tần suất và đúng trạng thái warm/cold.

### Bước 2 — Tính norm theo nhóm item

Với mỗi item, notebook tính độ dài vector embedding rồi tổng hợp theo:

- Popular.
- Medium.
- Rare.
- Cold.

### Bước 3 — Tính Spearman giữa frequency và norm

Notebook tính tương quan giữa:

```text
train frequency ↔ embedding norm
```

Phân tích chính chỉ dùng warm item. Kết quả tính trên toàn catalog có cold được lưu bổ sung để tham khảo, nhưng không dùng làm kết luận chính vì cold embedding chưa được huấn luyện.

### Bước 4 — Tính cosine similarity theo category

Notebook lấy mẫu cân bằng các cặp item và so sánh:

1. Warm item cùng category.
2. Warm item khác category.
3. Cold item – warm item cùng category.
4. Cold item – warm item khác category.

Mỗi nhóm có 50.000 cặp trên mỗi model-seed.

### Bước 5 — Tính khoảng cách tới centroid category

Notebook tính vector trung bình của các warm item trong mỗi category. Vector trung bình này gọi là centroid.

Sau đó tính:

- Khoảng cách item tới centroid.
- Cosine giữa item và centroid.
- Chênh lệch khoảng cách giữa popular, rare và cold.

### Bước 6 — Trực quan hóa bằng t-SNE

Embedding 64 chiều được giảm xuống 2 chiều để vẽ.

Notebook tạo hình:

- Tô màu theo popular/medium/rare/cold.
- Tô màu theo category.
- Đánh dấu cold item bằng ký hiệu riêng.

t-SNE được chạy cho cả seed 2024 và 2025 để kiểm tra xu hướng có ổn định không.

---

## 4. Norm embedding

### 4.1. Norm là gì?

Nếu embedding của item là:

```text
e_i = [x1, x2, ..., x64]
```

thì L2 norm là:

```text
||e_i||₂ = sqrt(x1² + x2² + ... + x64²)
```

Norm có thể hiểu là **độ dài của vector embedding**.

### 4.2. Dùng để làm gì?

File 07 dùng norm để kiểm tra:

- Các nhóm popularity có độ dài embedding khác nhau không?
- Rare item có biểu diễn khác popular item không?
- Cold embedding có độ lớn giống warm embedding không?

---

## 5. Kết quả norm theo nhóm popularity

### GRU4Rec

| Nhóm | Norm trung bình |
|---|---:|
| Popular | 0,998 |
| Medium | 1,186 |
| Rare | 1,357 |
| Cold | 0,159 |

### Kết quả tổng hợp qua ba mô hình

| Mô hình | Popular | Medium | Rare | Cold |
|---|---:|---:|---:|---:|
| GRU4Rec | 0,998 | 1,186 | **1,357** | **0,159** |
| NARM | 1,019 | 1,110 | **1,218** | **0,159** |
| SASRec | 0,830 | 0,961 | **1,069** | **0,159** |

Xu hướng chung:

```text
Rare norm > Medium norm > Popular norm >> Cold norm
```

### Kết luận

> Rare item thường có embedding dài hơn popular item. Cold item có norm rất nhỏ vì embedding của chúng được khởi tạo nhưng không nhận được gradient để học.

Điều này cho thấy sự khác biệt giữa popular, rare và cold item tồn tại ngay trong không gian embedding, trước khi xét đến số lần item được đề xuất.

---

## 6. Tương quan Spearman giữa frequency và norm

### 6.1. Spearman là gì?

Spearman đo mối liên hệ về thứ hạng giữa hai đại lượng:

```text
Tần suất item trong train ↔ độ dài embedding
```

Giá trị nằm trong khoảng `[-1, +1]`:

- Gần `+1`: frequency tăng thì norm có xu hướng tăng.
- Gần `0`: chưa thấy quan hệ thứ hạng rõ.
- Gần `-1`: frequency tăng thì norm có xu hướng giảm.

### 6.2. Kết quả warm-only

| Mô hình | Spearman trung bình | Độ lệch chuẩn qua 2 seed |
|---|---:|---:|
| GRU4Rec | **-0,7449** | 0,0119 |
| NARM | **-0,6675** | 0,0026 |
| SASRec | **-0,7066** | 0,0069 |

Các khoảng tin cậy bootstrap 95% của từng seed đều nằm hoàn toàn ở phía âm. P-value được báo cáo rất nhỏ, xấp xỉ `p < 0,0001`.

### 6.3. Cách diễn giải

Kết quả cho thấy:

```text
Tần suất train tăng
→ norm embedding có xu hướng giảm
```

Hoặc:

```text
Rare item thường có norm lớn hơn popular item
```

### 6.4. Không được diễn giải thành quan hệ nhân quả

Không được nói:

> Tần suất cao chắc chắn làm norm giảm.

Cách nói đúng:

> Trong dữ liệu và ba mô hình được thử nghiệm, frequency có mối liên hệ âm mạnh với embedding norm. Đây là tương quan, chưa phải bằng chứng frequency trực tiếp gây ra norm giảm.

Một yếu tố gây nhiễu có thể là tuổi item: item tồn tại lâu hơn thường có nhiều interaction hơn. Vì vậy, kết quả cần được diễn giải là quan hệ liên kết.

### 6.5. Kết quả nếu tính cả cold

File 07 có lưu thêm kết quả `all_catalog_incl_cold`, nhưng đây chỉ là kiểm tra bổ sung:

| Mô hình | Spearman all catalog, gồm cold |
|---|---:|
| GRU4Rec | 0,1112 |
| NARM | 0,1504 |
| SASRec | 0,1306 |

Kết quả này không dùng làm kết luận chính vì cold embedding chưa được huấn luyện và có norm khởi tạo rất nhỏ. Kết luận chính phải dùng `warm_only`.

---

## 7. Cosine similarity theo category

### 7.1. Cosine similarity là gì?

Cosine similarity đo mức độ giống nhau về hướng của hai vector:

- Gần `1`: hai vector gần cùng hướng.
- Gần `0`: hai vector gần như không liên quan về hướng.
- Gần `-1`: hai vector có hướng ngược nhau.

Cosine không tập trung vào độ dài vector mà tập trung vào hướng biểu diễn.

### 7.2. Dùng để làm gì?

File 07 kiểm tra xem mô hình có tự hình thành cấu trúc category không:

```text
Embedding item cùng category có giống hướng nhau hơn không?
```

Category không được đưa trực tiếp vào mô hình như một feature đầu vào. Vì vậy, nếu cosine cùng category cao hơn khác category, đó là dấu hiệu mô hình đã học được tín hiệu category từ chuỗi tương tác.

### 7.3. Kết quả warm item

| Mô hình | Cùng category | Khác category | Category cosine gap |
|---|---:|---:|---:|
| GRU4Rec | 0,605 | 0,175 | **0,429** |
| NARM | 0,551 | 0,113 | **0,437** |
| SASRec | 0,556 | 0,146 | **0,410** |

Trong cả ba mô hình, cosine cùng category cao hơn rõ rệt so với khác category. Các kiểm định Welch t-test và Mann–Whitney đều cho thấy chênh lệch warm cùng/khác category có ý nghĩa thống kê rất mạnh (`p < 0,0001`).

### 7.4. Kết luận warm item

> Các mô hình đã học được một phần cấu trúc category từ chuỗi tương tác. Item cùng category có xu hướng được đặt gần nhau hơn về hướng trong không gian embedding.

Điều này không có nghĩa embedding đã phân loại category hoàn hảo. Nó chỉ cho thấy có tín hiệu category đáng kể.

---

## 8. Phân tích cold item bằng cosine

File 07 so sánh cold item với warm item:

1. Cold–warm cùng category.
2. Cold–warm khác category.

Kết quả:

| Mô hình | Cold–warm cùng category | Cold–warm khác category | Chênh lệch |
|---|---:|---:|---:|
| GRU4Rec | 0,0026 | 0,0007 | 0,0019 |
| NARM | -0,0006 | 0,0003 | -0,0008 |
| SASRec | 0,0011 | 0,0000 | 0,0011 |

Các giá trị đều gần bằng 0 và chênh lệch gần như không đáng kể. Kiểm định Mann–Whitney cho cold cùng/khác category không cho thấy một xu hướng ổn định qua cả hai seed.

### Kết luận

> Cold item không hình thành được quan hệ category có ý nghĩa trong embedding.

Nguyên nhân là:

- Cold item không xuất hiện trong train.
- Embedding của cold item không nhận được gradient.
- Mô hình chỉ dùng item ID.
- Mô hình không có category hoặc metadata để suy ra vị trí cold item.

Vì vậy, việc cold item không gần warm item cùng category là giới hạn được dự đoán của baseline ID-based, không phải lỗi chạy mô hình.

---

## 9. Category cosine gap

Category cosine gap được tính:

```text
cosine gap
= cosine trung bình cùng category
  - cosine trung bình khác category
```

Giá trị càng lớn nghĩa là embedding càng phân biệt được item cùng category với item khác category.

Kết quả warm:

| Mô hình | Warm category cosine gap |
|---|---:|
| GRU4Rec | 0,4292 |
| NARM | 0,4374 |
| SASRec | 0,4105 |

Kết quả cold:

| Mô hình | Cold category cosine gap |
|---|---:|
| GRU4Rec | 0,0019 |
| NARM | -0,0008 |
| SASRec | 0,0011 |

### Kết luận

> Category structure thể hiện rõ ở warm item nhưng gần như biến mất ở cold item.

---

## 10. Khoảng cách tới centroid category

### 10.1. Centroid là gì?

Giả sử một category có ba item với embedding `A`, `B`, `C`. Centroid là vector trung bình:

```text
centroid = (A + B + C) / 3
```

Centroid đại diện cho vị trí trung tâm của category trong không gian embedding.

### 10.2. Khoảng cách tới centroid

File 07 đo khoảng cách từ embedding của từng item tới centroid category của nó.

- Khoảng cách nhỏ: item gần cấu trúc trung tâm của category.
- Khoảng cách lớn: item bị phân tán hoặc ít đại diện hơn cho category.

### 10.3. Kết quả

| Mô hình | Popular | Medium | Rare | Cold |
|---|---:|---:|---:|---:|
| GRU4Rec | 0,700 | 0,710 | 0,802 | **0,947** |
| NARM | 0,692 | 0,718 | 0,792 | **0,848** |
| SASRec | **0,577** | 0,612 | 0,686 | **0,735** |

Xu hướng chung:

```text
Popular gần centroid hơn rare
Rare xa centroid hơn popular
Cold thường xa centroid nhất
```

### Kết luận

> Popular item thường có biểu diễn ổn định và gần cấu trúc trung tâm của category hơn. Rare item phân tán hơn, còn cold item chưa có vị trí đáng tin cậy trong không gian embedding.

---

## 11. Centroid distance gap

File 07 cũng tính chênh lệch khoảng cách:

```text
rare minus popular
= distance rare tới centroid
  - distance popular tới centroid

cold minus popular
= distance cold tới centroid
  - distance popular tới centroid
```

Kết quả trung bình qua hai seed:

| Mô hình | Rare − Popular | Cold − Popular |
|---|---:|---:|
| GRU4Rec | 0,1016 | 0,2468 |
| NARM | 0,1001 | 0,1555 |
| SASRec | 0,1089 | 0,1586 |

Giá trị dương cho thấy rare và cold ở xa centroid hơn popular.

### Kết luận

> Cold item lệch khỏi cấu trúc trung tâm category nhiều nhất. Rare item cũng bị phân tán hơn popular nhưng mức chênh lệch nhỏ hơn cold.

---

## 12. Trực quan hóa bằng t-SNE

### 12.1. t-SNE là gì?

Embedding có 64 chiều nên không thể vẽ trực tiếp. t-SNE giảm từ 64 chiều xuống 2 chiều để quan sát.

Mỗi điểm trên hình là một item.

### 12.2. Các hình đã tạo

File 07 tạo hai nhóm hình:

1. t-SNE tô màu theo `popular`, `medium`, `rare`, `cold`.
2. t-SNE tô màu theo category và đánh dấu cold bằng ký hiệu `X`.

t-SNE được chạy trên cả seed 2024 và 2025.

### 12.3. Quan sát chính

- Warm item tạo thành các cụm category tương đối rõ.
- Popular, medium và rare vẫn đan xen nhau, không tạo thành ba cụm tách biệt hoàn toàn.
- Cold item thường nằm tách biệt hoặc phân tán.
- Xu hướng chính xuất hiện ở cả hai seed.

### 12.4. Giới hạn diễn giải

t-SNE chỉ ưu tiên bảo toàn cấu trúc lân cận cục bộ. Vì vậy:

- Khoảng cách giữa hai cụm trên hình không phản ánh trực tiếp khoảng cách trong không gian 64 chiều.
- Không được dùng riêng t-SNE để khẳng định nhóm nào gần nhóm nào về mặt định lượng.
- Kết luận chính phải dựa vào norm, cosine và khoảng cách tới centroid.

---

## 13. Độ ổn định qua hai seed

File 07 chạy mỗi mô hình với hai seed `2024` và `2025`, sau đó tính trung bình và độ lệch chuẩn.

Các xu hướng đều lặp lại:

- Spearman frequency–norm âm mạnh.
- Cosine warm cùng category cao hơn khác category.
- Popular gần centroid hơn rare.
- Cold có cosine gần 0 với cả warm cùng và khác category.
- Cold thường xa centroid hơn warm item.

Điều này cho thấy kết luận không chỉ xuất hiện do một lần khởi tạo ngẫu nhiên.

---

## 14. Các kết luận chính của file 07

### Kết luận 1 — Norm khác nhau theo popularity

```text
Rare norm > Medium norm > Popular norm >> Cold norm
```

Rare item thường có vector dài hơn popular item, còn cold item có vector rất nhỏ.

### Kết luận 2 — Frequency có tương quan âm mạnh với norm

Cả ba mô hình đều có Spearman âm mạnh:

| Mô hình | Spearman frequency–norm |
|---|---:|
| GRU4Rec | -0,7449 |
| NARM | -0,6675 |
| SASRec | -0,7066 |

Tần suất tăng thì norm có xu hướng giảm. Đây là tương quan, chưa phải quan hệ nhân quả.

### Kết luận 3 — Warm item có cấu trúc category

Cosine cùng category cao hơn khác category rõ rệt:

| Mô hình | Cùng category | Khác category |
|---|---:|---:|
| GRU4Rec | 0,605 | 0,175 |
| NARM | 0,551 | 0,113 |
| SASRec | 0,556 | 0,146 |

Điều này cho thấy mô hình tự học được một phần cấu trúc category từ hành vi session.

### Kết luận 4 — Cold item chưa học được cấu trúc category

Cosine cold–warm cùng category gần bằng 0 và gần như không khác cold–warm khác category.

Nguyên nhân là cold item chưa xuất hiện trong train và mô hình chỉ dùng item ID.

### Kết luận 5 — Popular item gần tâm category hơn

Popular item thường gần centroid hơn rare và cold. Điều này cho thấy popular item có biểu diễn ổn định hơn và đại diện tốt hơn cho cấu trúc trung tâm của category.

---

## 15. File 07 cho biết gì khi kết hợp với file 06?

File 06 phát hiện:

```text
Popular item được đề xuất nhiều hơn
Rare item được đề xuất ít hơn
```

File 07 phát hiện:

```text
Popular, rare và cold item có embedding khác nhau
Popular gần centroid hơn
Rare phân tán hơn
Cold gần như chưa có cấu trúc category
```

Kết luận kết hợp:

> Popularity bias không chỉ xuất hiện ở đầu ra recommendation. Nó đi kèm với sự khác biệt trong không gian embedding: popular item thường có biểu diễn ổn định hơn, rare item phân tán hơn, còn cold item gần như chưa được học.

Tuy nhiên, file 07 chưa chứng minh rằng sự khác biệt embedding là nguyên nhân duy nhất của popularity bias. Mối quan hệ trực tiếp giữa embedding và số lần recommendation được phân tích bổ sung trong file 08.

---

## 16. Những điều file 07 chưa thể kết luận

File 07 chưa thể kết luận:

- Norm embedding trực tiếp gây ra popularity bias.
- Item có norm lớn chắc chắn được đề xuất ít hơn.
- Cold item chắc chắn không thể được đề xuất trong mọi mô hình.
- Category là nguyên nhân duy nhất làm các embedding gần nhau.
- Khoảng cách nhìn thấy trên t-SNE là khoảng cách định lượng thật.

Các kết quả của file 07 là phân tích tương quan và hình học của baseline item-ID-only. Để kiểm tra quan hệ giữa norm và số lần recommendation sau khi kiểm soát frequency, cần dùng file 08.

---

## 17. Câu trình bày dùng trong thuyết trình

> Trong file 07, chúng tôi phân tích không gian embedding của ba mô hình với hai seed. Kết quả cho thấy rare item thường có norm lớn hơn popular item, còn cold item có norm rất nhỏ vì chưa được huấn luyện. Tương quan Spearman giữa tần suất trong train và norm embedding lần lượt là -0,7449 với GRU4Rec, -0,6675 với NARM và -0,7066 với SASRec, cho thấy item càng phổ biến thì norm có xu hướng càng nhỏ. Khi so sánh cosine, item cùng category có độ tương đồng cao hơn nhiều so với item khác category, chứng tỏ mô hình đã học được một phần cấu trúc category từ chuỗi tương tác. Tuy nhiên, cosine giữa cold item và warm item cùng category gần bằng 0, cho thấy cold item chưa học được cấu trúc category. Phân tích khoảng cách tới centroid cũng cho thấy popular item gần tâm category hơn rare và cold item. Vì vậy, popularity bias quan sát ở file 06 có liên quan đến sự khác biệt trong không gian embedding, nhưng quan hệ nhân quả cần được kiểm tra thận trọng hơn ở file 08.

---

## 18. Các file kết quả

### Bảng item-level và norm

- [`norm_frequency.csv`](../../results/tables/norm_frequency.csv)
- [`norm_frequency_seed_summary.csv`](../../results/tables/norm_frequency_seed_summary.csv)

### Bảng cosine

- [`cosine_category.csv`](../../results/tables/cosine_category.csv)
- [`cosine_seed_summary.csv`](../../results/tables/cosine_seed_summary.csv)
- [`cosine_tests.csv`](../../results/statistical_tests/cosine_tests.csv)

### Bảng centroid

- [`category_centroid.csv`](../../results/tables/category_centroid.csv)
- [`category_centroid_seed_summary.csv`](../../results/tables/category_centroid_seed_summary.csv)

### Bảng Spearman

- [`spearman_results.csv`](../../results/statistical_tests/spearman_results.csv)
- [`spearman_seed_summary.csv`](../../results/statistical_tests/spearman_seed_summary.csv)

### Tọa độ và hình t-SNE

- [`tsne_coordinates.parquet`](../../results/tables/tsne_coordinates.parquet)
- [`norm_boxplot.png`](../../results/figures/norm_boxplot.png)
- [`frequency_norm_scatter.png`](../../results/figures/frequency_norm_scatter.png)
- [`tsne_by_popularity.png`](../../results/figures/tsne_by_popularity.png)
- [`tsne_by_category.png`](../../results/figures/tsne_by_category.png)

