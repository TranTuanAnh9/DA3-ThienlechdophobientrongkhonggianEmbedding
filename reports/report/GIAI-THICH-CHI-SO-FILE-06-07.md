# Giải thích các chỉ số trong file 06 và file 07

## 1. Phạm vi của hai file

Hai notebook có nhiệm vụ khác nhau:

```text
File 06: Đánh giá mô hình và đo popularity bias ở đầu ra recommendation
File 07: Phân tích hình học embedding ở bên trong mô hình
```

Nói đơn giản:

- File 06 kiểm tra **mô hình đề xuất item nào và đề xuất có bị thiên lệch không**.
- File 07 kiểm tra **embedding của các item được tổ chức như thế nào trong không gian vector**.

Ba mô hình được phân tích là GRU4Rec, NARM và SASRec. Mỗi mô hình được chạy với hai seed `2024` và `2025`; kết quả thường được báo cáo bằng trung bình qua hai seed.

---

## 2. Các chỉ số của file 06

File 06 sử dụng kết quả Top-20 recommendation trên tập test. Với mỗi phiên hoặc mỗi truy vấn, mô hình tạo ra tối đa 20 item được đề xuất.

### 2.1. Recall@20

#### Recall@20 là gì?

Recall@20 là tỷ lệ truy vấn mà item đúng xuất hiện trong 20 item đầu tiên:

```text
Recall@20 = số truy vấn có target nằm trong Top-20 / tổng số truy vấn
```

Ví dụ: nếu có 100 truy vấn và 60 truy vấn chứa item đúng trong Top-20 thì:

```text
Recall@20 = 0,60
```

#### Dùng để làm gì?

Đo khả năng mô hình **không bỏ sót item đúng**.

Recall@20 cao nghĩa là mô hình thường đưa item mà người dùng thực sự xem tiếp vào danh sách Top-20.

#### Kết quả chính

| Mô hình | Recall@20 |
| ------- | --------: |
| GRU4Rec |    0,5570 |
| NARM    |    0,5625 |
| SASRec  |    0,5729 |

SASRec có Recall@20 cao nhất.

---

### 2.2. MRR@20

#### MRR@20 là gì?

MRR đánh giá vị trí của item đúng trong danh sách Top-20.

- Item đúng ở vị trí 1: điểm `1`.
- Item đúng ở vị trí 2: điểm `1/2`.
- Item đúng ở vị trí 10: điểm `1/10`.
- Item đúng không xuất hiện trong Top-20: điểm `0`.

Công thức ý tưởng:

```text
MRR = trung bình của 1 / thứ hạng item đúng
```

#### Dùng để làm gì?

Recall chỉ quan tâm item đúng có xuất hiện hay không. MRR quan tâm thêm item đúng đứng cao hay thấp.

Vì vậy MRR@20 trả lời:

> Mô hình có đưa item đúng lên gần đầu danh sách không?

#### Kết quả chính

| Mô hình | MRR@20 |
| ------- | -----: |
| GRU4Rec | 0,1989 |
| NARM    | 0,1978 |
| SASRec  | 0,2079 |

SASRec có MRR@20 cao nhất.

---

### 2.3. NDCG@20

#### NDCG@20 là gì?

NDCG cũng đánh giá vị trí của item đúng nhưng dùng hàm giảm dần theo thứ hạng. Item ở đầu danh sách được coi là quan trọng hơn item ở cuối danh sách.

NDCG được chuẩn hóa về khoảng từ 0 đến 1:

- Gần 1: item đúng thường nằm rất cao.
- Gần 0: item đúng thường bị xếp thấp hoặc không xuất hiện.

#### Dùng để làm gì?

Đo chất lượng **sắp xếp thứ hạng** của Top-20, không chỉ đo việc item đúng có được đưa vào danh sách hay không.

#### Kết quả chính

| Mô hình | NDCG@20 |
| ------- | ------: |
| GRU4Rec |  0,2781 |
| NARM    |  0,2786 |
| SASRec  |  0,2885 |

SASRec tiếp tục có kết quả tốt nhất.

---

### 2.4. Last-item Recall@20

#### Là gì?

Trong session-based recommendation, các event trước đó được dùng làm input và item tiếp theo được dùng làm target. Last-item Recall@20 kiểm tra riêng việc dự đoán item cuối cùng của session.

#### Dùng để làm gì?

Đây là một cách đánh giá gần với bài toán thực tế:

> Sau khi người dùng vừa xem xong item cuối cùng, mô hình có đề xuất đúng item tiếp theo trong Top-20 không?

Chỉ số này giúp kiểm tra mô hình có dự đoán tốt ở thời điểm kết thúc session hay không.

---

### 2.5. Effective Recall trên toàn bộ tập test

#### Là gì?

Đây là Recall@20 được tính trên toàn bộ truy vấn test theo giao thức đã chọn, bao gồm việc ghi rõ các truy vấn hợp lệ và các trường hợp target không thể đánh giá theo catalog ứng viên.

#### Dùng để làm gì?

Mục đích là tránh chỉ báo cáo một tập con thuận lợi và giúp kiểm tra độ bao phủ thực tế của giao thức đánh giá.

Chỉ số này được lưu trong bảng `final_model_comparison.csv` dưới tên `full_test_effective_recall@20`.

---

### 2.6. Recommendation count

#### Recommendation count là gì?

Đây là số lần một item xuất hiện trong toàn bộ các danh sách Top-20 được tạo ra trên tập test.

Ví dụ:

```text
Item A xuất hiện trong 4.000 danh sách Top-20
→ recommendation count của A = 4.000
```

Nếu có 40.001 truy vấn và mỗi truy vấn có 20 vị trí, tổng số recommendation slot là:

```text
40.001 × 20 = 800.020 slot
```

#### Dùng để làm gì?

Đây là đại lượng nền tảng để đo popularity bias đầu ra. Từ recommendation count, file 06 tính được:

- Item nào được đề xuất nhiều nhất.
- Item nào gần như không được đề xuất.
- Popular, medium, rare và cold item nhận được bao nhiêu exposure.

---

### 2.7. Warm catalog và warm coverage

#### Warm item là gì?

Warm item là item đã xuất hiện ít nhất một lần trong train và do đó mô hình đã có cơ hội học embedding cho item đó.

Coverage@20 là chỉ số đo độ bao phủ catalog của mô hình.

Catalog là danh sách toàn bộ item/sản phẩm mà hệ thống biết hoặc có thể sử dụng để recommendation.

#### Warm coverage là gì?

Warm Coverage@20 là tỷ lệ warm item đã xuất hiện ít nhất một lần trong các recommendation Top-20:

```text
Warm coverage
= số warm item được đề xuất ít nhất một lần
  / tổng số warm item
```

#### Dùng để làm gì?

Đo độ rộng của catalog mà mô hình thực sự sử dụng.

- Coverage cao: mô hình đề xuất đa dạng item hơn.
- Coverage thấp: recommendation tập trung vào một phần nhỏ catalog.

#### Kết quả chính

| Mô hình | Warm coverage |
| ------- | ------------: |
| GRU4Rec |        0,8851 |
| NARM    |        0,8858 |
| SASRec  |        0,9176 |

SASRec có coverage cao nhất.

---

### 2.8. Gini của recommendation

#### Gini là gì?

Gini đo mức độ tập trung của recommendation count giữa các item.

- Gini thấp: exposure phân bố tương đối đều.
- Gini cao: một số ít item nhận phần lớn recommendation.

Trong file 06, Gini được tính chủ yếu trên warm catalog để tránh cold item chưa học được làm sai diễn giải.

#### Dùng để làm gì?

Đo popularity bias ở đầu ra theo góc nhìn phân phối:

> Mô hình có dồn phần lớn recommendation vào một nhóm nhỏ item không?

#### Kết quả chính

| Mô hình | Warm recommendation Gini |
| ------- | -----------------------: |
| GRU4Rec |                   0,7036 |
| NARM    |                   0,7061 |
| SASRec  |                   0,6885 |

SASRec có Gini thấp nhất trong ba mô hình, còn NARM có Gini cao nhất.

Lưu ý: Gini đầu vào của EDA và recommendation Gini là hai đại lượng khác nhau. Gini đầu vào mô tả dữ liệu interaction; recommendation Gini mô tả đầu ra mô hình.

---

### 2.9. ARP — Average Recommendation Popularity

#### ARP là gì?

ARP là độ phổ biến trung bình của các item được đề xuất.

Có thể hiểu là:

```text
ARP cao
→ các item được đề xuất thường có tần suất train cao
→ mô hình thiên về item phổ biến hơn
```

#### Dùng để làm gì?

Đo popularity bias theo giá trị trung bình của các item được đưa vào Top-20.

#### Kết quả chính

| Mô hình | ARP@20 |
| ------- | -----: |
| GRU4Rec |  77,52 |
| NARM    |  76,57 |
| SASRec  |  73,90 |

SASRec có ARP thấp nhất, cho thấy độ phổ biến trung bình của item được đề xuất thấp hơn hai mô hình còn lại.

---

### 2.10. Exposure theo nhóm popularity

#### Exposure là gì?

Exposure là số cơ hội một item hoặc một nhóm item xuất hiện trong recommendation.

File 06 chia item thành các nhóm:

- Popular.
- Medium.
- Rare.
- Cold.

Sau đó tính tỷ lệ các vị trí Top-20 thuộc về từng nhóm.

#### Dùng để làm gì?

So sánh:

```text
Tỷ lệ item trong catalog
↔
Tỷ lệ vị trí recommendation mà nhóm đó nhận được
```

Nếu popular item chiếm 20% catalog nhưng nhận hơn 60% exposure, đó là dấu hiệu popularity bias rõ.

#### Kết quả chính

| Mô hình | Popular exposure | Rare exposure |
| ------- | ---------------: | ------------: |
| GRU4Rec |           63,96% |         4,27% |
| NARM    |           64,15% |         4,16% |
| SASRec  |           62,49% |         4,73% |

Kết luận:

> Cả ba mô hình đều ưu tiên popular item và dành rất ít exposure cho rare item. SASRec giảm nhẹ mức tập trung so với GRU4Rec và NARM nhưng chưa loại bỏ được bias.

---

## 3. Các chỉ số của file 07

File 07 không đánh giá trực tiếp độ chính xác recommendation. File 07 phân tích vector embedding của item sau huấn luyện.

### 3.1. Embedding norm

#### Norm là gì?

Norm là độ dài của vector embedding:

```text
embedding = [x1, x2, ..., xd]
norm = sqrt(x1² + x2² + ... + xd²)
```

#### Dùng để làm gì?

Kiểm tra xem các nhóm item có độ dài biểu diễn khác nhau không:

- Popular có norm lớn hay nhỏ?
- Rare có norm lớn hay nhỏ?
- Cold có norm ổn định không?

#### Kết quả chung

```text
rare norm > medium norm > popular norm >> cold norm
```

Ví dụ với GRU4Rec:

| Nhóm    | Norm trung bình gần đúng |
| ------- | -----------------------: |
| Popular |                    0,998 |
| Medium  |                    1,186 |
| Rare    |                    1,357 |
| Cold    |                    0,159 |

Kết luận:

> Rare item thường có vector dài hơn popular item, còn cold item có norm rất nhỏ vì chưa được học đầy đủ.

---

### 3.2. Spearman giữa train frequency và embedding norm

#### Là gì?

Đây là tương quan thứ hạng giữa:

```text
Tần suất item trong train ↔ norm embedding
```

Spearman nằm trong khoảng từ `-1` đến `+1`:

- Gần `+1`: frequency tăng thì norm thường tăng.
- Gần `0`: chưa thấy quan hệ thứ hạng rõ.
- Gần `-1`: frequency tăng thì norm thường giảm.

#### Dùng để làm gì?

Kiểm tra xem popularity có liên quan đến độ dài embedding hay không.

#### Kết quả

| Mô hình | Spearman frequency–norm |
| ------- | ----------------------: |
| GRU4Rec |                 -0,7449 |
| NARM    |                 -0,6675 |
| SASRec  |                 -0,7066 |

Kết luận:

> Cả ba mô hình đều có mối liên hệ âm mạnh giữa frequency và norm: item càng phổ biến thì norm có xu hướng càng nhỏ.

Đây là tương quan, không phải bằng chứng nhân quả. Không được nói rằng frequency chắc chắn gây ra norm giảm.

---

### 3.3. Cosine similarity

#### Cosine similarity là gì?

Cosine similarity đo độ giống nhau về hướng giữa hai vector:

```text
Gần 1  → hai vector cùng hướng
Gần 0  → hai vector gần như không liên quan về hướng
Gần -1 → hai vector có hướng ngược nhau
```

#### Dùng để làm gì?

Kiểm tra xem embedding có phản ánh category hay không.

File 07 so sánh:

- Warm item cùng category.
- Warm item khác category.
- Cold item và warm item cùng category.
- Cold item và warm item khác category.

#### Kết quả warm item

| Mô hình | Cùng category | Khác category | Chênh lệch |
| ------- | ------------: | ------------: | ---------: |
| GRU4Rec |         0,605 |         0,175 |      0,429 |
| NARM    |         0,551 |         0,113 |      0,437 |
| SASRec  |         0,556 |         0,146 |      0,410 |

Kết luận:

> Các item cùng category có hướng embedding giống nhau hơn rõ rệt so với item khác category. Vì vậy, mô hình đã học được một phần cấu trúc category từ chuỗi tương tác.

#### Kết quả cold item

Cosine cold–warm cùng category gần bằng 0:

| Mô hình | Cold–warm cùng category |
| ------- | ----------------------: |
| GRU4Rec |                  0,0026 |
| NARM    |                 -0,0006 |
| SASRec  |                  0,0011 |

Kết luận:

> Cold item chưa hình thành được quan hệ category có ý nghĩa vì không xuất hiện trong train và mô hình chỉ sử dụng item ID.

---

### 3.4. Category cosine gap

#### Là gì?

Category cosine gap là chênh lệch giữa cosine trung bình của cặp cùng category và cosine trung bình của cặp khác category:

```text
category cosine gap
= cosine cùng category - cosine khác category
```

#### Dùng để làm gì?

Tóm tắt mức độ mô hình phân biệt item cùng category với item khác category.

- Gap lớn: cấu trúc category thể hiện rõ hơn.
- Gap gần 0: embedding không phân biệt rõ category.

Kết quả:

- GRU4Rec: khoảng `0,429`.
- NARM: khoảng `0,437`.
- SASRec: khoảng `0,410`.
- Cold category gap của cả ba mô hình gần `0`.

---

### 3.5. Khoảng cách tới centroid category

#### Centroid là gì?

Centroid là vector trung bình của các item trong cùng một category.

Nếu category có các item `A`, `B`, `C`, centroid được tính bằng trung bình các vector của A, B và C.

#### Khoảng cách tới centroid là gì?

Đây là khoảng cách từ embedding của từng item tới centroid của category đó.

#### Dùng để làm gì?

Kiểm tra nhóm item nào gần cấu trúc trung tâm của category và nhóm nào bị phân tán hơn.

#### Kết quả chung

```text
Popular gần centroid hơn rare
Rare xa centroid hơn popular
Cold thường xa centroid nhất
```

Ví dụ với GRU4Rec:

| Nhóm    | Khoảng cách tới centroid |
| ------- | -----------------------: |
| Popular |                    0,700 |
| Rare    |                    0,802 |
| Cold    |                    0,947 |

Kết luận:

> Popular item thường có biểu diễn ổn định và đại diện hơn cho category. Rare item phân tán hơn, còn cold item chưa có vị trí đáng tin cậy.

---

### 3.6. Centroid distance gap

#### Là gì?

File 07 cũng tính chênh lệch khoảng cách giữa các nhóm:

```text
rare minus popular
= khoảng cách rare tới centroid - khoảng cách popular tới centroid

cold minus popular
= khoảng cách cold tới centroid - khoảng cách popular tới centroid
```

#### Dùng để làm gì?

Định lượng mức độ rare và cold item bị lệch khỏi cấu trúc trung tâm so với popular item.

- Giá trị dương: nhóm đó xa centroid hơn popular.
- Giá trị càng lớn: khác biệt hình học càng rõ.

---

### 3.7. t-SNE

#### t-SNE là gì?

t-SNE giảm embedding nhiều chiều xuống hai chiều để vẽ.

Mỗi điểm trên hình là một item.

#### Dùng để làm gì?

File 07 dùng t-SNE để quan sát trực quan:

- Item cùng category có tạo thành cụm không.
- Popular, medium, rare có tách biệt không.
- Cold item nằm ở đâu.

#### Kết quả quan sát

- Warm item tạo thành các cụm category tương đối rõ.
- Popular, medium và rare vẫn đan xen nhau.
- Cold item thường nằm tách biệt hoặc phân tán.
- Xu hướng tương tự xuất hiện ở cả hai seed.

#### Lưu ý

t-SNE chỉ là công cụ minh họa. Không được dùng khoảng cách giữa các cụm trên hình t-SNE làm bằng chứng định lượng chính. Bằng chứng chính vẫn là norm, Spearman, cosine và centroid distance.

---

### 3.8. P-value và khoảng tin cậy

File 07 lưu thêm p-value và khoảng tin cậy cho một số phân tích.

#### P-value là gì?

P-value giúp kiểm tra xem quan hệ quan sát được có khó xuất hiện nếu thực tế không có quan hệ hay không.

Thông thường:

- p-value nhỏ: có bằng chứng thống kê rằng quan hệ không chỉ do ngẫu nhiên.
- p-value lớn: chưa đủ bằng chứng thống kê.

P-value không cho biết quan hệ mạnh hay yếu. Độ mạnh phải xem qua Spearman, cosine gap hoặc effect size.

#### Khoảng tin cậy là gì?

Khoảng tin cậy 95% cho biết khoảng giá trị hợp lý của ước lượng khi lặp lại việc lấy mẫu.

Khoảng tin cậy hẹp hơn thường cho thấy ước lượng ổn định hơn.

---

### 3.9. Độ ổn định qua hai seed

File 07 tính trung bình và độ lệch chuẩn qua hai seed.

#### Dùng để làm gì?

Một kết quả chỉ xuất hiện ở một seed có thể là do khởi tạo ngẫu nhiên. Nếu cùng xu hướng xuất hiện ở cả hai seed, kết luận đáng tin cậy hơn.

File 07 cho thấy các xu hướng chính đều lặp lại:

- Spearman frequency–norm âm mạnh.
- Cosine cùng category cao hơn khác category.
- Popular gần centroid hơn rare.
- Cold không có cấu trúc category rõ.

---

## 4. Bảng phân biệt nhanh file 06 và file 07

| Chỉ số/đại lượng               | File 06 |   File 07    | Câu hỏi trả lời                               |
| ------------------------------ | :-----: | :----------: | --------------------------------------------- |
| Recall@20                      |   Có    |    Không     | Mô hình có tìm thấy item đúng không?          |
| MRR@20                         |   Có    |    Không     | Item đúng đứng cao đến đâu?                   |
| NDCG@20                        |   Có    |    Không     | Thứ hạng recommendation có tốt không?         |
| Recommendation count           |   Có    |   Dùng lại   | Item nào được đề xuất nhiều?                  |
| Coverage@20                    |   Có    | Tổng hợp lại | Mô hình dùng rộng catalog đến đâu?            |
| Recommendation Gini            |   Có    | Tổng hợp lại | Recommendation tập trung đến mức nào?         |
| ARP                            |   Có    | Tổng hợp lại | Item được đề xuất phổ biến trung bình ra sao? |
| Exposure theo popularity group |   Có    | Tổng hợp lại | Popular/rare nhận bao nhiêu vị trí?           |
| Embedding norm                 |  Không  |      Có      | Vector item dài bao nhiêu?                    |
| Spearman frequency–norm        |  Không  |      Có      | Frequency có liên quan norm không?            |
| Cosine similarity              |  Không  |      Có      | Item cùng category có gần nhau không?         |
| Category cosine gap            |  Không  |      Có      | Mức phân biệt category là bao nhiêu?          |
| Centroid distance              |  Không  |      Có      | Item cách tâm category bao xa?                |
| t-SNE                          |  Không  |      Có      | Không gian embedding trông như thế nào?       |

---

## 5. Kết luận chung từ các chỉ số

File 06 cho thấy:

> Cả ba mô hình đều có popularity bias ở đầu ra. Popular item được đề xuất nhiều hơn đáng kể, trong khi rare item nhận rất ít exposure. SASRec có accuracy, coverage và mức tập trung recommendation tốt nhất, nhưng vẫn chưa loại bỏ được bias.

File 07 cho thấy:

> Sự khác biệt giữa popular, rare và cold item tồn tại ngay trong không gian embedding. Warm item cùng category có embedding gần nhau hơn, popular item thường gần tâm category hơn, còn cold item gần như chưa học được vị trí phù hợp.

Khi kết hợp hai file:

```text
File 06: phát hiện bias ở đầu ra recommendation
File 07: tìm thấy khác biệt tương ứng trong không gian embedding
```

Vì vậy, hướng cải thiện hợp lý là:

- Giảm sự phụ thuộc vào frequency khi huấn luyện hoặc reranking.
- Tăng khả năng học cho rare item.
- Bổ sung category/metadata để tạo embedding tốt hơn cho cold item.
- Đánh giá đồng thời accuracy và các chỉ số fairness/popularity bias.

---

## 6. Các file kết quả liên quan

### Kết quả của file 06

- `results/tables/model_accuracy.csv`
- `results/tables/popularity_bias.csv`
- `results/tables/exposure_by_group.csv`

### Kết quả của file 07

- `results/tables/norm_frequency.csv`
- `results/tables/norm_frequency_seed_summary.csv`
- `results/tables/cosine_category.csv`
- `results/tables/cosine_seed_summary.csv`
- `results/tables/category_centroid.csv`
- `results/tables/category_centroid_seed_summary.csv`
- `results/statistical_tests/spearman_results.csv`
- `results/statistical_tests/spearman_seed_summary.csv`
- `results/statistical_tests/cosine_tests.csv`
- `results/tables/tsne_coordinates.parquet`
