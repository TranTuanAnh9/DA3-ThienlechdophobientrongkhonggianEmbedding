# BTL-K03: Thiên lệch độ phổ biến và hình học không gian embedding trong mô hình SBR

## 1. Tóm tắt đề tài trong một câu

Đề tài yêu cầu huấn luyện ba mô hình gợi ý theo phiên **GRU4Rec, NARM và SASRec** trên dữ liệu **Diginetica**, đo mức độ các mô hình ưu tiên item phổ biến ở danh sách gợi ý, sau đó phân tích embedding của item phổ biến, item hiếm và item cold để tìm hiểu xem thiên lệch ở đầu ra có đi kèm với những khác biệt hình học trong không gian embedding hay không.

Quy trình tổng quát:

```text
Dữ liệu Diginetica
        ↓
Tiền xử lý và thiết lập giao thức cold-start 20%
        ↓
Huấn luyện GRU4Rec, NARM, SASRec với 2 seed
        ↓
Sinh danh sách gợi ý top-20
        ↓
Đo popularity bias ở đầu ra
        ↓
Trích xuất và phân tích item embedding
        ↓
Liên hệ hình học embedding với tần suất item được gợi ý
        ↓
So sánh ba mô hình và đưa ra kết luận chẩn đoán
```

Đây là đề tài **chẩn đoán và phân tích**, không phải đề tài xây dựng một thuật toán khử thiên lệch mới.

---

## 2. Các khái niệm nền tảng

### 2.1. Item là gì?

Item là đối tượng mà hệ thống muốn gợi ý. Trong Diginetica, item thường là một sản phẩm trên website thương mại điện tử.

Ví dụ:

```text
item_101 = một chiếc điện thoại
item_205 = một bộ sạc
item_327 = một chiếc ốp lưng
```

### 2.2. Session là gì?

Session là chuỗi hành động xảy ra trong một lần truy cập. Một session có thể chứa nhiều item theo thứ tự thời gian.

Ví dụ:

```text
Session 01: điện thoại → ốp lưng → sạc dự phòng → tai nghe
```

Từ ba item đầu, mô hình cố gắng dự đoán item tiếp theo là `tai nghe`.

### 2.3. SBR là gì?

**SBR — Session-Based Recommendation** là hệ thống gợi ý dựa trên chuỗi hành động trong phiên hiện tại. Mục tiêu thường là dự đoán item tiếp theo mà người dùng sẽ xem, nhấp hoặc mua.

Đầu vào của mô hình có dạng:

```text
[item_12, item_35, item_08]
```

Đầu ra là điểm dự đoán cho các item ứng viên và một danh sách top-k, chẳng hạn top-20:

```text
[item_41, item_03, item_92, ..., item_17]
```

### 2.4. Popular, rare, warm và cold item

- **Popular item:** xuất hiện nhiều lần trong tập huấn luyện.
- **Rare item:** xuất hiện ít lần trong tập huấn luyện.
- **Warm item:** đã từng xuất hiện trong dữ liệu huấn luyện.
- **Cold item:** không xuất hiện trong dữ liệu huấn luyện, nên mô hình chưa có tương tác để học biểu diễn của item đó.

Cold item có thể được xem là trường hợp cực đoan của phổ tần suất:

```text
Popular: tần suất rất lớn
Medium:  tần suất trung bình
Rare:    tần suất nhỏ nhưng lớn hơn 0
Cold:    tần suất bằng 0 trong train
```

### 2.5. Popularity bias là gì?

Popularity bias, hay thiên lệch độ phổ biến, là hiện tượng mô hình có xu hướng gợi ý các item vốn đã phổ biến nhiều hơn mức hợp lý, trong khi item hiếm hoặc item mới ít được xuất hiện.

Vòng lặp có thể xảy ra:

```text
Item đã phổ biến
      ↓
Được mô hình gợi ý nhiều
      ↓
Nhận thêm nhiều tương tác
      ↓
Càng trở nên phổ biến
```

Đề tài không chỉ muốn biết mô hình dự đoán đúng hay sai, mà còn muốn biết **lượt hiển thị trong top-20 được phân phối cho các item như thế nào**.

### 2.6. Embedding là gì?

Mỗi item ID được mô hình biểu diễn bằng một vector nhiều chiều:

```text
item A = [0.2, 0.8, -0.1, ...]
item B = [0.4, 0.7,  0.0, ...]
```

Nếu embedding dimension bằng 64 thì mỗi item có một vector gồm 64 số. Toàn bộ các vector tạo thành một không gian embedding 64 chiều.

Trong quá trình huấn luyện, vector của các warm item được cập nhật bằng gradient. Item xuất hiện nhiều có thể nhận nhiều lần cập nhật hơn item hiếm. Cold item không xuất hiện trong train thường không nhận được cập nhật có ý nghĩa.

---

## 3. Câu chuyện nghiên cứu của đề tài

Đề tài có hai góc nhìn chính.

### Góc nhìn 1: Đầu ra của mô hình

Câu hỏi là:

> Danh sách top-20 có bị tập trung quá nhiều vào một số item phổ biến hay không?

Các chỉ số chính:

- Catalog Coverage@20.
- Gini của tần suất được gợi ý.
- Average Recommendation Popularity.

### Góc nhìn 2: Bên trong mô hình

Câu hỏi là:

> Embedding của popular, rare và cold item có khác nhau như thế nào?

Các phân tích chính:

- Tương quan giữa tần suất train và norm embedding.
- Cosine similarity của item cùng category và khác category.
- Vị trí của cold item so với warm item cùng category.
- Trực quan hóa bằng t-SNE.

### Phần kết nối hai góc nhìn

Đây là phần tạo chiều sâu cho đề tài:

> Mô hình có popularity bias mạnh ở đầu ra có đồng thời thể hiện sự phân hóa mạnh hơn giữa embedding của popular và rare item hay không?

Cần lưu ý rằng mối liên hệ thống kê không tự động chứng minh quan hệ nhân quả. Kết luận nên dùng các cách diễn đạt như:

- “Có mối liên hệ”.
- “Đi kèm với”.
- “Kết quả gợi ý một cơ chế khả dĩ”.

Không nên kết luận:

- “Norm lớn chắc chắn gây ra popularity bias”.
- “t-SNE chứng minh item phổ biến tạo thành cụm”.

---

## 4. Mục tiêu và câu hỏi nghiên cứu

### 4.1. Mục tiêu tổng quát

Định lượng popularity bias ở đầu ra và liên hệ nó với cấu trúc hình học của item embedding trong ba mô hình SBR, bao gồm vị trí của cold item.

### 4.2. Các câu hỏi bắt buộc từ mô tả đề tài

1. Mô hình nào có popularity bias lớn nhất theo Gini và Coverage@20?
2. Norm embedding có tương quan với tần suất xuất hiện trong train không? Mức độ tương quan ở mỗi mô hình ra sao?
3. Cold item có vị trí như thế nào so với warm item cùng category?

### 4.3. Các câu hỏi phân tích giúp kết nối kết quả

1. Item xuất hiện nhiều trong train có được gợi ý nhiều hơn không?
2. Item có norm embedding lớn có được gợi ý nhiều hơn không?
3. Quan hệ giữa norm và số lần được gợi ý có còn tồn tại khi so sánh các item có tần suất train gần nhau không?
4. Item cùng category có gần nhau hơn item khác category không?
5. Rare item có nằm xa tâm category hơn popular item không?
6. Các kết luận có ổn định qua hai random seed không?

Các câu hỏi ở mục 4.3 làm phần phân tích chặt chẽ hơn, nhưng cần giữ phạm vi phù hợp với thời gian của bài tập lớn.

---

## 5. Dữ liệu Diginetica

### 5.1. Diginetica là gì?

Diginetica là dữ liệu hành vi thương mại điện tử được công bố cho **CIKM Cup 2016**. Dữ liệu gồm các session, lượt xem hoặc tương tác với sản phẩm, thời gian và metadata sản phẩm đã được ẩn danh.

Các nguồn tham khảo:

- [Trang CIKM Cup 2016 trên CodaLab](https://competitions.codalab.org/competitions/11161)
- [Kho RecSysDatasets của RecBole](https://github.com/RUCAIBox/RecSysDatasets)
- [Hướng dẫn chuyển đổi Diginetica](https://github.com/RUCAIBox/RecSysDatasets/blob/master/conversion_tools/usage/DIGINETICA.md)
- [Bản dữ liệu được liên kết trong hướng dẫn RecBole](https://drive.google.com/open?id=0B7XZSACQf0KdXzZFS21DblRxQ3c)

### 5.2. Các file quan trọng

Theo hướng dẫn chuyển đổi của RecBole, ba file quan trọng là:

| File | Nội dung chính | Mục đích sử dụng |
|---|---|---|
| `train-item-views.csv` | Lượt xem item theo session và thời gian | Tạo chuỗi để huấn luyện SBR |
| `product-categories.csv` | Ánh xạ item với category | Phân tích cosine và category |
| `products.csv` | Metadata sản phẩm | Kiểm tra, đối chiếu hoặc mở rộng |

Biểu diễn khái quát của một lượt tương tác:

```text
session_id | item_id | timestamp/order | category
```

### 5.3. Vì sao đề tài chọn Diginetica?

#### Phù hợp với SBR

Dữ liệu chứa chuỗi hành động theo session, phù hợp trực tiếp với bài toán dự đoán item tiếp theo.

#### Có phân bố popular–rare

Dữ liệu thương mại điện tử thường có phân bố đuôi dài: một số ít item xuất hiện nhiều, trong khi phần lớn item xuất hiện ít. Đây là điều kiện cần để nghiên cứu popularity bias.

#### Có category

Ánh xạ category cho phép kiểm tra xem không gian embedding có đặt các item cùng loại gần nhau hay không.

#### Là benchmark phổ biến

Diginetica được sử dụng trong nhiều nghiên cứu SBR, đồng thời có công cụ chuyển đổi để dùng với RecBole.

#### Kích thước khả thi

Dữ liệu đủ lớn để thể hiện thiên lệch nhưng vẫn có thể chạy trên Colab hoặc Kaggle. Số session và item sau tiền xử lý có thể khác giữa các nghiên cứu vì quy tắc lọc khác nhau. Mô tả đề tài nhắc tới khoảng 40.000 item sau khi lọc.

### 5.4. Nên dùng dữ liệu thô hay dữ liệu đã xử lý?

Đề tài cần category và cần tự tạo giao thức cold-start 20%, vì vậy nên ưu tiên dữ liệu thô rồi tiền xử lý theo đúng tài liệu môn học.

Bản RecBole đã xử lý sẵn có thể tiện để chạy thử, nhưng có rủi ro:

- Đã lọc item hoặc session theo quy tắc khác.
- Không giữ đầy đủ category.
- Không sử dụng đúng cold-start 20%.
- Thay đổi tập item được dùng làm mẫu số của Coverage và Gini.

Nếu Mục 1.6 của tài liệu môn học cung cấp dữ liệu hoặc code chia tập riêng thì phải ưu tiên tài liệu đó.

---

## 6. Ba mô hình cần huấn luyện

### 6.1. GRU4Rec

GRU4Rec dùng mạng GRU để đọc chuỗi item trong session. Trạng thái ẩn của GRU tóm tắt những gì đã xảy ra trong phiên và được dùng để dự đoán item tiếp theo.

Khái quát:

```text
item_1 → item_2 → item_3
             ↓ GRU
       biểu diễn session
             ↓
      điểm cho các item
```

### 6.2. NARM

NARM kết hợp GRU và attention. Ngoài việc tóm tắt toàn bộ session, mô hình cố gắng nhận ra item nào trong phiên có ảnh hưởng lớn hơn đến ý định hiện tại.

### 6.3. SASRec

SASRec dùng self-attention để mô hình hóa quan hệ giữa các vị trí trong chuỗi. Nó có thể tập trung vào nhiều vị trí khác nhau mà không chỉ phụ thuộc vào trạng thái tuần tự như GRU.

### 6.4. Yêu cầu so sánh công bằng

Ba mô hình cần được huấn luyện trên cùng:

- Tập train, validation và test.
- Candidate set.
- Quy tắc lọc item/session.
- Cách định nghĩa popular, medium, rare và cold.
- Embedding dimension nếu có thể so sánh thống nhất.
- Top-k bằng 20.
- Hai random seed đã xác định trước.

Các siêu tham số riêng của mô hình có thể khác, nhưng mọi khác biệt phải được ghi lại.

---

## 7. Tiền xử lý dữ liệu và giao thức cold-start

### 7.1. Tiền xử lý session

Các bước thường gặp:

1. Đọc `train-item-views.csv`.
2. Sắp xếp tương tác theo thời gian trong từng session.
3. Loại các bản ghi lỗi hoặc thiếu item ID.
4. Loại session chỉ có một tương tác vì không tạo được cặp đầu vào–mục tiêu.
5. Có thể loại item quá hiếm nếu tài liệu yêu cầu.
6. Ánh xạ item sang category.
7. Chia train, validation và test theo đúng giao thức.

Một session:

```text
[A, B, C, D]
```

có thể sinh các mẫu next-item:

```text
[A]       → B
[A, B]    → C
[A, B, C] → D
```

### 7.2. Không được tự chọn cách chia cold-start tùy ý

“Cold-start 20%” có thể được hiểu theo nhiều cách:

- Chọn 20% item làm cold rồi loại mọi tương tác của chúng khỏi train.
- Dùng chia theo thời gian và xem item chỉ xuất hiện ở test là cold.
- Chọn cold theo category hoặc theo một quy tắc khác.

Do đó phải kiểm tra **Mục 1.6 và Option B** trong tài liệu gốc. Đây là chi tiết quyết định tính đúng đắn của toàn bộ thí nghiệm.

### 7.3. Vấn đề kỹ thuật của cold item trong mô hình ID-based

Nếu một item hoàn toàn không có trong train:

- Nó có thể không được RecBole đưa vào vocabulary.
- Nếu vẫn được đăng ký trong vocabulary, embedding của nó thường chỉ là giá trị khởi tạo vì không nhận gradient.
- Mô hình không có thông tin để hiểu quan hệ giữa cold item và các item khác.
- Việc đưa cold item vào danh sách ứng viên có thể không có ý nghĩa đối với baseline chỉ dựa trên ID.

Vì mô tả đề tài ghi “cold item chỉ để minh họa Option B”, nên cách hiểu an toàn là:

- Phân tích định lượng chính tập trung vào warm item.
- Cold item được giữ để đánh dấu và minh họa trong không gian embedding.
- Không dùng hiệu suất gợi ý cold item làm kết luận chính nếu giao thức không hỗ trợ hợp lệ.

---

## 8. Đánh giá chất lượng dự đoán

Đây không phải trọng tâm chính của đề tài, nhưng nên có một số chỉ số làm **sanity check** để chứng minh mô hình hoạt động bình thường.

Các chỉ số có thể dùng:

- Recall@20.
- MRR@20.
- NDCG@20.

Lý do cần sanity check:

> Một mô hình có Gini thấp chưa chắc tốt; nó có thể gợi ý gần như ngẫu nhiên nên phân phối rất đều nhưng dự đoán không chính xác.

Cần báo cáo đồng thời chất lượng và thiên lệch, nhưng không biến bài toán thành cuộc thi tối ưu accuracy.

---

## 9. Đo popularity bias ở đầu ra

Sau khi huấn luyện, mỗi mô hình sinh top-20 cho các session test. Từ toàn bộ danh sách này, đếm số lần mỗi item được gợi ý.

### 9.1. Recommendation count và exposure share

Với item \(i\):

\[
rec\_count_i = \text{số lần item } i \text{ xuất hiện trong tất cả danh sách top-20}
\]

Nếu có \(N\) session test thì tổng số vị trí gợi ý là:

\[
N \times 20
\]

Exposure share:

\[
exposure\_share_i = \frac{rec\_count_i}{N \times 20}
\]

Ví dụ có 10.000 session test:

```text
Tổng vị trí gợi ý = 10.000 × 20 = 200.000
```

Nếu item A xuất hiện 8.000 lần thì nó chiếm 4% tổng lượt hiển thị.

### 9.2. Catalog Coverage@20

\[
Coverage@20 =
\frac{\text{số item khác nhau xuất hiện trong các top-20}}
{\text{tổng số item hợp lệ trong catalog}}
\]

Diễn giải:

- Coverage cao: mô hình sử dụng nhiều item trong catalog.
- Coverage thấp: mô hình lặp lại một nhóm nhỏ item.

Cần ghi rõ mẫu số là:

- Toàn bộ warm candidate item, hay
- Toàn bộ item catalog bao gồm cold item.

Trong baseline ID-based, dùng tập candidate hợp lệ thường dễ diễn giải hơn. Phải áp dụng cùng một định nghĩa cho cả ba mô hình.

### 9.3. Gini của tần suất được gợi ý

Gini đo mức độ tập trung của `rec_count`:

- Gini gần 0: exposure tương đối đồng đều.
- Gini cao: exposure tập trung vào một số ít item.

Khi tính Gini, cần quyết định có đưa các item được gợi ý 0 lần vào vector hay không. Đối với đo catalog concentration, nên tính cả các candidate item hợp lệ có `rec_count = 0`, đồng thời ghi rõ định nghĩa trong báo cáo.

### 9.4. Average Recommendation Popularity

Chỉ số này đo độ phổ biến trong tập train của các item được gợi ý:

\[
ARP = \frac{1}{N\times K}
\sum_{u=1}^{N}\sum_{i\in Rec_K(u)} popularity(i)
\]

Trong đó `popularity(i)` thường là số lần item xuất hiện trong train. Nếu dùng `log(1 + frequency)` để giảm độ lệch, phải ghi rõ và dùng thống nhất.

Diễn giải:

- ARP cao: mô hình thường gợi ý item vốn đã phổ biến.
- ARP thấp: danh sách có nhiều item ít phổ biến hơn.

### 9.5. Cách kết luận popularity bias

Một mô hình thường được xem là thiên lệch mạnh hơn nếu nó có tổ hợp:

```text
Coverage thấp
+ Gini cao
+ Average Recommendation Popularity cao
```

Không nên dựa vào một chỉ số duy nhất.

---

## 10. Hình học không gian embedding

### 10.1. Trích xuất embedding

Sau khi huấn luyện, lấy ma trận item embedding từ checkpoint của từng mô hình. Trong RecBole, các mô hình thường có tham số tương tự `item_embedding.weight`, nhưng cần kiểm tra đúng implementation và loại bỏ hàng padding nếu có.

Cần bảo đảm:

- Ánh xạ từ internal item ID về raw item ID là chính xác.
- Không nhầm item embedding với position embedding hoặc session representation.
- Embedding được lấy từ đúng checkpoint tốt nhất.
- Embedding của ba mô hình được phân tích theo cùng quy trình.

### 10.2. Norm embedding

Với embedding \(e_i\) có \(d\) chiều:

\[
\lVert e_i\rVert_2 =
\sqrt{e_{i1}^2 + e_{i2}^2 + \dots + e_{id}^2}
\]

Norm là độ dài của vector. Phân tích chính:

\[
Spearman(train\_frequency_i, \lVert e_i\rVert_2)
\]

Diễn giải hệ số Spearman \(\rho\):

- Gần `+1`: item phổ biến hơn có xu hướng có norm lớn hơn.
- Gần `0`: chưa thấy quan hệ đơn điệu rõ ràng.
- Gần `-1`: item phổ biến hơn có xu hướng có norm nhỏ hơn.

Cần báo cáo `p-value` hoặc khoảng tin cậy. Với số lượng item rất lớn, `p-value` có thể rất nhỏ ngay cả khi hiệu ứng yếu, nên phải báo cáo cả độ lớn của \(\rho\).

### 10.3. Vì sao norm có thể liên quan đến điểm gợi ý?

Nhiều mô hình tính điểm gần với dot product:

\[
score(i) = h^\top e_i
\]

Trong đó:

- \(h\) là vector biểu diễn session.
- \(e_i\) là item embedding.

Về mặt hình học:

\[
h^\top e_i = \lVert h\rVert \lVert e_i\rVert \cos(\theta)
\]

Vì vậy, nếu hướng tương đối thuận lợi, item có norm lớn có thể dễ đạt điểm cao hơn. Tuy nhiên, đây chỉ là cơ chế khả dĩ; score còn phụ thuộc vào hướng vector, kiến trúc, bias term và quá trình huấn luyện.

### 10.4. Cosine similarity

Cosine similarity giữa hai embedding:

\[
cos(e_i,e_j)=
\frac{e_i^\top e_j}
{\lVert e_i\rVert\lVert e_j\rVert}
\]

So sánh tối thiểu:

- Warm–warm cùng category.
- Warm–warm khác category.

Có thể bổ sung để minh họa:

- Cold–warm cùng category.

Nếu cosine trung bình của cặp cùng category cao hơn cặp khác category, mô hình đã học được một phần cấu trúc category từ dữ liệu tương tác.

Không nên tạo toàn bộ ma trận cosine cho khoảng 40.000 item:

\[
40.000 \times 40.000 = 1{,}6 \text{ tỷ cặp}
\]

Nên lấy mẫu số lượng cặp bằng nhau giữa các nhóm, cố định seed lấy mẫu và báo cáo khoảng tin cậy.

### 10.5. Khoảng cách đến tâm category

Đây là phân tích bổ sung hữu ích. Tâm của category \(c\):

\[
centroid_c = \frac{1}{N_c}\sum_{i\in c}e_i
\]

Sau đó đo khoảng cách hoặc cosine giữa từng item với tâm category của nó.

Ví dụ giả định:

| Nhóm | Khoảng cách trung bình đến tâm category |
|---|---:|
| Popular warm | 0.42 |
| Medium warm | 0.58 |
| Rare warm | 0.79 |
| Cold | 1.31 |

Nếu popular item gần tâm category hơn rare item, điều đó cho thấy chất lượng hoặc mức độ ổn định của biểu diễn có thể khác theo tần suất.

### 10.6. t-SNE

t-SNE giảm embedding 64/128 chiều xuống hai chiều để trực quan hóa. Mỗi điểm là một item.

Nên tạo ít nhất:

1. Hình tô màu theo bin tần suất: popular, medium, rare.
2. Hình tô màu theo category.
3. Đánh dấu cold item bằng ký hiệu riêng.

Chỉ lấy mẫu khoảng 5.000–10.000 item. Cần ghi:

- Số điểm.
- Perplexity.
- Random seed.
- Các bước tiền xử lý như PCA trước t-SNE nếu có.

Giới hạn diễn giải:

- t-SNE ưu tiên bảo toàn cấu trúc lân cận cục bộ.
- Khoảng cách giữa hai cụm trên hình không nhất thiết phản ánh đúng khoảng cách trong không gian gốc.
- Hình có thể thay đổi theo seed và perplexity.
- Kết luận định lượng phải dựa vào norm, cosine, centroid hoặc các số đo trong không gian gốc.

---

## 11. Bảng phân tích item-level

Sau khi huấn luyện mỗi mô hình, nên tạo một bảng trung tâm. Đây là dữ liệu nối phần đầu ra với phần embedding.

| Cột | Ý nghĩa |
|---|---|
| `model` | GRU4Rec, NARM hoặc SASRec |
| `seed` | Seed thí nghiệm |
| `item_id` | ID item gốc |
| `category_id` | Category của item |
| `train_freq` | Số lần xuất hiện trong train |
| `popularity_group` | popular, medium, rare hoặc cold |
| `is_cold` | Item có cold hay không |
| `embedding_norm` | Norm L2 của embedding |
| `rec_count` | Số lần xuất hiện trong các top-20 |
| `exposure_share` | Tỷ trọng trên toàn bộ vị trí top-20 |
| `distance_to_category_centroid` | Khoảng cách đến tâm category, nếu tính |

Ví dụ minh họa:

| item_id | category | train_freq | group | cold | norm | rec_count | exposure |
|---|---|---:|---|---:|---:|---:|---:|
| 101 | phone | 5.000 | popular | 0 | 4.82 | 8.200 | 0.0410 |
| 102 | phone | 120 | medium | 0 | 2.10 | 250 | 0.00125 |
| 103 | phone | 5 | rare | 0 | 0.91 | 2 | 0.00001 |
| 104 | phone | 0 | cold | 1 | 0.36 | NA | NA |

### Quy tắc chia popularity group

Không có một cách duy nhất. Có thể dùng:

- Quantile theo `train_freq`.
- Quantile theo `log1p(train_freq)`.
- Nhóm head/middle/tail theo tỷ trọng tương tác tích lũy.
- Các khoảng tần suất cố định nếu tài liệu môn học quy định.

Điều quan trọng là:

- Xác định quy tắc trước khi so sánh mô hình.
- Dùng cùng quy tắc cho cả ba mô hình và hai seed.
- Cold luôn có `train_freq = 0` và tách khỏi các quantile warm.
- Ghi rõ số item và số tương tác trong mỗi nhóm.

---

## 12. Liên hệ hình học embedding với tần suất được gợi ý

### 12.1. Ba quan hệ cốt lõi

Từ bảng item-level, tính ba tương quan Spearman:

| Tương quan | Câu hỏi |
|---|---|
| `train_freq ↔ embedding_norm` | Item phổ biến có norm lớn hơn không? |
| `train_freq ↔ rec_count` | Item phổ biến có được gợi ý nhiều hơn không? |
| `embedding_norm ↔ rec_count` | Item có norm lớn có được gợi ý nhiều hơn không? |

Ví dụ giả định:

| Mô hình | freq–norm | freq–rec | norm–rec |
|---|---:|---:|---:|
| GRU4Rec | 0.31 | 0.61 | 0.35 |
| NARM | 0.52 | 0.70 | 0.54 |
| SASRec | 0.76 | 0.84 | 0.73 |

Nếu SASRec đồng thời có:

- Gini cao nhất.
- Coverage thấp nhất.
- Tương quan `freq–norm` mạnh nhất.
- Tương quan `norm–rec` mạnh nhất.

thì có thể nói popularity bias ở đầu ra **đi kèm với** sự phân hóa embedding mạnh hơn theo tần suất.

### 12.2. Biến gây nhiễu: train frequency

Một item phổ biến có thể vừa có norm lớn vừa được gợi ý nhiều chỉ vì nó xuất hiện nhiều trong train:

```text
train_freq ──→ embedding_norm
     │
     └───────→ rec_count
```

Khi đó tương quan `norm–rec` chưa chứng minh norm có vai trò độc lập.

#### Cách kiểm tra đơn giản

Trong từng popularity bin, tính lại tương quan giữa norm và rec_count. Khi các item có tần suất gần nhau, ảnh hưởng của frequency được giảm bớt.

#### Cách kiểm tra nâng cao

Dùng mô hình hồi quy:

\[
\log(1+rec\_count_i)=
\beta_0+
\beta_1\log(1+train\_freq_i)+
\beta_2 norm_i
\]

Nếu \(\beta_2\) vẫn dương sau khi kiểm soát frequency, có bằng chứng mạnh hơn rằng norm chứa thông tin liên quan đến mức exposure. Tuy nhiên, đây vẫn là nghiên cứu quan sát, không phải thí nghiệm nhân quả.

### 12.3. Các biểu đồ nên có

1. Scatter plot `log train_freq` và `embedding_norm`.
2. Scatter plot `log train_freq` và `log(1 + rec_count)`.
3. Scatter plot `embedding_norm` và `log(1 + rec_count)`, tô màu theo popularity group.
4. Boxplot norm theo popular/medium/rare/cold.
5. Boxplot rec_count hoặc exposure theo popularity group.
6. Biểu đồ t-SNE theo frequency bin.
7. Biểu đồ t-SNE theo category, đánh dấu cold item.

---

## 13. Cách đọc kết quả cold item

Cold item không được học từ tương tác nên embedding của nó thường là vector khởi tạo. Vì vậy:

- Norm của cold item có thể phản ánh initializer hơn là dữ liệu.
- Vị trí t-SNE có thể ngẫu nhiên.
- Cosine với warm item cùng category có thể gần 0.
- Category của cold item không tự động được mô hình ID-based sử dụng.

Một kết luận phù hợp:

> Cold item không nằm gần các warm item cùng category và không hình thành cấu trúc category rõ ràng. Kết quả minh họa giới hạn của baseline chỉ dựa trên item ID: khi item không xuất hiện trong train, mô hình không có tín hiệu để học một biểu diễn có ý nghĩa cho nó.

Không nên kết luận:

> Cold item nằm xa vì mô hình cố tình thiên vị nó.

Cold-start và popularity bias liên quan nhưng không hoàn toàn giống nhau:

- Popularity bias nói về sự phân bổ attention/exposure theo độ phổ biến.
- Cold-start nói về việc mô hình thiếu thông tin để biểu diễn hoặc dự đoán item hoàn toàn mới.

---

## 14. Ví dụ về cách tổng hợp kết quả

Giả sử thí nghiệm cho bảng sau:

| Mô hình | Recall@20 | Coverage@20 ↑ | Gini ↓ | ARP ↓ | Spearman freq–norm |
|---|---:|---:|---:|---:|---:|
| GRU4Rec | 0.41 | 0.32 | 0.62 | 125 | 0.31 |
| NARM | 0.44 | 0.25 | 0.70 | 158 | 0.52 |
| SASRec | 0.46 | 0.17 | 0.81 | 213 | 0.76 |

Một kết luận hợp lý:

> Trong kết quả giả định, SASRec đạt Recall@20 cao nhất nhưng có Coverage@20 thấp nhất, Gini và ARP cao nhất. Điều này cho thấy mô hình tạo danh sách chính xác hơn nhưng tập trung mạnh hơn vào item phổ biến. SASRec cũng có tương quan mạnh nhất giữa tần suất train và norm embedding. Phân tích item-level cho thấy item có norm lớn xuất hiện thường xuyên hơn trong top-20, kể cả khi quan hệ suy yếu sau khi kiểm soát tần suất. Các bằng chứng cho thấy popularity bias ở đầu ra đi kèm với sự phân hóa embedding theo độ phổ biến, nhưng chưa đủ để kết luận norm là nguyên nhân trực tiếp.

Lưu ý toàn bộ số trong bảng chỉ là ví dụ minh họa, không phải kết quả dự kiến phải đạt được.

---

## 15. Hai seed và độ ổn định

Đề tài yêu cầu kiểm tra trên hai seed. Mỗi mô hình cần chạy ít nhất hai lần với cùng cấu hình nhưng seed khác nhau.

Ví dụ:

```text
seed = 2024
seed = 2025
```

Cần kiểm tra:

- Thứ hạng mô hình theo Gini có đổi không?
- Thứ hạng theo Coverage có đổi không?
- Dấu và độ lớn của Spearman có ổn định không?
- Hình t-SNE có thay đổi quá nhiều không?

Báo cáo có thể dùng:

```text
mean ± standard deviation qua 2 seed
```

Với chỉ hai seed, độ lệch chuẩn chỉ cung cấp dấu hiệu ban đầu, không phải ước lượng độ bất định rất chắc chắn. Có thể dùng bootstrap trên item để bổ sung khoảng tin cậy cho các thống kê item-level.

---

## 16. Công cụ dự kiến

### RecBole/PyTorch

- Huấn luyện GRU4Rec, NARM, SASRec.
- Sinh top-20.
- Lưu checkpoint.
- Trích xuất embedding.

### pandas và NumPy

- Tiền xử lý dữ liệu.
- Đếm tần suất train và recommendation count.
- Tạo bảng item-level.
- Tính Coverage và Gini.

### SciPy

- `scipy.stats.spearmanr` để tính Spearman và p-value.

### scikit-learn

- PCA nếu cần giảm chiều sơ bộ.
- t-SNE để trực quan hóa.
- Cosine similarity hoặc nearest neighbors.

### Matplotlib/Seaborn

- Boxplot, scatter plot, histogram và t-SNE plot.

### Môi trường chạy

- GPU Colab hoặc Kaggle cho huấn luyện.
- CPU có thể chạy phần thống kê và t-SNE mẫu 5.000–10.000 item.

Ước lượng trong đề bài: ba mô hình × hai seed khoảng 5–10 giờ GPU, nhưng thời gian thực tế phụ thuộc cấu hình và cách tiền xử lý.

---

## 17. Kế hoạch thực hiện trong 10 tuần

### Tuần 1–2: Dữ liệu và thiết kế thí nghiệm

- Tải dữ liệu thô.
- Đọc Mục 1.5, 1.6 và Option B.
- Khảo sát schema và thống kê dữ liệu.
- Xác định cách chia train/validation/test.
- Xác định cold-start 20%.
- Xác định popularity bins và candidate catalog.
- Viết tài liệu định nghĩa các chỉ số trước khi chạy.

### Tuần 3: Mô hình đầu tiên

- Chuyển dữ liệu sang định dạng RecBole.
- Huấn luyện một mô hình, chẳng hạn GRU4Rec.
- Sinh top-20.
- Tính Coverage, Gini và ARP.
- Kiểm tra Recall/MRR/NDCG như sanity check.

Mốc 1: một mô hình chạy hoàn chỉnh và có chỉ số đầu ra.

### Tuần 4–5: Hoàn thành ba mô hình

- Huấn luyện NARM và SASRec.
- Chuẩn hóa việc lưu checkpoint, top-20 và log.
- Chạy seed thứ nhất cho cả ba mô hình.
- Kiểm tra sự công bằng của candidate set và cách đánh giá.

### Tuần 6: Norm và tần suất

- Trích xuất embedding đúng item mapping.
- Tính norm.
- Tạo bảng item-level.
- Tính Spearman frequency–norm.
- Vẽ scatter và boxplot.

Mốc 2: đủ ba mô hình và có phân tích norm–frequency.

### Tuần 7: Cosine và category

- Lấy mẫu cặp cùng/khác category.
- Tính cosine và khoảng tin cậy.
- Nếu đủ thời gian, tính khoảng cách đến category centroid.
- Phân tích warm/rare/cold một cách thận trọng.

### Tuần 8: Liên hệ embedding–recommendation

- Tính rec_count và exposure share theo item.
- Tính ba tương quan cốt lõi.
- Kiểm tra `norm–rec_count` trong từng frequency bin.
- Tổng hợp bảng so sánh ba mô hình.

### Tuần 9: t-SNE và seed thứ hai

- Chạy đủ seed thứ hai.
- Tạo t-SNE với tham số được ghi lại.
- Kiểm tra kết luận có ổn định không.
- Chọn hình và bảng quan trọng.

### Tuần 10: Báo cáo và slide

- Viết báo cáo 12–16 trang.
- Hoàn thiện notebook.
- Viết giới hạn nghiên cứu.
- Chuẩn bị slide và kịch bản thuyết trình.

---

## 18. Phân công cho hai sinh viên

Theo mô tả đề tài:

### Sinh viên 1: Đầu ra và popularity bias

- Tiền xử lý và huấn luyện GRU4Rec, NARM, SASRec.
- Sinh top-20.
- Tính Coverage@20, Gini và ARP.
- Phân tích chỉ số theo nhóm item.
- Lưu kết quả theo mô hình và seed.

### Sinh viên 2: Embedding geometry

- Trích xuất item embedding.
- Tính norm và Spearman.
- Phân tích cosine cùng/khác category.
- Trực quan hóa t-SNE.
- Đánh dấu và diễn giải cold item.

### Làm chung

- Chốt protocol trước khi chạy.
- Xây dựng bảng item-level thống nhất.
- Liên hệ output bias với embedding geometry.
- Kiểm tra kết luận qua hai seed.
- Viết báo cáo và slide.

Hai phần không được làm độc lập hoàn toàn. Cả hai thành viên phải thống nhất item mapping, split, popularity group và seed để có thể nối kết quả.

---

## 19. Sản phẩm cần nộp

Theo mô tả đề tài:

1. Notebook tính chỉ số popularity bias.
2. Notebook phân tích embedding và tạo hình t-SNE.
3. Báo cáo khoảng 12–16 trang.
4. Slide thuyết trình.

Nên lưu thêm để tái lập:

- File cấu hình từng mô hình.
- Danh sách seed.
- Thống kê split.
- Danh sách cold item.
- Bảng item-level của từng mô hình/seed.
- File top-20 hoặc recommendation count.
- Phiên bản thư viện và môi trường chạy.

---

## 20. Cấu trúc báo cáo đề xuất

### 1. Giới thiệu

- SBR là gì?
- Popularity bias là gì?
- Vì sao cần nhìn cả đầu ra và embedding?
- Mục tiêu và câu hỏi nghiên cứu.

### 2. Kiến thức liên quan

- GRU4Rec, NARM, SASRec.
- Item embedding.
- Popularity bias.
- Norm, cosine, Spearman và t-SNE.

### 3. Dữ liệu và phương pháp

- Diginetica.
- Tiền xử lý.
- Giao thức cold-start 20%.
- Train/validation/test.
- Định nghĩa popularity groups.
- Cấu hình mô hình và seed.
- Định nghĩa chính xác từng chỉ số.

### 4. Kết quả đầu ra

- Sanity-check accuracy.
- Coverage@20.
- Gini.
- ARP.
- So sánh ba mô hình và hai seed.

### 5. Phân tích embedding

- Norm–frequency.
- Cosine cùng/khác category.
- Khoảng cách centroid nếu có.
- t-SNE.
- Cold item.

### 6. Liên hệ hai góc nhìn

- Frequency–recommendation count.
- Norm–recommendation count.
- Phân tích trong từng frequency bin.
- Mô hình nào có cả output bias và embedding separation mạnh nhất?

### 7. Thảo luận và giới hạn

- Không khẳng định nhân quả.
- Hạn chế của t-SNE.
- Cold embedding chưa được học.
- Chỉ hai seed.
- Kết quả phụ thuộc preprocessing và định nghĩa catalog.

### 8. Kết luận

- Trả lời trực tiếp ba câu hỏi bắt buộc.
- Tóm tắt bằng số liệu.
- Không tuyên bố vượt quá bằng chứng.

---

## 21. Các hướng cải tiến có thể suy ra nhưng nằm ngoài phạm vi

Mục này dùng để hiểu ý nghĩa ứng dụng của kết quả. Mô tả đề tài ghi rõ **không đề xuất hoặc triển khai phương pháp khử thiên lệch và không hiệu chỉnh embedding**. Vì vậy không đưa các hướng sau thành đóng góp chính nếu chưa được giảng viên cho phép.

### Nếu popular item có norm quá lớn

Các hướng nghiên cứu sau này có thể gồm:

- Chuẩn hóa embedding và dùng cosine scoring.
- Regularization norm.
- Ràng buộc độ lớn embedding.

### Nếu bias mạnh nhưng không liên quan đến norm

Có thể nghiên cứu:

- Frequency-aware sampling.
- Weighted loss cho rare item.
- Inverse-propensity weighting.
- Điều chỉnh negative sampling.

### Nếu top-20 quá tập trung

Có thể dùng re-ranking với penalty độ phổ biến:

\[
score'(i)=score(i)-\lambda\log(1+frequency_i)
\]

Hoặc tối ưu đồng thời accuracy, diversity và catalog exposure.

### Nếu rare item có cấu trúc category yếu

Có thể nghiên cứu:

- Auxiliary category loss.
- Contrastive learning cho item cùng category.
- Chia sẻ biểu diễn qua metadata.

### Nếu cold item có embedding ngẫu nhiên

Mô hình chỉ dùng item ID không thể xử lý triệt để item hoàn toàn mới. Cần thêm side information:

- Category.
- Tên hoặc mô tả sản phẩm.
- Giá, brand và thuộc tính.
- Hình ảnh sản phẩm.

Khi đó có thể dùng mô hình hybrid hoặc content-aware để sinh embedding cho item mới.

Các hướng trên là bước nghiên cứu tiếp theo, không thuộc sản phẩm bắt buộc của BTL-K03.

---

## 22. Những lỗi dễ gặp

### Lỗi 1: Gọi mọi item hiếm là cold

Rare item vẫn có mặt trong train; cold item không có mặt trong train. Hai nhóm không giống nhau.

### Lỗi 2: Để cold item lọt vào train

Nếu item được chọn là cold nhưng vẫn có một số tương tác trong train, giao thức bị rò rỉ.

### Lỗi 3: RecBole loại bỏ cold item khỏi vocabulary

Khi đó không thể trích embedding để minh họa. Phải thiết kế data pipeline theo Option B.

### Lỗi 4: So sánh ba mô hình trên candidate set khác nhau

Kết quả Coverage và Gini sẽ không còn công bằng.

### Lỗi 5: Tính Gini nhưng bỏ các item có rec_count bằng 0

Việc này có thể làm thay đổi mạnh ý nghĩa đo catalog concentration.

### Lỗi 6: Kết luận chỉ dựa trên t-SNE

t-SNE chỉ là hình minh họa. Kết luận phải dựa vào số đo trong không gian gốc.

### Lỗi 7: Xem p-value nhỏ là hiệu ứng mạnh

Với hàng chục nghìn item, một tương quan rất nhỏ vẫn có thể có p-value nhỏ. Phải nhìn cả \(\rho\) và khoảng tin cậy.

### Lỗi 8: Khẳng định quan hệ nhân quả

Tương quan giữa frequency, norm và rec_count không chứng minh norm gây ra bias.

### Lỗi 9: Chỉ báo cáo bias mà không kiểm tra mô hình có hoạt động

Nên có Recall/MRR/NDCG làm sanity check để tránh khen một mô hình “ít bias” chỉ vì gợi ý kém.

### Lỗi 10: Vượt phạm vi sang xây dựng phương pháp debias

Đề tài chỉ chẩn đoán. Không triển khai debiasing trước khi hoàn thành đầy đủ ba mốc bắt buộc và có sự cho phép của giảng viên.

---

## 23. Checklist hoàn thành

### Dữ liệu và protocol

- [ ] Có dữ liệu Diginetica và ánh xạ category.
- [ ] Đã đọc Mục 1.5, Mục 1.6 và Option B.
- [ ] Có mô tả rõ train/validation/test.
- [ ] Có danh sách cold item và xác nhận không rò rỉ vào train.
- [ ] Có định nghĩa candidate catalog.
- [ ] Có định nghĩa popular/medium/rare.

### Huấn luyện

- [ ] GRU4Rec chạy đủ hai seed.
- [ ] NARM chạy đủ hai seed.
- [ ] SASRec chạy đủ hai seed.
- [ ] Lưu checkpoint, config và log.
- [ ] Có sanity-check accuracy.

### Popularity bias

- [ ] Coverage@20 cho ba mô hình.
- [ ] Gini cho ba mô hình.
- [ ] Average Recommendation Popularity cho ba mô hình.
- [ ] Mọi chỉ số dùng cùng catalog và định nghĩa.

### Embedding

- [ ] Ánh xạ embedding về đúng raw item ID.
- [ ] Tính norm cho từng item.
- [ ] Spearman norm–frequency kèm p-value hoặc khoảng tin cậy.
- [ ] Cosine cùng category và khác category.
- [ ] t-SNE có perplexity, seed và chú thích giới hạn.
- [ ] Cold item được đánh dấu và diễn giải thận trọng.

### Liên hệ hai góc nhìn

- [ ] Có train frequency, norm và rec_count trên cùng bảng item-level.
- [ ] Có tương quan frequency–rec_count.
- [ ] Có tương quan norm–rec_count.
- [ ] Nếu có thể, kiểm tra norm–rec_count trong cùng frequency bin.
- [ ] So sánh kết luận qua hai seed.
- [ ] Không khẳng định quan hệ nhân quả.

### Sản phẩm

- [ ] Notebook popularity bias.
- [ ] Notebook embedding geometry và t-SNE.
- [ ] Báo cáo 12–16 trang.
- [ ] Slide.
- [ ] Kết luận trả lời trực tiếp ba câu hỏi bắt buộc.

---

## 24. Phát biểu cuối cùng để ghi nhớ

Đề tài không chỉ làm:

```text
Train 3 mô hình → tính vài chỉ số → vẽ t-SNE
```

Mà phải làm thành chuỗi lập luận:

```text
Tần suất xuất hiện trong train
            ↓
Đặc điểm hình học của embedding
            ↓
Tần suất xuất hiện trong top-20
            ↓
Coverage, Gini và độ phổ biến trung bình
```

Mục tiêu cuối là xác định:

1. Mô hình nào thiên lệch nhất ở đầu ra?
2. Popular, rare và cold item khác nhau như thế nào trong không gian embedding?
3. Những khác biệt embedding đó có đi kèm với số lần item được gợi ý hay không?
4. Kết luận có ổn định qua hai seed không?

Đề tài dừng ở việc cung cấp bằng chứng chẩn đoán. Các giải pháp như normalization, weighted loss, re-ranking hoặc mô hình content-aware chỉ là hướng nghiên cứu tiếp theo và không thuộc phạm vi triển khai bắt buộc.
