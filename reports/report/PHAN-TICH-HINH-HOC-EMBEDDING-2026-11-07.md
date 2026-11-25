# Phân tích thực nghiệm hình học item embedding — 07/11/2026

## Phạm vi và cách thực hiện

Đã chạy notebook 07 và 08 trên các artifact huấn luyện hiện có; không huấn luyện lại mô hình. Cả sáu lần chạy GRU4Rec/NARM/SASRec × seed 2024/2025 đều được phân tích. Các bảng trong báo cáo là trung bình số đo của hai seed, không phải trung bình trực tiếp các ma trận embedding.

Mỗi ma trận có 37.583 item × 64 chiều, gồm 29.936 warm và 7.647 controlled cold, không chứa hàng padding. Đã kiểm tra thứ tự ID khớp mapping toàn cục và mapping của từng lần chạy, giá trị hữu hạn, norm tính lại khớp bảng item-level, tổng exposure đúng 800.020 và cold không có tần suất train/exposure. Cả sáu lần đều đạt; độ trôi cold embedding được artifact huấn luyện ghi nhận bằng 0.

Các phép tương quan chính dùng warm item. Controlled cold chỉ dùng để đối chiếu biểu diễn chưa học. Candidate set chính loại cold; không dùng exposure bằng 0 của cold để kết luận về khả năng xếp hạng cold.

## 1. Item phổ biến có norm lớn hơn không?

**Không. Kết quả hiện tại cho xu hướng ngược lại trong warm catalog.**

| Mô hình | Norm popular | Norm medium | Norm rare | Norm cold |
|---|---:|---:|---:|---:|
| GRU4Rec | 0,9978 | 1,1858 | 1,3574 | 0,1594 |
| NARM | 1,0186 | 1,1098 | 1,2176 | 0,1593 |
| SASRec | 0,8300 | 0,9609 | 1,0694 | 0,1593 |

Các giá trị là trung bình norm trên item trong nhóm, sau đó trung bình qua hai seed. Ở cả sáu lần chạy: rare > medium > popular > cold.

| Mô hình | Spearman(tần suất train, norm), warm |
|---|---:|
| GRU4Rec | −0,7449 |
| NARM | −0,6675 |
| SASRec | −0,7066 |

Hệ số âm nghĩa là item có tần suất cao hơn có xu hướng mang norm thấp hơn. Dấu âm nhất quán qua hai seed. Không có cơ sở cho lời giải thích “item phổ biến được gợi ý nhiều vì embedding luôn dài hơn”.

Norm tuyệt đối giữa các kiến trúc phụ thuộc cả thang biểu diễn và tối ưu hóa; norm của SASRec thấp hơn không tự động có nghĩa embedding tốt hơn hoặc ít thiên lệch hơn.

### Vì sao phải tách cold?

Nếu gộp cold, tương quan tần suất–norm đổi thành dương nhẹ: khoảng +0,1112 ở GRU4Rec, +0,1504 ở NARM, +0,1306 ở SASRec. Cold đồng thời có tần suất 0 và norm rất nhỏ, nên việc trộn hai trạng thái “đã học/chưa học” làm thay đổi kết luận.

Cold được khởi tạo với độ lệch chuẩn 0,02 trong 64 chiều và không học; norm khoảng 0,16 phù hợp thang khởi tạo (sqrt(64) × 0,02). Không diễn giải norm nhỏ của cold thành tín hiệu độ phổ biến đã học.

Hình: [Phân bố norm](../../results/figures/norm_boxplot.png), [Tần suất và norm](../../results/figures/frequency_norm_scatter.png).

## 2. Nếu rare có norm lớn hơn, tại sao vẫn ít được gợi ý?

| Mô hình | Spearman(tần suất, rec_count) | Spearman(norm, rec_count) |
|---|---:|---:|
| GRU4Rec | +0,6798 | −0,4875 |
| NARM | +0,6807 | −0,3980 |
| SASRec | +0,6689 | −0,4222 |

Warm item phổ biến hơn vẫn có xu hướng xuất hiện nhiều hơn trong top-20. Quan hệ tổng thể giữa norm và exposure lại âm, nên norm riêng lẻ không giải thích được mức exposure.

Điểm xếp hạng là:

`score(i | session) = sᵀeᵢ = ||s|| × ||eᵢ|| × cos(s,eᵢ)`.

Norm và hướng cùng tham gia chấm điểm. Rare có norm lớn hơn nhưng không bảo đảm cosine với các session test đủ cao để lọt top-20. Đây là khả năng phù hợp với công thức, chưa phải cơ chế đã chứng minh: phân tích này chưa đo phân bố góc session–item hay thực hiện can thiệp chuẩn hóa norm.

### Kiểm soát tần suất để tránh đọc nhầm tương quan tổng thể

Phân tích bổ sung đưa norm và rec_count về hạng trên warm catalog, rồi trừ trung bình hạng trong từng nhóm item có **đúng cùng tần suất train**. Pearson giữa hai phần dư là:

| Mô hình | Tương quan hạng sau loại trung bình theo từng tần suất |
|---|---:|
| GRU4Rec | +0,0753 |
| NARM | +0,1133 |
| SASRec | +0,1405 |

Sau điều chỉnh này, quan hệ norm–exposure đổi thành dương yếu, nhất quán qua hai seed. Điều này khác tương quan tổng thể âm và cho thấy frequency là yếu tố cần kiểm soát. Các con số này là tương quan của phần dư hạng, không gọi là Spearman gốc hay bằng chứng nhân quả.

Hồi quy mô tả `log(1 + rec_count) ~ log(1 + train_freq) + norm` cũng cho hệ số norm dương trong sáu lần chạy, R² khoảng 0,497–0,516. Không so độ lớn hệ số norm thô giữa các mô hình như một thước đo thiên lệch; chưa chuẩn hóa thang đo và mô hình quan sát chưa kiểm soát mọi yếu tố.

Trong các bin rộng rare/medium/popular, tần suất vẫn khác nhau bên trong bin; vì vậy không gọi phân tích chia ba bin là đã kiểm soát hoàn toàn frequency.

Hình: [Norm và số lần được gợi ý](../../results/figures/norm_recommendation_relationship.png).

## 3. Embedding có cấu trúc liên quan category không?

Lấy mẫu 50.000 cặp cho mỗi nhóm, mỗi lần chạy; không ghép item với chính nó trong nhóm warm cùng category.

| Mô hình | Cosine warm cùng category | Cosine warm khác category | Chênh lệch |
|---|---:|---:|---:|
| GRU4Rec | 0,6046 | 0,1754 | 0,4292 |
| NARM | 0,5508 | 0,1134 | 0,4374 |
| SASRec | 0,5565 | 0,1460 | 0,4105 |

Ở cả ba mô hình và cả hai seed, các cặp warm cùng category có cosine trung bình cao hơn rõ. Biểu diễn đã học có cấu trúc liên hệ với category dù category không được đưa vào làm đặc trưng đầu vào.

Đây là thống kê theo thiết kế lấy mẫu hiện có: chọn category với trọng số số warm item trong nhóm cùng category, không lấy đều trên mọi cặp khả dĩ. Các cặp có thể dùng chung item, nên không coi 50.000 cặp là 50.000 đơn vị quan sát độc lập để khẳng định quá mạnh về p-value/CI.

NARM có khoảng cách cosine cùng–khác category lớn nhất trong bảng này, nhưng không có Recall tốt nhất. Khả năng phân biệt category không đồng nhất với khả năng chọn đúng item tiếp theo trong category.

## 4. Cold item có gần warm item cùng category không?

| Mô hình | Cosine cold–warm cùng category | Cosine cold–warm khác category |
|---|---:|---:|
| GRU4Rec | +0,0026 | +0,0007 |
| NARM | −0,0006 | +0,0003 |
| SASRec | +0,0011 | +0,0000 |

Các số đều rất gần 0, trái với mức 0,55–0,60 ở cặp warm cùng category. Chênh lệch cold cùng–khác category nhỏ và không nhất quán về dấu qua các mô hình/seed.

Kết quả phù hợp với thiết kế: cold embedding bị đóng băng ở khởi tạo; metadata category không có đường đi vào encoder để đặt cold gần warm cùng loại. Kết luận áp dụng cho baseline ID-based và giao thức cold hiện tại, không khái quát cho mô hình có nội dung/metadata.

### Kiểm tra bằng tâm category, loại tự đóng góp

Warm item bị loại khỏi tâm dùng để đánh giá chính nó (leave-one-out); category chỉ có một warm item không có điểm warm hợp lệ. Cold so với tâm tất cả warm trong category. Sau đó lấy trung bình đều trên **627 category** có đủ cả hai số đo hợp lệ:

| Mô hình | Cosine warm với tâm warm còn lại | Cosine cold với tâm warm |
|---|---:|---:|
| GRU4Rec | 0,8050 | +0,0046 |
| NARM | 0,7707 | +0,0004 |
| SASRec | 0,7992 | −0,0013 |

Chênh lệch vẫn lớn sau khi bỏ lợi thế warm tự có mặt trong tâm và cân bằng category giữa hai nhóm. Cosine tới tâm và cosine từng cặp là hai số đo khác nhau; không so trực tiếp độ lớn giữa hai bảng.

Khoảng cách Euclidean bị ảnh hưởng bởi norm; không dùng riêng khoảng cách để kết luận cold gần hay xa về nội dung.

## 5. Đọc hình t-SNE

Đã chạy cho seed mô hình 2024, mỗi mô hình lấy 8.000 item gồm 2.000 mỗi nhóm; PCA xuống 32 chiều trước t-SNE, perplexity 30, random seed 2024.

- [Tô màu theo popularity](../../results/figures/tsne_by_popularity.png): các warm group có đan xen; cold xuất hiện thành một vùng dày riêng.
- [Tô màu theo top category](../../results/figures/tsne_by_category.png): warm của nhiều category được minh họa thành các cụm; cold khác màu category trộn lẫn trong vùng cold.

Vùng cold không có nghĩa cold giống nhau về nội dung. Các vector này có norm nhỏ và đều chưa học; biểu diễn t-SNE theo Euclidean/PCA chịu ảnh hưởng của đặc điểm đó. Không dùng khoảng cách cụm 2D để thay thế định lượng trong 64 chiều. Hình hiện chỉ là một seed mô hình và một thiết lập chiếu, chưa phải kiểm tra độ ổn định t-SNE qua nhiều cấu hình.

## 6. Nối hình học với kết quả đầu ra

| Mô hình | Recall@20 | Coverage warm@20 | Gini warm@20 | Spearman frequency–norm |
|---|---:|---:|---:|---:|
| GRU4Rec | 55,70% | 88,51% | 0,7036 | −0,7449 |
| NARM | 56,25% | 88,58% | 0,7061 | −0,6675 |
| SASRec | 57,29% | 91,76% | 0,6885 | −0,7066 |

SASRec giữ lợi thế Recall/coverage/Gini trong các cấu hình đã chạy, nhưng không có mức cosine category cao nhất hay độ lớn tương quan frequency–norm nhỏ nhất. Không có một số đo hình học riêng lẻ trong phân tích này giải thích đầy đủ thứ hạng chất lượng/thiên lệch của ba mô hình.

Chưa nên tuyên bố mô hình nào “gây thiên lệch vì norm” hay “ít thiên lệch vì category cluster tốt hơn”. Chỉ có ba kiến trúc, hai seed và các khác biệt huấn luyện; cần thêm phân tích session–item và thí nghiệm can thiệp để xác định cơ chế.

## 7. Đoạn kết luận dùng để báo cáo

“Phân tích embedding cho thấy tần suất train tương quan âm khá mạnh với norm trên warm item ở cả ba mô hình, trái với giả thuyết đơn giản rằng item phổ biến có vector dài hơn. Trong khi đó, tần suất vẫn tương quan dương với số lần được gợi ý. Sau khi loại trung bình theo từng tần suất, norm có quan hệ dương yếu với exposure, cho thấy cần xét frequency khi diễn giải kết quả.

Embedding warm thể hiện cấu trúc category: cosine cùng category cao hơn khác category. Cold embedding bị đóng băng gần như không có sự thẳng hướng với warm cùng category; kết quả này vẫn giữ sau kiểm tra tâm category loại tự đóng góp. SASRec có chất lượng và độ phủ tốt hơn, nhưng các số đo hình học hiện tại chưa đủ chứng minh nguyên nhân trực tiếp của chênh lệch popularity bias.”

## Artifact và tái lập

- Notebook đã chạy: `notebooks/07_embedding_geometry.ipynb`, `notebooks/08_final_analysis.ipynb` — tổng 9 code cell, không có lỗi.
- Lệnh chạy lại: `python -B scripts/run_notebooks.py` và `python -B scripts/audit.py` từ thư mục gốc.
- Bảng norm/cosine: `results/tables/norm_frequency.csv`, `results/tables/cosine_category.csv`.
- Ba tương quan: `results/tables/bias_geometry_relationship.csv`.
- Tách seed và tổng hợp: `results/tables/seed_stability.csv`, `results/tables/final_model_comparison.csv`.
- Kiểm soát bổ sung: `results/tables/embedding_frequency_category_controls.csv`, `results/tables/category_centroid_leave_one_out.csv`.
- Kiểm tra artifact: `results/tables/embedding_artifact_checks.csv`.
- CI Spearman trong notebook dùng bootstrap trên các hạng tính trước (xấp xỉ bootstrap Spearman), không tái xếp hạng từng mẫu bootstrap. Báo cáo dựa vào hướng/mức độ hiệu ứng và tính nhất quán giữa seed; không dùng CI/p-value để khẳng định tính nhân quả hay độ ổn định qua mọi lần huấn luyện.
