# Báo cáo tiến độ ngày 06/11/2026

**Đề tài:** Thiên lệch độ phổ biến và hình học không gian embedding trong mô hình gợi ý theo phiên (SBR).

Nội dung được đối chiếu với các kết quả đã lưu trong dự án; chưa chạy lại thí nghiệm trong lần chuẩn bị báo cáo này.

## 1. Những phần đã đạt được

### Phân tích dữ liệu Diginetica

- Đã thực thi notebook EDA, xuất bảng thống kê và biểu đồ về chất lượng dữ liệu, độ dài phiên, thời gian, category và độ phổ biến của item.
- Trên dữ liệu thô, 10% item phổ biến nhất chiếm khoảng **62,53% lượt tương tác**, cho thấy dữ liệu có phân bố đuôi dài và phù hợp để nghiên cứu thiên lệch độ phổ biến.

### Tiền xử lý và thiết lập thí nghiệm

- Đã xây dựng pipeline lọc dữ liệu, sắp thứ tự tương tác trong phiên, chia train/validation/test theo thời gian và tạo mẫu prefix → next item.
- Tập train cuối có **150.783 phiên**, **29.936 warm item**, **649.632 tương tác**, tạo **498.849 mẫu huấn luyện**.
- Validation có **29.828 mẫu**; test warm có **40.001 mẫu**, từ **12.791 phiên**.
- Đã giữ riêng **7.647 controlled cold item**, tương ứng gần 20% tập item tại bước lựa chọn cold; các item này không xuất hiện trong train. Không lấy 7.647 chia cho catalog cuối để diễn giải tỷ lệ thiết kế 20%.
- Giao thức cold hiện là **giả định được ghi rõ trong cấu hình**: chọn item theo category, giữ lại warm item cùng category để đối chiếu. Do chưa có nội dung gốc Mục 1.6/Option B trong tài liệu đã cung cấp, cần đối chiếu quy định môn học trước khi chốt giao thức chính thức.
- Đã phân nhóm popular/medium/rare theo tần suất trong train và lưu mapping item/category.
- Bảng audit tiền xử lý ghi nhận **23/23 kiểm tra đạt**, gồm không trùng phiên giữa các tập, đúng ranh giới thời gian, cold không lọt vào train và mapping hợp lệ.

### Huấn luyện và lưu kết quả

- Đã thực thi huấn luyện **GRU4Rec, NARM và SASRec**, mỗi mô hình với **2 seed: 2024 và 2025**, tổng cộng **6 lần chạy**.
- Mỗi lần chạy có checkpoint, lịch sử huấn luyện, cấu hình, chỉ số test, top-20 và item embedding đã xuất.
- Đã đo Recall@20, MRR@20, NDCG@20 và các chỉ số popularity bias: Coverage@20, Gini, ARP, exposure theo nhóm item.
- Đã có biểu đồ so sánh coverage, Gini, độ phổ biến trung bình và phân bố exposure.

## 2. Kết quả định lượng bước đầu

Các số dưới đây là **trung bình cộng của hai seed**, trên cùng giao thức **warm test, all-prefix**, gồm 40.001 mẫu. Coverage và Gini tính trên catalog 29.936 warm item.

| Mô hình | Recall@20 | MRR@20 | Coverage@20 | Gini | ARP |
|---|---:|---:|---:|---:|---:|
| GRU4Rec | 55,70% | 0,1989 | 88,51% | 0,7036 | 77,52 |
| NARM | 56,25% | 0,1978 | 88,58% | 0,7061 | 76,57 |
| SASRec | 57,29% | 0,2079 | 91,76% | 0,6885 | 73,90 |

Diễn giải để trình bày:

- **SASRec đang cho kết quả tốt nhất trong các cấu hình đã chạy:** Recall và coverage cao nhất, Gini và ARP thấp nhất. Xu hướng này xuất hiện ở cả hai seed.
- Nhóm popular chiếm khoảng **20,42% warm catalog**, nhận khoảng **62,46–64,23% lượt hiển thị gợi ý** qua sáu lần chạy. Nhóm này vốn chiếm **60,65% tương tác train**, nên cần so với cả phân bố train khi đánh giá mức khuếch đại thiên lệch.
- Nhóm rare chiếm khoảng **23,00% warm catalog**, nhưng chỉ nhận khoảng **4,11–4,78% lượt hiển thị**, thấp hơn tỷ trọng **5,78% tương tác train**.
- Gini thể hiện mức tập trung exposure; kết hợp ARP và exposure theo nhóm để diễn giải thiên lệch về item phổ biến.
- Chênh lệch GRU4Rec và NARM nhỏ, thứ tự coverage/Gini thay đổi theo seed; chưa nên khẳng định mô hình nào trong hai mô hình luôn thiên lệch hơn.

## 3. Phần còn lại

- Đã có mã phân tích embedding và embedding xuất từ sáu lần chạy, nhưng notebook `07_embedding_geometry.ipynb` chưa có kết quả thực thi được lưu.
- Cần chạy phân tích tương quan tần suất–norm, cosine/khoảng cách theo category và trực quan hóa embedding, gồm vị trí cold item so với warm item.
- Notebook `08_final_analysis.ipynb` chưa có kết quả thực thi được lưu; cần tổng hợp độ biến động giữa seed và liên hệ cấu trúc embedding với exposure đầu ra.
- Chưa có cơ sở kết luận hình học embedding gây ra popularity bias. Đây là câu hỏi cần phân tích tiếp.

Lưu ý: Các chỉ số chính hiện tại đánh giá warm item. Cold item được giữ để phân tích biểu diễn và không nằm trong candidate set warm; việc cold nhận 0 exposure không phải bằng chứng thực nghiệm độc lập cho thấy mô hình đã thử và thất bại khi gợi ý cold.

## 4. Đề xuất 5 slide báo cáo tiến độ

1. **Mục tiêu:** đo thiên lệch ở top-20 và tìm mối liên hệ với hình học item embedding.
2. **Dữ liệu và pipeline:** một biểu đồ đuôi dài, sơ đồ chia tập và các số liệu train/test chính.
3. **Phần đã hoàn thành:** EDA, preprocessing, sáu lần huấn luyện và các artifact đã xuất.
4. **Kết quả bước đầu:** bảng so sánh ba mô hình cùng biểu đồ Gini hoặc coverage; nhấn mạnh kết quả SASRec và exposure của popular/rare.
5. **Bước tiếp theo:** hoàn thành phân tích embedding, đối chiếu hai seed, tổng hợp kết luận và giới hạn.

Các tài liệu thuyết trình cũ trong thư mục này mới phản ánh giai đoạn EDA/preprocessing; cần cập nhật trạng thái huấn luyện khi tái sử dụng.

## 5. Lời trình bày khoảng 2 phút

“Đến thời điểm hiện tại, em đã hoàn thành phân tích dữ liệu, tiền xử lý và huấn luyện ba mô hình GRU4Rec, NARM, SASRec trên Diginetica.

Ở bước phân tích dữ liệu, em ghi nhận phân bố tương tác không đồng đều: 10% item phổ biến nhất chiếm khoảng 62,5% tương tác thô. Em đã xây dựng quy trình chia tập theo thời gian, tách gần 20% item ở bước lựa chọn làm controlled cold và kiểm tra không có cold item trong train. Tập train cuối gồm khoảng 151 nghìn phiên và gần 499 nghìn mẫu dự đoán item tiếp theo.

Về thực nghiệm, em đã chạy ba mô hình, mỗi mô hình hai seed, tổng cộng sáu lần chạy. Ngoài các chỉ số chất lượng dự đoán, em đã tính coverage, Gini, độ phổ biến trung bình và tỷ lệ hiển thị theo nhóm item.

Kết quả bước đầu cho thấy SASRec có Recall@20 trung bình khoảng 57,29%, coverage khoảng 91,76% và Gini khoảng 0,6885. Trong các cấu hình đã chạy, SASRec vừa dự đoán tốt hơn vừa phân phối lượt hiển thị ít tập trung hơn hai mô hình còn lại. Tuy vậy, item phổ biến vẫn chiếm phần lớn lượt gợi ý, trong khi nhóm item hiếm nhận tỷ lệ hiển thị thấp hơn tỷ trọng tương tác của chúng trong train.

Phần còn lại là chạy phân tích hình học embedding để kiểm tra tương quan giữa tần suất, norm và exposure, đồng thời khảo sát vị trí cold item so với warm item cùng category. Hiện em đã có embedding và mã phân tích, nhưng chưa có kết quả để kết luận mối liên hệ này.”

## 6. Nguồn số liệu trong dự án

- `data/processed/diginetica/dataset_statistics.json`
- `results/tables/preprocessing_checks.csv`
- `results/tables/eda_popularity_concentration.csv`
- `results/tables/model_accuracy.csv`
- `results/tables/popularity_bias.csv`
- `runs/{gru4rec,narm,sasrec}/seed_{2024,2025}/test_metrics.json`
