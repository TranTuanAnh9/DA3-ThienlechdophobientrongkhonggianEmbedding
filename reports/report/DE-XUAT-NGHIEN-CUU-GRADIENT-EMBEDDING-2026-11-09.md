# Đề xuất nghiên cứu: điều chỉnh xung đột hướng giữa hai vai trò của item embedding

Ngày rà soát: 09/11/2026. Trạng thái: đề xuất nghiên cứu có pilot chẩn đoán, chưa có kết quả huấn luyện phương pháp mới và chưa xác nhận tính mới tuyệt đối.

## 1. Tên và phạm vi

**Điều chỉnh xung đột gradient theo vai trò và hình học cho embedding dùng chung trong gợi ý theo phiên.**

Tên tiếng Anh dự kiến: *Role-aware radial–tangential gradient adjustment for tied-embedding session-based recommendation*.

Không tuyên bố weight tying, gradient surgery, chuẩn hóa vector, hay debiasing bằng gradient là phát minh của đề tài. Đóng góp ứng viên nằm ở việc tách các đường đạo hàm của cùng một loss thành vai trò đọc lịch sử và chấm điểm; xác định tác động riêng lên độ dài/hướng; kiểm tra điều chỉnh có chọn lọc đối với chất lượng gợi ý và exposure của item đuôi dài.

## 2. Vấn đề xuất phát từ code và kết quả thực tế

GRU4Rec/NARM/SASRec hiện dùng một ma trận E cho cả tra item đầu vào và chấm điểm đầu ra. Với prefix x:

`s = encoder(x; E, θ)` và `z = s Eᵀ`.

Một hàng embedding nhận gradient qua hai đường, kể cả khi chỉ có một cross-entropy loss:

`g_total(i) = g_input(i) + g_output(i)`.

Các đường này không phải hai nhãn hay hai nhiệm vụ giám sát độc lập. Có thể tách bằng stop-gradient trên một nhánh trong khi giữ nguyên giá trị forward.

Phân tích trước đó cho thấy norm rare > medium > popular trong cả sáu lần chạy, còn frequency–exposure vẫn dương. Đây là hiện tượng cần giải thích, không tự nó chứng minh weight tying là nguyên nhân.

## 3. Công trình gần nhất và ranh giới tính mới

| Công trình | Phần đã được nghiên cứu | Khác biệt mà đề xuất cần chứng minh |
|---|---|---|
| [NISER, 2019](https://arxiv.org/abs/1909.04276) | Chuẩn hóa item/session representations để xử lý popularity bias trong session recommendation | Không nhận chuẩn hóa là mới; nghiên cứu đường gradient và can thiệp có chọn lọc thay vì chỉ đổi norm |
| [Gradient Surgery / PCGrad, 2020](https://arxiv.org/abs/2001.06782) | Chiếu các task gradient xung đột | Ở đây là hai đường đạo hàm của một loss qua bảng dùng chung; phải so với áp dụng phép chiếu đầy đủ đơn giản |
| [Ren et al., ICDM 2022](https://pdxscholar.library.pdx.edu/compsci_fac/307/) | Phân tích popularity bias từ gradient và điều chỉnh embedding trong recommendation | Không nhận phân tích gradient là mới; cần chỉ rõ hai vai trò input/output và thành phần radial/tangential mang thêm thông tin gì |
| [Weight Tying Biases Token Embeddings Towards the Output Space, 2026](https://arxiv.org/html/2603.26663v1) | Weight tying trong language models; gradient đầu ra chi phối; tăng gradient đầu vào bằng hằng số | Không nhận cân bằng input/output gradient là mới; cần cơ chế chọn theo xung đột/hình học và bằng chứng về tail ranking trong SBR |
| [DDC, KDD 2026](https://arxiv.org/html/2512.10688v6) | Phân rã cập nhật BPR của user theo hướng preference và popularity; fine-tuning bất đối xứng | Đề xuất xét item embedding dùng chung của encoder/classifier, full-softmax CE; phân rã theo hướng radial/tangential của từng item |

Rà soát đã tìm cả các truy vấn chính xác về input-output gradient conflict, tied embeddings/popularity, tangential gradient và role-specific adjustment trong recommendation. Trong các tài liệu tìm được và các phần đã đọc, chưa thấy mô tả đúng toàn bộ thiết kế ở trên. Đây là bằng chứng rà soát ban đầu, không phải chứng minh không tồn tại công trình trùng. Cần tiếp tục truy vết trích dẫn và đọc toàn văn các bài gần nhất trước khi viết tuyên bố novelty trong bài nộp.

Lưu ý quan trọng: công trình weight tying năm 2026 đã phân tích norm–frequency và thấy dạng phi tuyến; không gọi riêng norm nghịch biến hay output-gradient dominance là phát hiện chưa từng được biết.

## 4. Pilot đã thực hiện

Chạy `python -B scripts/probe_gradients.py`.

- Cùng 2.048 mẫu TRAIN lấy ngẫu nhiên với seed cố định 20261109 cho sáu checkpoint tốt nhất.
- Batch 256, CPU, dropout tắt, không optimizer step, không thay checkpoint.
- Tách đạo hàm qua input/output bằng detach.
- Kiểm tra trên batch đầu từng checkpoint: g_total = g_input + g_output; sai số tuyệt đối tối đa qua sáu lần chạy khoảng 1,86e-9.
- Gradient các batch được lấy trung bình trước khi tính cosine giữa hai vai trò ở từng item.
- Có 4.538 warm item mang input gradient khác 0 trong mẫu, gồm 2.538 popular, 1.724 medium, 276 rare.

Tỷ lệ item có hai gradient ngược chiều (cosine < 0), trung bình hai seed:

| Mô hình | Warm có input gradient | Rare có input gradient |
|---|---:|---:|
| GRU4Rec | 53,02% | 55,80% |
| NARM | 54,86% | 58,51% |
| SASRec | 56,63% | 45,83% |

Cosine trung bình của toàn warm chỉ khoảng −0,006 đến −0,023. Hiệu ứng góc trung bình nhỏ; tỷ lệ âm quanh 50% không tự chứng minh gây hại vì các vector gần trực giao trong nhiều chiều cũng có thể có tính chất tương tự. Không suy ra rare luôn chịu xung đột nặng hơn; SASRec cho kết quả ngược điều đó trong pilot.

Phép chiếu gradient đầu ra lên hướng radial, đổi dấu để xét xu hướng cập nhật SGD vi phân, cho rare trung bình hướng ra ngoài và popular trung bình hướng vào trong ở cả sáu checkpoint. Đây là chẩn đoán tại checkpoint cuối trên một mẫu train, không phải phép đo toàn bộ lịch sử học và không phải bước cập nhật Adam thực tế.

Bảng đầy đủ: `results/tables/embedding_gradient_role_probe.csv`.

## 5. Ba câu hỏi nghiên cứu có thể bác bỏ

**RQ1 — Cơ chế:** Sau kiểm soát tần suất, độ dài prefix và category, xung đột input/output có dự báo suy giảm chất lượng gợi ý rare không? Quan hệ có nhất quán qua các giai đoạn train và kiến trúc không?

**RQ2 — Vị trí xung đột:** Xung đột trong thành phần thay đổi hướng có quan hệ với chất lượng khác với thành phần thay đổi norm không? Có trường hợp xung đột là regularization có lợi không?

**RQ3 — Can thiệp:** Điều chỉnh có chọn lọc thành phần xung đột có cải thiện Recall/MRR rare và exposure mà không vượt mức giảm chất lượng tổng thể đã đặt trước không? Có hơn tăng gradient đầu vào bằng hằng số và chiếu toàn gradient không?

Không khóa câu trả lời thành “phải tăng rare exposure”. Nếu chỉ tăng số lượt hiển thị nhưng không cải thiện chất lượng liên quan, chưa đạt mục tiêu.

## 6. Thiết kế phương pháp ứng viên

Với item embedding e khác 0, đặt `u = e / ||e||`.

Tách từng gradient thành:

`g_input = r_input u + t_input`

`g_output = r_output u + t_output`, trong đó `t_input ⟂ u` và `t_output ⟂ u`.

- Radial r u: liên quan thay đổi norm ở bậc một dưới một bước SGD nhỏ.
- Tangential t: liên quan thay đổi hướng ở bậc một.

Một ứng viên cụ thể để thử: chỉ giảm thành phần tangential của output đi ngược tangential input khi xung đột đủ mạnh; mức điều chỉnh có thể phụ thuộc frequency/số bằng chứng đầu vào. Không mặc định g_input luôn đúng: bắt buộc có đối chứng bảo vệ output và điều chỉnh đối xứng.

Với tích vô hướng tangential âm:

`t_output_new = t_output − λ_i [(t_output · t_input)/(||t_input||² + ε)] t_input`

`g_new = (r_input + r_output)u + t_input + t_output_new`.

λ_i trong [0,1]; chọn trên validation. λ_i=0 khôi phục baseline. λ_i=1 loại thành phần đối nghịch theo phép chiếu khi ε đủ nhỏ; không khẳng định chính xác tuyệt đối với ε hữu hạn.

Phép toán giữ nguyên radial component của gradient tổng trước optimizer. **Điều này không bảo đảm norm thực tế được giữ nguyên sau Adam**, vì momentum và preconditioning thay đổi hướng cập nhật. Nếu muốn phát biểu bảo đảm về bước cập nhật cần làm ở không gian update đã qua optimizer hoặc phân tích SGD riêng. Không sử dụng từ “bảo đảm giữ Recall” hay “chứng minh giảm bias” khi chưa có định lý đúng điều kiện.

Mục tiêu triển khai là thay đổi quá trình huấn luyện, giữ nguyên hàm dự đoán và số tham số suy luận. Chi phí train tăng do tách đường gradient; phải đo thời gian/bộ nhớ và so cả ngân sách epoch lẫn ngân sách thời gian.

## 7. Thí nghiệm bắt buộc

### Đối chứng

1. Baseline dùng chung E như hiện tại.
2. Tách bảng input/output; kiểm soát thêm ngân sách tham số để tránh gán hiệu quả do tăng capacity cho phương pháp.
3. Nhân input gradient với hằng số, chọn hệ số trên validation.
4. Chiếu toàn gradient input/output theo kiểu PCGrad, báo rõ cách thích nghi từ multitask sang hai đường đạo hàm.
5. Chỉ radial adjustment.
6. Chỉ tangential adjustment, không frequency gate.
7. Đề xuất đầy đủ; thêm đối chứng gate ngẫu nhiên/hoán đổi để kiểm tra hiệu ứng chỉ do giảm step size hoặc đổi regularization.

### Kiểm soát

- Giữ candidate set, split, prefix protocol và định nghĩa popularity như nhau.
- Kiểm tra sample-frequency và target-frequency: không đồng nhất số lần input xuất hiện với số lần item là target trong all-prefix.
- Thống kê theo nhiều batch độc lập; theo dõi checkpoint đầu/giữa/cuối, không suy luận động lực học chỉ từ checkpoint cuối.
- So hiệu ứng với gradient được ghép ngẫu nhiên có cùng nhóm frequency và độ lớn; cosine âm đơn thuần chưa là bằng chứng cơ chế.
- Cold tiếp tục bị đóng băng và mask; không gộp cold vào phép đo warm để tạo hiệu ứng.
- Chọn siêu tham số trên validation; không chỉnh theo test. Kết quả exploratory trên test hiện tại phải được xác nhận trên dữ liệu/holdout độc lập.

### Dữ liệu và seed

Pilot đầu huấn luyện GRU4Rec vì cấu trúc đơn giản. Nếu qua kiểm tra cơ chế mới mở sang NARM/SASRec. Đánh giá xác nhận trên Diginetica và ít nhất một dataset session khác, với nhiều seed (mục tiêu 3–5 nếu tài nguyên cho phép).

### Chỉ số

Recall/MRR/NDCG tổng thể và theo target group; macro recall theo item để tránh popular target chi phối; coverage/Gini/ARP; rare exposure; calibration nếu được định nghĩa rõ; thời gian và bộ nhớ.

Lấy mẫu bootstrap theo session để giữ các prefix cùng phiên đi cùng nhau. Khoảng tin cậy trên tập đánh giá không thay thế việc chạy nhiều seed để đo biến thiên do huấn luyện.

## 8. Giá trị và điều kiện dừng

Giá trị kỳ vọng:

- Cơ chế: phân biệt nguồn cập nhật và phần hình học liên hệ với giảm chất lượng đuôi dài.
- Thuật toán: điều chỉnh cục bộ quá trình học, có thể cắm vào ba kiến trúc hiện có.
- Triển khai: không yêu cầu metadata mới hoặc tăng số tham số lúc suy luận.
- Công cụ: probe tái lập để đánh giá liệu weight tying đang có lợi hay có hại theo từng nhóm item.

Đây là giá trị kỳ vọng, chưa phải hiệu quả đã được chứng minh.

Dừng hoặc đổi hướng nếu: không có tín hiệu vượt đối chứng ngẫu nhiên; tác dụng không hơn scaling đơn giản; improvement biến mất khi kiểm soát budget; chỉ có trên một seed/dataset; hoặc rare exposure tăng bằng cách đẩy item không phù hợp. Không cứu giả thuyết bằng cách chỉ báo các cấu hình thuận lợi.

## 9. Quyết định thẩm định

**Đủ cơ sở cho một đề xuất nghiên cứu có rủi ro được kiểm soát và một pilot sâu hơn. Chưa đủ cơ sở gọi là thuật toán mới đã được xác nhận hoặc hứa hẹn công bố.**

Câu novelty có thể dùng sau rà soát đầy đủ và có thực nghiệm: “Chúng tôi nghiên cứu sự tương tác giữa hai đường gradient qua item embedding dùng chung trong full-softmax session recommendation, tách tác động radial/tangential và kiểm chứng điều chỉnh theo vai trò đối với chất lượng đuôi dài.” Chỉ dùng “đầu tiên” khi có bằng chứng rà soát đủ mạnh; hiện chưa dùng.
