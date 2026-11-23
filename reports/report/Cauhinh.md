1. Những thứ được dùng chung
   Ba mô hình đều dùng chung:
   Thành phần Giá trị
   Dataset Diginetica
   Temporal split Train/validation/test giống nhau
   Số seed 2024 và 2025
   Kích thước embedding 64
   Độ dài sequence tối đa 50
   Loss Cross-Entropy
   Optimizer Adam
   Learning rate 0,001
   Weight decay 0
   Batch train 512
   Batch validation/test 1.024
   Số epoch tối đa 30
   Gradient clipping 5,0
   Early stopping monitor MRR@20
   Early stopping patience 5
   Top-K đánh giá 20
   Candidate item Warm item
   Accuracy metrics Recall@20, MRR@20, NDCG@20
   Popularity metrics Coverage@20, Gini@20, ARP@20
   Cold-start protocol Controlled cold 20%, embedding bị đóng băng

Điều này có nghĩa là ba mô hình:

- Được huấn luyện trên cùng dữ liệu.
- Dùng cùng train/validation/test.
- Dùng cùng số seed.
- Dùng cùng learning rate, batch size, optimizer và loss.
- Được đánh giá bằng cùng các chỉ số.
- Dùng cùng Top-20 candidate protocol.
  Vì vậy, việc so sánh accuracy và popularity bias giữa ba mô hình là tương đối công bằng.

2. Những thứ khác nhau do kiến trúc mô hình
   GRU4Rec
   Tham số Giá trị
   Embedding size 64
   Hidden size 128
   Số GRU layer 1
   Embedding dropout 0,25
   Hidden dropout 0,20

GRU4Rec dùng GRU để xử lý chuỗi item theo thứ tự thời gian.
NARM
Tham số Giá trị
Embedding size 64
Hidden size 128
Số layer 1
Embedding dropout 0,25
Context/local attention dropout 0,50

NARM có thêm cơ chế attention để tập trung vào các item quan trọng trong session. Vì vậy nó có thêm ct_dropout, là tham số riêng của NARM.
SASRec
Tham số Giá trị
Embedding size 64
Số Transformer layer 2
Số attention head 2
Inner size 256
Hidden dropout 0,20
Attention dropout 0,20
Activation GELU

SASRec dùng self-attention/Transformer nên không có hidden_size và GRU layer giống GRU4Rec/NARM. Thay vào đó, nó có:

- Số Transformer layer.
- Số attention head.
- Kích thước tầng feed-forward.
- Attention dropout.

3. Vì sao không đặt tất cả tham số giống hệt nhau?
   Vì ba mô hình có kiến trúc khác nhau.

Ví dụ:
GRU4Rec:
cần hidden_size và số GRU layer

NARM:
cần hidden_size và attention

SASRec:
cần Transformer layer, attention head và inner_size
Nếu cố ép mọi tham số giống nhau thì không còn phù hợp với kiến trúc của từng mô hình.

Điều cần công bằng không phải là mọi con số đều giống nhau tuyệt đối, mà là:
Cùng dữ liệu

- cùng quy trình chia tập
- cùng loss
- cùng optimizer
- cùng learning rate
- cùng batch size
- cùng số seed
- cùng giao thức đánh giá
  Còn các tham số như số layer hoặc số attention head phải khác nhau để mô hình hoạt động đúng.

4. Một điểm khác biệt: Mixed Precision
   Trong common.yaml, cấu hình chung là:
   use_amp: true
   Nhưng SASRec có override:
   use_amp: false
   AMP là Mixed Precision, chủ yếu ảnh hưởng đến:

- Tốc độ chạy.
- Mức sử dụng GPU.
- Độ chính xác số học trong quá trình tính toán.
  Nó không phải là thay đổi kiến trúc hay mục tiêu tối ưu chính. SASRec tắt AMP có thể do vấn đề ổn định khi chạy Transformer.

Nếu báo cáo, nên ghi rõ:
Ba mô hình dùng cùng giao thức huấn luyện và đánh giá. Một ngoại lệ kỹ thuật là SASRec tắt mixed precision để đảm bảo ổn định khi huấn luyện.

5. Early stopping có giống nhau không?
   Có. Cả ba mô hình đều:

- Theo dõi MRR@20 trên validation.
- Chọn checkpoint có validation MRR tốt nhất.
- Dừng nếu không cải thiện sau 5 epoch.
- Tối đa 30 epoch.
  Tuy nhiên, epoch tốt nhất của mỗi mô hình khác nhau:
  Mô hình Seed 2024 Seed 2025
  GRU4Rec 16 15
  NARM 15 20
  SASRec 12 8

Điều này là bình thường vì mỗi kiến trúc hội tụ với tốc độ khác nhau.

Không được hiểu rằng SASRec chỉ được huấn luyện 8 hoặc 12 epoch cố định. Đúng hơn là:
Mỗi mô hình được phép huấn luyện tối đa 30 epoch và dùng early stopping để chọn epoch tốt nhất trên validation.

6. Kết luận về tính công bằng
   Cách nói chính xác là:
   Ba mô hình không dùng mọi siêu tham số giống hệt nhau, vì mỗi mô hình có kiến trúc riêng. Tuy nhiên, chúng dùng cùng dữ liệu, cùng temporal split, cùng embedding dimension, cùng optimizer, learning rate, batch size, loss, số seed, early stopping và giao thức đánh giá. Vì vậy, thí nghiệm kiểm soát được các yếu tố chính để so sánh mô hình.

Hoặc nói ngắn gọn khi thuyết trình:
Chúng tôi giữ cố định tất cả thành phần có thể so sánh trực tiếp, còn các tham số đặc thù như số layer, attention head hay hidden size được cấu hình theo kiến trúc của từng mô hình. Do đó, đây là so sánh công bằng về giao thức thực nghiệm, nhưng không phải là việc ép ba kiến trúc dùng cùng một cấu hình nội bộ.
