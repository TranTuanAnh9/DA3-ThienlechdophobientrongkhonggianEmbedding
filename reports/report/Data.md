Dữ liệu được chia theo thời gian
Raw data
→ sắp xếp session
→ chia theo thời gian
→ lọc train
→ tạo controlled cold
→ lọc train lần cuối
→ tạo validation/test warm
→ tạo mẫu input–target

1. Khoảng thời gian của dữ liệu raw
   01/01/2016 đến 01/06/2016
   Tổng cộng 152 ngày quan sát
   Được chia theo thứ tự

Train: 01/01/2016 – 18/05/2016
Validation: 19/05/2016 – 25/05/2016
Test: 26/05/2016 – 01/06/2016

Sơ đồ
Quá khứ Tương lai
│ │
01/01 ───────────── 18/05 │ 19/05 ─ 25/05 │ 26/05 ─ 01/06
TRAIN VALIDATION TEST

2. Một session không bị tách giữa các tập
   sessionId → timeframe → thứ tự dòng gốc
   Và một session có thể đi qua nửa đêm ví dụ
   23:59 ngày 18/05
   00:05 ngày 19/05

Nhưng đây vẫn là một session duy nhất.

Để tránh làm rò rỉ thông tin giữa các tập, toàn bộ session được giữ trong cùng một split. Session được gán theo ngày của event cuối cùng.

Session bắt đầu ngày 18/05
và kết thúc ngày 19/05
→ đưa toàn bộ vào validation

3. Quy mô ban đầu của ba tập sau khi chia theo thời gian
   Đây là số liệu ngay sau temporal split, trước khi lọc item và tạo controlled cold:
   Tập Thời gian Event Session Item
   Train ban đầu 01/01–18/05 990.708 189.774 114.768
   Validation ban đầu 19/05–25/05 65.533 12.701 25.611
   Test ban đầu 26/05–01/06 88.445 17.155 30.290

Đây chưa phải số liệu cuối cùng được đưa trực tiếp vào mô hình, vì sau đó train còn được lọc và tạo cold-start.

4. Lọc train bằng ngưỡng frequency tối thiểu
   Trên train, pipeline dùng:
   item frequency ≥ 5
   Nghĩa là chỉ giữ item xuất hiện ít nhất 5 lần trong train.

Ngoài ra:

- Loại session chỉ có một event.
- Sau khi loại item, nếu session còn dưới 2 event thì loại tiếp.
- Lặp lại đến khi dữ liệu ổn định, gọi là fixed-point filtering.
  Sau bước lọc train lần đầu:
  Giai đoạn Event Session Item
  Train sau temporal split 990.708 189.774 114.768
  Train sau fixed-point 841.402 174.691 38.237

Lúc này còn 38.237 item đủ điều kiện để xem xét làm catalog train.

5. Tạo controlled cold bằng 20% catalog train
   Từ 38.237 item sau lọc, pipeline chọn 20% làm controlled cold:
   38.237 × 20% ≈ 7.647 item
   Controlled cold được chọn ở cấp item, không phải cấp session hay cấp query.
   Controlled cold nghĩa là gì?
   Đây là những item:

- Ban đầu đủ điều kiện xuất hiện trong train.
- Sau đó bị chủ động loại khỏi train.
- Không được nhận gradient trong quá trình huấn luyện.
- Được giữ lại để nghiên cứu hình học embedding của item chưa được học.
  Sau khi loại 7.647 controlled cold item khỏi train, train được lọc lại một lần nữa vì việc xóa item có thể làm session ngắn đi hoặc làm tần suất item khác giảm xuống.

6. Final train sau khi tạo controlled cold
   Sau khi loại controlled cold và chạy fixed-point lần cuối:
   Tập train cuối Event Session Item Category Mẫu next-item
   Final train 649.632 150.783 29.936 945 498.849

Đây là tập train thực sự được dùng để huấn luyện GRU4Rec, NARM và SASRec.

Vì vậy, catalog chính của mô hình gồm:
29.936 warm item

- 7.647 controlled cold item
  = 37.583 item
  Trong đó:

* 29.936 warm item: được phép học.
* 7.647 controlled cold item: có trong mapping nhưng bị đóng băng, không học.

7. Validation và test chính chỉ giữ warm item
   Sau khi xác định final warm catalog từ train, validation và test được lọc để chỉ giữ những item mà mô hình có khả năng học.
   Validation warm
   Chỉ số Giá trị
   Thời gian 19/05–25/05/2016
   Event 39.450
   Session 9.622
   Item 12.227
   Mẫu next-item 29.828

Test warm
Chỉ số Giá trị
Thời gian 26/05–01/06/2016
Event 52.792
Session 12.791
Item 13.847
Mẫu next-item 40.001

Đây là các tập chính dùng để:

- Chọn checkpoint tốt nhất trên validation.
- Đánh giá Recall, MRR, NDCG trên test.
- Tính popularity bias trong file 06.



12. Warm/cold và popular/medium/rare trong split
Sau final train, popularity được tính lại chỉ từ final train.
Nhóm	Điều kiện
Cold	Tần suất train = 0
Rare	Tần suất train từ 5 đến 6
Medium	Tần suất train từ 7 đến 26
Popular	Tần suất train từ 27 trở lên


Kết quả:
Nhóm	Số item
Warm rare	6.884
Warm medium	16.940
Warm popular	6.112
Controlled cold	7.6474




