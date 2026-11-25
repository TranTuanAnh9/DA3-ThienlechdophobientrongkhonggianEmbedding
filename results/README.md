# Kết quả thí nghiệm

Mỗi nhóm thí nghiệm có thư mục riêng; mỗi lần chạy được giữ theo mã thời gian để không ghi đè kết quả cũ.

| Thư mục | Nội dung |
|---|---|
| `contrastive/20261121_191806/` | Kết quả chạy đầy đủ SASRec: baseline, uniform, head; hai seed |
| `diagnostics/norm_audit_20261122_215518/` | Kiểm tra frequency–norm, checkpoint và hình học embedding |
| `directional_calibration/20261122_220947/` | Hiệu chỉnh hướng trên validation; chưa đánh giá test độc lập |
| `figures/` | Hình phân tích baseline |
| `item_level/` | Dữ liệu phân tích ở mức item |

## Quy ước thư mục

Trong các lần chạy diagnostics và directional calibration:

```text
<run_id>/
├── summary.json          # Kết luận và phạm vi đánh giá, nếu có
├── *.json                # Protocol, policy khóa và kiểm tra
├── tables/               # Bảng CSV tổng hợp
├── figures/              # Hình PNG và SVG
├── data/                 # Dữ liệu trung gian dạng Parquet, nếu có
├── predictions/          # Dự đoán và dữ liệu riêng từng seed, nếu có
│   └── seed_<seed>/
└── scripts/              # Mã tái lập phép đo và hình
```

Contrastive giữ cấu trúc do notebook 14 tạo: `full/<method>/seed_<seed>/`, `smoke/` và các bảng tổng hợp ở gốc lần chạy. Giữ nguyên để notebook và bộ Kaggle tiếp tục đọc được kết quả.

Hai lần chạy thử CPU `20261121_153830` và `20261121_161946` đã chuyển sang `.cache/archive/contrastive-smoke/`. Chúng chỉ chạy một epoch trên 64 mẫu, có checkpoint và dự đoán trùng nhau; không dùng làm kết quả nghiên cứu. Kho lưu trữ này được Git bỏ qua.

## Tái lập

Chạy từ thư mục gốc dự án:

```powershell
python -B results/diagnostics/norm_audit_20261122_215518/scripts/reproduce_audit.py
python -B results/diagnostics/norm_audit_20261122_215518/scripts/reproduce_plots.py
python -B results/directional_calibration/20261122_220947/scripts/reproduce_screening.py
python -B results/directional_calibration/20261122_220947/scripts/reproduce_plots.py
```

Các script audit/screening tạo lần chạy mới; script plots đọc lần chạy chứa chính script và cập nhật hình/bảng ở đó. Cần dữ liệu processed, checkpoint baseline và các thư viện trong `requirements.txt`.

Không đổi tên thư mục đã công bố tùy ý hoặc sửa các JSON ghi lại đường dẫn lịch sử. Giữ protocol, metrics, dự đoán và mã tái lập cùng một lần chạy. File CSV, Parquet và hình được kiểm tra hash trước/sau khi sắp xếp để bảo toàn nội dung.
