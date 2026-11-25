# Cấu trúc dự án BTL-K03

```text
.
├── README.md                         # Hướng dẫn và kết quả hiện tại
├── BTL-K03-tong-quan-de-tai.md        # Nội dung đề tài gốc
├── CAU-TRUC-DU-AN.md                 # Tài liệu này
├── requirements.txt
├── pytest.ini
├── configs/                          # Cấu hình baseline, contrastive và pilot
│   └── topology/                     # Cấu hình pilot và full study
├── data/                             # Diginetica raw/interim/processed
├── src/btl_k03/                      # Model, metrics, calibration và thống kê
│   └── topology/                     # Persistent homology và gradient theo vai trò
├── scripts/                          # Các lệnh thực thi bên dưới
├── tests/                            # Kiểm thử dữ liệu, metrics và phương pháp
├── notebooks/                        # 00–08 và các notebook nghiên cứu 12–14
│   └── legacy/                       # Notebook 09–11 cũ
├── runs/                             # Checkpoint baseline và cosine legacy
├── results/                          # Bảng, hình và kết quả có thể kiểm chứng
│   ├── contrastive/                  # Chạy đầy đủ theo method/seed
│   ├── diagnostics/                  # Audit: tables, figures, data, scripts
│   ├── directional_calibration/      # Tables, figures, predictions, scripts
│   ├── figures/                      # Hình baseline
│   └── item_level/                   # Phân tích từng item
├── kaggle/                           # Bộ notebook theo thí nghiệm/tài khoản
├── reports/
│   ├── report/                      # Báo cáo nội dung đề tài
│   ├── figures/                     # Hình phục vụ báo cáo/slide
│   └── slides/                      # Giữ các bản PowerPoint và backup
└── .cache/                          # Cache và archive chạy thử; Git bỏ qua nội dung
```

## Scripts cần thiết

| Lệnh | Công dụng |
|---|---|
| `scripts/train.py` | Huấn luyện và so sánh baseline/PH/gradient theo vai trò |
| `scripts/report.py` | Tổng hợp kết quả và bootstrap ghép cặp theo session |
| `scripts/visualize.py` | t-SNE và số đo không gian gốc |
| `scripts/run_notebooks.py` | Chạy lại notebook phân tích 07–08 |

Chạy từ thư mục gốc. Notebook 13 là điểm đọc kết quả mới nhất.
Không dùng `legacy` như implementation PH hiện tại. Không gộp controlled cold
vào warm để kết luận về popularity bias.

Các export embedding lặp trong pilot đã được bỏ; checkpoint giữ nguyên.
`src/btl_k03/artifacts.py` đọc được cả run có NPY và run chỉ còn checkpoint.

Chi tiết kết quả và lệnh tái lập: [results/README.md](results/README.md).

Hai lần smoke CPU cũ được lưu tại `.cache/archive/contrastive-smoke/`; chỉ lần chạy đầy đủ `results/contrastive/20261121_191806/` nằm trong thư mục kết quả contrastive hiện tại.
