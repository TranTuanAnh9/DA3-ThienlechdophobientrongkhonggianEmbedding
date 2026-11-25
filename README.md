# BTL-K03 — Thiên lệch độ phổ biến và hình học embedding

GRU4Rec, NARM và SASRec trên Diginetica. Xem [đề tài gốc](BTL-K03-tong-quan-de-tai.md)
và [cấu trúc thư mục](CAU-TRUC-DU-AN.md).

## Bắt đầu từ đâu?

- `notebooks/00–08`: dữ liệu, huấn luyện baseline và phân tích đề tài gốc.
- [Notebook 12](notebooks/12_persistent_homology.ipynb): PH H₀/H₁ thật và pilot đầu tiên.
- [Notebook 13](notebooks/13_role_gradients_and_measured_results.ipynb): kết quả thực nghiệm mới nhất, đã chạy.
- [Notebook 14](notebooks/14_popularity_aware_contrastive_sasrec.ipynb): contrastive SASRec đã sửa và kiểm tra chạy.
- `notebooks/legacy/`: ba notebook cosine cũ, chỉ lưu để đối chiếu lịch sử.

## Kết quả hiện có

- PH-guided gradient: 22 lượt pilot, chưa tăng rare hit so với CE ở cả 6 cặp model/seed.
- Hiệu chỉnh điểm theo popularity: Rare Recall tăng **1,877–3,142 điểm phần trăm**,
  Recall tổng giảm **0,145–0,405 điểm phần trăm** trên test; không cần train lại.
- Hệ số được chọn trên validation trước khi đánh giá test. Đây là đối chứng đã có
  tiền lệ, không phải thuật toán mới hoặc bằng chứng PH có ích. Chưa có xác nhận
  trên dataset độc lập hay hiệu quả kinh doanh.

Kết quả hiện có được tổ chức tại [results/README.md](results/README.md):

| Thư mục | Nội dung |
|---|---|
| `results/contrastive/20261121_191806/` | Chạy đầy đủ ba phương pháp, hai seed; dự đoán test và bootstrap |
| `results/diagnostics/norm_audit_20261122_215518/` | Audit frequency–norm, dữ liệu hình học và mã tái lập |
| `results/directional_calibration/20261122_220947/` | Hiệu chỉnh hướng; xác nhận nội bộ trên validation |
| `results/figures/`, `results/item_level/` | Hình và phân tích baseline ở mức item |

Các đường dẫn `results/persistent_homology/` và `results/topology/` được nhắc trong notebook/báo cáo cũ hiện không có trong bản làm việc này. Số liệu PH và popularity calibration ở phần trên là kết quả đã ghi trong notebook; muốn tái lập cần bổ sung artifact và công cụ gốc tương ứng.

## Chạy lại

Notebook 14 dùng `src/btl_k03/contrastive.py` và `configs/contrastive.yaml`.
Local mặc định chỉ chạy kiểm tra nhỏ; Kaggle chạy CE/uniform/head-aware trên hai
seed, cùng budget và validation MRR. Có checkpoint, lịch sử, dự đoán và bootstrap.
Chưa mặc định phương pháp head-aware tốt hơn các đối chứng.

Bản Kaggle mang theo source/config đã kiểm tra tại `kaggle/contrastive-sasrec/`:

```powershell
kaggle kernels push -p kaggle/contrastive-sasrec --accelerator NvidiaTeslaT4
```

Chạy các lệnh từ thư mục gốc. Mỗi thí nghiệm cần một thư mục đầu ra mới.

```powershell
python -m pip install -r requirements.txt
python -B scripts/run_notebooks.py --help
```

Pilot gradient/PH:

```powershell
python -B scripts/train.py --models gru4rec --methods baseline role_input_scale role_tangent ph_h0_h1 role_ph --seeds 2024 2025 --epochs 2 --train-limit 4096 --validation-limit 8192 --learning-rate 0.0001 --config configs/topology/role_gradient_pilot.yaml --init-existing --output results/persistent_homology/role_pilot_new
```

Notebook Kaggle cho thí nghiệm contrastive: `kaggle/contrastive-sasrec/contrastive-sasrec.ipynb`.
Mỗi lần chạy bắt đầu bằng kiểm tra nhỏ; chỉ train đầy đủ khi kiểm tra thành công.

## Quy ước lưu trữ

- `src/btl_k03/`: code dùng lại; `scripts/`: 4 entry point cần thiết.
- `data/`: dữ liệu Diginetica; `runs/`: baseline/checkpoint gốc; `results/`: số đo và thí nghiệm.
- `reports/`: báo cáo, slide, hình và nhật ký dọn thư mục.
- `.cache/`: file tạm, được Git bỏ qua ngoại trừ README hướng dẫn.

Các pilot giữ checkpoint, mapping, config, log và dự đoán. Bản `item_embeddings.npy`
lặp đã được bỏ sau khi kiểm tra trùng chính xác với checkpoint. Đọc embedding bằng
`btl_k03.artifacts.load_item_embeddings(run_dir)`; hàm tự lấy trọng số từ checkpoint
khi không có NPY. Dữ liệu gốc, checkpoint và slide được giữ lại.
