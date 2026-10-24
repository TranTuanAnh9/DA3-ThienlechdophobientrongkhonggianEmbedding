# Cấu trúc dự án BTL-K03

## 1. Mục đích của tài liệu

Tài liệu này mô tả toàn bộ cấu trúc thư mục của dự án:

> **BTL-K03 — Thiên lệch độ phổ biến và hình học không gian embedding trong mô hình SBR**

Mục tiêu của cấu trúc này là:

- Tách dữ liệu gốc khỏi dữ liệu đã xử lý.
- Tách code/notebook khỏi kết quả chạy.
- Tách từng lần huấn luyện theo mô hình và random seed.
- Giúp hai sinh viên làm việc song song mà không ghi đè kết quả của nhau.
- Cho phép chạy lại thí nghiệm và kiểm tra nguồn gốc của từng bảng, hình hoặc checkpoint.
- Phục vụ trực tiếp cho báo cáo, slide và hai notebook bắt buộc của đề tài.

---

## 2. Quy ước trạng thái file và thư mục

Trong tài liệu này sử dụng các ký hiệu sau:

| Ký hiệu | Ý nghĩa |
|---|---|
| `[TẠO TRƯỚC]` | File hoặc thư mục được tạo ngay khi khởi tạo dự án |
| `[TẢI VỀ]` | Dữ liệu được tải thủ công từ nguồn bên ngoài |
| `[SINH RA]` | Được chương trình hoặc notebook tự động tạo khi chạy |
| `[VIẾT SAU]` | Được hoàn thiện trong quá trình thực hiện đề tài |
| `[ĐÃ CÓ]` | File hiện đã tồn tại trong dự án |

Các file kết quả `[SINH RA]` không cần tạo thủ công và không nên tạo dữ liệu giả chỉ để lấp đầy thư mục.

---

## 3. Cây thư mục tổng thể

```text
DA-3/
│
├── README.md
├── BTL-K03-tong-quan-de-tai.md
├── CAU-TRUC-DU-AN.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── common.yaml
│   ├── data.yaml
│   ├── gru4rec.yaml
│   ├── narm.yaml
│   └── sasrec.yaml
│
├── data/
│   ├── raw/
│   │   └── diginetica/
│   │       ├── train-item-views.csv
│   │       ├── product-categories.csv
│   │       └── products.csv
│   │
│   ├── interim/
│   │   ├── interactions_clean.parquet
│   │   ├── items_clean.parquet
│   │   └── category_mapping.csv
│   │
│   └── processed/
│       ├── recbole/
│       │   └── diginetica/
│       │       ├── diginetica.train.inter
│       │       ├── diginetica.valid.inter
│       │       ├── diginetica.test.inter
│       │       └── diginetica.item
│       │
│       ├── cold_items.csv
│       ├── item_mapping.csv
│       ├── popularity_groups.csv
│       └── dataset_statistics.json
│
├── notebooks/
│   ├── 00_environment_setup.ipynb
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_train_gru4rec.ipynb
│   ├── 04_train_narm.ipynb
│   ├── 05_train_sasrec.ipynb
│   ├── 06_popularity_bias.ipynb
│   ├── 07_embedding_geometry.ipynb
│   └── 08_final_analysis.ipynb
│
├── src/
│   └── btl_k03/
│       ├── __init__.py
│       ├── paths.py
│       ├── popularity_metrics.py
│       ├── embedding_analysis.py
│       └── utils.py
│
├── runs/
│   ├── gru4rec/
│   │   ├── seed_2024/
│   │   └── seed_2025/
│   ├── narm/
│   │   ├── seed_2024/
│   │   └── seed_2025/
│   └── sasrec/
│       ├── seed_2024/
│       └── seed_2025/
│
├── results/
│   ├── item_level/
│   ├── tables/
│   ├── figures/
│   └── statistical_tests/
│
├── reports/
│   ├── report/
│   └── slides/
│
└── tests/
    ├── test_data_split.py
    ├── test_item_mapping.py
    ├── test_no_cold_leakage.py
    └── test_popularity_metrics.py
```

Lưu ý: một số file trong cây trên chỉ xuất hiện sau khi chạy notebook. Các thư mục hiện đang rỗng được giữ lại bằng file `.gitkeep`.

---

## 4. Các file ở thư mục gốc

### 4.1. `README.md`

Trạng thái: `[TẠO TRƯỚC]`, hiện để trống.

Đây là trang giới thiệu nhanh của repository. Khi hoàn thiện, file nên chứa:

- Tên đề tài.
- Thành viên nhóm.
- Mục tiêu ngắn gọn.
- Hướng dẫn cài đặt môi trường.
- Vị trí đặt dữ liệu Diginetica.
- Thứ tự chạy các notebook.
- Danh sách đầu ra quan trọng.

`README.md` dành cho người muốn chạy dự án, còn tài liệu tổng quan đề tài dành cho việc hiểu nội dung nghiên cứu.

### 4.2. `BTL-K03-tong-quan-de-tai.md`

Trạng thái: `[ĐÃ CÓ]`.

Đây là tài liệu giải thích học thuật và nghiệp vụ của đề tài, bao gồm:

- SBR và popularity bias.
- Diginetica.
- GRU4Rec, NARM và SASRec.
- Coverage, Gini và Average Recommendation Popularity.
- Norm, cosine similarity và t-SNE.
- Cách liên hệ output bias với embedding geometry.
- Cold item và warm item.
- Kế hoạch thực hiện, sản phẩm và giới hạn phạm vi.

### 4.3. `CAU-TRUC-DU-AN.md`

Trạng thái: `[ĐÃ CÓ]`.

Đây chính là tài liệu hiện tại. File mô tả cách tổ chức dự án, nhiệm vụ của từng thư mục và luồng dữ liệu giữa các notebook.

### 4.4. `requirements.txt`

Trạng thái: `[TẠO TRƯỚC]`, hiện để trống.

Sau này file sẽ ghi các thư viện Python cần thiết, ví dụ:

- PyTorch.
- RecBole.
- pandas.
- NumPy.
- SciPy.
- scikit-learn.
- Matplotlib.
- Seaborn.
- PyArrow nếu sử dụng Parquet.

Mục đích là giúp cài cùng một môi trường trên máy cá nhân, Colab hoặc Kaggle.

### 4.5. `.gitignore`

Trạng thái: `[TẠO TRƯỚC]`, hiện để trống.

File này sẽ khai báo những dữ liệu không đưa lên Git, đặc biệt:

- Dữ liệu Diginetica thô.
- Dữ liệu trung gian và processed có kích thước lớn.
- Model checkpoint.
- Embedding dạng NumPy.
- Top-20 dạng Parquet.
- Cache và file môi trường Python.

Không đưa checkpoint hoặc dữ liệu lớn lên Git giúp repository nhẹ và tránh vi phạm điều kiện phân phối dữ liệu.

---

## 5. Thư mục `configs/`

Trạng thái thư mục: `[TẠO TRƯỚC]`.

Thư mục này lưu toàn bộ cấu hình thí nghiệm. Mục đích là tránh ghi trực tiếp các tham số quan trọng rải rác trong notebook.

### 5.1. `configs/common.yaml`

Chứa cấu hình chung cho cả ba mô hình, chẳng hạn:

- Hai random seed.
- Embedding dimension.
- Số epoch tối đa.
- Train batch size.
- Evaluation batch size.
- Top-k bằng 20.
- Các chỉ số Recall, MRR và NDCG.
- Thiết bị CPU/GPU.
- Quy tắc early stopping.

Các tham số chung phải nhất quán để so sánh ba mô hình công bằng.

### 5.2. `configs/data.yaml`

Chứa cấu hình dữ liệu:

- Tên dataset.
- Đường dẫn dữ liệu raw và processed.
- Tên các cột session, item, time và category.
- Ngưỡng lọc session/item.
- Cách chia train, validation và test.
- Cold ratio bằng 20% nếu đúng theo tài liệu gốc.
- Cách định nghĩa popular, medium và rare.
- Candidate catalog dùng trong đánh giá.

Các giá trị chính thức chỉ được điền sau khi kiểm tra Mục 1.6 và Option B của tài liệu môn học.

### 5.3. `configs/gru4rec.yaml`

Chứa các tham số riêng của GRU4Rec, ví dụ:

- Hidden size.
- Số lớp GRU.
- Dropout.
- Learning rate.
- Loss type.

### 5.4. `configs/narm.yaml`

Chứa các tham số riêng của NARM, ví dụ:

- Hidden size.
- Số lớp.
- Kích thước attention.
- Dropout.
- Learning rate.

### 5.5. `configs/sasrec.yaml`

Chứa các tham số riêng của SASRec, ví dụ:

- Số attention head.
- Số Transformer layer.
- Inner size.
- Dropout.
- Maximum sequence length.
- Learning rate.

### Quy tắc sử dụng `configs/`

- Không thay cấu hình giữa hai seed của cùng một mô hình, ngoại trừ seed.
- Mỗi lần huấn luyện cần sao chép cấu hình thực tế vào thư mục run.
- Nếu thay đổi cấu hình quan trọng, nên tạo lần chạy mới thay vì ghi đè checkpoint cũ.

---

## 6. Thư mục `data/`

Thư mục `data/` chứa toàn bộ dữ liệu của dự án theo ba cấp:

```text
raw → interim → processed
```

### 6.1. `data/raw/`

Trạng thái: thư mục `[TẠO TRƯỚC]`, dữ liệu `[TẢI VỀ]`.

Đây là nơi lưu dữ liệu nguyên bản, chưa chỉnh sửa. Nguyên tắc bắt buộc:

> Notebook chỉ được đọc từ `data/raw/`, không được ghi đè hoặc sửa trực tiếp file gốc.

#### `data/raw/diginetica/`

Nơi đặt ba file Diginetica:

##### `train-item-views.csv`

Chứa lượt tương tác của item theo session và thời gian. Đây là nguồn chính để tạo chuỗi SBR.

##### `product-categories.csv`

Chứa ánh xạ giữa item và category. File được dùng trong phân tích cosine và cấu trúc category của embedding.

##### `products.csv`

Chứa metadata sản phẩm. Đề tài chính không nhất thiết đưa metadata vào mô hình, nhưng file cần được giữ để kiểm tra và phục vụ mở rộng nếu cần.

### 6.2. `data/interim/`

Trạng thái: thư mục `[TẠO TRƯỚC]`, nội dung `[SINH RA]`.

Đây là dữ liệu trung gian sau khi làm sạch nhưng trước khi chuyển hoàn toàn sang định dạng huấn luyện.

#### `interactions_clean.parquet`

Dự kiến chứa các tương tác đã:

- Chuẩn hóa tên cột.
- Loại bản ghi lỗi.
- Sắp xếp theo thời gian trong từng session.
- Xử lý trùng lặp theo quy tắc đã chọn.
- Lọc session/item theo protocol.

#### `items_clean.parquet`

Chứa danh sách item đã làm sạch và các thuộc tính cần giữ.

#### `category_mapping.csv`

Chứa ánh xạ item–category sau khi giải quyết thiếu category, trùng category hoặc item không khớp giữa các file nguồn.

Dữ liệu trong `interim/` có thể xóa và tạo lại từ `raw/` bất kỳ lúc nào.

### 6.3. `data/processed/`

Trạng thái: thư mục `[TẠO TRƯỚC]`, nội dung `[SINH RA]`.

Đây là dữ liệu cuối cùng được dùng trực tiếp cho huấn luyện và phân tích.

#### `data/processed/recbole/diginetica/`

Chứa các atomic file theo định dạng RecBole.

##### `diginetica.train.inter`

Chứa tương tác dùng để huấn luyện. Cold item không được xuất hiện trong file này nếu protocol yêu cầu cold item hoàn toàn chưa thấy.

##### `diginetica.valid.inter`

Chứa dữ liệu validation dùng để chọn epoch/checkpoint và early stopping.

##### `diginetica.test.inter`

Chứa dữ liệu test dùng để đánh giá cuối cùng.

##### `diginetica.item`

Chứa thông tin item cần thiết cho RecBole, có thể gồm raw item token và category tùy cấu hình.

#### `cold_items.csv`

Danh sách item được xác định là cold theo giao thức 20%.

File nên chứa tối thiểu:

```text
item_id
category_id
```

Nếu có thêm thông tin, cần ghi rõ cách tạo.

#### `item_mapping.csv`

Ánh xạ giữa raw item ID của Diginetica và internal item ID do RecBole sử dụng.

Đây là một trong những file quan trọng nhất của dự án. Nếu mapping sai:

- Embedding sẽ bị gắn nhầm item.
- Tần suất train sẽ gắn nhầm embedding.
- Category sẽ bị gắn nhầm.
- Toàn bộ phân tích item-level không còn giá trị.

#### `popularity_groups.csv`

Chứa tần suất train và nhóm phổ biến của từng item:

```text
item_id
train_freq
popularity_group
is_cold
```

Các nhóm có thể là `popular`, `medium`, `rare` và `cold`.

#### `dataset_statistics.json`

Lưu thống kê sau tiền xử lý, chẳng hạn:

- Số session.
- Số item.
- Số interaction.
- Độ dài session trung bình.
- Số item theo nhóm popularity.
- Số cold item.
- Số lượng train, validation và test.

File này giúp kiểm tra hai người có đang sử dụng cùng một phiên bản dữ liệu hay không.

---

## 7. Thư mục `notebooks/`

Trạng thái: `[TẠO TRƯỚC]`.

Các notebook hiện là file `.ipynb` hợp lệ nhưng chưa có cell. Tên notebook bắt đầu bằng số để thể hiện thứ tự chạy.

Nguyên tắc:

- Notebook dùng để điều phối, chạy thí nghiệm và trình bày kết quả.
- Hàm dùng lặp lại nhiều lần nên chuyển sang `src/btl_k03/`.
- Mỗi notebook cần ghi rõ đầu vào, đầu ra và thời gian chạy.
- Không phụ thuộc vào biến chỉ tồn tại do chạy ngẫu nhiên một notebook trước đó; dữ liệu trao đổi phải được lưu thành file.

### 7.1. `00_environment_setup.ipynb`

Nhiệm vụ:

- Kiểm tra phiên bản Python.
- Cài RecBole và các thư viện.
- Kiểm tra GPU/CUDA.
- Kết nối Google Drive nếu chạy Colab.
- Thiết lập đường dẫn gốc của dự án.
- Kiểm tra các thư mục cần thiết.

Đầu vào: không có dữ liệu nghiên cứu.

Đầu ra: môi trường chạy sẵn sàng; không cần tạo kết quả phân tích.

### 7.2. `01_data_exploration.ipynb`

Nhiệm vụ:

- Đọc ba file Diginetica thô.
- Kiểm tra schema và kiểu dữ liệu.
- Đếm session, item và interaction.
- Khảo sát độ dài session.
- Khảo sát tần suất item.
- Kiểm tra category thiếu hoặc không khớp.
- Kiểm tra trùng lặp và timestamp.

Đầu vào:

```text
data/raw/diginetica/
```

Đầu ra dự kiến:

```text
results/tables/raw_dataset_statistics.csv
results/figures/raw_item_frequency.png
results/figures/session_length_distribution.png
```

Notebook này không được sửa dữ liệu raw.

### 7.3. `02_data_preprocessing.ipynb`

Nhiệm vụ:

- Làm sạch dữ liệu.
- Sắp xếp tương tác theo thời gian.
- Lọc session và item theo protocol.
- Ánh xạ category.
- Tạo train, validation và test.
- Tạo cold-start 20% theo Option B.
- Chia popular, medium và rare.
- Xuất dữ liệu sang định dạng RecBole.
- Lưu item mapping và thống kê dataset.

Đầu vào:

```text
data/raw/diginetica/
configs/data.yaml
```

Đầu ra:

```text
data/interim/
data/processed/
```

Đây là notebook duy nhất chịu trách nhiệm chính cho việc tạo dữ liệu huấn luyện.

### 7.4. `03_train_gru4rec.ipynb`

Nhiệm vụ:

- Đọc dữ liệu processed và cấu hình GRU4Rec.
- Huấn luyện với hai seed.
- Chọn checkpoint tốt nhất theo validation metric.
- Đánh giá trên test.
- Sinh top-20 cho các session test.
- Trích xuất item embedding.
- Lưu mọi đầu ra vào `runs/gru4rec/`.

Đầu vào:

```text
configs/common.yaml
configs/data.yaml
configs/gru4rec.yaml
data/processed/
```

Đầu ra:

```text
runs/gru4rec/seed_2024/
runs/gru4rec/seed_2025/
```

### 7.5. `04_train_narm.ipynb`

Nhiệm vụ tương tự notebook GRU4Rec nhưng dành cho NARM.

Đầu ra:

```text
runs/narm/seed_2024/
runs/narm/seed_2025/
```

### 7.6. `05_train_sasrec.ipynb`

Nhiệm vụ tương tự hai notebook trên nhưng dành cho SASRec.

Đầu ra:

```text
runs/sasrec/seed_2024/
runs/sasrec/seed_2025/
```

### 7.7. `06_popularity_bias.ipynb`

Đây là notebook chính về thiên lệch đầu ra.

Nhiệm vụ:

- Đọc top-20 của ba mô hình và hai seed.
- Tính recommendation count theo item.
- Tính exposure share.
- Tính Catalog Coverage@20.
- Tính Gini.
- Tính Average Recommendation Popularity.
- So sánh ba mô hình.
- Kết hợp với Recall/MRR/NDCG để kiểm tra trade-off chất lượng–thiên lệch.

Đầu vào:

```text
runs/*/seed_*/top20.parquet
runs/*/seed_*/test_metrics.json
data/processed/popularity_groups.csv
```

Đầu ra dự kiến:

```text
results/tables/model_accuracy.csv
results/tables/popularity_bias.csv
results/tables/exposure_by_group.csv
results/figures/catalog_coverage.png
results/figures/gini_comparison.png
results/figures/average_popularity.png
results/figures/exposure_distribution.png
```

### 7.8. `07_embedding_geometry.ipynb`

Đây là notebook chính về hình học embedding.

Nhiệm vụ:

- Đọc item embedding của từng model/seed.
- Ánh xạ embedding về raw item ID.
- Tính norm embedding.
- Tính Spearman giữa train frequency và norm.
- So sánh norm của popular, medium, rare và cold.
- Tính cosine của cặp cùng/khác category.
- Có thể tính khoảng cách đến category centroid.
- Chạy t-SNE trên mẫu item.
- Đánh dấu cold item.
- Xây dựng bảng item-level.

Đầu vào:

```text
runs/*/seed_*/item_embeddings.npy
runs/*/seed_*/item_embedding_mapping.csv
data/processed/item_mapping.csv
data/processed/popularity_groups.csv
data/processed/cold_items.csv
```

Đầu ra dự kiến:

```text
results/item_level/
results/tables/norm_frequency.csv
results/tables/cosine_category.csv
results/tables/category_centroid.csv
results/statistical_tests/spearman_results.csv
results/statistical_tests/cosine_tests.csv
results/figures/norm_boxplot.png
results/figures/frequency_norm_scatter.png
results/figures/tsne_by_popularity.png
results/figures/tsne_by_category.png
```

### 7.9. `08_final_analysis.ipynb`

Nhiệm vụ:

- Nối kết quả output bias với embedding geometry.
- Tính quan hệ `train_freq ↔ rec_count`.
- Tính quan hệ `embedding_norm ↔ rec_count`.
- Kiểm tra quan hệ norm–rec_count trong từng frequency bin nếu thực hiện.
- So sánh kết quả qua hai seed.
- Xây dựng bảng tổng hợp cuối cùng.
- Trả lời trực tiếp các câu hỏi nghiên cứu.

Đầu vào:

```text
results/item_level/
results/tables/
results/statistical_tests/
```

Đầu ra dự kiến:

```text
results/tables/final_model_comparison.csv
results/tables/bias_geometry_relationship.csv
results/tables/seed_stability.csv
results/figures/norm_recommendation_relationship.png
results/figures/final_model_comparison.png
```

---

## 8. Thư mục `src/btl_k03/`

Trạng thái: `[TẠO TRƯỚC]`, các file hiện để trống.

Thư mục này chứa các hàm Python được nhiều notebook dùng chung. Ban đầu có thể viết trực tiếp trong notebook; khi một hàm bắt đầu bị lặp lại thì chuyển vào `src/`.

### 8.1. `__init__.py`

Đánh dấu `btl_k03` là một Python package, giúp notebook import các module dùng chung.

### 8.2. `paths.py`

Dự kiến quản lý đường dẫn chuẩn:

- Project root.
- Raw data.
- Processed data.
- Runs.
- Results.
- Figures.

Mục tiêu là tránh hard-code các đường dẫn khác nhau trong từng notebook.

### 8.3. `popularity_metrics.py`

Dự kiến chứa các hàm:

- Tính Coverage@k.
- Tính Gini.
- Tính Average Recommendation Popularity.
- Tính recommendation count.
- Tính exposure share.
- Tổng hợp chỉ số theo popularity group.

### 8.4. `embedding_analysis.py`

Dự kiến chứa các hàm:

- Tính norm embedding.
- Tính Spearman.
- Lấy mẫu cặp item cùng/khác category.
- Tính cosine similarity.
- Tính category centroid.
- Tạo dữ liệu đầu vào cho t-SNE.
- Kết hợp embedding với item metadata.

### 8.5. `utils.py`

Dự kiến chứa tiện ích chung:

- Thiết lập random seed.
- Đọc/ghi JSON, CSV và Parquet.
- Tạo thư mục đầu ra nếu chưa tồn tại.
- Ghi metadata của lần chạy.
- Kiểm tra các cột bắt buộc.

### Quy tắc cho `src/`

- Không đặt code chỉ dùng đúng một lần nếu nó làm dự án khó hiểu hơn.
- Hàm tính metric cần có đầu vào/đầu ra rõ ràng.
- Không để hàm tự ghi file vào vị trí bất ngờ.
- Notebook quyết định file đầu vào và nơi lưu đầu ra.

---

## 9. Thư mục `runs/`

Trạng thái: thư mục cấp mô hình `[TẠO TRƯỚC]`, nội dung `[SINH RA]`.

`runs/` lưu đầu ra trực tiếp của từng lần huấn luyện. Mỗi mô hình và seed có thư mục riêng để tránh ghi đè.

Cấu trúc chuẩn của một lần chạy:

```text
runs/{model}/seed_{seed}/
├── config.yaml
├── checkpoint.pth
├── training.log
├── training_history.csv
├── test_metrics.json
├── top20.parquet
├── item_embeddings.npy
└── item_embedding_mapping.csv
```

Ví dụ:

```text
runs/sasrec/seed_2024/
```

### 9.1. `config.yaml`

Bản sao cấu hình thực tế đã dùng trong lần chạy. File giúp xác định checkpoint được tạo bằng tham số nào.

### 9.2. `checkpoint.pth`

Trọng số đã học của mô hình. Đây không phải code; nó là sản phẩm của quá trình huấn luyện.

### 9.3. `training.log`

Log huấn luyện, bao gồm epoch, loss, validation metric, thời gian và thông báo lỗi/cảnh báo.

### 9.4. `training_history.csv`

Bảng lịch sử theo epoch, thuận tiện để vẽ learning curve.

Ví dụ các cột:

```text
epoch
train_loss
validation_score
elapsed_time
```

### 9.5. `test_metrics.json`

Chứa Recall@20, MRR@20, NDCG@20 và các metric kiểm tra chất lượng dự đoán.

### 9.6. `top20.parquet`

Chứa top-20 của các session test. Đây là đầu vào chính của notebook popularity bias.

Các cột dự kiến:

```text
session_id
rank
item_id
score
```

### 9.7. `item_embeddings.npy`

Ma trận item embedding trích từ checkpoint tốt nhất.

### 9.8. `item_embedding_mapping.csv`

Xác định hàng thứ bao nhiêu trong `item_embeddings.npy` tương ứng với item nào. File này cần được lưu cùng embedding để tránh nhầm mapping.

### Phân biệt `runs/` với `results/`

```text
runs/    = đầu ra trực tiếp của từng lần huấn luyện
results/ = kết quả tổng hợp và phân tích từ nhiều lần chạy
```

---

## 10. Thư mục `results/`

Trạng thái: thư mục `[TẠO TRƯỚC]`, nội dung `[SINH RA]`.

### 10.1. `results/item_level/`

Chứa bảng phân tích từng item cho mỗi mô hình và seed.

Ví dụ:

```text
gru4rec_seed2024.parquet
gru4rec_seed2025.parquet
narm_seed2024.parquet
narm_seed2025.parquet
sasrec_seed2024.parquet
sasrec_seed2025.parquet
```

Các cột dự kiến:

```text
model
seed
item_id
category_id
train_freq
popularity_group
is_cold
embedding_norm
rec_count
exposure_share
distance_to_category_centroid
```

Đây là cầu nối giữa phần popularity bias và embedding geometry.

### 10.2. `results/tables/`

Chứa các bảng nhỏ được dùng trong báo cáo:

```text
raw_dataset_statistics.csv
model_accuracy.csv
popularity_bias.csv
exposure_by_group.csv
norm_frequency.csv
cosine_category.csv
category_centroid.csv
final_model_comparison.csv
bias_geometry_relationship.csv
seed_stability.csv
```

### 10.3. `results/figures/`

Chứa hình dùng trong notebook, báo cáo và slide:

- Phân bố tần suất item.
- Phân bố độ dài session.
- So sánh Coverage.
- So sánh Gini.
- So sánh ARP.
- Boxplot norm theo nhóm popularity.
- Scatter frequency–norm.
- Scatter norm–recommendation count.
- t-SNE theo popularity.
- t-SNE theo category.
- Hình so sánh cuối cùng giữa ba mô hình.

Tên file hình cần thể hiện nội dung; không dùng tên như `figure1.png` nếu chưa có quy ước rõ ràng.

### 10.4. `results/statistical_tests/`

Chứa kết quả thống kê chi tiết:

- Spearman coefficient và p-value.
- Bootstrap confidence interval nếu có.
- Kiểm định cosine cùng/khác category.
- Kết quả hồi quy hoặc partial analysis nếu thực hiện.

Tách phần này khỏi `tables/` giúp bảng báo cáo ngắn gọn trong khi vẫn giữ đủ kết quả để kiểm tra.

---

## 11. Thư mục `reports/`

Trạng thái: `[TẠO TRƯỚC]`, nội dung `[VIẾT SAU]`.

### 11.1. `reports/report/`

Chứa báo cáo 12–16 trang và các file nguồn liên quan.

Có thể gồm:

```text
report.docx
report.pdf
```

hoặc file LaTeX nếu nhóm sử dụng LaTeX.

### 11.2. `reports/slides/`

Chứa slide thuyết trình:

```text
slides.pptx
slides.pdf
```

Hình trong báo cáo/slide nên lấy từ `results/figures/`, không sao chép nhiều phiên bản không rõ nguồn gốc.

---

## 12. Thư mục `tests/`

Trạng thái: `[TẠO TRƯỚC]`, các file hiện để trống.

Thư mục này chứa các kiểm tra tự động cho những phần dễ làm sai nhất.

### 12.1. `test_data_split.py`

Kiểm tra:

- Train/validation/test được chia đúng.
- Không có bản ghi bị mất ngoài dự kiến.
- Thứ tự thời gian phù hợp nếu dùng temporal split.
- Các session thỏa mãn độ dài tối thiểu.

### 12.2. `test_item_mapping.py`

Kiểm tra:

- Raw item ID ánh xạ nhất quán sang internal ID.
- Không có hai raw ID ánh xạ sai vào cùng internal ID.
- Hàng embedding khớp đúng với item.
- Category được nối đúng item.

### 12.3. `test_no_cold_leakage.py`

Kiểm tra điều kiện quan trọng:

```text
cold_items ∩ train_items = ∅
```

Nếu giao thức Option B có quy định khác, test phải phản ánh đúng định nghĩa chính thức.

### 12.4. `test_popularity_metrics.py`

Kiểm tra Coverage, Gini và ARP trên các ví dụ nhỏ có thể tính bằng tay. Mục đích là tránh công thức chạy được nhưng mang ý nghĩa sai.

---

## 13. File `.gitkeep`

Git không theo dõi thư mục rỗng. Vì thế, các thư mục chưa có dữ liệu thật đang chứa một file rỗng tên `.gitkeep`.

Ví dụ:

```text
results/figures/.gitkeep
```

`.gitkeep`:

- Không chứa code.
- Không tham gia vào quá trình huấn luyện.
- Không ảnh hưởng đến dữ liệu hoặc kết quả.
- Chỉ giúp cấu trúc thư mục tồn tại khi đưa lên Git.

Khi thư mục đã có file thật, có thể xóa hoặc giữ `.gitkeep`; cả hai đều không ảnh hưởng đến chương trình.

---

## 14. Luồng dữ liệu giữa các thư mục

```text
data/raw/
    │
    │  01_data_exploration.ipynb
    │  02_data_preprocessing.ipynb
    ▼
data/interim/
    │
    ▼
data/processed/
    │
    │  03_train_gru4rec.ipynb
    │  04_train_narm.ipynb
    │  05_train_sasrec.ipynb
    ▼
runs/{model}/seed_{seed}/
    │
    │  06_popularity_bias.ipynb
    │  07_embedding_geometry.ipynb
    ▼
results/item_level/
results/tables/
results/figures/
results/statistical_tests/
    │
    │  08_final_analysis.ipynb
    ▼
reports/report/
reports/slides/
```

Quy tắc dễ nhớ:

```text
Raw data       → data/raw/
Clean data     → data/interim/
Training data  → data/processed/
Model outputs  → runs/
Analysis       → results/
Deliverables   → reports/
```

---

## 15. Thứ tự chạy dự án

### Bước 1: Chuẩn bị môi trường

```text
00_environment_setup.ipynb
```

### Bước 2: Đặt dữ liệu

Tải Diginetica vào:

```text
data/raw/diginetica/
```

### Bước 3: Khảo sát dữ liệu

```text
01_data_exploration.ipynb
```

### Bước 4: Tiền xử lý

```text
02_data_preprocessing.ipynb
```

Kiểm tra dữ liệu trong `data/processed/` trước khi huấn luyện.

### Bước 5: Huấn luyện

```text
03_train_gru4rec.ipynb
04_train_narm.ipynb
05_train_sasrec.ipynb
```

Mỗi notebook phải hoàn thành cả hai seed và lưu vào `runs/`.

### Bước 6: Đo popularity bias

```text
06_popularity_bias.ipynb
```

### Bước 7: Phân tích embedding

```text
07_embedding_geometry.ipynb
```

### Bước 8: Tổng hợp

```text
08_final_analysis.ipynb
```

### Bước 9: Viết báo cáo và slide

```text
reports/report/
reports/slides/
```

---

## 16. Những file cần sao lưu sau mỗi lần huấn luyện

Nếu chạy trên Colab hoặc Kaggle, phiên làm việc có thể bị ngắt. Sau mỗi lần huấn luyện thành công, cần sao lưu tối thiểu:

```text
checkpoint.pth
config.yaml
training.log
test_metrics.json
top20.parquet
item_embeddings.npy
item_embedding_mapping.csv
```

Không chỉ sao lưu checkpoint. Nếu mất item mapping hoặc top-20 thì việc phân tích có thể không tái lập được dù vẫn còn trọng số mô hình.

---

## 17. Quy tắc đặt tên kết quả

Tên file cần chứa đủ thông tin để nhận diện mô hình và seed khi cần.

Ví dụ tốt:

```text
sasrec_seed2024.parquet
gru4rec_seed2025_norm_frequency.csv
narm_seed2024_tsne_popularity.png
```

Không nên dùng:

```text
result.csv
final.csv
final_new.csv
figure1.png
test2.npy
```

Nếu đã chia thư mục theo mô hình và seed trong `runs/`, các file bên trong có thể dùng tên thống nhất như `checkpoint.pth`.

---

## 18. Phân công làm việc theo cấu trúc

### Sinh viên 1

Phụ trách chính:

```text
configs/
data/
03_train_gru4rec.ipynb
04_train_narm.ipynb
05_train_sasrec.ipynb
06_popularity_bias.ipynb
runs/
```

### Sinh viên 2

Phụ trách chính:

```text
07_embedding_geometry.ipynb
src/btl_k03/embedding_analysis.py
results/item_level/
results/statistical_tests/
```

### Làm chung

```text
02_data_preprocessing.ipynb
08_final_analysis.ipynb
results/tables/
results/figures/
reports/
tests/
```

Hai thành viên phải thống nhất:

- Item mapping.
- Cold item list.
- Popularity groups.
- Candidate catalog.
- Seed.
- Tên cột của bảng item-level.

---

## 19. Những nguyên tắc bắt buộc

1. Không sửa file trong `data/raw/`.
2. Không ghi đè kết quả của seed khác.
3. Không dùng checkpoint nếu không biết cấu hình tạo ra nó.
4. Luôn lưu mapping cùng item embedding.
5. Cold item không được rò rỉ vào train nếu protocol yêu cầu strict cold.
6. Các mô hình phải dùng cùng split và candidate catalog.
7. Notebook phân tích phải đọc kết quả đã lưu, không phụ thuộc vào biến trong RAM của notebook huấn luyện.
8. Không dùng `runs/` để lưu hình báo cáo.
9. Không dùng `results/` để lưu checkpoint.
10. Mọi bảng/hình trong báo cáo phải truy được về notebook và file đầu vào đã tạo ra nó.

---

## 20. Tóm tắt ngắn

```text
configs/   → cấu hình thí nghiệm
data/      → dữ liệu raw, interim và processed
notebooks/ → các bước chạy chính
src/       → hàm Python dùng chung
runs/      → checkpoint, log, top-20 và embedding theo model/seed
results/   → bảng, hình và kiểm định đã tổng hợp
reports/   → báo cáo và slide
tests/     → kiểm tra split, mapping, cold leakage và metric
```

Pipeline chính:

```text
Diginetica raw
→ tiền xử lý
→ huấn luyện 3 mô hình × 2 seed
→ lưu checkpoint/top-20/embedding
→ đo popularity bias
→ phân tích embedding geometry
→ liên hệ hai kết quả
→ báo cáo và slide
```
