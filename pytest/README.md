# Thư mục runtime cục bộ

Thư mục này gom các file tạm sinh ra khi kiểm thử hoặc chạy notebook để không làm rối thư mục gốc của dự án.

- `cache/`: cache của pytest.
- `tmp/`: thư mục tạm của pytest.
- `jupyter_runtime/`: connection file và runtime file của Jupyter.
- `ipython/`: profile và file tạm của IPython.

Ba thư mục runtime trên được bỏ qua bởi Git và sẽ tự sinh khi cần.

Khi chạy notebook từ PowerShell, đặt:

```powershell
$env:JUPYTER_RUNTIME_DIR = "D:\DA-3\pytest\jupyter_runtime"
$env:IPYTHONDIR = "D:\DA-3\pytest\ipython"
```
