# Runtime cục bộ

Thư mục này chứa cache và file tạm; chỉ README được quản lý bằng Git.
`pytest.ini` đưa cache vào `pytest/` và dữ liệu kiểm thử tạm vào `pytest-tmp/`.

Khi cần thực thi notebook bằng kernel riêng từ PowerShell:

```powershell
$env:JUPYTER_RUNTIME_DIR = Join-Path (Get-Location) '.cache/jupyter_runtime'
$env:IPYTHONDIR = Join-Path (Get-Location) '.cache/ipython'
```
