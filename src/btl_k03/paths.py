from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Tìm thư mục gốc của dự án dựa trên file BTL-K03-tong-quan-de-tai.md."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "BTL-K03-tong-quan-de-tai.md").exists():
            return candidate
    raise FileNotFoundError(
        "Không tìm thấy thư mục gốc dự án BTL-K03. Hãy chạy từ bên trong repository."
    )


PROJECT_ROOT = find_project_root()
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIGI_DIR = DATA_DIR / "raw" / "diginetica"
INTERIM_DIGI_DIR = DATA_DIR / "interim" / "diginetica"
PROCESSED_DIGI_DIR = DATA_DIR / "processed" / "diginetica"
RECBOLE_DIGI_DIR = DATA_DIR / "processed" / "recbole" / "diginetica"

RUNS_DIR = PROJECT_ROOT / "runs"
RESULTS_DIR = PROJECT_ROOT / "results"
ITEM_LEVEL_DIR = RESULTS_DIR / "item_level"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
STAT_TESTS_DIR = RESULTS_DIR / "statistical_tests"


def get_run_dir(model_name: str, seed: int) -> Path:
    """Trả về đường dẫn thư mục lưu kết quả huấn luyện cho từng mô hình và seed."""
    path = RUNS_DIR / model_name.lower() / f"seed_{int(seed)}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_output_dirs() -> None:
    """Đảm bảo tất cả các thư mục đầu ra đều tồn tại."""
    for directory in (
        RUNS_DIR,
        ITEM_LEVEL_DIR,
        TABLES_DIR,
        FIGURES_DIR,
        STAT_TESTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
