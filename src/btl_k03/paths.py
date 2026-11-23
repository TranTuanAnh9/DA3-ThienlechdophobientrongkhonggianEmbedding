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

RAW_YOO_DIR = DATA_DIR / "raw"
RAW_YOO_CLICKS_FILE = RAW_YOO_DIR / "yoochoose-clicks.dat" / "yoochoose-clicks.dat"
RAW_YOO_BUYS_FILE = RAW_YOO_DIR / "yoochoose-buys.dat" / "yoochoose-buys.dat"
INTERIM_YOO_DIR = DATA_DIR / "interim" / "yoochoose"
PROCESSED_YOO_DIR = DATA_DIR / "processed" / "yoochoose"

RUNS_DIR = PROJECT_ROOT / "runs"
RESULTS_DIR = PROJECT_ROOT / "results"
ITEM_LEVEL_DIR = RESULTS_DIR / "item_level"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
STAT_TESTS_DIR = RESULTS_DIR / "statistical_tests"

RUNS_YOO_DIR = RUNS_DIR / "yoochoose"
RESULTS_YOO_DIR = RESULTS_DIR / "yoochoose"
ITEM_LEVEL_YOO_DIR = RESULTS_YOO_DIR / "item_level"
TABLES_YOO_DIR = RESULTS_YOO_DIR / "tables"
FIGURES_YOO_DIR = RESULTS_YOO_DIR / "figures"
STAT_TESTS_YOO_DIR = RESULTS_YOO_DIR / "statistical_tests"


def get_dataset_paths(dataset: str = "diginetica") -> dict[str, Path]:
    """Trả về bộ đường dẫn chuẩn theo tên dataset ('diginetica' hoặc 'yoochoose')."""
    ds = dataset.lower()
    if ds == "diginetica":
        return {
            "config_dir": CONFIGS_DIR,
            "raw_dir": RAW_DIGI_DIR,
            "interim_dir": INTERIM_DIGI_DIR,
            "processed_dir": PROCESSED_DIGI_DIR,
            "runs_dir": RUNS_DIR,
            "results_dir": RESULTS_DIR,
            "item_level_dir": ITEM_LEVEL_DIR,
            "tables_dir": TABLES_DIR,
            "figures_dir": FIGURES_DIR,
            "stat_tests_dir": STAT_TESTS_DIR,
        }
    if ds == "yoochoose":
        return {
            "config_dir": CONFIGS_DIR / "yoochoose",
            "raw_dir": RAW_YOO_DIR,
            "raw_clicks_file": RAW_YOO_CLICKS_FILE,
            "raw_buys_file": RAW_YOO_BUYS_FILE,
            "interim_dir": INTERIM_YOO_DIR,
            "processed_dir": PROCESSED_YOO_DIR,
            "runs_dir": RUNS_YOO_DIR,
            "results_dir": RESULTS_YOO_DIR,
            "item_level_dir": ITEM_LEVEL_YOO_DIR,
            "tables_dir": TABLES_YOO_DIR,
            "figures_dir": FIGURES_YOO_DIR,
            "stat_tests_dir": STAT_TESTS_YOO_DIR,
        }
    raise ValueError(f"Dataset không được hỗ trợ: {dataset}")


def get_run_dir(model_name: str, seed: int, dataset: str = "diginetica") -> Path:
    """Trả về đường dẫn thư mục lưu kết quả huấn luyện cho từng mô hình và seed."""
    base_runs = RUNS_DIR if dataset.lower() == "diginetica" else get_dataset_paths(dataset)["runs_dir"]
    path = base_runs / model_name.lower() / f"seed_{int(seed)}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_output_dirs(dataset: str = "diginetica") -> None:
    """Đảm bảo tất cả các thư mục đầu ra đều tồn tại."""
    paths = get_dataset_paths(dataset)
    for key in ("interim_dir", "processed_dir", "runs_dir", "item_level_dir", "tables_dir", "figures_dir", "stat_tests_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)
