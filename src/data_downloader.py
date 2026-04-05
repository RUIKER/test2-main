"""
自动下载 NGAFID 基准子集数据。
优先使用 Zenodo 压缩包，失败时回退到 Google Drive。
"""

import json
import re
import tarfile
import urllib.parse
import urllib.request
import zipfile
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Any

ALT_DATASET_SOURCES = [
    "https://doi.org/10.5281/zenodo.6624956",
    "https://www.kaggle.com/datasets/hooong/aviation-maintenance-dataset-from-the-ngafid",
]

ZENODO_RECORD_API = "https://zenodo.org/api/records/6624956"
DEFAULT_DATASETS = ("2days",)


def _extract_filename(content_disposition: str, default_name: str) -> str:
    filename_star = re.search(r"filename\*=UTF-8''([^;]+)", content_disposition, re.IGNORECASE)
    if filename_star:
        return urllib.parse.unquote(filename_star.group(1)).strip()

    filename = re.search(r'filename="?([^";]+)"?', content_disposition, re.IGNORECASE)
    if filename:
        return filename.group(1).strip()

    return default_name


def _stream_to_file(response, output_path: Path):
    with open(output_path, "wb") as file_handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            file_handle.write(chunk)


def _download_url_to_file(url: str, output_path: Path):
    with urllib.request.urlopen(url, timeout=600) as response:
        _stream_to_file(response, output_path)


def _download_google_drive_file(file_id: str, download_dir: Path) -> Path:
    base_url = "https://drive.google.com/uc?export=download"
    cookie_jar = CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))

    first_url = f"{base_url}&id={urllib.parse.quote(file_id)}"
    with opener.open(first_url, timeout=120) as response:
        content_disposition = response.headers.get("Content-Disposition", "")

        if "attachment" in content_disposition.lower():
            filename = _extract_filename(content_disposition, f"{file_id}.bin")
            output_path = download_dir / filename
            _stream_to_file(response, output_path)
            return output_path

        html = response.read().decode("utf-8", errors="ignore")

    confirm_match = re.search(r"confirm=([0-9A-Za-z_]+)", html)
    if not confirm_match:
        raise RuntimeError(
            f"无法获取 Google Drive 确认令牌，file_id={file_id}。可能是链接失效或网络受限。"
        )

    confirm_token = confirm_match.group(1)
    second_url = f"{base_url}&id={urllib.parse.quote(file_id)}&confirm={confirm_token}"
    with opener.open(second_url, timeout=120) as response:
        content_disposition = response.headers.get("Content-Disposition", "")
        filename = _extract_filename(content_disposition, f"{file_id}.bin")
        output_path = download_dir / filename
        _stream_to_file(response, output_path)
        return output_path


def _extract_drive_ids_from_text(text: str) -> list[str]:
    patterns = [
        r"https?://drive\.google\.com/uc\?id=([a-zA-Z0-9_-]+)",
        r"https?://drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)",
        r"https?://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)",
    ]

    ids: list[str] = []
    for pattern in patterns:
        for file_id in re.findall(pattern, text):
            if file_id not in ids:
                ids.append(file_id)
    return ids


def _extract_file_ids_from_notebook(notebook_path: Path) -> list[str]:
    with open(notebook_path, "r", encoding="utf-8") as file_handle:
        notebook_content = json.load(file_handle)

    ids: list[str] = []
    for cell in notebook_content.get("cells", []):
        source = "".join(cell.get("source", []))
        for file_id in _extract_drive_ids_from_text(source):
            if file_id not in ids:
                ids.append(file_id)
    return ids


def _extract_file_ids_from_dataset_py(dataset_py_path: Path) -> list[str]:
    if not dataset_py_path.exists():
        return []

    text = dataset_py_path.read_text(encoding="utf-8")
    return _extract_drive_ids_from_text(text)


def _extract_named_file_ids_from_dataset_py(dataset_py_path: Path) -> dict[str, str]:
    if not dataset_py_path.exists():
        return {}

    text = dataset_py_path.read_text(encoding="utf-8")
    entries = re.findall(
        r'"([a-zA-Z0-9_]+)"\s*:\s*"https?://drive\.google\.com/uc\?id=([a-zA-Z0-9_-]+)"',
        text,
    )
    return {name: file_id for name, file_id in entries}


def _find_dataset_root(download_dir: Path, dataset_name: str) -> Path | None:
    if not download_dir.exists():
        return None

    candidates = [download_dir] + [path for path in download_dir.rglob("*") if path.is_dir()]

    if dataset_name == "2days":
        required_files = {"flight_data.pkl", "flight_header.csv", "stats.csv"}
        for candidate in candidates:
            files = {path.name for path in candidate.iterdir() if path.is_file()}
            if required_files.issubset(files):
                return candidate
        return None

    if dataset_name == "all_flights":
        for candidate in candidates:
            files = {path.name for path in candidate.iterdir() if path.is_file()}
            dirs = {path.name for path in candidate.iterdir() if path.is_dir()}
            if "flight_header.csv" in files and "one_parq" in dirs:
                return candidate
        return None

    return None


def _extract_archive(archive_path: Path, destination: Path):
    lower_name = archive_path.name.lower()
    if lower_name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive_path, "r:*") as tar_file:
            tar_file.extractall(destination)
        return
    if lower_name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zip_file:
            zip_file.extractall(destination)
        return
    raise ValueError(f"不支持的压缩格式: {archive_path.name}")


def _select_zenodo_file(files: list[dict[str, Any]], dataset_name: str) -> dict[str, Any] | None:
    archive_exts = (".tar.gz", ".tgz", ".tar", ".zip")
    keywords = [dataset_name]
    if dataset_name == "all_flights":
        keywords.append("all_flight")

    for file_info in files:
        key = str(file_info.get("key", "")).lower()
        if any(keyword in key for keyword in keywords) and key.endswith(archive_exts):
            return file_info

    return None


def _download_dataset_from_zenodo(download_dir: Path, dataset_name: str, files: list[dict[str, Any]]) -> bool:
    selected_file = _select_zenodo_file(files, dataset_name)
    if selected_file is None:
        raise RuntimeError(f"Zenodo 中未找到 {dataset_name} 对应压缩包。")

    archive_name = str(selected_file.get("key", "subset_data_archive.tar.gz"))
    links = selected_file.get("links", {})
    download_url = links.get("download") or links.get("self")
    if not download_url:
        raise RuntimeError("Zenodo 文件记录缺少下载链接。")

    archive_path = download_dir / archive_name
    print(f"正在下载: {archive_name}")
    _download_url_to_file(download_url, archive_path)
    print(f"下载完成，开始解压: {archive_path.name}")
    _extract_archive(archive_path, download_dir)
    try:
        archive_path.unlink(missing_ok=True)
    except OSError:
        pass

    dataset_root = _find_dataset_root(download_dir, dataset_name)
    if dataset_root is None:
        raise RuntimeError(f"{dataset_name} 已解压但未找到预期目录结构。")

    print(f"Zenodo 数据准备完成: {dataset_root}")
    return True


def _maybe_extract_downloaded_file(saved_path: Path, download_dir: Path):
    lower_name = saved_path.name.lower()
    if lower_name.endswith((".tar.gz", ".tgz", ".tar", ".zip")):
        print(f"检测到压缩包，开始解压: {saved_path.name}")
        _extract_archive(saved_path, download_dir)
        try:
            saved_path.unlink(missing_ok=True)
        except OSError:
            pass


def extract_and_download_subset(dataset_names: tuple[str, ...] = DEFAULT_DATASETS):
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent

    download_dir = project_root / "data" / "subset_data"
    download_dir.mkdir(parents=True, exist_ok=True)

    pending_datasets: list[str] = []
    for dataset_name in dataset_names:
        root = _find_dataset_root(download_dir, dataset_name)
        if root is None:
            pending_datasets.append(dataset_name)
        else:
            print(f"检测到本地数据已存在，跳过下载 {dataset_name}: {root}")

    if not pending_datasets:
        print("所有目标数据集均已就绪。")
        return

    print(f"目标下载数据集: {pending_datasets}")

    zenodo_failed: list[str] = []
    try:
        print(f"尝试从 Zenodo 记录下载数据: {ZENODO_RECORD_API}")
        with urllib.request.urlopen(ZENODO_RECORD_API, timeout=60) as response:
            record = json.loads(response.read().decode("utf-8"))
        files = record.get("files", [])
        if not files:
            raise RuntimeError("Zenodo 记录中未找到可下载文件。")

        for dataset_name in pending_datasets:
            try:
                _download_dataset_from_zenodo(download_dir, dataset_name, files)
            except Exception as item_exc:
                zenodo_failed.append(dataset_name)
                print(f"Zenodo 下载失败 {dataset_name}: {item_exc}")
    except Exception as exc:
        print(f"Zenodo 下载阶段失败，回退到 Google Drive 方案: {exc}")
        zenodo_failed = pending_datasets.copy()

    if not zenodo_failed:
        print("目标数据集已通过 Zenodo 下载完成。")
        return

    print("正在解析 Google Drive 下载链接...")

    repo_dir = project_root / "data"
    notebook_path = repo_dir / "NGAFID_DATASET_TF_EXAMPLE.ipynb"

    named_ids: dict[str, str] = {}
    file_ids: list[str] = []
    if notebook_path.exists():
        file_ids = _extract_file_ids_from_notebook(notebook_path)

    if file_ids:
        # Notebook 中若只有单个下载链接，按 2days 处理（当前默认任务）。
        if len(file_ids) == 1:
            named_ids["2days"] = file_ids[0]

    if not file_ids:
        dataset_py_path = project_root / "data" / "ngafiddataset" / "dataset" / "dataset.py"
        named_ids = _extract_named_file_ids_from_dataset_py(dataset_py_path)
        if not named_ids:
            file_ids = _extract_file_ids_from_dataset_py(dataset_py_path)
            if file_ids:
                named_ids = {f"file_{i}": file_id for i, file_id in enumerate(file_ids)}

    if not named_ids:
        print("未找到可用下载链接，请检查 Notebook 或 data/ngafiddataset/dataset/dataset.py。")
        return

    failed_datasets: list[str] = []
    for dataset_name in zenodo_failed:
        dataset_id = named_ids.get(dataset_name)
        if dataset_id is None and dataset_name == "all_flights":
            dataset_id = named_ids.get("all_flight")
        if dataset_id is None:
            failed_datasets.append(dataset_name)
            print(f"未找到 {dataset_name} 的 Google Drive file_id，跳过。")
            continue

        print(f"正在下载 {dataset_name} (file_id={dataset_id}) ...")
        try:
            saved_path = _download_google_drive_file(dataset_id, download_dir)
            print(f"已保存: {saved_path.name}")
            _maybe_extract_downloaded_file(saved_path, download_dir)
            if _find_dataset_root(download_dir, dataset_name) is None:
                failed_datasets.append(dataset_name)
                print(f"{dataset_name} 下载后结构校验失败。")
        except Exception as exc:
            failed_datasets.append(dataset_name)
            print(f"下载失败 {dataset_name}: {exc}")

    if failed_datasets:
        print(f"完成但有失败项: {failed_datasets}")
        print("可尝试从以下官方来源手动获取数据后放入 data/subset_data:")
        for src in ALT_DATASET_SOURCES:
            print(f"- {src}")
    else:
        print("目标数据集自动下载完成！")


if __name__ == "__main__":
    extract_and_download_subset()
