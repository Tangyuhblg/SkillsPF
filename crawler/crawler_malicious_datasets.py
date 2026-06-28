# -*- coding: utf-8 -*-

import os
import csv
import json
import logging
import re
import zipfile
from typing import Dict, List, Tuple
import requests


# ========== 路径配置 ==========
CLASSIFICATION_CSV = "./data/skills_dataset.csv"
ZIP_DIR = r"../datasets/zip"
REPO_DIR = r"../datasets/repo"
OUTPUT_JSON = r"./data/malicious_skills.json"

REQUEST_TIMEOUT = 60
CHUNK_SIZE = 1024 * 1024  # 1MB

SCRIPT_EXTENSIONS = {
    ".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx",
    ".sh", ".bash", ".zsh", ".fish", ".ps1",
    ".rb", ".php", ".pl", ".lua", ".r"
}
SKIP_DIR_NAMES = {
    ".git", "node_modules", "dist", "build", "coverage",
    "__pycache__", ".venv", "venv", "env", ".idea", ".vscode"
}


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s\t%(levelname)s\t%(message)s"
    )


def save_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_url(url: str) -> str:
    if not isinstance(url, str) or not url.strip():
        return ""
    url = url.strip().split("#")[0].split("?")[0].rstrip("/")
    return url.lower()


def sanitize_filename(name: str, fallback: str) -> str:
    """
    将 skill_name 转成可作为 zip 文件名 / 解压目录名的安全字符串。

    说明：
    - Windows 文件名不能包含 <>:"/\\|?* 等字符；
    - 空白字符统一替换为下划线；
    - 如果 skill_name 为空，则使用 fallback。
    """
    name = (name or "").strip()
    if not name:
        name = fallback

    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", name)
    name = re.sub(r"\s+", "_", name)
    name = name.strip(" ._")

    return name or fallback


def read_classification_csv(csv_path: str) -> List[dict]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"分类 CSV 不存在: {csv_path}")

    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def filter_suspicious_rows(rows: List[dict]) -> List[dict]:
    suspicious = []
    for row in rows:
        cls = (row.get("classification") or "").strip().lower()
        if cls == "suspicious":
            suspicious.append(row)
    return suspicious


def download_file(url: str, save_path: str) -> Tuple[bool, int, str]:
    status_code = None
    try:
        response = requests.get(
            url,
            stream=True,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )
        status_code = response.status_code

        if status_code != 200:
            return False, status_code, f"HTTP {status_code}"

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

        return True, status_code, ""
    except Exception as e:
        return False, status_code or -1, str(e)


def extract_zip_to_repo(zip_path: str, repo_dir: str) -> Tuple[bool, str]:
    """
    把 zip 解压到 repo/<zip_stem> 目录
    返回: (success, extract_dir_or_error)
    """
    if not os.path.exists(zip_path):
        return False, f"zip 不存在: {zip_path}"

    if not zipfile.is_zipfile(zip_path):
        return False, f"不是合法 zip: {zip_path}"

    zip_stem = os.path.splitext(os.path.basename(zip_path))[0]
    extract_dir = os.path.join(repo_dir, zip_stem)

    os.makedirs(repo_dir, exist_ok=True)

    if os.path.exists(extract_dir) and os.listdir(extract_dir):
        return True, extract_dir

    if os.path.exists(extract_dir):
        try:
            import shutil
            shutil.rmtree(extract_dir)
        except Exception:
            pass

    os.makedirs(extract_dir, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        return True, extract_dir
    except Exception as e:
        return False, str(e)


def download_and_extract_unique_suspicious_zips(
    suspicious_rows: List[dict],
    zip_dir: str,
    repo_dir: str,
):
    """
    1) 读取分类 CSV 并筛 suspicious
    2) 下载并解压所有唯一 suspicious zip
    """
    os.makedirs(zip_dir, exist_ok=True)
    os.makedirs(repo_dir, exist_ok=True)

    unique_by_url = {}
    for row in suspicious_rows:
        url = normalize_url(row.get("url", ""))
        if not url:
            continue
        if url not in unique_by_url:
            unique_by_url[url] = row

    download_log: Dict[str, dict] = {}
    used_skill_ids = set()

    for idx, (url, row) in enumerate(unique_by_url.items(), 1):
        raw_skill_name = (row.get("skill_name") or "").strip()
        base_skill_id = sanitize_filename(raw_skill_name, fallback=f"skill_{idx:06d}")

        # 如果不同 URL 的 skill_name 相同，避免覆盖已有 zip。
        skill_id = base_skill_id
        duplicate_idx = 2
        while skill_id in used_skill_ids:
            skill_id = f"{base_skill_id}_{duplicate_idx}"
            duplicate_idx += 1
        used_skill_ids.add(skill_id)

        filename = f"{skill_id}.zip"
        save_path = os.path.join(zip_dir, filename)

        if os.path.exists(save_path):
            ok = True
            status_code = 200
            err = ""
            skipped_existing = True
            logging.info("[%d/%d] 已存在，跳过下载: %s", idx, len(unique_by_url), save_path)
        else:
            ok, status_code, err = download_file(url, save_path)
            skipped_existing = False
            if ok:
                logging.info("[%d/%d] 下载成功: %s -> %s", idx, len(unique_by_url), url, save_path)
            else:
                logging.error("[%d/%d] 下载失败: %s | %s", idx, len(unique_by_url), url, err)

        extracted = False
        extract_path = ""
        extract_error = ""

        if ok:
            extracted, extract_result = extract_zip_to_repo(save_path, repo_dir)
            if extracted:
                extract_path = extract_result
                logging.info("[%d/%d] 解压成功: %s -> %s", idx, len(unique_by_url), save_path, extract_path)
            else:
                extract_error = extract_result
                logging.error("[%d/%d] 解压失败: %s | %s", idx, len(unique_by_url), save_path, extract_error)

        download_log[url] = {
            "skill_id": skill_id,
            "skill_name": raw_skill_name,
            "downloaded": ok,
            "skipped_existing": skipped_existing,
            "zip_path": save_path if ok else "",
            "status_code": status_code,
            "error": err,
            "extracted": extracted,
            "extract_path": extract_path,
            "extract_error": extract_error,
        }

    return download_log


def is_script_file(file_path: str) -> bool:
    suffix = os.path.splitext(file_path)[1].lower()
    if suffix in SCRIPT_EXTENSIONS:
        return True

    if suffix == "":
        try:
            with open(file_path, "rb") as f:
                first_line = f.readline(256)
            return first_line.startswith(b"#!")
        except Exception:
            return False

    return False


def find_repo_root(extract_path: str) -> str:
    """
    若解压目录下只有一个顶层文件夹，则返回该文件夹；否则返回解压目录本身。
    """
    if not extract_path or not os.path.exists(extract_path):
        return ""

    try:
        entries = os.listdir(extract_path)
    except Exception:
        return extract_path

    dirs = [d for d in entries if os.path.isdir(os.path.join(extract_path, d))]
    files = [f for f in entries if os.path.isfile(os.path.join(extract_path, f))]

    if len(dirs) == 1 and len(files) == 0:
        return os.path.join(extract_path, dirs[0])
    return extract_path


def normalize_scan_root(repo_root: str) -> Tuple[str, str]:
    """
    输出风格与 all_skills_data.json / all_skills_data_with_scripts.json 保持一致：
    1) 优先返回包含 SKILL.md 的目录
    2) 多个时取第一个
    3) 否则回退 repo_root
    """
    if not repo_root or not os.path.exists(repo_root):
        return "", "unresolved"

    candidates = []
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIR_NAMES]
        lower_files = {f.lower() for f in files}
        if "skill.md" in lower_files:
            candidates.append(root)

    if len(candidates) == 1:
        return os.path.normpath(candidates[0]), "skill_path"
    if len(candidates) > 1:
        return os.path.normpath(candidates[0]), "skill_path"
    return os.path.normpath(repo_root), "repo_path"


def find_script_files(scan_root: str) -> List[str]:
    if not scan_root or not os.path.exists(scan_root):
        return []

    found: List[str] = []
    for root, dirs, files in os.walk(scan_root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIR_NAMES]
        for filename in files:
            full_path = os.path.join(root, filename)
            if is_script_file(full_path):
                rel_path = os.path.relpath(full_path, scan_root)
                found.append(os.path.normpath(rel_path))
    return sorted(set(found))


def build_record(row: dict, idx: int, download_log: Dict[str, dict]) -> dict:
    """
    每个下载文件夹输出 1 条记录。
    字段风格与 all_skills_data.json 一致，并额外补充 repo_path / skill_path /
    scan_root / path_source / script_count / script_files。
    """
    s_url = normalize_url(row.get("url", ""))
    download = download_log.get(s_url, {})
    raw_skill_name = (row.get("skill_name") or "").strip()
    skill_id = download.get("skill_id") or sanitize_filename(raw_skill_name, fallback=f"skill_{idx:06d}")

    record = {
        "id": skill_id,
        "slug": raw_skill_name,
        "name": raw_skill_name,
        "label": True,
        "source_url": "",
        "r2_zip_key": row.get("url", ""),
        "created_at": "",
        "updated_at": "",
        "request": "",
        "data_source": row.get("source", ""),
    }

    if download.get("extracted") and download.get("extract_path") and os.path.exists(download["extract_path"]):
        repo_path = os.path.normpath(find_repo_root(download["extract_path"]))
        scan_root, path_source = normalize_scan_root(repo_path)
        script_files = find_script_files(scan_root)

        record.update({
            "repo_path": repo_path,
            "skill_path": scan_root,
            "scan_root": scan_root,
            "path_source": path_source,
            "script_count": len(script_files),
            "script_files": script_files,
        })
    else:
        record.update({
            "repo_path": "",
            "skill_path": "",
            "scan_root": "",
            "path_source": "",
            "script_count": 0,
            "script_files": [],
        })

    return record


def main():
    setup_logging()

    csv_rows = read_classification_csv(CLASSIFICATION_CSV)
    suspicious_rows = filter_suspicious_rows(csv_rows)

    logging.info("CSV 总行数: %d", len(csv_rows))
    logging.info("classification=suspicious 的行数: %d", len(suspicious_rows))

    download_log = download_and_extract_unique_suspicious_zips(
        suspicious_rows=suspicious_rows,
        zip_dir=ZIP_DIR,
        repo_dir=REPO_DIR,
    )

    output_records: List[dict] = []
    for idx, row in enumerate(suspicious_rows, 1):
        output_records.append(build_record(row, idx, download_log))

    save_json(OUTPUT_JSON, output_records)

    print("\n===== 完成 =====")
    print(f"输出 JSON: {OUTPUT_JSON}")
    print(f"总记录数: {len(output_records)}")


if __name__ == "__main__":
    main()
    """
        1
        抓取csv中suspicious的压缩包并解压
    """