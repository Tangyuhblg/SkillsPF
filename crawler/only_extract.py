# -*- coding: utf-8 -*-
"""
extract_all_zips_to_repo.py

功能：
1. 遍历 /root/BATaint/datasets/zip 下所有 .zip 文件；
2. 将每个 zip 解压到 /root/BATaint/datasets/repo/<zip文件名不含后缀>/；
3. 如果目标解压目录已经存在且非空，默认跳过，避免重复覆盖；
4. 支持安全解压，防止 zip 内部路径穿越；
5. 生成 JSON 解压日志。

示例：
/root/BATaint/datasets/zip/example-skill.zip
解压到：
/root/BATaint/datasets/repo/example-skill/

运行：
python extract_all_zips_to_repo.py
"""

import os
import json
import shutil
import zipfile
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List


# ===================== 配置区 =====================

ZIP_DIR = r"/root/BATaint/datasets/zip"
REPO_DIR = r"/root/BATaint/datasets/repo"

# True：如果目标目录已存在且非空，则跳过
# False：如果目标目录已存在，则删除后重新解压
SKIP_EXISTING_NONEMPTY_DIR = True

# True：真正解压
# False：只预览，不真正解压
DO_EXTRACT = True

LOG_JSON = r"/root/BATaint/datasets/extract_all_zips_to_repo_log.json"


# ===================== 日志配置 =====================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(levelname)s\t%(message)s"
)


# ===================== 基础函数 =====================

def normalize_path(path: str) -> str:
    return os.path.abspath(os.path.normpath(path))


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def list_zip_files(zip_dir: str) -> List[str]:
    """
    列出 zip_dir 下所有 .zip 文件。
    只扫描 zip_dir 当前目录，不递归扫描子目录。
    """
    zip_dir = normalize_path(zip_dir)

    if not os.path.exists(zip_dir):
        raise FileNotFoundError(f"ZIP_DIR 不存在: {zip_dir}")

    if not os.path.isdir(zip_dir):
        raise NotADirectoryError(f"ZIP_DIR 不是目录: {zip_dir}")

    zip_files = []

    for name in sorted(os.listdir(zip_dir)):
        path = normalize_path(os.path.join(zip_dir, name))
        if os.path.isfile(path) and name.lower().endswith(".zip"):
            zip_files.append(path)

    return zip_files


def is_safe_zip_member(target_dir: str, member_name: str) -> bool:
    """
    检查 zip 成员路径是否安全，防止 ../ 路径穿越。
    """
    target_dir = normalize_path(target_dir)
    member_path = normalize_path(os.path.join(target_dir, member_name))

    try:
        common = os.path.commonpath([target_dir, member_path])
        return common == target_dir
    except Exception:
        return False


def safe_extract_zip(zip_path: str, extract_dir: str) -> Dict:
    """
    安全解压 zip 文件到 extract_dir。
    """
    zip_path = normalize_path(zip_path)
    extract_dir = normalize_path(extract_dir)

    record = {
        "zip_path": zip_path,
        "extract_dir": extract_dir,
        "success": False,
        "skipped": False,
        "file_count_in_zip": 0,
        "extracted_member_count": 0,
        "unsafe_member_count": 0,
        "error": "",
    }

    if not os.path.exists(zip_path):
        record["error"] = f"zip 文件不存在: {zip_path}"
        return record

    if not zipfile.is_zipfile(zip_path):
        record["error"] = f"不是合法 zip 文件: {zip_path}"
        return record

    if os.path.exists(extract_dir) and os.listdir(extract_dir):
        if SKIP_EXISTING_NONEMPTY_DIR:
            record["success"] = True
            record["skipped"] = True
            logging.info("目标目录已存在且非空，跳过: %s", extract_dir)
            return record

        logging.info("目标目录已存在，删除后重新解压: %s", extract_dir)
        if DO_EXTRACT:
            shutil.rmtree(extract_dir)

    if DO_EXTRACT:
        ensure_dir(extract_dir)

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.infolist()
            record["file_count_in_zip"] = len(members)

            for member in members:
                if not is_safe_zip_member(extract_dir, member.filename):
                    record["unsafe_member_count"] += 1
                    logging.warning("跳过不安全 zip 成员: %s | %s", zip_path, member.filename)
                    continue

                if DO_EXTRACT:
                    zf.extract(member, extract_dir)

                record["extracted_member_count"] += 1

        record["success"] = True
        logging.info("解压成功: %s -> %s", zip_path, extract_dir)
        return record

    except Exception as e:
        record["error"] = f"{type(e).__name__}: {e}"
        logging.error("解压失败: %s | %s", zip_path, record["error"])
        return record


def extract_all_zips(zip_dir: str, repo_dir: str) -> Dict:
    """
    遍历 zip_dir 下所有 zip 文件，并解压到 repo_dir。
    """
    zip_dir = normalize_path(zip_dir)
    repo_dir = normalize_path(repo_dir)

    ensure_dir(repo_dir)

    zip_files = list_zip_files(zip_dir)

    result = {
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "zip_dir": zip_dir,
        "repo_dir": repo_dir,
        "do_extract": DO_EXTRACT,
        "skip_existing_nonempty_dir": SKIP_EXISTING_NONEMPTY_DIR,
        "zip_total_count": len(zip_files),
        "records": [],
        "summary": {},
    }

    for idx, zip_path in enumerate(zip_files, 1):
        zip_name = os.path.basename(zip_path)
        zip_stem = os.path.splitext(zip_name)[0]
        extract_dir = normalize_path(os.path.join(repo_dir, zip_stem))

        logging.info("[%d/%d] 处理 zip: %s", idx, len(zip_files), zip_path)

        record = safe_extract_zip(
            zip_path=zip_path,
            extract_dir=extract_dir
        )
        result["records"].append(record)

    success_count = sum(1 for r in result["records"] if r["success"] and not r["skipped"])
    skipped_count = sum(1 for r in result["records"] if r["success"] and r["skipped"])
    failed_count = sum(1 for r in result["records"] if not r["success"])
    unsafe_member_count = sum(r["unsafe_member_count"] for r in result["records"])

    result["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result["summary"] = {
        "zip_total_count": len(zip_files),
        "success_extract_count": success_count,
        "skipped_existing_count": skipped_count,
        "failed_count": failed_count,
        "unsafe_member_count": unsafe_member_count,
    }

    return result


def save_json(path: str, data: Dict):
    path = normalize_path(path)
    ensure_dir(os.path.dirname(path))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    print("===== 解压所有 ZIP 到 repo =====")
    print(f"ZIP_DIR : {ZIP_DIR}")
    print(f"REPO_DIR: {REPO_DIR}")
    print(f"DO_EXTRACT: {DO_EXTRACT}")
    print(f"SKIP_EXISTING_NONEMPTY_DIR: {SKIP_EXISTING_NONEMPTY_DIR}")

    result = extract_all_zips(
        zip_dir=ZIP_DIR,
        repo_dir=REPO_DIR
    )

    save_json(LOG_JSON, result)

    print("\n===== 完成 =====")
    print(f"ZIP 总数: {result['summary']['zip_total_count']}")
    print(f"成功解压数量: {result['summary']['success_extract_count']}")
    print(f"已存在跳过数量: {result['summary']['skipped_existing_count']}")
    print(f"失败数量: {result['summary']['failed_count']}")
    print(f"不安全成员跳过数量: {result['summary']['unsafe_member_count']}")
    print(f"日志文件: {LOG_JSON}")


if __name__ == "__main__":
    main()
