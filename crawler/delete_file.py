# -*- coding: utf-8 -*-
"""
clean_repo_keep_py_and_skill_md.py

功能：
1. 遍历 /root/BATaint/datasets/repo；
2. 只保留：
   - .py 文件
   - 文件名为 skill.md / SKILL.md 的文件
3. 删除其它所有文件，包括 README.md、CHANGELOG.md、.js、.sh、.json 等；
4. 删除文件后，检查空文件夹并删除；
5. 不删除根目录 /root/BATaint/datasets/repo 本身；
6. 生成 JSON 日志。

注意：
- 这里保留的是文件名为 skill.md 的 Markdown 文件；
- 不是保留所有 .md 文件。
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime


# ===================== 配置区 =====================

ROOT_DIR = r"/root/BATaint/datasets/repo"

# True：只预览，不真正删除
# False：真正删除
DRY_RUN = False

# 是否删除空文件夹
REMOVE_EMPTY_DIRS = True

# 是否保留 ROOT_DIR 根目录本身
KEEP_ROOT_DIR = True

# 日志输出文件
LOG_JSON = "clean_repo_keep_py_and_skill_md_log.json"


# ===================== 日志配置 =====================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(levelname)s\t%(message)s"
)


def normalize_path(path: str) -> str:
    """
    统一路径格式。
    """
    return os.path.abspath(os.path.normpath(path))


def should_keep_file(file_path: str) -> bool:
    """
    判断文件是否需要保留。

    只保留：
    1. 后缀为 .py 的 Python 文件；
    2. 文件名为 skill.md 的文件，不区分大小写，例如：
       - skill.md
       - SKILL.md
       - Skill.md

    不保留：
    - README.md
    - CHANGELOG.md
    - requirements.txt
    - package.json
    - .js / .sh / .json / .yaml 等其它文件
    """
    filename = os.path.basename(file_path).lower()
    suffix = Path(file_path).suffix.lower()

    if suffix == ".py":
        return True

    if filename == "skill.md":
        return True

    return False


def delete_file(file_path: str, dry_run: bool = True) -> dict:
    """
    删除单个文件。

    如果 file_path 是符号链接文件，只删除链接本身，不删除链接指向的真实文件。
    """
    file_path = normalize_path(file_path)

    record = {
        "path": file_path,
        "deleted": False,
        "dry_run": dry_run,
        "error": ""
    }

    try:
        if dry_run:
            logging.info("[DRY-RUN] 将删除文件: %s", file_path)
            return record

        os.remove(file_path)
        record["deleted"] = True
        logging.info("已删除文件: %s", file_path)
        return record

    except Exception as e:
        record["error"] = str(e)
        logging.error("删除文件失败: %s | %s", file_path, e)
        return record


def remove_empty_dirs(root_dir: str, dry_run: bool = True, keep_root: bool = True) -> list:
    """
    删除空文件夹。

    逻辑：
    - 从最底层目录往上检查；
    - 如果目录中没有任何文件和子目录，则删除；
    - 默认不删除 root_dir 本身。
    """
    root_dir = normalize_path(root_dir)
    removed_dirs = []

    for current_root, dirs, files in os.walk(root_dir, topdown=False):
        current_root = normalize_path(current_root)

        if keep_root and current_root == root_dir:
            continue

        record = {
            "path": current_root,
            "deleted": False,
            "dry_run": dry_run,
            "error": ""
        }

        try:
            if not os.listdir(current_root):
                if dry_run:
                    logging.info("[DRY-RUN] 将删除空文件夹: %s", current_root)
                    removed_dirs.append(record)
                else:
                    os.rmdir(current_root)
                    record["deleted"] = True
                    logging.info("已删除空文件夹: %s", current_root)
                    removed_dirs.append(record)

        except Exception as e:
            record["error"] = str(e)
            logging.error("删除空文件夹失败: %s | %s", current_root, e)
            removed_dirs.append(record)

    return removed_dirs


def clean_repo_files(root_dir: str, dry_run: bool = True) -> dict:
    """
    删除 root_dir 下所有非 .py / skill.md 文件，然后删除空文件夹。
    """
    root_dir = normalize_path(root_dir)

    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"目录不存在: {root_dir}")

    if not os.path.isdir(root_dir):
        raise NotADirectoryError(f"不是目录: {root_dir}")

    result = {
        "root_dir": root_dir,
        "dry_run": dry_run,
        "keep_rule": "keep files whose suffix is .py OR whose filename is skill.md case-insensitively",
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "kept_files": [],
        "deleted_files": [],
        "error_files": [],
        "removed_empty_dirs": [],
        "summary": {}
    }

    total_files = 0

    # ========== 第一步：删除非 .py / skill.md 文件 ==========
    for current_root, dirs, files in os.walk(root_dir):
        # 不跟随符号链接目录，避免误删外部目录内容
        dirs[:] = [
            d for d in dirs
            if not os.path.islink(os.path.join(current_root, d))
        ]

        for filename in files:
            file_path = normalize_path(os.path.join(current_root, filename))
            total_files += 1

            if should_keep_file(file_path):
                result["kept_files"].append(file_path)
                logging.info("保留文件: %s", file_path)
            else:
                delete_record = delete_file(file_path, dry_run=dry_run)

                if delete_record["error"]:
                    result["error_files"].append(delete_record)
                else:
                    result["deleted_files"].append(delete_record)

    # ========== 第二步：检查并删除空文件夹 ==========
    if REMOVE_EMPTY_DIRS:
        result["removed_empty_dirs"] = remove_empty_dirs(
            root_dir=root_dir,
            dry_run=dry_run,
            keep_root=KEEP_ROOT_DIR
        )

    result["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    result["summary"] = {
        "total_files_scanned": total_files,
        "kept_file_count": len(result["kept_files"]),
        "deleted_file_count": len(result["deleted_files"]),
        "error_file_count": len(result["error_files"]),
        "removed_empty_dir_count": len(result["removed_empty_dirs"]),
    }

    return result


def save_json(path: str, data: dict):
    """
    保存清理日志。
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    print("===== 清理 repo：只保留 .py 和 skill.md，并删除空文件夹 =====")
    print(f"扫描目录: {ROOT_DIR}")
    print("保留规则: .py 文件 或 文件名为 skill.md/SKILL.md 的文件")
    print(f"DRY_RUN: {DRY_RUN}")
    print(f"REMOVE_EMPTY_DIRS: {REMOVE_EMPTY_DIRS}")

    if DRY_RUN:
        print("\n当前是预览模式，不会真正删除文件或文件夹。")
        print("确认输出无误后，请把 DRY_RUN 改为 False 再运行。\n")
    else:
        print("\n当前是真正删除模式，请确认路径正确！\n")

    result = clean_repo_files(ROOT_DIR, dry_run=DRY_RUN)
    save_json(LOG_JSON, result)

    print("\n===== 完成 =====")
    print(f"扫描文件总数: {result['summary']['total_files_scanned']}")
    print(f"保留文件数: {result['summary']['kept_file_count']}")
    print(f"删除文件数: {result['summary']['deleted_file_count']}")
    print(f"删除失败数: {result['summary']['error_file_count']}")
    print(f"删除空文件夹数: {result['summary']['removed_empty_dir_count']}")
    print(f"日志文件: {LOG_JSON}")


if __name__ == "__main__":
    main()
    """
        4
        删除无关文件，只保留.py, skill.md
        
        注意：
        正常抓取只需要4-7
        处理文献下载的csv需要1-7
    """