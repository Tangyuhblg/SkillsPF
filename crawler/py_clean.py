# -*- coding: utf-8 -*-
"""
clean_repo_keep_skill_md_and_scripts_py.py

功能：
1. 遍历 /root/BATaint/datasets/repo；
2. 识别 Skill 根目录：直接包含 skill.md / SKILL.md / Skill.md 的目录；
3. 对每个 Skill 根目录只保留：
   - 根目录下的 skill.md / SKILL.md / Skill.md；
   - 根目录下的 scripts 文件夹；
   - scripts 文件夹内部的 .py 文件；
4. 删除：
   - Skill 根目录下除 skill.md 和 scripts 之外的所有文件；
   - Skill 根目录下除 scripts 之外的所有文件夹；
   - scripts 文件夹内部除 .py 之外的所有文件；
   - scripts 文件夹内部清理后产生的空子文件夹；
5. 不删除 /root/BATaint/datasets/repo 根目录本身；
6. 生成 JSON 清理日志。

最终目标结构：
/root/BATaint/datasets/repo/<skill_name>/
├── skill.md
└── scripts/
    ├── xxx.py
    └── yyy.py

注意：
- 默认 DRY_RUN=True，只预览不删除；
- 确认日志无误后，把 DRY_RUN 改为 False 再执行真正删除。
"""

import os
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List


# ===================== 配置区 =====================

ROOT_DIR = r"/root/BATaint/datasets/repo"

# True：只预览，不真正删除
# False：真正删除
DRY_RUN = False

# 是否在 scripts 目录内部递归保留 .py 文件
# True ：scripts/a/b.py 会被保留，scripts/a/empty_dir 清空后删除
# False：只保留 scripts 目录第一层的 .py 文件，删除 scripts 下所有子目录
KEEP_PY_RECURSIVELY_IN_SCRIPTS = True

# 是否删除 scripts 内部清理后产生的空子目录
REMOVE_EMPTY_DIRS_IN_SCRIPTS = True

# 日志输出文件
LOG_JSON = "clean_repo_keep_skill_md_and_scripts_py_log.json"

# 不进入这些目录，避免误处理外部依赖或缓存
SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".idea",
    ".vscode",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".eggs",
    "site-packages",
    ".cache",
}


# ===================== 日志配置 =====================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(levelname)s\t%(message)s"
)


# ===================== 基础函数 =====================

def normalize_path(path: str) -> str:
    return os.path.abspath(os.path.normpath(path))


def norm_key(path: str) -> str:
    return os.path.normcase(normalize_path(path))


def is_python_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".py"


def find_skill_roots(root_dir: str) -> List[Path]:
    """
    在 ROOT_DIR 下递归查找所有 Skill 根目录。
    Skill 根目录定义：目录下直接包含 skill.md / SKILL.md / Skill.md。
    """
    root = Path(normalize_path(root_dir))

    if not root.exists():
        raise FileNotFoundError(f"目录不存在: {root}")

    if not root.is_dir():
        raise NotADirectoryError(f"不是目录: {root}")

    skill_roots: List[Path] = []

    for current_root, dirs, files in os.walk(root):
        dirs[:] = [
            d for d in dirs
            if d not in SKIP_DIRS and not os.path.islink(os.path.join(current_root, d))
        ]

        lower_files = {f.lower() for f in files}
        if "skill.md" in lower_files:
            skill_roots.append(Path(normalize_path(current_root)))

    # 先处理更深层目录，再处理浅层目录，降低嵌套 Skill 被父目录误删的风险
    skill_roots = sorted(set(skill_roots), key=lambda p: len(p.parts), reverse=True)
    return skill_roots


def delete_path(path: Path, dry_run: bool, reason: str) -> Dict:
    """
    删除文件、符号链接或目录。
    """
    path = Path(normalize_path(str(path)))

    record = {
        "path": str(path),
        "reason": reason,
        "deleted": False,
        "dry_run": dry_run,
        "error": "",
    }

    try:
        if dry_run:
            logging.info("[DRY-RUN] 将删除: %s | 原因: %s", path, reason)
            return record

        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        else:
            record["error"] = "path does not exist"
            return record

        record["deleted"] = True
        logging.info("已删除: %s | 原因: %s", path, reason)
        return record

    except Exception as e:
        record["error"] = f"{type(e).__name__}: {e}"
        logging.error("删除失败: %s | %s", path, record["error"])
        return record


def remove_empty_dirs(root: Path, dry_run: bool, keep_root: bool = True) -> List[Dict]:
    """
    从底层向上删除空目录。
    """
    root = Path(normalize_path(str(root)))
    records: List[Dict] = []

    if not root.exists() or not root.is_dir():
        return records

    for current_root, dirs, files in os.walk(root, topdown=False):
        current = Path(normalize_path(current_root))

        if keep_root and norm_key(str(current)) == norm_key(str(root)):
            continue

        record = {
            "path": str(current),
            "reason": "empty directory after cleaning",
            "deleted": False,
            "dry_run": dry_run,
            "error": "",
        }

        try:
            if current.exists() and current.is_dir() and not any(current.iterdir()):
                if dry_run:
                    logging.info("[DRY-RUN] 将删除空目录: %s", current)
                else:
                    current.rmdir()
                    record["deleted"] = True
                    logging.info("已删除空目录: %s", current)

                records.append(record)

        except Exception as e:
            record["error"] = f"{type(e).__name__}: {e}"
            logging.error("删除空目录失败: %s | %s", current, record["error"])
            records.append(record)

    return records


# ===================== scripts 目录清理 =====================

def clean_scripts_dir(scripts_dir: Path, dry_run: bool) -> Dict:
    """
    清理 scripts 文件夹。

    保留：
    - .py 文件

    删除：
    - 非 .py 文件
    - 如果 KEEP_PY_RECURSIVELY_IN_SCRIPTS=False，则删除 scripts 下所有子目录
    - 如果 KEEP_PY_RECURSIVELY_IN_SCRIPTS=True，则递归保留子目录中的 .py 文件，
      删除非 .py 文件，并删除空子目录。
    """
    scripts_dir = Path(normalize_path(str(scripts_dir)))

    result = {
        "scripts_dir": str(scripts_dir),
        "kept_python_files": [],
        "deleted_items": [],
        "error_items": [],
        "removed_empty_dirs": [],
    }

    if not scripts_dir.exists() or not scripts_dir.is_dir():
        return result

    if KEEP_PY_RECURSIVELY_IN_SCRIPTS:
        # 递归扫描 scripts 内部：只保留 .py 文件
        for current_root, dirs, files in os.walk(scripts_dir):
            dirs[:] = [
                d for d in dirs
                if d not in SKIP_DIRS and not os.path.islink(os.path.join(current_root, d))
            ]

            for filename in files:
                file_path = Path(current_root) / filename

                if is_python_file(file_path):
                    result["kept_python_files"].append(str(normalize_path(str(file_path))))
                    logging.info("保留 scripts 内 Python 文件: %s", file_path)
                else:
                    record = delete_path(
                        file_path,
                        dry_run=dry_run,
                        reason="non-python file inside scripts",
                    )
                    if record["error"]:
                        result["error_items"].append(record)
                    else:
                        result["deleted_items"].append(record)

        if REMOVE_EMPTY_DIRS_IN_SCRIPTS:
            result["removed_empty_dirs"] = remove_empty_dirs(
                scripts_dir,
                dry_run=dry_run,
                keep_root=True,
            )

    else:
        # 只保留 scripts 第一层 .py 文件，删除所有子目录
        for child in scripts_dir.iterdir():
            if child.is_file() and child.suffix.lower() == ".py":
                result["kept_python_files"].append(str(normalize_path(str(child))))
                logging.info("保留 scripts 第一层 Python 文件: %s", child)
            else:
                record = delete_path(
                    child,
                    dry_run=dry_run,
                    reason="only top-level .py files are kept inside scripts",
                )
                if record["error"]:
                    result["error_items"].append(record)
                else:
                    result["deleted_items"].append(record)

    return result


# ===================== Skill 根目录清理 =====================

def clean_one_skill_root(skill_root: Path, dry_run: bool) -> Dict:
    """
    对一个 Skill 根目录执行清理。

    保留：
    - 根目录下 skill.md / SKILL.md / Skill.md
    - 根目录下 scripts 文件夹
    - scripts 文件夹内部 .py 文件

    删除：
    - 根目录下除 skill.md 和 scripts 之外的其它文件/文件夹
    - scripts 内部非 .py 文件
    """
    skill_root = Path(normalize_path(str(skill_root)))

    result = {
        "skill_root": str(skill_root),
        "kept_skill_md": [],
        "kept_scripts_dirs": [],
        "deleted_root_items": [],
        "error_root_items": [],
        "scripts_results": [],
        "warning": [],
    }

    if not skill_root.exists() or not skill_root.is_dir():
        result["warning"].append("skill root does not exist or is not a directory")
        return result

    scripts_dirs: List[Path] = []

    for child in list(skill_root.iterdir()):
        # 保留根目录下的 skill.md
        if child.is_file() and child.name.lower() == "skill.md":
            result["kept_skill_md"].append(str(normalize_path(str(child))))
            logging.info("保留 skill.md: %s", child)
            continue

        # 保留根目录下的 scripts 文件夹，并稍后清理内部内容
        if child.is_dir() and child.name.lower() == "scripts":
            scripts_dirs.append(child)
            result["kept_scripts_dirs"].append(str(normalize_path(str(child))))
            logging.info("保留 scripts 文件夹: %s", child)
            continue

        # 其它所有根目录下的文件或文件夹删除
        record = delete_path(
            child,
            dry_run=dry_run,
            reason="not skill.md and not scripts directory under skill root",
        )

        if record["error"]:
            result["error_root_items"].append(record)
        else:
            result["deleted_root_items"].append(record)

    if not scripts_dirs:
        result["warning"].append("no scripts directory under this skill root")

    # 清理 scripts 内部，只保留 .py
    for scripts_dir in scripts_dirs:
        result["scripts_results"].append(
            clean_scripts_dir(scripts_dir, dry_run=dry_run)
        )

    return result


def clean_repo(root_dir: str, dry_run: bool) -> Dict:
    """
    清理整个 repo。
    """
    root_dir = normalize_path(root_dir)
    skill_roots = find_skill_roots(root_dir)

    result = {
        "root_dir": root_dir,
        "dry_run": dry_run,
        "rule": (
            "For each skill root, keep only root-level skill.md and scripts directory; "
            "inside scripts, keep only .py files."
        ),
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "skill_root_count": len(skill_roots),
        "skill_results": [],
        "summary": {},
    }

    logging.info("索引到 Skill 根目录数量: %d", len(skill_roots))

    for skill_root in skill_roots:
        logging.info("开始清理 Skill: %s", skill_root)
        result["skill_results"].append(
            clean_one_skill_root(skill_root, dry_run=dry_run)
        )

    result["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    kept_skill_md_count = 0
    kept_scripts_dir_count = 0
    kept_py_count = 0
    deleted_count = 0
    error_count = 0
    warning_count = 0
    removed_empty_dir_count = 0

    for skill_result in result["skill_results"]:
        kept_skill_md_count += len(skill_result["kept_skill_md"])
        kept_scripts_dir_count += len(skill_result["kept_scripts_dirs"])
        deleted_count += len(skill_result["deleted_root_items"])
        error_count += len(skill_result["error_root_items"])
        warning_count += len(skill_result["warning"])

        for scripts_result in skill_result["scripts_results"]:
            kept_py_count += len(scripts_result["kept_python_files"])
            deleted_count += len(scripts_result["deleted_items"])
            error_count += len(scripts_result["error_items"])
            removed_empty_dir_count += len(scripts_result["removed_empty_dirs"])

    result["summary"] = {
        "skill_root_count": len(skill_roots),
        "kept_skill_md_count": kept_skill_md_count,
        "kept_scripts_dir_count": kept_scripts_dir_count,
        "kept_python_files_inside_scripts_count": kept_py_count,
        "deleted_item_count": deleted_count,
        "error_item_count": error_count,
        "warning_count": warning_count,
        "removed_empty_dir_count": removed_empty_dir_count,
    }

    return result


def save_json(path: str, data: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    print("===== 清理 repo：只保留 skill.md + scripts/*.py =====")
    print(f"扫描目录: {ROOT_DIR}")
    print("保留规则:")
    print("  1. Skill 根目录下的 skill.md / SKILL.md / Skill.md")
    print("  2. Skill 根目录下的 scripts 文件夹")
    print("  3. scripts 文件夹内部的 .py 文件")
    print(f"DRY_RUN: {DRY_RUN}")
    print(f"KEEP_PY_RECURSIVELY_IN_SCRIPTS: {KEEP_PY_RECURSIVELY_IN_SCRIPTS}")

    if DRY_RUN:
        print("\n当前是预览模式，不会真正删除文件或文件夹。")
        print("确认日志无误后，请把 DRY_RUN 改为 False 再运行。\n")
    else:
        print("\n当前是真正删除模式，请确认路径正确！\n")

    result = clean_repo(ROOT_DIR, dry_run=DRY_RUN)
    save_json(LOG_JSON, result)

    print("\n===== 完成 =====")
    print(f"Skill 根目录数量: {result['summary']['skill_root_count']}")
    print(f"保留 skill.md 数量: {result['summary']['kept_skill_md_count']}")
    print(f"保留 scripts 文件夹数量: {result['summary']['kept_scripts_dir_count']}")
    print(f"保留 scripts 内 .py 文件数量: {result['summary']['kept_python_files_inside_scripts_count']}")
    print(f"删除项目数量: {result['summary']['deleted_item_count']}")
    print(f"删除失败数量: {result['summary']['error_item_count']}")
    print(f"警告数量: {result['summary']['warning_count']}")
    print(f"删除 scripts 内空目录数量: {result['summary']['removed_empty_dir_count']}")
    print(f"日志文件: {LOG_JSON}")


if __name__ == "__main__":
    main()
    """
        6
        删除无用.py
        保留
        repo
        ----skill.md
        ----scripts
        --------*.py
    """
