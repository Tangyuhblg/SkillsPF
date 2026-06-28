# -*- coding: utf-8 -*-
"""
normalize_repo_to_skill_py_md_keep_relative_paths.py

目标：
把 /root/BATaint/datasets/repo 规范化为如下结构：

repo/
  <skill_name>/
    skill.md
    <原相对路径>/*.py

与上一版的关键区别：
- 不再把子目录中的 .py 文件重命名为 scripts__main.py；
- 保留 Python 文件在 Skill 内部的原始相对路径；
- 例如：
    原始: repo/xxx/my-skill/scripts/main.py
    输出: repo/my-skill/scripts/main.py

说明：
如果要求“不重命名子目录中的 .py 文件”，就不能再强制所有 .py 文件都平铺到
repo/<skill_name>/ 下，否则不同子目录中同名文件会冲突，例如：
    scripts/main.py
    utils/main.py

因此本版本采用“保留相对路径”的方式：
    repo/<skill_name>/scripts/main.py
    repo/<skill_name>/utils/main.py

这样既满足“不重命名”，又保证不会覆盖文件。

具体行为：
1. 递归扫描原始 repo，查找所有直接包含 skill.md / SKILL.md / Skill.md 的目录；
2. 每个包含 skill.md 的目录视为一个 Skill 根目录；
3. 以 Skill 根目录的文件夹名作为 skill_name；
4. 为每个 Skill 创建标准目录：repo/<skill_name>/；
5. 复制该 Skill 的 skill.md 到 repo/<skill_name>/skill.md；
6. 复制该 Skill 根目录下所有 .py 文件到 repo/<skill_name>/ 下，并保留原相对路径；
7. 如果一个 Skill 内部嵌套另一个 Skill，扫描父 Skill 时会跳过子 Skill，避免重复复制；
8. 默认只保留含有 .py 文件的 Skill；
9. 构建成功后，不再备份原 repo，直接用清洗后的 repo 替换原 repo。

重要说明：
- 先生成临时目录 repo_clean_tmp；
- 只有临时目录生成成功后，才删除原 repo，并将 repo_clean_tmp 移动为新的 repo；
- 不再生成 repo_backup_时间戳 备份目录；
- 默认 DRY_RUN = True，只预览，不真正替换。
"""

import os
import re
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set


# ===================== 配置区 =====================

# 原始 repo 路径
REPO_ROOT = r"/root/BATaint/datasets/repo"

# 临时清洗目录
TMP_CLEAN_ROOT = r"/root/BATaint/datasets/repo_clean_tmp"

# 如果不想直接替换原 repo，可以设置 REPLACE_ORIGINAL_REPO = False，
# 此时会输出到 OUTPUT_CLEAN_ROOT，而不移动原 repo。
OUTPUT_CLEAN_ROOT = r"/root/BATaint/datasets/repo_clean"

# 是否用清洗后的目录替换原 REPO_ROOT
# True:
#   删除原 repo
#   repo_clean_tmp -> repo
# False:
#   repo_clean_tmp -> repo_clean
REPLACE_ORIGINAL_REPO = True

# True：只预览，不复制、不删除、不替换
# False：真正执行
DRY_RUN = False

# 是否保留没有 .py 文件的 Skill
# True：没有 .py 的 Skill 也会被复制，只包含 skill.md
# False：没有 .py 的 Skill 会跳过
KEEP_SKILLS_WITHOUT_PY = False

# 是否清理已存在的临时目录 / 输出目录
CLEAN_EXISTING_OUTPUT = True

# 日志文件
SUMMARY_JSON = r"/root/BATaint/datasets/normalize_repo_to_skill_py_md_summary.json"

# 扫描时跳过的目录，避免依赖、缓存、构建产物进入数据集
SKIP_DIRS = {
    ".git", "__pycache__", ".idea", ".vscode",
    "node_modules", "dist", "build", ".next",
    ".venv", "venv", "env",
    ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".tox", ".eggs", "site-packages", ".cache",
}

# Python 后缀
PY_EXT = ".py"


# ===================== 日志配置 =====================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(levelname)s\t%(message)s"
)


# ===================== 基础函数 =====================

def normalize_path(path: str) -> str:
    return os.path.abspath(os.path.normpath(path))


def norm_key(path: str) -> str:
    """
    用于路径去重。
    """
    return os.path.normcase(normalize_path(path))


def is_skill_md(filename: str) -> bool:
    """
    判断文件名是否为 skill.md，不区分大小写。
    """
    return filename.lower() == "skill.md"


def find_skill_md_file(skill_root: str) -> str:
    """
    在 skill_root 目录下查找直接存在的 skill.md / SKILL.md。
    找到则返回绝对路径，否则返回空字符串。
    """
    try:
        for filename in os.listdir(skill_root):
            full_path = os.path.join(skill_root, filename)
            if os.path.isfile(full_path) and is_skill_md(filename):
                return normalize_path(full_path)
    except Exception:
        return ""
    return ""


def sanitize_name(name: str, fallback: str = "unknown_skill") -> str:
    """
    将 skill_name 转成安全目录名。

    注意：
    这里仅用于 Skill 输出目录名，不用于 Python 文件名。
    Python 文件名和相对路径会尽量保持原样。
    """
    if not name:
        return fallback

    name = str(name).strip()
    name = re.sub(r'[\\/:*?"<>|\s]+', "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("._- ")

    return name or fallback


def safe_relative_path(rel_path: str) -> str:
    """
    清理相对路径，防止路径穿越，同时尽量保持原始目录结构和文件名不变。

    例如：
    scripts/main.py -> scripts/main.py
    ../a.py -> a.py
    ./utils/helper.py -> utils/helper.py

    注意：
    不会把 scripts/main.py 改名为 scripts__main.py。
    """
    rel_path = os.path.normpath(rel_path)

    safe_parts = []
    for part in rel_path.split(os.sep):
        if part in ("", ".", ".."):
            continue
        safe_parts.append(part)

    if not safe_parts:
        return ""

    return os.path.join(*safe_parts)


def is_descendant_path(child: str, parent: str) -> bool:
    """
    判断 child 是否是 parent 的子路径。
    """
    try:
        child_key = norm_key(child)
        parent_key = norm_key(parent)
        common = os.path.commonpath([child_key, parent_key])
        return common == parent_key and child_key != parent_key
    except Exception:
        return False


# ===================== 查找 Skill 和 Python 文件 =====================

def find_all_skill_roots(repo_root: str) -> List[str]:
    """
    递归查找 repo_root 下所有包含 skill.md / SKILL.md 的目录。
    每个这样的目录被视为一个 Skill 根目录。
    """
    repo_root = normalize_path(repo_root)

    if not os.path.exists(repo_root):
        raise FileNotFoundError(f"repo 路径不存在: {repo_root}")

    if not os.path.isdir(repo_root):
        raise NotADirectoryError(f"repo 不是目录: {repo_root}")

    skill_roots = []

    for current_root, dirs, files in os.walk(repo_root):
        dirs[:] = [
            d for d in dirs
            if d not in SKIP_DIRS and not os.path.islink(os.path.join(current_root, d))
        ]

        lower_files = {f.lower() for f in files}
        if "skill.md" in lower_files:
            skill_roots.append(normalize_path(current_root))

    return sorted(set(skill_roots))


def find_python_files_for_skill(skill_root: str, all_skill_roots: List[str]) -> List[str]:
    """
    查找一个 Skill 根目录下的所有 .py 文件。

    注意：
    如果当前 Skill 根目录内部嵌套另一个 Skill 根目录，
    则扫描当前 Skill 时跳过嵌套 Skill，避免重复复制。
    """
    skill_root = normalize_path(skill_root)

    nested_skill_roots = {
        norm_key(p)
        for p in all_skill_roots
        if is_descendant_path(p, skill_root)
    }

    py_files = []

    for current_root, dirs, files in os.walk(skill_root):
        dirs[:] = [
            d for d in dirs
            if d not in SKIP_DIRS and not os.path.islink(os.path.join(current_root, d))
        ]

        # 跳过嵌套 Skill 根目录
        filtered_dirs = []
        for d in dirs:
            child_dir = normalize_path(os.path.join(current_root, d))
            if norm_key(child_dir) in nested_skill_roots:
                continue
            filtered_dirs.append(d)
        dirs[:] = filtered_dirs

        for filename in files:
            full_path = normalize_path(os.path.join(current_root, filename))
            if Path(full_path).suffix.lower() == PY_EXT:
                py_files.append(full_path)

    return sorted(set(py_files))


# ===================== 构建标准 repo =====================

def prepare_empty_dir(path: str, dry_run: bool):
    """
    准备空目录。
    """
    path = normalize_path(path)

    if os.path.exists(path):
        if CLEAN_EXISTING_OUTPUT:
            if dry_run:
                logging.info("[DRY-RUN] 将删除已存在目录: %s", path)
            else:
                shutil.rmtree(path)
        else:
            raise FileExistsError(f"目录已存在: {path}")

    if dry_run:
        logging.info("[DRY-RUN] 将创建目录: %s", path)
    else:
        os.makedirs(path, exist_ok=True)


def copy_file(src: str, dst: str, dry_run: bool):
    """
    复制文件。
    """
    src = normalize_path(src)
    dst = normalize_path(dst)

    if dry_run:
        logging.info("[DRY-RUN] 复制文件: %s -> %s", src, dst)
        return

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def copy_python_file_keep_relative_path(
    py_file: str,
    skill_root: str,
    target_skill_dir: str,
    dry_run: bool
) -> dict:
    """
    复制 Python 文件，并保留其在 Skill 根目录内的相对路径。

    原始：
      skill_root/scripts/main.py

    输出：
      target_skill_dir/scripts/main.py

    不进行重命名。
    """
    py_file = normalize_path(py_file)
    skill_root = normalize_path(skill_root)
    target_skill_dir = normalize_path(target_skill_dir)

    rel_path = os.path.relpath(py_file, skill_root)
    rel_path = safe_relative_path(rel_path)

    if not rel_path:
        raise ValueError(f"非法 Python 相对路径: {py_file}")

    target_py_path = normalize_path(os.path.join(target_skill_dir, rel_path))

    # 安全校验：确保目标路径仍在 target_skill_dir 内部
    target_dir_key = norm_key(target_skill_dir)
    target_file_key = norm_key(target_py_path)
    common = os.path.commonpath([target_dir_key, target_file_key])
    if common != target_dir_key:
        raise ValueError(f"目标路径越界: {target_py_path}")

    copy_file(py_file, target_py_path, dry_run=dry_run)

    return {
        "src": py_file,
        "dst": target_py_path,
        "original_relative_path": os.path.normpath(rel_path).replace(os.sep, "/"),
        "kept_relative_path": os.path.normpath(rel_path).replace(os.sep, "/"),
        "renamed": False,
    }


def build_clean_repo(repo_root: str, clean_root: str, dry_run: bool) -> dict:
    """
    从原始 repo_root 构建标准化 clean_root。

    输出结构：
    clean_root/
      <skill_name>/
        skill.md
        原相对路径/*.py
    """
    repo_root = normalize_path(repo_root)
    clean_root = normalize_path(clean_root)

    skill_roots = find_all_skill_roots(repo_root)

    summary = {
        "repo_root": repo_root,
        "clean_root": clean_root,
        "total_skill_roots_found": len(skill_roots),
        "skills_copied": [],
        "skills_skipped_no_py": [],
        "skills_failed": [],
        "duplicate_skill_names": {},
        "total_py_files_copied": 0,
        "python_file_copy_policy": "keep original relative path under each skill root; do not rename .py files",
    }

    prepare_empty_dir(clean_root, dry_run=dry_run)

    used_skill_dir_names: Dict[str, int] = {}

    for idx, skill_root in enumerate(skill_roots, 1):
        try:
            raw_skill_name = os.path.basename(skill_root)
            base_skill_name = sanitize_name(raw_skill_name, fallback=f"skill_{idx:06d}")

            used_skill_dir_names[base_skill_name] = used_skill_dir_names.get(base_skill_name, 0) + 1
            if used_skill_dir_names[base_skill_name] == 1:
                skill_dir_name = base_skill_name
            else:
                # 注意：这里仅在 skill_name 目录重名时，为避免覆盖，给 Skill 目录加编号；
                # Python 文件本身不重命名。
                skill_dir_name = f"{base_skill_name}_{used_skill_dir_names[base_skill_name]}"
                summary["duplicate_skill_names"].setdefault(base_skill_name, []).append(skill_dir_name)

            skill_md_src = find_skill_md_file(skill_root)
            if not skill_md_src:
                raise FileNotFoundError(f"未找到 skill.md: {skill_root}")

            py_files = find_python_files_for_skill(skill_root, all_skill_roots=skill_roots)

            if not py_files and not KEEP_SKILLS_WITHOUT_PY:
                summary["skills_skipped_no_py"].append({
                    "skill_name": raw_skill_name,
                    "skill_root": skill_root,
                    "reason": "no .py files"
                })
                continue

            target_skill_dir = normalize_path(os.path.join(clean_root, skill_dir_name))

            if dry_run:
                logging.info("[DRY-RUN] 将创建 Skill 目录: %s", target_skill_dir)
            else:
                os.makedirs(target_skill_dir, exist_ok=True)

            # 统一复制为 skill.md，小写
            skill_md_dst = os.path.join(target_skill_dir, "skill.md")
            copy_file(skill_md_src, skill_md_dst, dry_run=dry_run)

            copied_py_files = []

            for py_file in py_files:
                copied_info = copy_python_file_keep_relative_path(
                    py_file=py_file,
                    skill_root=skill_root,
                    target_skill_dir=target_skill_dir,
                    dry_run=dry_run
                )
                copied_py_files.append(copied_info)

            summary["skills_copied"].append({
                "skill_name": raw_skill_name,
                "output_skill_name": skill_dir_name,
                "skill_root": skill_root,
                "target_dir": target_skill_dir,
                "skill_md_src": skill_md_src,
                "skill_md_dst": normalize_path(skill_md_dst),
                "py_file_count": len(copied_py_files),
                "py_files": copied_py_files,
            })

            summary["total_py_files_copied"] += len(copied_py_files)

        except Exception as e:
            logging.error("处理 Skill 失败: %s | %s", skill_root, e)
            summary["skills_failed"].append({
                "skill_root": skill_root,
                "error": str(e)
            })

    return summary


def replace_original_repo_without_backup(repo_root: str, clean_root: str, dry_run: bool) -> dict:
    """
    用 clean_root 替换 repo_root，不生成备份目录。

    执行逻辑：
    1. 确认 clean_root 已经生成成功；
    2. 直接删除原 repo_root；
    3. 将 clean_root 移动为新的 repo_root。

    注意：
    - 该函数不会创建 repo_backup_时间戳；
    - 删除原 repo 后无法通过本脚本恢复，请确认 clean_root 生成结果正确后再执行。
    """
    repo_root = normalize_path(repo_root)
    clean_root = normalize_path(clean_root)

    result = {
        "repo_root": repo_root,
        "clean_root": clean_root,
        "backup_enabled": False,
        "deleted_original_repo": False,
        "moved_clean_repo": False,
        "dry_run": dry_run,
        "error": "",
    }

    if dry_run:
        logging.info("[DRY-RUN] 不再备份原 repo")
        logging.info("[DRY-RUN] 将删除原 repo: %s", repo_root)
        logging.info("[DRY-RUN] 将替换 repo: %s -> %s", clean_root, repo_root)
        return result

    try:
        if not os.path.exists(clean_root):
            raise FileNotFoundError(f"清洗目录不存在，不能替换: {clean_root}")

        if not os.path.isdir(clean_root):
            raise NotADirectoryError(f"清洗路径不是目录: {clean_root}")

        if os.path.exists(repo_root):
            shutil.rmtree(repo_root)
            result["deleted_original_repo"] = True
            logging.info("已删除原 repo: %s", repo_root)

        shutil.move(clean_root, repo_root)
        result["moved_clean_repo"] = True
        logging.info("已将清洗目录移动为新 repo: %s -> %s", clean_root, repo_root)

        return result

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        logging.error("替换 repo 失败: %s", result["error"])
        raise


def move_clean_to_output(clean_root: str, output_root: str, dry_run: bool):
    """
    不替换原 repo 时，将临时清洗目录移动到输出目录。
    """
    clean_root = normalize_path(clean_root)
    output_root = normalize_path(output_root)

    if dry_run:
        logging.info("[DRY-RUN] 将输出清洗目录: %s -> %s", clean_root, output_root)
        return

    if os.path.exists(output_root):
        if CLEAN_EXISTING_OUTPUT:
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(f"输出目录已存在: {output_root}")

    shutil.move(clean_root, output_root)


def save_json(path: str, data: dict):
    """
    保存 JSON。
    """
    path = normalize_path(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    start_time = datetime.now()

    print("===== Normalize repo without backup: repo/<skill_name>/skill.md + repo/<skill_name>/<relative_path>.py =====")
    print(f"原始 repo: {REPO_ROOT}")
    print(f"临时清洗目录: {TMP_CLEAN_ROOT}")
    print(f"是否替换原 repo: {REPLACE_ORIGINAL_REPO}")
    print(f"DRY_RUN: {DRY_RUN}")
    print(f"只保留含 .py 的 Skill: {not KEEP_SKILLS_WITHOUT_PY}")
    print("Python 文件复制策略: 保留原相对路径，不重命名 .py 文件")

    if DRY_RUN:
        print("\n当前是 DRY_RUN=True，代码只预览，不会真的复制、删除或替换。")
        print("确认日志无误后，把 DRY_RUN 改为 False 再运行。\n")
    else:
        print("\n当前是真正执行模式：不会备份，会直接删除原 repo 并替换，请确认路径正确！\n")

    summary = build_clean_repo(
        repo_root=REPO_ROOT,
        clean_root=TMP_CLEAN_ROOT,
        dry_run=DRY_RUN
    )

    replace_result = {}

    if REPLACE_ORIGINAL_REPO:
        replace_result = replace_original_repo_without_backup(
            repo_root=REPO_ROOT,
            clean_root=TMP_CLEAN_ROOT,
            dry_run=DRY_RUN
        )
        final_repo_path = normalize_path(REPO_ROOT)
    else:
        move_clean_to_output(
            clean_root=TMP_CLEAN_ROOT,
            output_root=OUTPUT_CLEAN_ROOT,
            dry_run=DRY_RUN
        )
        final_repo_path = normalize_path(OUTPUT_CLEAN_ROOT)

    end_time = datetime.now()

    summary.update({
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "dry_run": DRY_RUN,
        "replace_original_repo": REPLACE_ORIGINAL_REPO,
        "backup_enabled": False,
        "replace_result": replace_result,
        "final_repo_path": final_repo_path,
        "expected_final_structure": "repo/<skill_name>/skill.md and repo/<skill_name>/<original_relative_path>.py",
        "config": {
            "KEEP_SKILLS_WITHOUT_PY": KEEP_SKILLS_WITHOUT_PY,
            "CLEAN_EXISTING_OUTPUT": CLEAN_EXISTING_OUTPUT,
            "SKIP_DIRS": sorted(SKIP_DIRS),
            "PY_EXT": PY_EXT,
        }
    })

    save_json(SUMMARY_JSON, summary)

    print("\n===== 完成 =====")
    print(f"发现 Skill 根目录数量: {summary['total_skill_roots_found']}")
    print(f"复制 Skill 数量: {len(summary['skills_copied'])}")
    print(f"跳过无 .py 的 Skill 数量: {len(summary['skills_skipped_no_py'])}")
    print(f"失败 Skill 数量: {len(summary['skills_failed'])}")
    print(f"复制 .py 文件总数: {summary['total_py_files_copied']}")
    print(f"最终 repo 路径: {final_repo_path}")
    print("原始 repo 备份路径: 未生成备份，已关闭备份功能")
    print(f"summary JSON: {SUMMARY_JSON}")

    if DRY_RUN:
        print("\n注意：当前只是预览，没有真正修改 repo。确认无误后设置 DRY_RUN = False。")


if __name__ == "__main__":
    main()
    """
        5
        清洗repo路径
        repo/<skill_name>/skill.md
        repo/<skill_name>/*.py
    """
