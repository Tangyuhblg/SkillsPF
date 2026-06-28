# -*- coding: utf-8 -*-

import os
import json
import logging

# ========= 路径配置 =========
JSON_PATH = r"/root/BATaint/crawler/data/malicious_skills_with_scripts.json" # /root/Skill/crawler/data/all_skills_data_with_scripts.json
TARGET_NEW_REPO_ROOT = r"/root/BATaint/datasets/repo" # /data/workspace/repo


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s\t%(levelname)s\t%(message)s"
    )


def load_json(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_dir_path(path_str: str):
    """
    将路径规范化；如果传入的是文件路径，则返回其父目录。
    """
    if not path_str:
        return None
    path_str = os.path.normpath(path_str)
    if os.path.splitext(path_str)[1]:
        return os.path.dirname(path_str)
    return path_str


def safe_skill_field_name(skill_name: str):
    if not skill_name:
        return "unknown_skill"

    chars = []
    for ch in skill_name:
        if ch.isalnum():
            chars.append(ch.lower())
        else:
            chars.append("_")

    name = "".join(chars)
    while "__" in name:
        name = name.replace("__", "_")
    return name.strip("_") or "unknown_skill"


def get_old_repo_parent(record: dict):
    """
    JSON 中 repo_path 形如:
        .../workspace/workspace/repo/<id>/<repo_name>
    我们需要它的父目录:
        .../workspace/workspace/repo/<id>
    这样才能映射到:
        .../workspace/workspace/new_repo/<id>
    """
    repo_path = normalize_dir_path(record.get("repo_path", ""))
    if not repo_path:
        return None
    return os.path.dirname(repo_path)


def build_skill_field_to_old_root(record: dict):
    """
    根据 all_skill_paths 构造:
        {
            "idea_refine_script_files": "旧的 skill 根目录",
            "agent_browser_2_script_files": "旧的 skill 根目录",
            ...
        }

    兼容两种 all_skill_paths 结构：
    1) 旧结构: [".../skills/idea-refine", ".../skills/agent-browser"]
    2) 新结构:
       [
         {"skill_name": "idea-refine", "skill_code_path": ".../skills/idea-refine"},
         {"skill_name": "idea-refine", "skill_code_path": ".../skills/idea-refine/scripts/idea-refine.sh"}
       ]

    注意：
    - 如果 skill_code_path 指向的是脚本文件，则自动取其父目录作为 skill 根目录
    - 对同一 skill 的多个脚本文件，只保留一个唯一 skill 根目录
    """
    result = {}
    grouped_roots = {}

    def add_unique_root(base_name: str, root_path: str):
        grouped_roots.setdefault(base_name, [])
        if root_path not in grouped_roots[base_name]:
            grouped_roots[base_name].append(root_path)

    all_skill_paths = record.get("all_skill_paths", [])

    for item in all_skill_paths:
        skill_name = ""
        raw_path = ""

        # 新结构：dict
        if isinstance(item, dict):
            skill_name = str(item.get("skill_name", "")).strip()
            raw_path = str(item.get("skill_code_path", "")).strip()

        # 旧结构：str
        elif isinstance(item, str):
            raw_path = item.strip()

        else:
            continue

        if not raw_path:
            continue

        path_str = os.path.normpath(raw_path)

        # 如果是文件路径，则回退到父目录；如果是目录，则直接使用
        if os.path.splitext(path_str)[1]:
            # 这里的 item 可能已经是具体脚本文件路径
            # 对于新结构，build_keep_files() 会直接优先使用文件绝对路径，
            # 这里只作为旧兼容分支使用
            old_root = os.path.dirname(path_str)
        else:
            old_root = path_str

        old_root = os.path.normpath(old_root)

        if not skill_name:
            skill_name = os.path.basename(old_root)

        base_name = safe_skill_field_name(skill_name)
        add_unique_root(base_name, old_root)

    # 只为 record 中真实存在的 xxx_script_files 字段建立映射
    script_field_names = [
        k for k, v in record.items()
        if k.endswith("_script_files") and isinstance(v, list)
    ]

    for field_name in script_field_names:
        prefix = field_name[:-len("_script_files")]

        # 兼容:
        #   idea_refine_script_files
        #   agent_browser_2_script_files
        parts = prefix.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            base_name = parts[0]
            index = int(parts[1]) - 1   # _2 -> 第2个 -> 索引1
        else:
            base_name = prefix
            index = 0

        roots = grouped_roots.get(base_name, [])
        if index < len(roots):
            result[field_name] = roots[index]

    return result


def build_keep_files(record: dict, dst_repo_root: str):
    """
    从 JSON 记录中构建应该保留的目标文件集合（绝对路径）。

    优先兼容当前新结构：
    all_skill_paths = [
        {"skill_name": "...", "skill_code_path": "旧 repo 下的脚本绝对路径"},
        ...
    ]

    其次兼容旧结构：
    1) xxx_script_files + all_skill_paths(目录)
    2) script_files + scan_root/skill_path/repo_path
    """
    keep_files = set()

    old_repo_parent = get_old_repo_parent(record)
    if not old_repo_parent:
        return keep_files

    # --------------------------------------------------
    # 1) 最新结构：all_skill_paths 里直接就是脚本文件绝对路径
    # --------------------------------------------------
    all_skill_paths = record.get("all_skill_paths", [])
    if isinstance(all_skill_paths, list):
        for item in all_skill_paths:
            if not isinstance(item, dict):
                continue

            old_abs_file = str(item.get("skill_code_path", "")).strip()
            if not old_abs_file:
                continue

            old_abs_file = os.path.normpath(old_abs_file)

            # 只处理文件路径
            if not os.path.splitext(old_abs_file)[1]:
                continue

            try:
                rel_file = os.path.relpath(old_abs_file, old_repo_parent)
            except Exception:
                continue

            new_abs_file = os.path.normpath(os.path.join(dst_repo_root, rel_file))
            keep_files.add(new_abs_file)

    if keep_files:
        return keep_files

    # --------------------------------------------------
    # 2) 中间结构：xxx_script_files + all_skill_paths(目录)
    # --------------------------------------------------
    field_to_old_root = build_skill_field_to_old_root(record)

    if field_to_old_root:
        for field_name, old_skill_root in field_to_old_root.items():
            rel_files = record.get(field_name, [])
            if not isinstance(rel_files, list):
                continue

            try:
                rel_root = os.path.relpath(old_skill_root, old_repo_parent)
            except Exception:
                continue

            new_skill_root = os.path.normpath(os.path.join(dst_repo_root, rel_root))

            for rel_file in rel_files:
                keep_files.add(os.path.normpath(os.path.join(new_skill_root, rel_file)))

    if keep_files:
        return keep_files

    # --------------------------------------------------
    # 3) 旧结构：只有 script_files
    # --------------------------------------------------
    script_files = record.get("script_files", [])
    if isinstance(script_files, list) and script_files:
        base_old = None
        for key in ["scan_root", "skill_path", "repo_path"]:
            p = record.get(key, "")
            if p:
                base_old = normalize_dir_path(p)
                break

        if base_old:
            try:
                rel_root = os.path.relpath(base_old, old_repo_parent)
                base_new = os.path.normpath(os.path.join(dst_repo_root, rel_root))

                for rel_file in script_files:
                    keep_files.add(os.path.normpath(os.path.join(base_new, rel_file)))
            except Exception:
                pass

    return keep_files


# def keep_only_scripts_for_one_skill(record: dict):
#     """
#     对单个 skill_id：
#     - 在 new_repo/<id> 下只保留脚本文件
#     - 删除其他所有文件
#     - 删除空目录
#     """
#     skill_id = str(record.get("id", "")).strip()
#     if not skill_id:
#         return {
#             "id": "",
#             "status": "skipped",
#             "reason": "empty id"
#         }
#
#     dst_repo_root = os.path.abspath(
#         os.path.normpath(os.path.join(TARGET_NEW_REPO_ROOT, skill_id))
#     )
#
#     if not os.path.exists(dst_repo_root):
#         return {
#             "id": skill_id,
#             "status": "skipped",
#             "reason": "new_repo target folder not found",
#             "dst_repo_root": dst_repo_root
#         }
#
#     # 直接根据 JSON 记录构建最终应保留的文件
#     keep_files = build_keep_files(record, dst_repo_root)
#
#     if not keep_files:
#         return {
#             "id": skill_id,
#             "status": "skipped",
#             "reason": "no script files resolved from record"
#         }
#
#     existing_keep_files = [p for p in keep_files if os.path.isfile(p)]
#     missing_keep_files = [p for p in keep_files if not os.path.isfile(p)]
#
#     if not existing_keep_files:
#         return {
#             "id": skill_id,
#             "status": "skipped",
#             "reason": "no keep_files found in new_repo, skip to avoid destructive deletion",
#             "keep_files": sorted(list(keep_files))
#         }
#
#     removed_files = 0
#
#     for current_root, dirs, files in os.walk(dst_repo_root, topdown=False):
#         for file_name in files:
#             current_file = os.path.normpath(os.path.join(current_root, file_name))
#             if current_file not in keep_files:
#                 try:
#                     os.remove(current_file)
#                     removed_files += 1
#                 except Exception as e:
#                     logging.error("删除文件失败 | ID=%s | %s | %s", skill_id, current_file, e)
#
#     removed_empty_dirs = remove_empty_dirs(dst_repo_root)
#
#     return {
#         "id": skill_id,
#         "status": "processed",
#         "kept_count": len(existing_keep_files),
#         "missing_keep_count": len(missing_keep_files),
#         "removed_files_count": removed_files,
#         "removed_empty_dirs_count": removed_empty_dirs,
#         "kept_files": sorted(existing_keep_files),
#         "missing_keep_files": sorted(missing_keep_files)
#     }

# def choose_base_root(record: dict):
#     """
#     script_files 是相对于 scan_root/skill_path 的相对路径。
#     现在 JSON 里的路径已经是 new_repo 下的路径，所以直接取：
#     1) scan_root
#     2) skill_path
#     3) repo_path
#     """
#     for key in ["scan_root", "skill_path", "repo_path"]:
#         p = record.get(key, "")
#         if p:
#             return normalize_dir_path(p), key
#     return None, "unresolved"

# def choose_base_root(record: dict, dst_repo_root: str):
#     for key in ["scan_root", "skill_path", "repo_path"]:
#         p = record.get(key, "")
#         if p:
#             return normalize_dir_path(p), key
#
#     # fallback: 如果 JSON 没有路径字段，就直接用 new_repo/<id>
#     if os.path.exists(dst_repo_root):
#         return dst_repo_root, "dst_repo_root_fallback"
#
#     return None, "unresolved"


def remove_empty_dirs(root_dir: str):
    """
    自底向上删除空目录
    """
    removed_count = 0
    for current_root, dirs, files in os.walk(root_dir, topdown=False):
        if current_root == root_dir:
            continue
        if not os.listdir(current_root):
            os.rmdir(current_root)
            removed_count += 1
    return removed_count


def keep_only_scripts_for_one_skill(record: dict):
    """
    对单个 skill_id：
    - 在 new_repo/<id> 下只保留脚本文件
    - 删除其他所有文件
    - 删除空目录
    """
    skill_id = str(record.get("id", "")).strip()
    if not skill_id:
        return {
            "id": "",
            "status": "skipped",
            "reason": "empty id"
        }

    dst_repo_root = os.path.abspath(
        os.path.normpath(os.path.join(TARGET_NEW_REPO_ROOT, skill_id))
    )

    if not os.path.exists(dst_repo_root):
        return {
            "id": skill_id,
            "status": "skipped",
            "reason": "new_repo target folder not found",
            "dst_repo_root": dst_repo_root
        }

    # 直接根据 JSON 记录构建最终应保留的文件
    keep_files = build_keep_files(record, dst_repo_root)

    if not keep_files:
        return {
            "id": skill_id,
            "status": "skipped",
            "reason": "no script files resolved from record"
        }

    existing_keep_files = [p for p in keep_files if os.path.isfile(p)]
    missing_keep_files = [p for p in keep_files if not os.path.isfile(p)]

    if not existing_keep_files:
        return {
            "id": skill_id,
            "status": "skipped",
            "reason": "no keep_files found in new_repo, skip to avoid destructive deletion",
            "keep_files": sorted(list(keep_files))
        }

    removed_files = 0

    for current_root, dirs, files in os.walk(dst_repo_root, topdown=False):
        for file_name in files:
            current_file = os.path.normpath(os.path.join(current_root, file_name))
            if current_file not in keep_files:
                try:
                    os.remove(current_file)
                    removed_files += 1
                except Exception as e:
                    logging.error("删除文件失败 | ID=%s | %s | %s", skill_id, current_file, e)

    removed_empty_dirs = remove_empty_dirs(dst_repo_root)

    return {
        "id": skill_id,
        "status": "processed",
        "kept_count": len(existing_keep_files),
        "missing_keep_count": len(missing_keep_files),
        "removed_files_count": removed_files,
        "removed_empty_dirs_count": removed_empty_dirs,
        "kept_files": sorted(existing_keep_files),
        "missing_keep_files": sorted(missing_keep_files)
    }


# ========= 代码语言分类配置 =========
LANGUAGE_EXTENSIONS = {
    "python": {".py"},
    # "javascript": {".js", ".mjs", ".cjs"},
    # "shell": {".sh", ".bash", ".fish"}
}


def get_code_language(file_path: str):
    """
    根据文件后缀判断代码语言类型。
    返回:
        python / javascript / shell / None
    """
    suffix = os.path.splitext(file_path)[1].lower()

    for language, extensions in LANGUAGE_EXTENSIONS.items():
        if suffix in extensions:
            return language

    return None


def count_code_files_by_language(root_dir: str):
    """
    递归统计 root_dir 下 Python、JavaScript、Shell 代码文件数量。

    返回:
    {
        "root_dir": "...",
        "python_count": 10,
        "javascript_count": 5,
        "shell_count": 2,
        "total_code_files": 17,
        "details": [...]
    }
    """
    result = {
        "root_dir": os.path.abspath(os.path.normpath(root_dir)),
        "python_count": 0,
        "javascript_count": 0,
        "shell_count": 0,
        "total_code_files": 0,
        "details": []
    }

    if not os.path.exists(root_dir):
        logging.error("统计目录不存在: %s", root_dir)
        return result

    for current_root, dirs, files in os.walk(root_dir):
        # 可选：跳过一些无关目录
        dirs[:] = [
            d for d in dirs
            if d not in {
                ".git", "__pycache__", "node_modules",
                ".venv", "venv", "env",
                "dist", "build", ".idea", ".vscode"
            }
        ]

        for file_name in files:
            file_path = os.path.normpath(os.path.join(current_root, file_name))
            language = get_code_language(file_path)

            if language is None:
                continue

            result[f"{language}_count"] += 1
            result["total_code_files"] += 1

            result["details"].append({
                "language": language,
                "file_path": os.path.abspath(file_path)
            })

    return result


def count_code_files_by_skill_folder(root_dir: str):
    """
    按 new_repo/<skill_id> 一级文件夹统计代码文件数量。
    """
    summary = {
        "root_dir": os.path.abspath(os.path.normpath(root_dir)),
        "total_skill_folders": 0,
        "python_count": 0,
        "javascript_count": 0,
        "shell_count": 0,
        "total_code_files": 0,
        "skill_folder_details": []
    }

    if not os.path.exists(root_dir):
        logging.error("统计目录不存在: %s", root_dir)
        return summary

    for name in sorted(os.listdir(root_dir)):
        skill_folder = os.path.join(root_dir, name)

        if not os.path.isdir(skill_folder):
            continue

        summary["total_skill_folders"] += 1

        one_result = count_code_files_by_language(skill_folder)

        summary["python_count"] += one_result["python_count"]
        summary["javascript_count"] += one_result["javascript_count"]
        summary["shell_count"] += one_result["shell_count"]
        summary["total_code_files"] += one_result["total_code_files"]

        summary["skill_folder_details"].append({
            "skill_id": name,
            "skill_folder": os.path.abspath(os.path.normpath(skill_folder)),
            "python_count": one_result["python_count"],
            "javascript_count": one_result["javascript_count"],
            "shell_count": one_result["shell_count"],
            "total_code_files": one_result["total_code_files"]
        })

    return summary


def save_json(path: str, data):
    """
    保存 JSON 结果。
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    setup_logging()

    data = load_json(JSON_PATH)

    processed = []
    skipped = []

    total = len(data)
    logging.info("准备处理 %d 个 skill文件夹数据", total)

    for idx, record in enumerate(data, 1):
        result = keep_only_scripts_for_one_skill(record)

        if result["status"] == "processed":
            processed.append(result)
            logging.info("[%d/%d] ID=%s 处理完成，保留 %d 个脚本文件，删除 %d 个非脚本文件",
                         idx, total, result["id"], result["kept_count"], result["removed_files_count"])
        else:
            skipped.append(result)
            logging.warning("[%d/%d] ID=%s 跳过：%s",
                            idx, total, result.get("id", ""), result.get("reason", ""))

    logging.info("完成！processed=%d, skipped=%d, total=%d",
                 len(processed), len(skipped), total)


    target_root = os.path.abspath(os.path.normpath(TARGET_NEW_REPO_ROOT))
    logging.info("开始统计代码文件数量，目标目录: %s", target_root)
    summary = count_code_files_by_skill_folder(target_root)

    logging.info(f"[*] Python 文件数量 (.py): {summary['python_count']}")
    logging.info(f"[*] JavaScript 文件数量 (.js/.mjs/.cjs): {summary['javascript_count']}")
    logging.info(f"[*] Shell 文件数量 (.sh/.bash/.fish): {summary['shell_count']}")
    logging.info(f"[*] 三类代码文件总数: {summary['total_code_files']}")



if __name__ == "__main__":
    main()
    '''
        3
        解压完先备份再删
        cp -r /data/workspace/repo /data/workspace/repo_backup
        只保留.py .sh .js文件其它文件全部删除
        感觉.sh不需要留，先看看占比
    '''