# -*- coding: utf-8 -*-
"""
extract_skill_scripts_python_code_to_csv.py

功能：
根据 skills_dataset.csv 中的 skill_name、classification、url，
到 /root/BATaint/datasets/repo/<skill_name>/scripts/ 下提取 Python 代码，
并将每个 Skill 的所有 scripts/*.py 文件合并保存到一个 CSV 文件中。

适配的数据目录结构：
/root/BATaint/datasets/repo/<skill_name>/
└── scripts/
    ├── xxx.py
    └── yyy.py

输出 CSV：
/root/BATaint/datasets/skills_ground_truth_files_dataset.csv

输出字段：
1. url
2. skill_name
3. file-label
4. python_code

核心规则：
- 一个 skill_name 只输出一行；
- 只读取 scripts 文件夹中的 .py 文件；
- 不读取 skill 根目录下其它位置的 .py 文件；
- 不读取 skill.md 内容；
- 如果 skills_dataset.csv 中同一个 skill_name 重复出现：
  - malicious > suspicious > safe / benign / normal；
  - 优先保留风险更高的 classification；
  - 例如同一个 skill_name 同时有 safe 和 suspicious，则保留 suspicious；
- 如果 repo 下没有对应 skill_name 文件夹，则跳过；
- 如果没有 scripts 文件夹，则跳过；
- 如果 scripts 文件夹中没有 .py 文件，则跳过；
- 生成 summary JSON 便于检查统计结果。
"""

import os
import csv
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple


# ===================== 配置区 =====================

SKILLS_DATASET_CSV = r"/root/BATaint/crawler/data/skills_dataset.csv"
REPO_ROOT = r"/root/BATaint/datasets/repo"
OUTPUT_DIR = r"/root/BATaint/datasets"

OUTPUT_CSV = os.path.join(OUTPUT_DIR, "skills_ground_truth_files_dataset.csv")
OUTPUT_SUMMARY_JSON = os.path.join(OUTPUT_DIR, "skills_ground_truth_files_dataset_summary.json")

OUTPUT_FIELDS = ["url", "skill_name", "file-label", "python_code"]

FIELD_SKILL_NAME = "skill_name"
FIELD_CLASSIFICATION = "classification"
FIELD_URL = "url"

# 是否递归扫描 scripts 子目录
# 如果你的结构严格是 scripts/*.py，建议保持 False
# 如果 scripts 下还有子目录，且也要读取其中 .py，可以改为 True
SCAN_SCRIPTS_RECURSIVE = False

# 多个 classification 冲突时的风险优先级
CLASSIFICATION_PRIORITY = {
    "safe": 0,
    "benign": 0,
    "normal": 0,
    "suspicious": 1,
    "malicious": 2,
}

# 多个 Python 文件合并到 python_code 字段时的分隔符
CODE_FILE_SEPARATOR_TEMPLATE = (
    "\n\n"
    "# ===== BEGIN PYTHON FILE: {relative_path} =====\n"
    "{code}\n"
    "# ===== END PYTHON FILE: {relative_path} =====\n"
)

# 如果 Python 代码特别大，可以设置最大字符数；None 表示不截断
MAX_CODE_CHARS_PER_SKILL = None


# ===================== 日志配置 =====================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(levelname)s\t%(message)s"
)


# ===================== 基础工具函数 =====================

def normalize_path(path: str) -> str:
    return os.path.abspath(os.path.normpath(path))


def normalize_skill_key(name: str) -> str:
    """
    用于匹配 CSV 中的 skill_name 和 repo 中的文件夹名。

    处理逻辑：
    - 转小写；
    - 去除首尾空白；
    - 将空格、下划线、特殊字符统一为短横线；
    - 连续分隔符合并。

    示例：
    - "jira-agile" -> "jira-agile"
    - "Jira Agile" -> "jira-agile"
    - "jira_agile" -> "jira-agile"
    """
    if name is None:
        return ""

    name = str(name).strip().lower()

    chars = []
    prev_sep = False

    for ch in name:
        if ch.isalnum():
            chars.append(ch)
            prev_sep = False
        else:
            if not prev_sep:
                chars.append("-")
                prev_sep = True

    return "".join(chars).strip("-")


def read_csv_rows(csv_path: str) -> List[dict]:
    """
    读取 skills_dataset.csv。
    """
    csv_path = normalize_path(csv_path)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")

    rows = []

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError(f"CSV 没有表头: {csv_path}")

        required_fields = {FIELD_SKILL_NAME, FIELD_CLASSIFICATION, FIELD_URL}
        missing = required_fields - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"CSV 缺少必要字段: {missing}; 当前字段: {reader.fieldnames}"
            )

        for row in reader:
            rows.append(dict(row))

    return rows


def choose_higher_risk_label(label_a: str, label_b: str) -> str:
    """
    如果同一个 skill_name 出现多个 classification，选择风险最高的标签。
    """
    label_a = (label_a or "").strip().lower()
    label_b = (label_b or "").strip().lower()

    score_a = CLASSIFICATION_PRIORITY.get(label_a, -1)
    score_b = CLASSIFICATION_PRIORITY.get(label_b, -1)

    if score_b > score_a:
        return label_b

    return label_a


def classification_to_file_label(classification: str) -> str:
    """
    根据 classification 生成 file-label。

    malicious / suspicious -> TRUE
    safe / benign / normal -> FALSE
    """
    classification = (classification or "").strip().lower()

    if classification in {"malicious", "suspicious"}:
        return "TRUE"

    if classification in {"safe", "benign", "normal"}:
        return "FALSE"

    raise ValueError(f"未知 classification: {classification}")


# ===================== CSV 标签去重 =====================

def build_skill_info_map(rows: List[dict]) -> Tuple[Dict[str, dict], Dict[str, list]]:
    """
    构造 skill_name -> 信息映射。

    去重规则：
    - 同一个 skill_name 只保留一条最终标签；
    - 如果同一个 skill_name 有多个 classification，则保留风险更高者；
    - malicious > suspicious > safe / benign / normal；
    - 例如 safe 和 suspicious 冲突时，最终保留 suspicious。
    """
    skill_info_map: Dict[str, dict] = {}
    conflict_map: Dict[str, list] = {}

    for row in rows:
        raw_skill_name = (row.get(FIELD_SKILL_NAME) or "").strip()
        classification = (row.get(FIELD_CLASSIFICATION) or "").strip().lower()
        url = (row.get(FIELD_URL) or "").strip()

        if not raw_skill_name:
            continue

        skill_key = normalize_skill_key(raw_skill_name)
        if not skill_key:
            continue

        if classification not in CLASSIFICATION_PRIORITY:
            logging.warning(
                "未知 classification: %s | skill_name=%s，跳过该行",
                classification,
                raw_skill_name
            )
            continue

        if skill_key not in skill_info_map:
            skill_info_map[skill_key] = {
                "skill_name": raw_skill_name,
                "classification": classification,
                "url": url,
                "rows": [row],
                "all_classifications": sorted({classification}),
                "all_urls": sorted({url}) if url else [],
                "removed_duplicate_classifications": [],
            }
            continue

        old_classification = skill_info_map[skill_key]["classification"]
        final_classification = choose_higher_risk_label(old_classification, classification)

        all_cls = set(skill_info_map[skill_key].get("all_classifications", []))
        all_cls.add(classification)
        skill_info_map[skill_key]["all_classifications"] = sorted(all_cls)

        all_urls = set(skill_info_map[skill_key].get("all_urls", []))
        if url:
            all_urls.add(url)
        skill_info_map[skill_key]["all_urls"] = sorted(all_urls)

        if len(all_cls) > 1:
            conflict_map[skill_key] = sorted(all_cls)

        # 新标签风险更高：覆盖旧记录
        if final_classification == classification and classification != old_classification:
            skill_info_map[skill_key]["removed_duplicate_classifications"].append(old_classification)
            skill_info_map[skill_key]["skill_name"] = raw_skill_name
            skill_info_map[skill_key]["classification"] = classification
            skill_info_map[skill_key]["url"] = url
            skill_info_map[skill_key]["rows"] = [row]
            continue

        # 旧标签风险更高：忽略新记录
        if final_classification == old_classification and classification != old_classification:
            skill_info_map[skill_key]["removed_duplicate_classifications"].append(classification)
            continue

        # 标签相同：记录来源行；如果原 url 为空，新 url 非空，则补充
        skill_info_map[skill_key]["rows"].append(row)
        if not skill_info_map[skill_key].get("url") and url:
            skill_info_map[skill_key]["url"] = url

    return skill_info_map, conflict_map


# ===================== repo 目录索引 =====================

def index_repo_skill_dirs(repo_root: str) -> Dict[str, List[str]]:
    """
    建立 skill_name -> skill_root 的索引。

    由于当前 repo 已经标准化为：
    /root/BATaint/datasets/repo/<skill_name>/scripts/*.py

    因此这里只索引 repo_root 的直接子目录。
    """
    repo_root = normalize_path(repo_root)

    if not os.path.exists(repo_root):
        raise FileNotFoundError(f"repo 目录不存在: {repo_root}")

    if not os.path.isdir(repo_root):
        raise NotADirectoryError(f"repo 路径不是目录: {repo_root}")

    index: Dict[str, List[str]] = {}

    for item in sorted(os.listdir(repo_root)):
        skill_root = normalize_path(os.path.join(repo_root, item))

        if not os.path.isdir(skill_root):
            continue

        skill_key = normalize_skill_key(item)
        if not skill_key:
            continue

        index.setdefault(skill_key, []).append(skill_root)

    return index


def compute_repo_csv_match_stats(
    skill_info_map: Dict[str, dict],
    skill_root_index: Dict[str, List[str]]
) -> dict:
    """
    以 /root/BATaint/datasets/repo 下的 Skill 文件夹为统计基准，
    计算这些 repo Skill 文件夹中有多少能在 skills_dataset.csv 中匹配到标签。

    注意：
    - repo_skill_folder_count 统计的是 repo 下的文件夹数量；
    - matched_repo_skill_folder_count 表示 repo 中能在 CSV 中找到对应 skill_name 的文件夹数量；
    - unmatched_repo_skill_folder_count 表示 repo 中不能在 CSV 中找到对应 skill_name 的文件夹数量；
    - 这里的“匹配/未匹配”不以 CSV 为基准，而是以 repo 文件夹为基准。
    """
    csv_skill_keys = set(skill_info_map.keys())

    repo_skill_folders = []
    matched_repo_skill_folders = []
    unmatched_repo_skill_folders = []

    for skill_key, roots in sorted(skill_root_index.items(), key=lambda x: x[0]):
        for root in sorted(roots):
            folder_name = os.path.basename(root)

            record = {
                "skill_key": skill_key,
                "skill_name_from_repo_folder": folder_name,
                "skill_root": root,
            }

            repo_skill_folders.append(record)

            if skill_key in csv_skill_keys:
                matched_info = skill_info_map[skill_key]
                record_with_csv = dict(record)
                record_with_csv.update({
                    "csv_skill_name": matched_info.get("skill_name", ""),
                    "classification": matched_info.get("classification", ""),
                    "file-label": classification_to_file_label(matched_info.get("classification", "")),
                    "url": matched_info.get("url", ""),
                    "all_classifications": matched_info.get("all_classifications", []),
                })
                matched_repo_skill_folders.append(record_with_csv)
            else:
                unmatched_repo_skill_folders.append(record)

    return {
        "repo_skill_folder_count": len(repo_skill_folders),
        "matched_repo_skill_folder_count": len(matched_repo_skill_folders),
        "unmatched_repo_skill_folder_count": len(unmatched_repo_skill_folders),
        "matched_repo_skill_folders": matched_repo_skill_folders,
        "unmatched_repo_skill_folders": unmatched_repo_skill_folders,
    }


# ===================== scripts Python 代码提取 =====================

def find_scripts_dir(skill_root: str) -> str:
    """
    查找 Skill 根目录下的 scripts 文件夹。

    默认支持大小写不敏感：
    - scripts
    - Scripts
    - SCRIPTS
    """
    skill_root = normalize_path(skill_root)

    if not os.path.isdir(skill_root):
        return ""

    for item in os.listdir(skill_root):
        path = os.path.join(skill_root, item)
        if os.path.isdir(path) and item.lower() == "scripts":
            return normalize_path(path)

    return ""


def find_python_files_in_scripts(skill_root: str) -> List[str]:
    """
    只查找 skill_root/scripts 下的 .py 文件。
    不扫描 Skill 根目录下其它位置的 .py 文件。
    """
    scripts_dir = find_scripts_dir(skill_root)

    if not scripts_dir:
        return []

    py_files = []

    if SCAN_SCRIPTS_RECURSIVE:
        for current_root, dirs, files in os.walk(scripts_dir):
            # 不跟随符号链接目录，避免误扫外部路径
            dirs[:] = [
                d for d in dirs
                if not os.path.islink(os.path.join(current_root, d))
            ]

            for filename in files:
                full_path = normalize_path(os.path.join(current_root, filename))
                if Path(full_path).suffix.lower() == ".py":
                    py_files.append(full_path)
    else:
        for filename in os.listdir(scripts_dir):
            full_path = normalize_path(os.path.join(scripts_dir, filename))
            if os.path.isfile(full_path) and Path(full_path).suffix.lower() == ".py":
                py_files.append(full_path)

    return sorted(set(py_files))


def read_source_code(file_path: str) -> str:
    """
    读取 Python 源码。
    """
    encodings = ["utf-8", "utf-8-sig", "gbk", "latin-1"]
    last_error = None

    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc, errors="replace") as f:
                return f.read()
        except Exception as e:
            last_error = e

    logging.error("读取源码失败: %s | %s", file_path, last_error)
    return ""


def merge_python_files_code(py_files: List[str], repo_root: str) -> str:
    """
    将同一个 Skill 的 scripts/*.py 文件代码合并为一个字符串。
    """
    merged_parts = []
    repo_root = normalize_path(repo_root)

    for py_file in py_files:
        rel_path = os.path.relpath(py_file, repo_root)
        rel_path = os.path.normpath(rel_path).replace(os.sep, "/")

        code = read_source_code(py_file)

        merged_parts.append(
            CODE_FILE_SEPARATOR_TEMPLATE.format(
                relative_path=rel_path,
                code=code
            )
        )

    merged_code = "".join(merged_parts)

    if MAX_CODE_CHARS_PER_SKILL is not None and len(merged_code) > MAX_CODE_CHARS_PER_SKILL:
        merged_code = merged_code[:MAX_CODE_CHARS_PER_SKILL]

    return merged_code


# ===================== 记录构造与保存 =====================

def build_output_records(
    skill_info_map: Dict[str, dict],
    skill_root_index: Dict[str, List[str]],
    repo_root: str
) -> Tuple[List[dict], dict]:
    """
    构造最终 CSV 记录。

    每个 skill_name 输出一行：
    - url
    - skill_name
    - file-label
    - python_code

    python_code 只来自：
    /root/BATaint/datasets/repo/<skill_name>/scripts/*.py
    """
    output_records = []

    skipped_missing_skills = []
    skipped_missing_scripts = []
    skipped_no_python_scripts = []
    matched_skills = []

    emitted_skill_keys = set()

    for skill_key, info in sorted(skill_info_map.items(), key=lambda x: x[0]):
        if skill_key in emitted_skill_keys:
            continue

        skill_name = info["skill_name"]
        classification = info["classification"]
        url = info["url"]

        skill_roots = skill_root_index.get(skill_key, [])

        if not skill_roots:
            skipped_missing_skills.append({
                "skill_name": skill_name,
                "classification": classification,
                "url": url,
                "reason": "skill_name not found under REPO_ROOT"
            })
            continue

        all_py_files_for_skill = []
        used_skill_roots = []
        missing_scripts_roots = []
        no_python_scripts_roots = []

        for skill_root in skill_roots:
            scripts_dir = find_scripts_dir(skill_root)

            if not scripts_dir:
                missing_scripts_roots.append(skill_root)
                continue

            py_files = find_python_files_in_scripts(skill_root)

            if not py_files:
                no_python_scripts_roots.append(skill_root)
                continue

            all_py_files_for_skill.extend(py_files)
            used_skill_roots.append(skill_root)

        all_py_files_for_skill = sorted(set(all_py_files_for_skill))

        if not used_skill_roots:
            if missing_scripts_roots:
                skipped_missing_scripts.append({
                    "skill_name": skill_name,
                    "classification": classification,
                    "url": url,
                    "skill_roots": skill_roots,
                    "missing_scripts_roots": missing_scripts_roots,
                    "reason": "skill_name found but scripts directory is missing"
                })

            if no_python_scripts_roots:
                skipped_no_python_scripts.append({
                    "skill_name": skill_name,
                    "classification": classification,
                    "url": url,
                    "skill_roots": skill_roots,
                    "no_python_scripts_roots": no_python_scripts_roots,
                    "reason": "scripts directory exists but no .py files found"
                })

            continue

        python_code = merge_python_files_code(
            py_files=all_py_files_for_skill,
            repo_root=repo_root
        )

        file_label = classification_to_file_label(classification)

        record = {
            "url": url,
            "skill_name": skill_name,
            "file-label": file_label,
            "python_code": python_code
        }

        output_records.append(record)
        emitted_skill_keys.add(skill_key)

        matched_skills.append({
            "skill_name": skill_name,
            "classification": classification,
            "file-label": file_label,
            "all_classifications": info.get("all_classifications", []),
            "removed_duplicate_classifications": info.get("removed_duplicate_classifications", []),
            "url": url,
            "skill_roots": used_skill_roots,
            "python_file_count": len(all_py_files_for_skill),
            "python_files": all_py_files_for_skill,
        })

    stats = {
        "matched_skills": matched_skills,
        "skipped_missing_skills": skipped_missing_skills,
        "skipped_missing_scripts": skipped_missing_scripts,
        "skipped_no_python_scripts": skipped_no_python_scripts,
    }

    return output_records, stats


def save_csv(path: str, rows: List[dict], fieldnames: List[str]):
    """
    保存 CSV 文件。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            quoting=csv.QUOTE_ALL
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def save_json(path: str, data: dict):
    """
    保存 JSON 文件。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ===================== 主函数 =====================

def main():
    start_time = datetime.now()

    print("===== Extract Skill scripts/*.py Code Dataset =====")
    print(f"skills_dataset.csv: {SKILLS_DATASET_CSV}")
    print(f"repo root: {REPO_ROOT}")
    print(f"output csv: {OUTPUT_CSV}")
    print(f"SCAN_SCRIPTS_RECURSIVE: {SCAN_SCRIPTS_RECURSIVE}")

    rows = read_csv_rows(SKILLS_DATASET_CSV)

    skill_info_map, conflict_map = build_skill_info_map(rows)
    skill_root_index = index_repo_skill_dirs(REPO_ROOT)

    # 以 repo 文件夹为基准，统计 repo 中有多少 Skill 能在 CSV 中匹配到标签
    repo_csv_match_stats = compute_repo_csv_match_stats(
        skill_info_map=skill_info_map,
        skill_root_index=skill_root_index
    )

    output_records, stats = build_output_records(
        skill_info_map=skill_info_map,
        skill_root_index=skill_root_index,
        repo_root=REPO_ROOT
    )

    save_csv(
        OUTPUT_CSV,
        output_records,
        fieldnames=OUTPUT_FIELDS
    )

    end_time = datetime.now()

    summary = {
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),

        "input_csv": normalize_path(SKILLS_DATASET_CSV),
        "repo_root": normalize_path(REPO_ROOT),
        "output_csv": normalize_path(OUTPUT_CSV),

        "csv_total_rows": len(rows),
        "csv_unique_skill_names_after_dedup": len(skill_info_map),

        # repo 目录统计：以 /root/BATaint/datasets/repo 下的 Skill 文件夹为基准
        "repo_direct_skill_key_count": len(skill_root_index),
        "repo_skill_folder_count": repo_csv_match_stats["repo_skill_folder_count"],
        "repo_csv_matched_skill_folder_count": repo_csv_match_stats["matched_repo_skill_folder_count"],
        "repo_csv_unmatched_skill_folder_count": repo_csv_match_stats["unmatched_repo_skill_folder_count"],

        "output_skill_rows": len(output_records),

        # 匹配成功的 Skills 数量：
        # 成功在 repo 中找到 skill_name，存在 scripts 文件夹，并且 scripts 中存在 .py 文件，
        # 最终被写入 skills_ground_truth_files_dataset.csv。
        "matched_skill_count": len(stats["matched_skills"]),

        # 未匹配的 Skills 数量：
        # 没有进入最终输出 CSV 的 skill_name 总数。
        # 包括：repo 中未找到、缺少 scripts、scripts 中没有 .py。
        "unmatched_skill_count": (
            len(stats["skipped_missing_skills"])
            + len(stats["skipped_missing_scripts"])
            + len(stats["skipped_no_python_scripts"])
        ),

        "skipped_missing_skill_count": len(stats["skipped_missing_skills"]),
        "skipped_missing_scripts_count": len(stats["skipped_missing_scripts"]),
        "skipped_no_python_scripts_count": len(stats["skipped_no_python_scripts"]),

        "classification_conflict_count": len(conflict_map),
        "classification_conflicts": conflict_map,
        "classification_priority": CLASSIFICATION_PRIORITY,

        "scan_scripts_recursive": SCAN_SCRIPTS_RECURSIVE,
        "output_fields": OUTPUT_FIELDS,

        "definition": {
            "url": "url field of the skill_name from skills_dataset.csv",
            "skill_name": "skill_name from skills_dataset.csv",
            "file-label": "TRUE for malicious/suspicious, FALSE for safe/benign/normal",
            "python_code": "merged source code from /root/BATaint/datasets/repo/<skill_name>/scripts/*.py only",
            "missing_skill_rule": "if skill_name is not found under REPO_ROOT, skip it",
            "missing_scripts_rule": "if skill_name is found but scripts directory is missing, skip it",
            "no_python_scripts_rule": "if scripts exists but no .py file exists, skip it"
        },

        "matched_skills": stats["matched_skills"],
        "skipped_missing_skills": stats["skipped_missing_skills"],
        "skipped_missing_scripts": stats["skipped_missing_scripts"],
        "skipped_no_python_scripts": stats["skipped_no_python_scripts"],

        # 以 repo 文件夹为基准的 CSV 匹配结果
        "repo_csv_matched_skill_folders": repo_csv_match_stats["matched_repo_skill_folders"],
        "repo_csv_unmatched_skill_folders": repo_csv_match_stats["unmatched_repo_skill_folders"],
    }

    save_json(OUTPUT_SUMMARY_JSON, summary)

    print("\n===== Done =====")
    print(f"CSV 总行数: {len(rows)}")
    print(f"repo 直接 skill key 数量: {len(skill_root_index)}")
    print(f"repo 下 skill 文件夹总数: {repo_csv_match_stats['repo_skill_folder_count']}")
    print(f"CSV 中匹配的 repo skills 数量: {repo_csv_match_stats['matched_repo_skill_folder_count']}")
    print(f"CSV 中未匹配的 repo skills 数量: {repo_csv_match_stats['unmatched_repo_skill_folder_count']}")
    matched_skill_count = len(stats["matched_skills"])
    unmatched_skill_count = (
        len(stats["skipped_missing_skills"])
        + len(stats["skipped_missing_scripts"])
        + len(stats["skipped_no_python_scripts"])
    )

    print(f"输出 skill_name 行数: {len(output_records)}")
    print(f"匹配的 skills 数量: {matched_skill_count}")
    print(f"未匹配的 skills 数量: {unmatched_skill_count}")
    print(f"  ├─ repo 中未找到 skill_name 数量: {len(stats['skipped_missing_skills'])}")
    print(f"  ├─ 缺少 scripts 文件夹数量: {len(stats['skipped_missing_scripts'])}")
    print(f"  └─ scripts 中无 .py 数量: {len(stats['skipped_no_python_scripts'])}")
    print(f"classification 冲突 skill_name 数量: {len(conflict_map)}")
    print(f"输出 CSV: {OUTPUT_CSV}")
    print(f"统计 JSON: {OUTPUT_SUMMARY_JSON}")


if __name__ == "__main__":
    main()
    """
        7
        注意：
        正常抓取只需要4-7
        处理文献下载的csv需要1-7
    """
