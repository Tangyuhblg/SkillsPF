# -*- coding: utf-8 -*-
"""
extract_data_bu_python_code_to_csv.py

功能：
递归提取 /root/crawler_data/datasets/data_bu 下每个 Skill 文件夹中的所有 Python 代码，
并按 Skill 粒度生成 CSV 数据集。

输入目录结构示例：
/root/crawler_data/datasets/data_bu/
├── skill_a/
│   ├── main.py
│   └── utils/helper.py
├── skill_b/
│   └── scripts/run.py
└── skill_c/
    └── README.md

输出 CSV：
/root/crawler_data/datasets/data_bu_python_code_dataset.csv

输出字段：
1. url
2. skill_name
3. file-label
4. python_code

核心规则：
- data_bu 下每个直接子文件夹视为一个 Skill；
- 一个 Skill 文件夹只输出 CSV 中一行；
- 递归提取该 Skill 文件夹下所有 .py 文件；
- 如果一个 Skill 中有多个 Python 文件，则合并到同一个 python_code 字段；
- url 全部为空字符串；
- file-label 全部为 TRUE；
- 没有 Python 文件的 Skill 文件夹跳过；
- 生成 summary JSON，便于检查统计结果。
"""

import os
import csv
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple


# ===================== 配置区 =====================

DATA_BU_ROOT = r"/root/crawler_data/datasets/data_bu"

OUTPUT_DIR = r"/root/crawler_data/datasets"

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "data_bu.csv"
)

OUTPUT_SUMMARY_JSON = os.path.join(
    OUTPUT_DIR
)

OUTPUT_FIELDS = [
    "url",
    "skill_name",
    "file-label",
    "python_code"
]

# 是否递归扫描 Skill 文件夹下所有子目录中的 .py 文件
SCAN_RECURSIVE = True

# 如果 Python 代码特别大，可以设置最大字符数；None 表示不截断
MAX_CODE_CHARS_PER_SKILL = None

# 多个 Python 文件合并到 python_code 字段时的分隔符
CODE_FILE_SEPARATOR_TEMPLATE = (
    "\n\n"
    "# ===== BEGIN PYTHON FILE: {relative_path} =====\n"
    "{code}\n"
    "# ===== END PYTHON FILE: {relative_path} =====\n"
)


# ===================== 日志配置 =====================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(levelname)s\t%(message)s"
)


# ===================== 基础工具函数 =====================

def normalize_path(path: str) -> str:
    """
    标准化路径。
    """
    return os.path.abspath(os.path.normpath(path))


def read_source_code(file_path: str) -> str:
    """
    读取 Python 源码。

    为了适配不同来源的数据集，依次尝试多种编码。
    """
    encodings = [
        "utf-8",
        "utf-8-sig",
        "gbk",
        "latin-1"
    ]

    last_error = None

    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc, errors="replace") as f:
                return f.read()
        except Exception as e:
            last_error = e

    logging.error("读取源码失败: %s | %s", file_path, last_error)
    return ""


# ===================== Skill 文件夹扫描 =====================

def list_skill_dirs(root_dir: str) -> List[str]:
    """
    获取 data_bu 根目录下的直接子文件夹。

    每个直接子文件夹视为一个 Skill。
    """
    root_dir = normalize_path(root_dir)

    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"输入目录不存在: {root_dir}")

    if not os.path.isdir(root_dir):
        raise NotADirectoryError(f"输入路径不是目录: {root_dir}")

    skill_dirs = []

    for item in sorted(os.listdir(root_dir)):
        full_path = normalize_path(os.path.join(root_dir, item))

        if os.path.isdir(full_path):
            skill_dirs.append(full_path)

    return skill_dirs


def find_python_files_in_skill(skill_root: str) -> List[str]:
    """
    查找某个 Skill 文件夹下的所有 .py 文件。

    当前规则：
    - 递归扫描整个 Skill 文件夹；
    - 只保留后缀为 .py 的文件；
    - 不跟随符号链接目录，避免误扫外部路径。
    """
    skill_root = normalize_path(skill_root)

    if not os.path.isdir(skill_root):
        return []

    py_files = []

    if SCAN_RECURSIVE:
        for current_root, dirs, files in os.walk(skill_root):
            # 不跟随符号链接目录
            dirs[:] = [
                d for d in dirs
                if not os.path.islink(os.path.join(current_root, d))
            ]

            for filename in files:
                full_path = normalize_path(os.path.join(current_root, filename))

                if not os.path.isfile(full_path):
                    continue

                if Path(full_path).suffix.lower() == ".py":
                    py_files.append(full_path)
    else:
        for filename in os.listdir(skill_root):
            full_path = normalize_path(os.path.join(skill_root, filename))

            if not os.path.isfile(full_path):
                continue

            if Path(full_path).suffix.lower() == ".py":
                py_files.append(full_path)

    return sorted(set(py_files))


# ===================== Python 代码合并 =====================

def merge_python_files_code(
    py_files: List[str],
    skill_root: str
) -> str:
    """
    将同一个 Skill 文件夹下的多个 Python 文件合并为一个字符串。

    合并时保留每个 Python 文件的相对路径，便于后续模型或人工分析区分文件来源。
    """
    merged_parts = []
    skill_root = normalize_path(skill_root)

    for py_file in py_files:
        rel_path = os.path.relpath(py_file, skill_root)
        rel_path = os.path.normpath(rel_path).replace(os.sep, "/")

        code = read_source_code(py_file)

        merged_parts.append(
            CODE_FILE_SEPARATOR_TEMPLATE.format(
                relative_path=rel_path,
                code=code
            )
        )

    merged_code = "".join(merged_parts)

    if (
        MAX_CODE_CHARS_PER_SKILL is not None
        and len(merged_code) > MAX_CODE_CHARS_PER_SKILL
    ):
        merged_code = merged_code[:MAX_CODE_CHARS_PER_SKILL]

    return merged_code


# ===================== 构造 CSV 记录 =====================

def build_output_records(
    data_bu_root: str
) -> Tuple[List[dict], dict]:
    """
    构造最终 CSV 记录。

    每个 Skill 输出一行：
    - url: 空字符串
    - skill_name: Skill 文件夹名称
    - file-label: TRUE
    - python_code: 该 Skill 下所有 .py 文件合并后的源码
    """
    skill_dirs = list_skill_dirs(data_bu_root)

    output_records = []
    matched_skills = []
    skipped_no_python_files = []

    for skill_root in skill_dirs:
        skill_name = os.path.basename(skill_root)

        py_files = find_python_files_in_skill(skill_root)

        if not py_files:
            skipped_no_python_files.append({
                "skill_name": skill_name,
                "skill_root": skill_root,
                "reason": "no .py files found under skill folder"
            })
            continue

        python_code = merge_python_files_code(
            py_files=py_files,
            skill_root=skill_root
        )

        record = {
            "url": "",
            "skill_name": skill_name,
            "file-label": "TRUE",
            "python_code": python_code
        }

        output_records.append(record)

        matched_skills.append({
            "skill_name": skill_name,
            "skill_root": skill_root,
            "file-label": "TRUE",
            "python_file_count": len(py_files),
            "python_files": py_files
        })

    stats = {
        "skill_folder_count": len(skill_dirs),
        "output_skill_count": len(output_records),
        "skipped_no_python_file_count": len(skipped_no_python_files),
        "matched_skills": matched_skills,
        "skipped_no_python_files": skipped_no_python_files
    }

    return output_records, stats


# ===================== 保存文件 =====================

def save_csv(
    path: str,
    rows: List[dict],
    fieldnames: List[str]
):
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



# ===================== 主函数 =====================

def main():
    start_time = datetime.now()

    print("===== Extract data_bu Python Code Dataset =====")
    print(f"data_bu root: {DATA_BU_ROOT}")
    print(f"output csv: {OUTPUT_CSV}")
    print(f"output summary json: {OUTPUT_SUMMARY_JSON}")
    print(f"SCAN_RECURSIVE: {SCAN_RECURSIVE}")
    print(f"MAX_CODE_CHARS_PER_SKILL: {MAX_CODE_CHARS_PER_SKILL}")

    output_records, stats = build_output_records(
        data_bu_root=DATA_BU_ROOT
    )

    save_csv(
        path=OUTPUT_CSV,
        rows=output_records,
        fieldnames=OUTPUT_FIELDS
    )

    end_time = datetime.now()

    # summary = {
    #     "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
    #     "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
    #
    #     "input_root": normalize_path(DATA_BU_ROOT),
    #     "output_csv": normalize_path(OUTPUT_CSV),
    #     "output_summary_json": normalize_path(OUTPUT_SUMMARY_JSON),
    #
    #     "output_fields": OUTPUT_FIELDS,
    #
    #     "field_definition": {
    #         "url": "empty string for all skills",
    #         "skill_name": "folder name under /root/crawler_data/datasets/data_bu",
    #         "file-label": "TRUE for all extracted skills",
    #         "python_code": "merged source code from all .py files under each skill folder"
    #     },
    #
    #     "scan_recursive": SCAN_RECURSIVE,
    #     "max_code_chars_per_skill": MAX_CODE_CHARS_PER_SKILL,
    #
    #     "skill_folder_count": stats["skill_folder_count"],
    #     "output_skill_count": stats["output_skill_count"],
    #     "skipped_no_python_file_count": stats["skipped_no_python_file_count"],
    #
    #     "matched_skills": stats["matched_skills"],
    #     "skipped_no_python_files": stats["skipped_no_python_files"]
    # }



    print("\n===== Done =====")
    print(f"Skill 文件夹总数: {stats['skill_folder_count']}")
    print(f"输出 Skill 数量: {stats['output_skill_count']}")
    print(f"无 Python 文件而跳过的 Skill 数量: {stats['skipped_no_python_file_count']}")
    print(f"输出 CSV: {OUTPUT_CSV}")
    # print(f"统计 JSON: {OUTPUT_SUMMARY_JSON}")


if __name__ == "__main__":
    main()