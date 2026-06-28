# -*- coding: utf-8 -*-
"""
preprocess_skill_python_ground_truth_lines.py

功能：
处理 skills_python_ground_truth_files_dataset.csv，把每个 skill_name 的 python_code 按行展开，生成行级 CSV 数据集。

输入 CSV 默认字段：
- url
- skill_name
- classification
- python_code

输出 CSV 字段：
- url
- skill_name
- code_line
- line_num
- is_comment
- is_blank
- file-label

classification -> file-label：
- malicious / suspicious -> TRUE
- safe / benign / normal -> FALSE
"""

import os
import re
import csv
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple
import sys


# ===================== 配置区 =====================

INPUT_CSV = r"/root/BATaint/datasets/skills_ground_truth_files_dataset.csv"
OUTPUT_DIR = r"/root/BATaint/datasets"

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "skills_ground_truth_lines_dataset.csv"
)

OUTPUT_SUMMARY_JSON = os.path.join(
    OUTPUT_DIR,
    "skills_ground_truth_lines_dataset_summary.json"
)

# 输入 CSV 字段名
FIELD_URL = "url"
FIELD_SKILL_NAME = "skill_name"
FIELD_CLASSIFICATION = "file-label"
FIELD_PYTHON_CODE = "python_code"

# 输出 CSV 字段名，严格按照需求
OUTPUT_FIELDS = [
    "url",
    "skill_name",
    "code_line",
    "line_num",
    "is_comment",
    "is_blank",
    "file-label",
]

# classification -> file-label
CLASSIFICATION_TO_FILE_LABEL = {
    "malicious": True,
    "suspicious": True,
    "safe": False,
    "benign": False,
    "normal": False,
}

# 是否预处理 code_line。
# True：字符串、数字、符号做轻量标准化，适合复现 groovy/java 预处理思想。
# False：保留原始代码行，仅去除首尾空白。
PREPROCESS_CODE_LINE = True

# 是否跳过之前合并 Python 文件时插入的人工边界行：
# # ===== BEGIN PYTHON FILE: xxx.py =====
# # ===== END PYTHON FILE: xxx.py =====
DROP_MERGE_SEPARATOR_LINES = True

# 是否保留空行和注释行
KEEP_BLANK_LINES = True
KEEP_COMMENT_LINES = True

CSV_ENCODING = "utf-8-sig"


# ===================== 日志配置 =====================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(levelname)s\t%(message)s"
)


# ===================== 基础工具 =====================

def normalize_path(path: str) -> str:
    return os.path.abspath(os.path.normpath(path))


def bool_to_str(value: bool) -> str:
    """
    输出 TRUE / FALSE，而不是 Python 的 True / False。
    """
    return "TRUE" if bool(value) else "FALSE"


def set_large_csv_field_limit():
    """
    解决：
    _csv.Error: field larger than field limit (131072)

    原因：
    skills_ground_truth_files_dataset.csv 中的 code_line 字段保存的是整段 Python 代码，
    单个字段可能超过 csv 模块默认限制。
    """
    max_size = sys.maxsize

    while True:
        try:
            csv.field_size_limit(max_size)
            logging.info("CSV field_size_limit set to %s", max_size)
            return
        except OverflowError:
            max_size = int(max_size / 10)

def read_input_csv(path: str) -> List[dict]:
    path = normalize_path(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"输入 CSV 不存在: {path}")

    # 关键修改：解决 code_line 字段过大导致的 csv 读取错误
    set_large_csv_field_limit()

    rows = []
    with open(path, "r", encoding=CSV_ENCODING, newline="") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError(f"CSV 没有表头: {path}")

        required_fields = {
            FIELD_URL,
            FIELD_SKILL_NAME,
            FIELD_CLASSIFICATION,
            FIELD_PYTHON_CODE,
        }
        missing = required_fields - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"输入 CSV 缺少必要字段: {missing}; 当前字段: {reader.fieldnames}"
            )

        for row in reader:
            rows.append(dict(row))

    return rows


def save_csv(path: str, rows: List[dict], fieldnames: List[str]):
    path = normalize_path(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding=CSV_ENCODING, newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_json(path: str, data: dict):
    path = normalize_path(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ===================== 标签处理 =====================

def parse_file_label(file_label: str) -> bool:
    """
    将输入 CSV 中的 file-label 转换为 bool。

    支持：
    - TRUE / FALSE
    - true / false
    - 1 / 0
    - yes / no
    - malicious / suspicious / safe / benign / normal
    """
    value = (file_label or "").strip().lower()

    if value in {"true", "t", "1", "yes", "y"}:
        return True

    if value in {"false", "f", "0", "no", "n"}:
        return False

    if value in CLASSIFICATION_TO_FILE_LABEL:
        return CLASSIFICATION_TO_FILE_LABEL[value]

    raise ValueError(f"未知 file-label: {file_label}")


# ===================== 行处理 =====================

def is_merge_separator_line(line: str) -> bool:
    """
    判断是否为人工插入的 BEGIN/END 文件边界行。
    """
    stripped = (line or "").strip()
    if not stripped.startswith("#"):
        return False

    patterns = [
        r"^#\s*=+\s*BEGIN PYTHON FILE:",
        r"^#\s*=+\s*END PYTHON FILE:",
    ]
    return any(re.match(p, stripped, flags=re.IGNORECASE) for p in patterns)


def is_blank_line(line: str) -> bool:
    return len((line or "").strip()) == 0


def update_triple_quote_state(
    line: str,
    in_triple_quote: bool,
    quote_type: str
) -> Tuple[bool, str, bool]:
    """
    简单识别 Python 三引号块。三引号块中的行视作注释/文档字符串行。
    这不是完整 AST 解析，但适合数据集行级预处理。
    """
    stripped = (line or "").strip()

    if not stripped:
        return in_triple_quote, quote_type, False

    if in_triple_quote:
        if quote_type in stripped:
            return False, "", True
        return True, quote_type, True

    triple_positions = []
    for q in ('"""', "'''"):
        pos = stripped.find(q)
        if pos != -1:
            triple_positions.append((pos, q))

    if not triple_positions:
        return False, "", False

    triple_positions.sort(key=lambda x: x[0])
    _, q = triple_positions[0]

    # 同一行出现一对三引号，视作文档字符串/注释行
    if stripped.count(q) >= 2:
        return False, "", True

    # 只出现一次，进入三引号块
    return True, q, True


def is_comment_line(
    line: str,
    in_triple_quote: bool,
    quote_type: str
) -> Tuple[bool, bool, str]:
    """
    Python 注释判断：
    1. 空行不是注释；
    2. strip 后以 # 开头是注释；
    3. 三引号块中的行视作文档字符串/注释行。
    """
    stripped = (line or "").strip()

    if not stripped:
        return False, in_triple_quote, quote_type

    new_in_triple_quote, new_quote_type, triple_comment = update_triple_quote_state(
        line=line,
        in_triple_quote=in_triple_quote,
        quote_type=quote_type,
    )

    if triple_comment:
        return True, new_in_triple_quote, new_quote_type

    if stripped.startswith("#"):
        return True, new_in_triple_quote, new_quote_type

    return False, new_in_triple_quote, new_quote_type


def preprocess_code_line(line: str) -> str:
    """
    对代码行做轻量标准化，参考上传代码中的 preprocess_code_line 思路：
    - 字符串常量替换为 <str>；
    - 独立数字删除；
    - 方括号内容删除；
    - 常见符号替换为空格；
    - 压缩多余空白。
    """
    if line is None:
        return ""

    code_line = str(line).strip()

    # 替换三引号、双引号、单引号字符串
    code_line = re.sub(r'""".*?"""', "<str>", code_line)
    code_line = re.sub(r"'''.*?'''", "<str>", code_line)
    code_line = re.sub(r'"(?:\\.|[^"\\])*"', "<str>", code_line)
    code_line = re.sub(r"'(?:\\.|[^'\\])*'", "<str>", code_line)

    # 删除独立数字
    code_line = re.sub(r"\b\d+\b", "", code_line)

    # 删除方括号内容
    code_line = re.sub(r"\[.*?\]", "", code_line)

    # 常见符号替换为空格
    code_line = re.sub(r"[.,:;{}()\[\]]", " ", code_line)

    chars_to_remove = [
        "+", "-", "*", "/", "=", "\\", "|", "&", "!", "<", ">", "%",
        "^", "~", "@",
    ]
    for ch in chars_to_remove:
        code_line = code_line.replace(ch, " ")

    code_line = re.sub(r"\s+", " ", code_line).strip()
    return code_line


def process_python_code_to_line_records(
    url: str,
    skill_name: str,
    classification: str,
    python_code: str,
) -> Tuple[List[dict], dict]:
    """
    将一个 skill_name 的 python_code 展开为多条行级记录。
    """
    file_label = parse_file_label(classification)
    original_lines = (python_code or "").splitlines()

    records = []
    in_triple_quote = False
    quote_type = ""

    skipped_separator_count = 0
    skipped_comment_count = 0
    skipped_blank_count = 0

    # line_num 表示该行在原 python_code 中的原始行号，从 1 开始
    for line_num, original_line in enumerate(original_lines, start=1):
        if DROP_MERGE_SEPARATOR_LINES and is_merge_separator_line(original_line):
            skipped_separator_count += 1
            continue

        blank = is_blank_line(original_line)
        comment, in_triple_quote, quote_type = is_comment_line(
            line=original_line,
            in_triple_quote=in_triple_quote,
            quote_type=quote_type,
        )

        if blank and not KEEP_BLANK_LINES:
            skipped_blank_count += 1
            continue

        if comment and not KEEP_COMMENT_LINES:
            skipped_comment_count += 1
            continue

        if comment:
            # 注释行保持文本语义，仅去除首尾空白
            code_line = original_line.strip()
        elif PREPROCESS_CODE_LINE:
            code_line = preprocess_code_line(original_line)
        else:
            code_line = original_line.strip()

        final_blank = is_blank_line(code_line)

        records.append({
            "url": url,
            "skill_name": skill_name,
            "code_line": code_line,
            "line_num": line_num,
            "is_comment": bool_to_str(comment),
            "is_blank": bool_to_str(final_blank),
            "file-label": bool_to_str(file_label),
        })

    stats = {
        "skill_name": skill_name,
        "file-label": bool_to_str(file_label),
        "file_label": bool_to_str(file_label),
        "original_line_count": len(original_lines),
        "output_line_count": len(records),
        "skipped_separator_count": skipped_separator_count,
        "skipped_comment_count": skipped_comment_count,
        "skipped_blank_count": skipped_blank_count,
    }

    return records, stats


# ===================== 主处理流程 =====================

def preprocess_dataset(input_csv: str, output_csv: str) -> dict:
    rows = read_input_csv(input_csv)

    all_records = []
    skill_stats = []
    error_rows = []

    classification_counter: Dict[str, int] = {}
    file_label_counter: Dict[str, int] = {"TRUE": 0, "FALSE": 0}

    for idx, row in enumerate(rows, start=1):
        url = (row.get(FIELD_URL) or "").strip()
        skill_name = (row.get(FIELD_SKILL_NAME) or "").strip()
        file_label_value = (row.get(FIELD_CLASSIFICATION) or "").strip()
        python_code = row.get(FIELD_PYTHON_CODE) or ""

        if not skill_name:
            error_rows.append({
                "row_index": idx,
                "error": "empty skill_name",
            })
            continue

        try:
            records, stats = process_python_code_to_line_records(
                url=url,
                skill_name=skill_name,
                classification=file_label_value,
                python_code=python_code,
            )

            all_records.extend(records)
            skill_stats.append(stats)

            classification_counter[file_label_value] = classification_counter.get(file_label_value, 0) + 1
            file_label_counter[stats["file_label"]] = file_label_counter.get(stats["file_label"], 0) + 1

        except Exception as e:
            logging.error("处理第 %d 行失败: skill_name=%s | %s", idx, skill_name, e)
            error_rows.append({
                "row_index": idx,
                "skill_name": skill_name,
                "classification": file_label_value,
                "error": str(e),
            })

    save_csv(output_csv, all_records, OUTPUT_FIELDS)

    summary = {
        "input_csv": normalize_path(input_csv),
        "output_csv": normalize_path(output_csv),
        "total_input_skill_rows": len(rows),
        "total_output_line_rows": len(all_records),
        "processed_skill_count": len(skill_stats),
        "error_skill_count": len(error_rows),
        "classification_counter": classification_counter,
        "file_label_counter": file_label_counter,
        "config": {
            "PREPROCESS_CODE_LINE": PREPROCESS_CODE_LINE,
            "DROP_MERGE_SEPARATOR_LINES": DROP_MERGE_SEPARATOR_LINES,
            "KEEP_BLANK_LINES": KEEP_BLANK_LINES,
            "KEEP_COMMENT_LINES": KEEP_COMMENT_LINES,
            "CLASSIFICATION_TO_FILE_LABEL": {
                k: bool_to_str(v) for k, v in CLASSIFICATION_TO_FILE_LABEL.items()
            },
            "OUTPUT_FIELDS": OUTPUT_FIELDS,
        },
        "skill_stats": skill_stats,
        "error_rows": error_rows,
    }

    return summary


def main():
    start_time = datetime.now()

    print("===== Preprocess Skill Python Ground Truth Lines =====")
    print(f"输入 CSV: {INPUT_CSV}")
    print(f"输出 CSV: {OUTPUT_CSV}")
    print(f"输出字段: {OUTPUT_FIELDS}")

    summary = preprocess_dataset(
        input_csv=INPUT_CSV,
        output_csv=OUTPUT_CSV,
    )

    end_time = datetime.now()
    summary["start_time"] = start_time.strftime("%Y-%m-%d %H:%M:%S")
    summary["end_time"] = end_time.strftime("%Y-%m-%d %H:%M:%S")

    save_json(OUTPUT_SUMMARY_JSON, summary)

    print("\n===== 完成 =====")
    print(f"输入 skill 行数: {summary['total_input_skill_rows']}")
    print(f"输出代码行数: {summary['total_output_line_rows']}")
    print(f"成功处理 skill 数量: {summary['processed_skill_count']}")
    print(f"失败 skill 数量: {summary['error_skill_count']}")
    print(f"classification 统计: {summary['classification_counter']}")
    print(f"file-label 统计: {summary['file_label_counter']}")
    print(f"输出 CSV: {OUTPUT_CSV}")
    print(f"summary JSON: {OUTPUT_SUMMARY_JSON}")


if __name__ == "__main__":
    main()
    """
        处理为行级形式url, skill_name, file-label, code_line
    """
