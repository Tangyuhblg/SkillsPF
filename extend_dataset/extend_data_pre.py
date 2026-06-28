# -*- coding: utf-8 -*-
"""
preprocess_data_bu_python_lines.py

功能：
将 /root/crawler_data/datasets/data_bu.csv 从 Skill 级形式转换为行级形式。

输入 CSV：
/root/crawler_data/datasets/data_bu.csv

输入字段：
1. url
2. skill_name
3. file-label
4. python_code

输出 CSV：
/root/crawler_data/datasets/data_bu_lines.csv

输出字段：
1. url
2. skill_name
3. code_line
4. line_num
5. is_comment
6. is_blank
7. file-label

核心规则：
- 一个 Skill 的 python_code 按行展开；
- 每一行代码输出为 CSV 中一行；
- file-label 继承输入 CSV 中的 file-label，当前 data_bu.csv 中一般全部为 TRUE；
- 跳过人工合并分隔行：
  # ===== BEGIN PYTHON FILE: xxx.py =====
  # ===== END PYTHON FILE: xxx.py =====
- 识别空行；
- 识别普通注释行和三引号文档字符串行；
- 普通代码行可进行轻量标准化处理；
- 不保存 summary JSON，只输出行级 CSV。
"""

import os
import re
import csv
import sys
import logging
from typing import Dict, List, Tuple


# ===================== 配置区 =====================

INPUT_CSV = r"/root/crawler_data/datasets/data_bu.csv"

OUTPUT_DIR = r"/root/crawler_data/datasets"

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "data_bu_lines.csv"
)

# 输入 CSV 字段名
FIELD_URL = "url"
FIELD_SKILL_NAME = "skill_name"
FIELD_FILE_LABEL = "file-label"
FIELD_PYTHON_CODE = "python_code"

# 输出 CSV 字段名
OUTPUT_FIELDS = [
    "url",
    "skill_name",
    "code_line",
    "line_num",
    "is_comment",
    "is_blank",
    "file-label",
]

# 是否对普通代码行进行轻量预处理
PREPROCESS_CODE_LINE = True

# 是否跳过合并 Python 文件时插入的人工边界行
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


# ===================== 基础工具函数 =====================

def normalize_path(path: str) -> str:
    """
    标准化路径。
    """
    return os.path.abspath(os.path.normpath(path))


def bool_to_str(value: bool) -> str:
    """
    输出 TRUE / FALSE。
    """
    return "TRUE" if bool(value) else "FALSE"


def normalize_file_label(file_label: str) -> str:
    """
    规范化 file-label。

    支持：
    - TRUE / FALSE
    - true / false
    - 1 / 0
    - yes / no

    当前 data_bu.csv 中 file-label 理论上全部为 TRUE。
    """
    value = (file_label or "").strip().lower()

    if value in {"true", "t", "1", "yes", "y"}:
        return "TRUE"

    if value in {"false", "f", "0", "no", "n"}:
        return "FALSE"

    raise ValueError(f"未知 file-label: {file_label}")


def set_large_csv_field_limit():
    """
    解决 csv 字段过大问题。

    data_bu.csv 中的 python_code 字段可能保存整个 Skill 的代码，
    容易超过 csv 模块默认字段大小限制。
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
    """
    读取 Skill 级 data_bu.csv。
    """
    path = normalize_path(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"输入 CSV 不存在: {path}")

    set_large_csv_field_limit()

    rows = []

    with open(path, "r", encoding=CSV_ENCODING, newline="") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError(f"CSV 没有表头: {path}")

        required_fields = {
            FIELD_URL,
            FIELD_SKILL_NAME,
            FIELD_FILE_LABEL,
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


def save_csv(
    path: str,
    rows: List[dict],
    fieldnames: List[str]
):
    """
    保存行级 CSV。
    """
    path = normalize_path(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding=CSV_ENCODING, newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            quoting=csv.QUOTE_ALL
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


# ===================== 行级处理函数 =====================

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

    return any(
        re.match(pattern, stripped, flags=re.IGNORECASE)
        for pattern in patterns
    )


def is_blank_line(line: str) -> bool:
    """
    判断是否为空行。
    """
    return len((line or "").strip()) == 0


def update_triple_quote_state(
    line: str,
    in_triple_quote: bool,
    quote_type: str
) -> Tuple[bool, str, bool]:
    """
    简单识别 Python 三引号块。

    三引号块中的内容视作文档字符串或注释语义。
    该方法不是完整 AST 解析，但适用于行级数据预处理。
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

    # 同一行中出现一对三引号
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
    判断当前行是否为注释行。

    规则：
    1. 空行不是注释；
    2. strip 后以 # 开头的是注释；
    3. 三引号块中的行视作文档字符串行。
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
    对普通代码行做轻量标准化。

    处理内容：
    - 字符串常量替换为 <str>；
    - 删除独立数字；
    - 删除方括号内容；
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
    file_label: str,
    python_code: str,
) -> Tuple[List[dict], dict]:
    """
    将一个 Skill 的 python_code 展开为多条行级记录。
    """
    normalized_label = normalize_file_label(file_label)

    original_lines = (python_code or "").splitlines()

    records = []

    in_triple_quote = False
    quote_type = ""

    skipped_separator_count = 0
    skipped_comment_count = 0
    skipped_blank_count = 0

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
            # 注释行保留文本语义，只去除首尾空白
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
            "file-label": normalized_label,
        })

    stats = {
        "skill_name": skill_name,
        "file-label": normalized_label,
        "original_line_count": len(original_lines),
        "output_line_count": len(records),
        "skipped_separator_count": skipped_separator_count,
        "skipped_comment_count": skipped_comment_count,
        "skipped_blank_count": skipped_blank_count,
    }

    return records, stats


# ===================== 主处理流程 =====================

def preprocess_dataset(
    input_csv: str,
    output_csv: str
) -> dict:
    """
    将 Skill 级 CSV 转换为行级 CSV。
    """
    rows = read_input_csv(input_csv)

    all_records = []
    skill_stats = []
    error_rows = []

    file_label_counter: Dict[str, int] = {
        "TRUE": 0,
        "FALSE": 0,
    }

    for idx, row in enumerate(rows, start=1):
        url = (row.get(FIELD_URL) or "").strip()
        skill_name = (row.get(FIELD_SKILL_NAME) or "").strip()
        file_label = (row.get(FIELD_FILE_LABEL) or "").strip()
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
                file_label=file_label,
                python_code=python_code,
            )

            all_records.extend(records)
            skill_stats.append(stats)

            label = stats["file-label"]
            file_label_counter[label] = file_label_counter.get(label, 0) + 1

        except Exception as e:
            logging.error(
                "处理第 %d 行失败: skill_name=%s | %s",
                idx,
                skill_name,
                e
            )

            error_rows.append({
                "row_index": idx,
                "skill_name": skill_name,
                "file-label": file_label,
                "error": str(e),
            })

    save_csv(
        path=output_csv,
        rows=all_records,
        fieldnames=OUTPUT_FIELDS
    )

    summary = {
        "total_input_skill_rows": len(rows),
        "total_output_line_rows": len(all_records),
        "processed_skill_count": len(skill_stats),
        "error_skill_count": len(error_rows),
        "file_label_counter": file_label_counter,
        "error_rows": error_rows,
    }

    return summary


# ===================== 主函数 =====================

def main():
    print("===== Preprocess data_bu Python Lines =====")
    print(f"输入 CSV: {INPUT_CSV}")
    print(f"输出 CSV: {OUTPUT_CSV}")
    print(f"输出字段: {OUTPUT_FIELDS}")
    print(f"PREPROCESS_CODE_LINE: {PREPROCESS_CODE_LINE}")
    print(f"DROP_MERGE_SEPARATOR_LINES: {DROP_MERGE_SEPARATOR_LINES}")
    print(f"KEEP_BLANK_LINES: {KEEP_BLANK_LINES}")
    print(f"KEEP_COMMENT_LINES: {KEEP_COMMENT_LINES}")

    summary = preprocess_dataset(
        input_csv=INPUT_CSV,
        output_csv=OUTPUT_CSV,
    )

    print("\n===== 完成 =====")
    print(f"输入 Skill 行数: {summary['total_input_skill_rows']}")
    print(f"输出代码行数: {summary['total_output_line_rows']}")
    print(f"成功处理 Skill 数量: {summary['processed_skill_count']}")
    print(f"失败 Skill 数量: {summary['error_skill_count']}")
    print(f"file-label 统计: {summary['file_label_counter']}")
    print(f"输出 CSV: {OUTPUT_CSV}")


    


if __name__ == "__main__":
    main()