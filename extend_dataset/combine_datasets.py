# -*- coding: utf-8 -*-
"""
merge_ground_truth_and_data_bu_lines.py

功能：
合并两个行级 CSV 数据集：

1. /root/BATaint/datasets/skills_ground_truth_lines_dataset.csv
2. /root/crawler_data/datasets/data_bu_lines.csv

合并后输出到：
/root/BATaint/datasets/skills_ground_truth_lines_dataset.csv

注意：
- 输出路径与第一个输入路径相同；
- 程序会先完整读取两个 CSV，再写入临时文件，最后替换原文件；
- 默认会备份原始 skills_ground_truth_lines_dataset.csv。
"""

import os
import csv
import sys
import shutil
import logging
from datetime import datetime
from typing import List


# ===================== 配置区 =====================

GROUND_TRUTH_LINES_CSV = r"/root/BATaint/datasets/skills_ground_truth_lines_dataset.csv"

DATA_BU_LINES_CSV = r"/root/crawler_data/datasets/data_bu_lines.csv"

OUTPUT_DIR = r"/root/BATaint/datasets"

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "skills_ground_truth_lines_dataset.csv"
)

OUTPUT_FIELDS = [
    "url",
    "skill_name",
    "code_line",
    "line_num",
    "is_comment",
    "is_blank",
    "file-label",
]

CSV_ENCODING = "utf-8-sig"

# 是否备份原始 ground truth 行级 CSV
CREATE_BACKUP = True


# ===================== 日志配置 =====================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(levelname)s\t%(message)s"
)


# ===================== 基础工具函数 =====================

def normalize_path(path: str) -> str:
    return os.path.abspath(os.path.normpath(path))


def set_large_csv_field_limit():
    """
    解决 CSV 字段过大问题。
    """
    max_size = sys.maxsize

    while True:
        try:
            csv.field_size_limit(max_size)
            logging.info("CSV field_size_limit set to %s", max_size)
            return
        except OverflowError:
            max_size = int(max_size / 10)


def check_file_exists(path: str):
    path = normalize_path(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV 文件不存在: {path}")

    if not os.path.isfile(path):
        raise IsADirectoryError(f"路径不是 CSV 文件: {path}")


def read_csv_rows(path: str, required_fields: List[str]) -> List[dict]:
    """
    读取 CSV，并检查字段是否完整。
    """
    path = normalize_path(path)
    check_file_exists(path)

    rows = []

    with open(path, "r", encoding=CSV_ENCODING, newline="") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError(f"CSV 没有表头: {path}")

        missing = set(required_fields) - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"CSV 缺少必要字段: {missing}; 文件: {path}; 当前字段: {reader.fieldnames}"
            )

        for row in reader:
            new_row = {}

            for field in required_fields:
                new_row[field] = row.get(field, "")

            rows.append(new_row)

    return rows


def save_csv(path: str, rows: List[dict], fieldnames: List[str]):
    """
    保存 CSV。
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


def backup_file(path: str) -> str:
    """
    备份原始 CSV。
    """
    path = normalize_path(path)

    if not os.path.exists(path):
        return ""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{path}.bak_{timestamp}"

    shutil.copy2(path, backup_path)

    return backup_path


# ===================== 主合并逻辑 =====================

def merge_csv_files():
    set_large_csv_field_limit()

    ground_truth_path = normalize_path(GROUND_TRUTH_LINES_CSV)
    data_bu_path = normalize_path(DATA_BU_LINES_CSV)
    output_path = normalize_path(OUTPUT_CSV)

    print("===== Merge Line-level CSV Datasets =====")
    print(f"ground truth csv: {ground_truth_path}")
    print(f"data_bu lines csv: {data_bu_path}")
    print(f"output csv: {output_path}")

    ground_truth_rows = read_csv_rows(
        path=ground_truth_path,
        required_fields=OUTPUT_FIELDS
    )

    data_bu_rows = read_csv_rows(
        path=data_bu_path,
        required_fields=OUTPUT_FIELDS
    )

    merged_rows = ground_truth_rows + data_bu_rows

    backup_path = ""

    if CREATE_BACKUP and os.path.exists(output_path):
        backup_path = backup_file(output_path)

    temp_output_path = output_path + ".tmp"

    save_csv(
        path=temp_output_path,
        rows=merged_rows,
        fieldnames=OUTPUT_FIELDS
    )

    os.replace(temp_output_path, output_path)

    print("\n===== Done =====")
    print(f"ground truth 原始行数: {len(ground_truth_rows)}")
    print(f"data_bu 行数: {len(data_bu_rows)}")
    print(f"合并后总行数: {len(merged_rows)}")
    print(f"输出 CSV: {output_path}")

    if backup_path:
        print(f"原始 CSV 备份: {backup_path}")


def main():
    merge_csv_files()


if __name__ == "__main__":
    main()