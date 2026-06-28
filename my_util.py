# -*- coding: utf-8 -*-
"""
my_util_taint_fusion.py

读取 Skill 行级数据集、按 skill_name 聚合、分层划分，并构建包含源码 token 与污点路径 token 的 Dataset。
"""

import csv
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from taint_feature_extract import TaintTextExtractor


def set_large_csv_field_limit():
    max_size = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_size)
            return max_size
        except OverflowError:
            max_size = int(max_size / 10)


def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_path(path: str) -> str:
    return os.path.abspath(os.path.normpath(path))


def resolve_data_csv(path: str) -> str:
    """默认路径不存在时，自动尝试常见替代文件名。"""
    path = normalize_path(path)
    if os.path.exists(path):
        return path

    dirname = os.path.dirname(path)
    fallback_names = [
        'skills_python_ground_truth_lines_dataset.csv',
        'skills_ground_truth_lines_dataset.csv',
    ]
    for name in fallback_names:
        candidate = os.path.join(dirname, name)
        if os.path.exists(candidate):
            print(f'[WARN] 指定数据集不存在，自动使用替代路径: {candidate}')
            return candidate

    raise FileNotFoundError(f'数据集不存在: {path}')


def parse_bool_label(value) -> int:
    v = str(value).strip().lower()
    if v in {'true', 't', '1', 'yes', 'y', 'malicious', 'suspicious'}:
        return 1
    if v in {'false', 'f', '0', 'no', 'n', 'safe', 'benign', 'normal'}:
        return 0
    raise ValueError(f'无法解析 file-label: {value}')


def bool_str(label: int) -> str:
    return 'TRUE' if int(label) == 1 else 'FALSE'


def is_true_like(value) -> bool:
    return str(value).strip().lower() in {'true', 't', '1', 'yes', 'y'}


@dataclass
class SkillExample:
    url: str
    skill_name: str
    lines: List[str]
    label: int


def load_skill_examples(
    csv_path: str,
    drop_blank: bool = True,
    drop_comment: bool = True,
    min_lines: int = 1,
) -> List[SkillExample]:
    csv_path = resolve_data_csv(csv_path)

    required_cols = {'url', 'skill_name', 'code_line', 'line_num', 'is_comment', 'is_blank', 'file-label'}

    set_large_csv_field_limit()
    df = pd.read_csv(csv_path, encoding='utf-8-sig', dtype=str, keep_default_na=False)

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f'CSV 缺少必要字段: {missing}; 当前字段: {list(df.columns)}')

    df['label_int'] = df['file-label'].apply(parse_bool_label)

    if drop_blank:
        df = df[~df['is_blank'].apply(is_true_like)]
    if drop_comment:
        df = df[~df['is_comment'].apply(is_true_like)]

    df['line_num_int'] = pd.to_numeric(df['line_num'], errors='coerce').fillna(0).astype(int)
    df = df.sort_values(['skill_name', 'line_num_int'], kind='stable')

    examples: List[SkillExample] = []
    conflict_count = 0

    for skill_name, group in df.groupby('skill_name', sort=False):
        urls = list(group['url'].unique())
        labels = sorted(set(group['label_int'].tolist()))
        if len(labels) > 1:
            conflict_count += 1

        # 标签冲突时采用高风险优先：只要该 Skill 有一行 TRUE，就认为 Skill 有缺陷。
        label = 1 if 1 in labels else 0
        lines = [str(x) for x in group['code_line'].tolist() if str(x).strip() != '']

        if len(lines) < min_lines:
            continue

        examples.append(SkillExample(url=urls[0] if urls else '', skill_name=str(skill_name), lines=lines, label=label))

    if conflict_count:
        print(f'[WARN] {conflict_count} 个 skill_name 存在标签冲突，已采用高风险优先策略。')
    if not examples:
        raise ValueError('没有读取到有效 Skill 样本，请检查 CSV、drop_blank/drop_comment 设置。')

    return examples


def summarize_examples(examples: List[SkillExample]) -> Dict:
    labels = [ex.label for ex in examples]
    line_counts = [len(ex.lines) for ex in examples]
    return {
        'skill_count': len(examples),
        'positive_count': int(sum(labels)),
        'negative_count': int(len(labels) - sum(labels)),
        'positive_ratio': float(sum(labels) / len(labels)) if labels else 0.0,
        'min_lines': int(np.min(line_counts)) if line_counts else 0,
        'max_lines': int(np.max(line_counts)) if line_counts else 0,
        'avg_lines': float(np.mean(line_counts)) if line_counts else 0.0,
    }


def _validate_stratify(labels: np.ndarray, split_name: str):
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2:
        raise ValueError(f'{split_name}: 只有一个类别，无法进行二分类训练和分层划分。')
    if np.min(counts) < 2:
        raise ValueError(f'{split_name}: 存在类别样本数小于 2，无法分层划分。类别统计: {dict(zip(unique, counts))}')


def stratified_split_examples(
    examples: List[SkillExample],
    train_ratio: float = 0.7,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[SkillExample], List[SkillExample], List[SkillExample]]:
    ratio_sum = train_ratio + valid_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f'train/valid/test 比例之和必须为 1，当前为 {ratio_sum}')

    labels = np.array([ex.label for ex in examples])
    _validate_stratify(labels, '完整数据集')

    temp_ratio = valid_ratio + test_ratio
    train_examples, temp_examples = train_test_split(
        examples,
        test_size=temp_ratio,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )

    temp_labels = np.array([ex.label for ex in temp_examples])
    _validate_stratify(temp_labels, '验证集+测试集临时集合')

    test_in_temp_ratio = test_ratio / (valid_ratio + test_ratio)
    valid_examples, test_examples = train_test_split(
        temp_examples,
        test_size=test_in_temp_ratio,
        random_state=seed,
        shuffle=True,
        stratify=temp_labels,
    )

    return train_examples, valid_examples, test_examples


def save_split_summary(output_dir: str, train_examples: List[SkillExample], valid_examples: List[SkillExample], test_examples: List[SkillExample]):
    output_dir = normalize_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    split_rows = []
    for split_name, examples in [('train', train_examples), ('valid', valid_examples), ('test', test_examples)]:
        for ex in examples:
            split_rows.append({
                'split': split_name,
                'url': ex.url,
                'skill_name': ex.skill_name,
                'file-label': bool_str(ex.label),
                'line_count': len(ex.lines),
            })

    pd.DataFrame(split_rows).to_csv(os.path.join(output_dir, 'skill_split.csv'), index=False, encoding='utf-8-sig')

    summary = {'train': summarize_examples(train_examples), 'valid': summarize_examples(valid_examples), 'test': summarize_examples(test_examples)}
    with open(os.path.join(output_dir, 'skill_split_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


class SkillCodeTaintDataset(Dataset):
    """返回源码 token 与污点路径 token 的 Skill 级 Dataset。"""

    def __init__(
        self,
        examples: List[SkillExample],
        tokenizer,
        repo_root: str,
        code_block_size: int = 75,
        taint_block_size: int = 256,
        max_lines: Optional[int] = 900,
        max_taint_paths: int = 32,
        taint_result_name: str = 'taint_result.json',
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.repo_root = repo_root
        self.code_block_size = code_block_size
        self.taint_block_size = taint_block_size
        self.max_lines = max_lines
        self.max_taint_paths = max_taint_paths
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1

        self.taint_extractor = TaintTextExtractor(
            repo_root=repo_root,
            taint_result_name=taint_result_name,
            max_taint_paths=max_taint_paths,
            enable_cache=True,
            enable_dir_index=True,
        )
        self.features = []
        self._build_features()

    def _encode_text(self, text: str, max_length: int) -> List[int]:
        return self.tokenizer.encode(
            str(text),
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding='max_length',
        )

    def _build_features(self):
        for ex in self.examples:
            lines = ex.lines[: self.max_lines] if self.max_lines else ex.lines
            if not lines:
                lines = ['']
            code_ids = [self._encode_text(line, self.code_block_size) for line in lines]

            taint_texts = self.taint_extractor.extract_for_skill(ex.skill_name)
            has_taint = len(taint_texts) > 0
            if has_taint:
                taint_texts = taint_texts[: self.max_taint_paths]
            else:
                # 占位；编码阶段根据 has_taint=False 使用零向量，不使用该文本语义。
                taint_texts = ['']
            taint_ids = [self._encode_text(text, self.taint_block_size) for text in taint_texts]

            self.features.append({
                'code_input_ids': torch.tensor(code_ids, dtype=torch.long),
                'taint_input_ids': torch.tensor(taint_ids, dtype=torch.long),
                'has_taint': bool(has_taint),
                'label': torch.tensor(float(ex.label), dtype=torch.float),
                'url': ex.url,
                'skill_name': ex.skill_name,
                'taint_path_count': 0 if not has_taint else len(taint_texts),
            })

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        item = self.features[idx]
        meta = {
            'url': item['url'],
            'skill_name': item['skill_name'],
            'has_taint': item['has_taint'],
            'taint_path_count': item['taint_path_count'],
        }
        return item['code_input_ids'], item['taint_input_ids'], item['has_taint'], item['label'], meta


def collate_skill_taint_batch(batch):
    code_input_ids_list = [item[0] for item in batch]
    taint_input_ids_list = [item[1] for item in batch]
    has_taint_list = [item[2] for item in batch]
    labels = torch.stack([item[3] for item in batch], dim=0)
    metas = [item[4] for item in batch]
    return code_input_ids_list, taint_input_ids_list, has_taint_list, labels, metas
