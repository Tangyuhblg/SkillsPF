# -*- coding: utf-8 -*-
"""
taint_feature_utils.py

从每个 Skill 的 taint_result.json 中提取 Source -> Sink 污点路径语义文本。
这些文本随后由 UniXCoder/CodeBERT 编码，并平均池化为 Skill 级污点路径语义向量。
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional


def normalize_path(path: str) -> str:
    return os.path.abspath(os.path.normpath(path))


def normalize_skill_key(name: str) -> str:
    """用于容错匹配目录名和 skill_name。"""
    name = str(name or '').strip().lower()
    name = re.sub(r'[^a-z0-9]+', '-', name)
    name = re.sub(r'-+', '-', name).strip('-')
    return name


def safe_read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None


def truncate_text(text: str, max_chars: int = 3000) -> str:
    text = str(text or '').strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + ' ... <TRUNCATED>'


def finding_to_taint_text(finding: dict, source_file: str = '', py_file_name: str = '') -> str:
    """将一条 finding 转换为可输入 UniXCoder 的半结构化文本。"""
    source = finding.get('source', '')
    sink = finding.get('sink', '')
    code = finding.get('code', '')
    line = finding.get('line', '')
    col = finding.get('col', '')
    path_labels = finding.get('path_labels', []) or []

    # 保留路径顺序，因为顺序表达 Source -> Sink 的传播语义。
    path_text = ' -> '.join(str(x) for x in path_labels)

    text = (
        'Taint flow in Python skill. '
        f'File: {py_file_name}. Source file: {source_file}. '
        f'Source: {source}. Sink: {sink}. '
        f'Sink location: line {line}, column {col}. '
        f'Sink code: {code}. '
        f'Propagation path: {path_text}.'
    )
    return truncate_text(text, max_chars=3000)


def extract_taint_texts_from_result_json(json_path: Path) -> List[str]:
    """从一个 taint_result.json 中提取所有 finding 的路径语义文本。"""
    payload = safe_read_json(json_path)
    if not isinstance(payload, dict):
        return []

    files = payload.get('files', {})
    if not isinstance(files, dict):
        return []

    taint_texts: List[str] = []
    for py_file_name, file_info in files.items():
        if not isinstance(file_info, dict):
            continue
        source_file = file_info.get('source_file', '')
        findings = file_info.get('findings', []) or []
        if not isinstance(findings, list):
            continue

        for finding in findings:
            if isinstance(finding, dict):
                taint_texts.append(
                    finding_to_taint_text(
                        finding=finding,
                        source_file=source_file,
                        py_file_name=str(py_file_name),
                    )
                )
    return taint_texts


class TaintTextExtractor:
    """
    根据 skill_name 查找对应 Skill 的 taint_result.json，并提取污点路径文本。

    默认优先查找：
        repo_root / skill_name / ** / taint_result.json

    若目录名与 skill_name 存在大小写、下划线、短横线差异，则启用规范化目录索引进行兜底匹配。
    """

    def __init__(
        self,
        repo_root: str = '/root/BATaint/datasets/repo',
        taint_result_name: str = 'taint_result.json',
        max_taint_paths: int = 32,
        enable_cache: bool = True,
        enable_dir_index: bool = True,
    ):
        self.repo_root = Path(normalize_path(repo_root))
        self.taint_result_name = taint_result_name
        self.max_taint_paths = max_taint_paths
        self.enable_cache = enable_cache
        self.enable_dir_index = enable_dir_index
        self._cache: Dict[str, List[str]] = {}
        self._dir_index: Optional[Dict[str, List[Path]]] = None

    def _build_dir_index(self) -> Dict[str, List[Path]]:
        index: Dict[str, List[Path]] = {}
        if not self.repo_root.exists():
            return index

        skip_dirs = {'.git', '__pycache__', '.venv', 'venv', 'env', 'node_modules', 'dist', 'build'}
        for root, dirs, files in os.walk(self.repo_root):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            root_path = Path(root)
            key = normalize_skill_key(root_path.name)
            if key:
                index.setdefault(key, []).append(root_path)
        return index

    def _get_dir_index(self) -> Dict[str, List[Path]]:
        if self._dir_index is None:
            self._dir_index = self._build_dir_index()
        return self._dir_index

    def candidate_skill_dirs(self, skill_name: str) -> List[Path]:
        skill_name = str(skill_name)
        candidates: List[Path] = []

        direct = self.repo_root / skill_name
        if direct.exists() and direct.is_dir():
            candidates.append(direct)

        # 数字目录常见：007、0007 等。直接路径不存在时尝试去掉/补齐前导 0。
        if skill_name.isdigit():
            variants = {skill_name.lstrip('0') or '0', skill_name.zfill(3), skill_name.zfill(4), skill_name.zfill(6)}
            for v in variants:
                p = self.repo_root / v
                if p.exists() and p.is_dir() and p not in candidates:
                    candidates.append(p)

        if self.enable_dir_index:
            key = normalize_skill_key(skill_name)
            for p in self._get_dir_index().get(key, []):
                if p not in candidates:
                    candidates.append(p)

        return candidates

    def find_taint_result_files(self, skill_name: str) -> List[Path]:
        result_files: List[Path] = []
        seen = set()
        skip_dirs = {'.git', '__pycache__', '.venv', 'venv', 'env', 'node_modules', 'dist', 'build'}

        for skill_dir in self.candidate_skill_dirs(skill_name):
            for root, dirs, files in os.walk(skill_dir):
                dirs[:] = [d for d in dirs if d not in skip_dirs]
                if self.taint_result_name in files:
                    p = Path(root) / self.taint_result_name
                    if p not in seen:
                        seen.add(p)
                        result_files.append(p)

        return sorted(result_files)

    def extract_for_skill(self, skill_name: str) -> List[str]:
        skill_name = str(skill_name)
        if self.enable_cache and skill_name in self._cache:
            return self._cache[skill_name]

        texts: List[str] = []
        for json_path in self.find_taint_result_files(skill_name):
            texts.extend(extract_taint_texts_from_result_json(json_path))

        # 去重且保持顺序
        seen = set()
        unique_texts: List[str] = []
        for text in texts:
            if text not in seen:
                seen.add(text)
                unique_texts.append(text)

        if self.max_taint_paths and len(unique_texts) > self.max_taint_paths:
            unique_texts = unique_texts[: self.max_taint_paths]

        if self.enable_cache:
            self._cache[skill_name] = unique_texts
        return unique_texts


def build_taint_text_summary(repo_root: str, skill_names: List[str], taint_result_name: str = 'taint_result.json') -> Dict:
    extractor = TaintTextExtractor(repo_root=repo_root, taint_result_name=taint_result_name)
    rows = []
    for skill_name in skill_names:
        texts = extractor.extract_for_skill(skill_name)
        rows.append({'skill_name': skill_name, 'taint_path_count': len(texts)})

    total = len(rows)
    with_taint = sum(1 for r in rows if r['taint_path_count'] > 0)
    return {
        'total_skill_count': total,
        'skills_with_taint_paths': with_taint,
        'skills_without_taint_paths': total - with_taint,
        'details': rows,
    }
